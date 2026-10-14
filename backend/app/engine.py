"""Tiled super-resolution with ONNX Runtime, plus optional GFPGAN face restoration.

Everything runs on ONNX Runtime (CPU, CUDA or CoreML). No PyTorch is needed:
Real-ESRGAN graphs are built from the official checkpoints at start-up, and
GFPGAN uses a published ONNX export.
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

import numpy as np
import onnxruntime as ort

from .config import settings
from .onnx_models import blend, build_rrdb, build_srvgg, load_pth

log = logging.getLogger("enhancer.engine")

# per model: context pixels around each tile, and the feathered overlap band
TILE_CONTEXT = {"general": (16, 8), "deep": (32, 12)}


def _providers() -> list[str]:
    avail = ort.get_available_providers()
    order = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in avail] or avail


def _ramp(n: int, lead: int, trail: int) -> np.ndarray:
    r = np.ones(n, np.float32)
    if lead:
        r[:lead] = np.minimum(r[:lead], np.arange(1, lead + 1) / (lead + 1))
    if trail:
        r[n - trail:] = np.minimum(r[n - trail:], np.arange(trail, 0, -1) / (trail + 1))
    return r


def tiled(run, rgb: np.ndarray, scale: int, tile: int, pad: int, feather: int) -> np.ndarray:
    """Runs `run` (NCHW float in, NCHW float out at `scale`) over feathered, padded tiles.

    Neighbouring tiles overlap by 2 * feather input pixels and are blended with
    linear ramps; each also gets `pad` pixels of context. Returns HxWx3 float.
    """
    h, w = rgb.shape[:2]
    x = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    acc = np.zeros((h * scale, w * scale, 3), np.float32)
    wsum = np.zeros((h * scale, w * scale, 1), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            ey0 = max(0, y0 - feather) if y0 > 0 else 0
            ex0 = max(0, x0 - feather) if x0 > 0 else 0
            ey1 = min(h, y1 + feather) if y1 < h else h
            ex1 = min(w, x1 + feather) if x1 < w else w
            py0, px0, py1, px1 = max(0, ey0 - pad), max(0, ex0 - pad), min(h, ey1 + pad), min(w, ex1 + pad)
            o = run(x[:, :, py0:py1, px0:px1])[0].transpose(1, 2, 0)
            oy, ox, hh, ww = (ey0 - py0) * scale, (ex0 - px0) * scale, (ey1 - ey0) * scale, (ex1 - ex0) * scale
            lead_y = (y0 - ey0 + min(feather, y1 - y0)) * scale if y0 > 0 else 0
            trail_y = (ey1 - y1 + min(feather, y1 - y0)) * scale if y1 < h else 0
            lead_x = (x0 - ex0 + min(feather, x1 - x0)) * scale if x0 > 0 else 0
            trail_x = (ex1 - x1 + min(feather, x1 - x0)) * scale if x1 < w else 0
            wt = (_ramp(hh, lead_y, trail_y)[:, None] * _ramp(ww, lead_x, trail_x)[None, :])[..., None]
            acc[ey0 * scale:ey1 * scale, ex0 * scale:ex1 * scale] += o[oy:oy + hh, ox:ox + ww] * wt
            wsum[ey0 * scale:ey1 * scale, ex0 * scale:ex1 * scale] += wt
    out = acc / np.maximum(wsum, 1e-8)
    if not np.isfinite(out).all():
        raise RuntimeError("model produced invalid pixels")
    return out


class Engine:
    """Holds checkpoints in memory and builds ONNX sessions on demand."""

    def __init__(self, weights_dir: str):
        self.dir = weights_dir
        self.general = load_pth(os.path.join(weights_dir, "realesr-general-x4v3.pth"))
        self.general_wdn = load_pth(os.path.join(weights_dir, "realesr-general-wdn-x4v3.pth"))
        x4p = os.path.join(weights_dir, "RealESRGAN_x4plus.pth")
        self.x4plus = load_pth(x4p) if os.path.exists(x4p) else None
        self.providers = _providers()
        self._lock = threading.Lock()
        self.face = FaceRestorer.try_load(weights_dir, self.providers)
        log.info("engine ready: providers=%s x4plus=%s gfpgan=%s", self.providers, self.x4plus is not None, self.face is not None)

    @property
    def model_name(self) -> str:
        return "Real-ESRGAN x4plus + general v3" if self.x4plus is not None else "Real-ESRGAN general v3"

    @lru_cache(maxsize=4)
    def session(self, kind: str, dni: float) -> ort.InferenceSession:
        if kind in ("deep", "x4plus"):
            if self.x4plus is None:
                raise FileNotFoundError("RealESRGAN_x4plus.pth is not installed")
            proto = build_rrdb(self.x4plus)
        else:
            proto = build_srvgg(blend(self.general, self.general_wdn, dni))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(proto.SerializeToString(), opts, providers=self.providers)

    def upscale(self, rgb: np.ndarray, denoise: float = 0.5, model: str = "general", tile: int = settings.tile) -> np.ndarray:
        """rgb: HxWx3 uint8. Returns the 4x result as HxWx3 uint8."""
        kind = "deep" if model in ("deep", "x4plus") else "general"
        dni = round(float(np.clip(denoise, 0, 1)) * 20) / 20
        sess = self.session(kind, 0.0 if kind == "deep" else dni)
        pad, feather = TILE_CONTEXT[kind]
        if kind == "deep":
            tile = min(tile, 192)
        with self._lock:  # one inference at a time per process keeps memory bounded
            out = tiled(lambda t: sess.run(None, {"input": t})[0], rgb, 4, tile, pad, feather)
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# Five-point template for a 512 x 512 aligned face (FFHQ alignment used by GFPGAN):
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
FFHQ_512 = np.array([
    [192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935],
    [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares similarity transform (Umeyama) mapping src points onto dst, as a 2x3 matrix."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    var = (s0 ** 2).sum() / len(src)
    if var < 1e-6:
        return None
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    c = (S * np.diag(D)).sum() / var
    t = md - c * R @ ms
    return np.hstack([c * R, t[:, None]]).astype(np.float32)


class FaceRestorer:
    """GFPGAN v1.4 through ONNX Runtime.

    Generative face restoration can change likeness. The restored face is
    blended with the identity-preserving super-resolution result by `strength`,
    inside a feathered mask, and only when the client asks for it.
    """

    def __init__(self, sess: ort.InferenceSession):
        self.sess = sess
        self.inp = sess.get_inputs()[0].name

    @classmethod
    def try_load(cls, weights_dir: str, providers: list[str]):
        path = os.path.join(weights_dir, "gfpgan_1.4.onnx")
        if not settings.enable_gfpgan or not os.path.exists(path):
            return None
        try:
            import cv2  # noqa: F401
        except Exception as e:  # pragma: no cover
            log.warning("GFPGAN needs opencv-python-headless: %s", e)
            return None
        return cls(ort.InferenceSession(path, providers=providers))

    def restore(self, rgb: np.ndarray, faces: list[np.ndarray], strength: float) -> tuple[np.ndarray, int]:
        import cv2
        out = rgb.astype(np.float32)
        h, w = rgb.shape[:2]
        done = 0
        for lm in faces:
            lm = np.asarray(lm, np.float32).reshape(5, 2)
            M = similarity(lm, FFHQ_512)
            if M is None:
                continue
            crop = cv2.warpAffine(rgb, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            x = (crop.astype(np.float32) / 255.0 - 0.5) / 0.5
            y = self.sess.run(None, {self.inp: x.transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
            restored = np.clip((y * 0.5 + 0.5) * 255.0, 0, 255)
            inv = cv2.invertAffineTransform(M)
            back = cv2.warpAffine(restored, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = np.zeros((512, 512), np.float32)
            cv2.ellipse(mask, (256, 290), (190, 230), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), 24)
            m = cv2.warpAffine(mask, inv, (w, h), flags=cv2.INTER_LINEAR)[..., None] * float(np.clip(strength, 0, 1))
            out = out * (1 - m) + back * m
            done += 1
        return np.clip(out + 0.5, 0, 255).astype(np.uint8), done


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine(settings.weights_dir)
    return _engine
