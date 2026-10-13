"""Tiled super-resolution with ONNX Runtime, plus optional GFPGAN."""
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


def _providers() -> list[str]:
    avail = ort.get_available_providers()
    order = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in avail] or avail


class Engine:
    """Holds checkpoints in memory and builds ONNX sessions per denoise level."""

    def __init__(self, weights_dir: str):
        self.dir = weights_dir
        self.general = load_pth(os.path.join(weights_dir, "realesr-general-x4v3.pth"))
        self.general_wdn = load_pth(os.path.join(weights_dir, "realesr-general-wdn-x4v3.pth"))
        x4p = os.path.join(weights_dir, "RealESRGAN_x4plus.pth")
        self.x4plus = load_pth(x4p) if os.path.exists(x4p) else None
        self.providers = _providers()
        self._lock = threading.Lock()
        self.face = FaceRestorer.try_load(weights_dir)
        log.info("engine ready: providers=%s x4plus=%s gfpgan=%s", self.providers, self.x4plus is not None, self.face is not None)

    @property
    def model_name(self) -> str:
        return "Real-ESRGAN x4plus + general-x4v3" if self.x4plus is not None else "Real-ESRGAN general-x4v3"

    @lru_cache(maxsize=4)
    def session(self, kind: str, dni: float) -> ort.InferenceSession:
        if kind == "x4plus":
            proto = build_rrdb(self.x4plus)
        else:
            proto = build_srvgg(blend(self.general, self.general_wdn, dni))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(proto.SerializeToString(), opts, providers=self.providers)

    def upscale(self, rgb: np.ndarray, denoise: float, tile: int = settings.tile, pad: int = 24) -> np.ndarray:
        """rgb: HxWx3 uint8. Returns 4x HxWx3 uint8.

        Low denoise on photos uses the larger x4plus model when available;
        otherwise the general model with the requested denoise blend.
        """
        dni = round(float(np.clip(denoise, 0, 1)) * 20) / 20
        kind = "x4plus" if self.x4plus is not None and dni >= 0.5 else "general"
        sess = self.session(kind, 0.0 if kind == "x4plus" else dni)
        h, w = rgb.shape[:2]
        x = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = np.zeros((h * 4, w * 4, 3), np.float32)
        with self._lock:  # one inference at a time per process keeps memory bounded
            for y0 in range(0, h, tile):
                for x0 in range(0, w, tile):
                    y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                    py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
                    py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
                    o = sess.run(None, {"input": x[:, :, py0:py1, px0:px1]})[0][0].transpose(1, 2, 0)
                    oy, ox = (y0 - py0) * 4, (x0 - px0) * 4
                    out[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = o[oy:oy + (y1 - y0) * 4, ox:ox + (x1 - x0) * 4]
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


class FaceRestorer:
    """Optional GFPGAN v1.4. Needs `pip install gfpgan` (and PyTorch).

    Generative face restoration can change likeness, so results are blended
    back with the identity-preserving upscale by `weight` (default 0.5) and
    the feature is off unless the client asks for it.
    """

    def __init__(self, restorer):
        self.r = restorer

    @classmethod
    def try_load(cls, weights_dir: str):
        path = os.path.join(weights_dir, "GFPGANv1.4.pth")
        if not settings.enable_gfpgan or not os.path.exists(path):
            return None
        try:
            from gfpgan import GFPGANer  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            log.warning("GFPGAN unavailable: %s", e)
            return None
        return cls(GFPGANer(model_path=path, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None))

    def restore(self, rgb: np.ndarray, weight: float) -> np.ndarray:  # pragma: no cover - needs torch
        bgr = rgb[:, :, ::-1].copy()
        _, _, out = self.r.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
        restored = out[:, :, ::-1]
        w = float(np.clip(weight, 0, 1)) * 0.6  # never fully replace the original face
        return (rgb * (1 - w) + restored * w).astype(np.uint8)


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine(settings.weights_dir)
    return _engine
