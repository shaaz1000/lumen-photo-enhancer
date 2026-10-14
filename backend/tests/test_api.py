"""API tests. Run from backend/: `pytest -q`.

The API tests use the stub provider (bicubic) so they run anywhere; the model
tests check the ONNX graphs against a NumPy reference and run real Real-ESRGAN
on CPU when the weights are present.
"""
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("ENHANCER_PROVIDER", "stub")
os.environ.setdefault("ENHANCER_RATE_PER_MINUTE", "600")
os.environ.setdefault("ENHANCER_RATE_BURST", "50")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from app.guard import RateLimiter, sniff  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights")


def png(w=64, h=48, mode="RGB", fmt="PNG", **kw):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (h, w, 4 if mode == "RGBA" else 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, fmt, **kw)
    return buf.getvalue()


def post(data, name="a.png", **form):
    return client.post("/api/enhance", files={"image": (name, data, "application/octet-stream")}, data={k: str(v) for k, v in form.items()})


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("scale", [2, 4])
def test_enhance_scales(scale):
    r = post(png(40, 30), scale=scale)
    assert r.status_code == 200, r.text
    out = Image.open(io.BytesIO(r.content))
    assert out.size == (40 * scale, 30 * scale)


def test_jpeg_and_webp_inputs():
    assert post(png(fmt="JPEG", quality=20), "a.jpg").status_code == 200
    assert post(png(fmt="WEBP", quality=30), "a.webp").status_code == 200


def test_alpha_is_preserved():
    r = post(png(20, 20, mode="RGBA"))
    assert Image.open(io.BytesIO(r.content)).mode == "RGBA"


def test_extreme_aspect_ratio_rejected():
    assert post(png(400, 10)).status_code == 422


def test_unsupported_and_corrupt_files():
    assert post(b"GIF89a" + b"\0" * 100, "a.gif").status_code == 415
    assert post(b"\x89PNG\r\n\x1a\n" + b"garbage" * 20).status_code == 422
    assert post(b"").status_code == 400


def test_disguised_file_rejected():
    # JPEG bytes claiming to be PNG by name still work; random bytes named .png do not
    assert post(png(fmt="JPEG"), "photo.png").status_code == 200
    assert post(os.urandom(500), "photo.png").status_code == 415


def test_bad_parameters():
    assert post(png(), scale=8).status_code == 422
    assert post(png(), denoise=2).status_code == 422


def test_output_cap():
    from app import main
    object.__setattr__(main.settings, "max_output_pixels", 1000)
    try:
        assert post(png(40, 30), scale=4).status_code == 413
    finally:
        object.__setattr__(main.settings, "max_output_pixels", 36_000_000)


def test_rate_limiter_token_bucket():
    rl = RateLimiter(per_minute=6, burst=2)
    assert rl.allow("a", 0) and rl.allow("a", 0)
    assert not rl.allow("a", 0)
    assert rl.allow("b", 0)  # separate clients
    assert rl.allow("a", 10.5)  # one token refills every 10 s


def test_sniff():
    assert sniff(b"\xff\xd8\xff\xe0") == "JPEG"
    assert sniff(b"RIFF\0\0\0\0WEBPVP8 ") == "WEBP"
    assert sniff(b"<svg") is None


# ---------------- model graphs ----------------

def _conv(x, w, b):
    C, H, W = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    o = np.zeros((w.shape[0], H, W), np.float64)
    for i in range(3):
        for j in range(3):
            o += np.einsum("oc,chw->ohw", w[:, :, i, j], p[:, i:i + H, j:j + W])
    return o + b[:, None, None]


def _lrelu(x):
    return np.where(x >= 0, x, 0.2 * x)


def _up2(x):
    return x.repeat(2, 1).repeat(2, 2)


def _ref_rrdb(sd, x):
    c = lambda t, k: _conv(t, sd[k + ".weight"], sd[k + ".bias"])  # noqa: E731

    def rdb(t, k):
        fs = [t]
        for j in range(1, 5):
            fs.append(_lrelu(c(np.concatenate(fs), f"{k}.conv{j}")))
        return c(np.concatenate(fs), f"{k}.conv5") * 0.2 + t

    feat = c(x, "conv_first")
    t = feat
    for i in range(len({k.split(".")[1] for k in sd if k.startswith("body.")})):
        t = rdb(rdb(rdb(t, f"body.{i}.rdb1"), f"body.{i}.rdb2"), f"body.{i}.rdb3") * 0.2 + t
    feat = feat + c(t, "conv_body")
    feat = _lrelu(c(_up2(feat), "conv_up1"))
    feat = _lrelu(c(_up2(feat), "conv_up2"))
    return c(_lrelu(c(feat, "conv_hr")), "conv_last")


def _tiny_rrdb(rng, nf=8, gc=4, blocks=2):
    sd = {}
    def add(k, o, i):
        sd[k + ".weight"] = rng.normal(0, 0.1, (o, i, 3, 3)).astype(np.float32)
        sd[k + ".bias"] = rng.normal(0, 0.01, o).astype(np.float32)
    add("conv_first", nf, 3)
    for b in range(blocks):
        for r in (1, 2, 3):
            for j in range(1, 5):
                add(f"body.{b}.rdb{r}.conv{j}", gc, nf + (j - 1) * gc)
            add(f"body.{b}.rdb{r}.conv5", nf, nf + 4 * gc)
    for k in ("conv_body", "conv_up1", "conv_up2", "conv_hr"):
        add(k, nf, nf)
    add("conv_last", 3, nf)
    return sd


def test_rrdb_graph_matches_numpy_reference():
    import onnxruntime as ort
    from app.onnx_models import build_rrdb
    rng = np.random.default_rng(1)
    sd = _tiny_rrdb(rng)
    x = rng.random((1, 3, 6, 7)).astype(np.float32)
    sess = ort.InferenceSession(build_rrdb(sd).SerializeToString(), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x})[0][0]
    want = _ref_rrdb(sd, x[0].astype(np.float64))
    assert got.shape == (3, 24, 28)
    assert np.abs(got - want).max() < 1e-3


def test_srvgg_graph_matches_reference_fixture():
    path = os.path.join(WEIGHTS, "realesr-general-x4v3.pth")
    fx = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "srvgg_ref.json")
    if not (os.path.exists(path) and os.path.exists(fx)):
        pytest.skip("weights not downloaded")
    import json
    import onnxruntime as ort
    from app.onnx_models import build_srvgg, load_pth
    ref = json.load(open(fx))
    sd = {k: v.astype(np.float16).astype(np.float32) for k, v in load_pth(path).items()}
    sess = ort.InferenceSession(build_srvgg(sd).SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.array(ref["input_hwc"], np.float32).reshape(ref["h"], ref["w"], 3).transpose(2, 0, 1)[None]
    got = sess.run(None, {"input": x})[0][0].transpose(1, 2, 0).ravel()
    assert np.abs(got - np.array(ref["output_hwc"])).max() < 1e-3


def test_real_engine_end_to_end():
    if not os.path.exists(os.path.join(WEIGHTS, "realesr-general-wdn-x4v3.pth")):
        pytest.skip("weights not downloaded")
    from app.engine import Engine
    e = Engine(WEIGHTS)
    rgb = (np.random.default_rng(2).random((20, 33, 3)) * 255).astype(np.uint8)
    out = e.upscale(rgb, denoise=0.3, tile=16)  # tiles and seams exercised
    assert out.shape == (80, 132, 3) and out.dtype == np.uint8
    whole = e.upscale(rgb, denoise=0.3, tile=64)
    assert np.abs(out.astype(int) - whole.astype(int)).max() <= 2


def test_weak_texture_source_is_stacked():
    r = post(png(40, 30), scale=2, weak=1)
    assert r.status_code == 200, r.text
    assert r.headers.get("x-lumen-stack") == "2"
    assert Image.open(io.BytesIO(r.content)).size == (80, 120)


def test_bad_model_and_landmarks_rejected():
    assert post(png(), model="huge").status_code == 422
    assert post(png(), landmarks="[[1,2,3]]").status_code == 422
    assert post(png(), landmarks="not json").status_code == 422


def test_info_header_reports_processing():
    from urllib.parse import unquote
    r = post(png(), scale=2)
    assert "Stub provider" in unquote(r.headers["x-lumen-info"])


def test_feathered_tiles_match_whole_image():
    from app.engine import tiled
    rng = np.random.default_rng(5)
    img = (rng.random((23, 37, 3)) * 255).astype(np.uint8)
    # a local operator with a 3x3 footprint, so 8 px of context is plenty
    def run(t):
        p = np.pad(t, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
        y = sum(p[:, :, i:i + t.shape[2], j:j + t.shape[3]] for i in range(3) for j in range(3)) / 9
        return y.repeat(2, 2).repeat(2, 3)
    a = tiled(run, img, 2, 8, 8, 4)
    b = tiled(run, img, 2, 64, 8, 4)
    assert np.abs(a - b).max() < 1e-5


def test_deep_model_end_to_end():
    if not os.path.exists(os.path.join(WEIGHTS, "RealESRGAN_x4plus.pth")):
        pytest.skip("weights not downloaded")
    from app.engine import Engine
    e = Engine(WEIGHTS)
    rgb = (np.random.default_rng(3).random((14, 18, 3)) * 255).astype(np.uint8)
    out = e.upscale(rgb, 0.5, "deep", tile=8)
    whole = e.upscale(rgb, 0.5, "deep", tile=64)
    assert out.shape == (56, 72, 3)
    assert np.abs(out.astype(int) - whole.astype(int)).max() <= 6


def test_face_alignment_similarity_fit():
    from app.engine import FFHQ_512, similarity
    th = 0.25
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = (FFHQ_512 - 256) @ R.T * 0.3 + [120, 90]
    M = similarity(src, FFHQ_512)
    assert np.abs(src @ M[:, :2].T + M[:, 2] - FFHQ_512).max() < 1e-3
    assert similarity(np.zeros((5, 2)), FFHQ_512) is None


def test_gfpgan_restores_only_inside_the_face():
    path = os.path.join(WEIGHTS, "gfpgan_1.4.onnx")
    if not os.path.exists(path):
        pytest.skip("GFPGAN not downloaded")
    import onnxruntime as ort
    from app.engine import FFHQ_512, FaceRestorer
    fr = FaceRestorer(ort.InferenceSession(path, providers=["CPUExecutionProvider"]))
    img = np.full((600, 600, 3), 128, np.uint8)
    lm = FFHQ_512 * 0.5 + [150, 150]   # a face about 256 px wide in the middle
    out, n = fr.restore(img, [lm], 1.0)
    assert n == 1 and out.shape == img.shape
    assert np.abs(out[:20, :20].astype(int) - 128).max() == 0          # corners untouched
    assert np.abs(out[250:350, 250:350].astype(int) - 128).mean() > 1  # face area changed
