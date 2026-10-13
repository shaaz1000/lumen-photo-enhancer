"""Lumen enhancement server.

Serves the same web app as the Claude artifact, plus /api/enhance, which runs
Real-ESRGAN (ONNX Runtime) and optional GFPGAN. Images are processed in memory
and never written to disk or logged.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from .config import settings
from .guard import RateLimiter, Rejected, decode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("enhancer")

app = FastAPI(title="Lumen enhancement server", docs_url=None, redoc_url=None)
limiter = RateLimiter(settings.rate_per_minute, settings.rate_burst)
slots = asyncio.Semaphore(settings.max_concurrent)


@app.middleware("http")
async def headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.exception_handler(Rejected)
async def rejected(_: Request, e: Rejected):
    return JSONResponse({"detail": e.detail}, status_code=e.status)


def client_key(request: Request) -> str:
    if settings.trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _engine():
    if settings.provider == "stub":
        return None
    from .engine import get_engine
    return get_engine()


@app.get("/api/health")
async def health():
    if settings.provider == "replicate":
        return {"status": "ok", "model": "Real-ESRGAN (hosted)", "face_restoration": False, "provider": "replicate"}
    if settings.provider == "stub":
        return {"status": "ok", "model": "stub (tests only, bicubic)", "face_restoration": False, "provider": "stub"}
    try:
        e = await run_in_threadpool(_engine)
    except FileNotFoundError:
        return JSONResponse({"status": "error", "detail": "Model weights are missing. Run scripts/download_weights.sh."}, status_code=503)
    return {"status": "ok", "model": e.model_name, "face_restoration": e.face is not None, "provider": "local"}


@app.post("/api/enhance")
async def enhance(
    request: Request,
    image: UploadFile = File(...),
    scale: int = Form(4),
    denoise: float = Form(0.5),
    face_restore: float = Form(0.0),
):
    if not limiter.allow(client_key(request)):
        raise Rejected(429, "Too many requests. Wait a minute and try again.")
    if scale not in (2, 4):
        raise Rejected(422, "Scale must be 2 or 4.")
    if not (0 <= denoise <= 1 and 0 <= face_restore <= 1):
        raise Rejected(422, "denoise and face_restore must be between 0 and 1.")
    data = await image.read(settings.max_upload_bytes + 1)
    im = decode(data)
    w, h = im.size
    if w * h * scale * scale > settings.max_output_pixels:
        raise Rejected(413, f"The result would exceed {settings.max_output_pixels / 1e6:.0f} megapixels. Choose 2x or a smaller image.")
    if slots.locked():
        raise Rejected(503, "The server is busy. Try again in a moment.")
    async with slots:
        try:
            out = await asyncio.wait_for(run_in_threadpool(_process, im, scale, denoise, face_restore), timeout=300)
        except asyncio.TimeoutError:
            raise Rejected(504, "Processing took too long. Try a smaller image.")
        except Rejected:
            raise
        except Exception:
            log.exception("inference failed")  # no image data is logged
            raise Rejected(500, "Enhancement failed on the server.")
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=False, compress_level=3)
    del im, out
    return Response(buf.getvalue(), media_type="image/png")


def _process(im: Image.Image, scale: int, denoise: float, face: float) -> Image.Image:
    alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") or "transparency" in im.info else None
    rgb = np.asarray(im.convert("RGB"))
    if settings.provider == "stub":
        up = np.asarray(Image.fromarray(rgb).resize((rgb.shape[1] * 4, rgb.shape[0] * 4), Image.BICUBIC))
    elif settings.provider == "replicate":
        up = _replicate(rgb)
    else:
        e = _engine()
        up = e.upscale(rgb, denoise)
        if face > 0 and e.face is not None:
            up = e.face.restore(up, face)
    out = Image.fromarray(up)
    if scale == 2:
        out = out.resize((rgb.shape[1] * 2, rgb.shape[0] * 2), Image.LANCZOS)
    if alpha is not None:
        out.putalpha(alpha.resize(out.size, Image.BICUBIC))
    return out


def _replicate(rgb: np.ndarray) -> np.ndarray:  # pragma: no cover - needs network and a key
    """Hosted inference. The API token stays on the server; the browser never sees it."""
    import httpx

    if not settings.replicate_token or not settings.replicate_version:
        raise Rejected(503, "Hosted inference is not configured.")
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    hdr = {"Authorization": f"Bearer {settings.replicate_token}", "Prefer": "wait=60"}
    with httpx.Client(timeout=120) as c:
        r = c.post("https://api.replicate.com/v1/predictions", headers=hdr,
                   json={"version": settings.replicate_version, "input": {"image": uri, "scale": 4, "face_enhance": False}})
        r.raise_for_status()
        pred = r.json()
        while pred.get("status") not in ("succeeded", "failed", "canceled"):
            import time
            time.sleep(1.5)
            pred = c.get(pred["urls"]["get"], headers=hdr).json()
        if pred["status"] != "succeeded":
            raise Rejected(502, "The hosted model failed.")
        url = pred["output"] if isinstance(pred["output"], str) else pred["output"][0]
        img = Image.open(io.BytesIO(c.get(url).content)).convert("RGB")
        # ask Replicate to delete the prediction's files promptly
        c.post(pred["urls"].get("cancel", pred["urls"]["get"]), headers=hdr)
    return np.asarray(img)


# ---------- the web app ----------
_SKELETON = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>{}</body></html>')

if os.path.isdir(settings.static_dir):
    _index = os.path.join(settings.static_dir, "index.html")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open(_index, encoding="utf-8") as f:
            return HTMLResponse(_SKELETON.format(f.read()))

    app.mount("/models", StaticFiles(directory=os.path.join(settings.static_dir, "models")), name="models")
else:
    @app.get("/")
    async def no_app():
        return JSONResponse({"detail": "Web app not built. Run `npm run build` and copy dist/ to backend/static."}, status_code=404)
