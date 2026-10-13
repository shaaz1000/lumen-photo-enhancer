"""Upload validation and abuse controls."""
from __future__ import annotations

import io
import threading
import time
import warnings

from PIL import Image

from .config import settings

Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression bomb guard; Pillow raises above 2x this
_SIGS = {b"\x89PNG\r\n\x1a\n": "PNG", b"\xff\xd8\xff": "JPEG"}


class Rejected(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def sniff(head: bytes) -> str | None:
    for sig, fmt in _SIGS.items():
        if head.startswith(sig):
            return fmt
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    return None


def decode(data: bytes) -> Image.Image:
    if not data:
        raise Rejected(400, "The upload was empty.")
    if len(data) > settings.max_upload_bytes:
        raise Rejected(413, f"Images must be under {settings.max_upload_bytes // 1048576} MB.")
    fmt = sniff(data[:16])
    if fmt is None:
        raise Rejected(415, "Only PNG, JPEG and WebP images are accepted.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            if im.format != fmt:
                raise Rejected(415, "The file contents do not match an accepted image format.")
            w, h = im.size
            if w * h > settings.max_input_pixels:
                raise Rejected(413, f"Images over {settings.max_input_pixels / 1e6:.0f} megapixels are not accepted.")
            if min(w, h) < 8 or max(w, h) / min(w, h) > 20:
                raise Rejected(422, "The image is too small or too long and thin to enhance.")
            im.load()
    except Rejected:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise Rejected(413, "The image is too large to decode safely.")
    except Exception:
        raise Rejected(422, "The image could not be decoded. It may be damaged.")
    from PIL import ImageOps
    return ImageOps.exif_transpose(im)


class RateLimiter:
    """Token bucket per client key. In-memory: use a shared store if you run several replicas."""

    def __init__(self, per_minute: int, burst: int):
        self.rate = per_minute / 60.0
        self.burst = burst
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            tokens, last = self.buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self.buckets[key] = (tokens - 1 if ok else tokens, now)
            if len(self.buckets) > 10_000:  # drop stale entries
                cutoff = now - 3600
                self.buckets = {k: v for k, v in self.buckets.items() if v[1] > cutoff}
            return ok
