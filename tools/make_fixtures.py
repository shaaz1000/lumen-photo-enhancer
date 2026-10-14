"""Synthetic test images for the analysis tests (no third-party photos)."""
import io, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

rng = np.random.default_rng(3)
W = H = 256
im = Image.new("RGB", (W, H), (200, 190, 170))
d = ImageDraw.Draw(im)
for _ in range(40):
    x, y = rng.integers(0, W, 2); r = int(rng.integers(6, 40))
    c = tuple(int(v) for v in rng.integers(20, 235, 3))
    (d.ellipse if rng.random() < 0.5 else d.rectangle)([x - r, y - r, x + r, y + r], fill=c)
for x in range(0, W, 5):
    d.line([x, 200, x + 20, 256], fill=(60, 50, 40), width=1)
for i in range(6):
    d.text((20, 20 + i * 14), "Ferry 07:45 North Pier", fill=(10, 10, 10))
gy = np.linspace(80, 180, H)[:, None].repeat(W, 1)

def save(name, img, fmt_bytes=None):
    rgba = np.asarray(img.convert("RGBA"), np.uint8)
    open(f"tests/fixtures/{name}.rgba", "wb").write(rgba.tobytes())
    return {"w": img.width, "h": img.height, "bytes": fmt_bytes}

meta = {}
meta["sharp"] = save("sharp", im, 60000)
meta["blurry"] = save("blurry", im.filter(ImageFilter.GaussianBlur(3.5)), 60000)
meta["soft"] = save("soft", im.filter(ImageFilter.GaussianBlur(0.9)), 60000)
a = np.asarray(im, np.float32)
motion = sum(np.roll(a, k, axis=1) for k in range(-6, 7)) / 13
meta["motion"] = save("motion", Image.fromarray(motion.astype(np.uint8)), 60000)
buf = io.BytesIO(); im.save(buf, "JPEG", quality=12)
meta["jpeg"] = save("jpeg", Image.open(io.BytesIO(buf.getvalue())), len(buf.getvalue()))
noisy = np.clip(np.asarray(im, np.float32) + rng.normal(0, 14, (H, W, 3)), 0, 255).astype(np.uint8)
meta["noisy"] = save("noisy", Image.fromarray(noisy), 60000)
json.dump(meta, open("tests/fixtures/analysis_meta.json", "w"), indent=1)
print(meta)
