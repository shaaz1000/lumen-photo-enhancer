"""Builds the benchmark cases: clean references plus synthetic degradations.

All references ship with scikit-image and are public domain or CC0
(astronaut: NASA; coffee: Rachel Michetti, CC0; chelsea: Stefan van der Walt,
CC0; rocket: SpaceX, public domain). Degradations are deterministic.

Writes bench/out/cases/<case>/{input.png|input.jpg, gt.png} and cases.json.
"""
import io, json, os
import numpy as np
from PIL import Image, ImageFilter
import skimage.data as d

OUT = os.path.join(os.path.dirname(__file__), "out", "cases")
rng = np.random.default_rng(2026)


def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return b.getvalue()


def disk_blur(im, r):
    from scipy.ndimage import convolve
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (xx ** 2 + yy ** 2 <= r * r).astype(np.float32); k /= k.sum()
    a = np.asarray(im, np.float32)
    return Image.fromarray(np.clip(np.stack([convolve(a[..., c], k, mode="reflect") for c in range(3)], -1) + 0.5, 0, 255).astype(np.uint8))


def save(name, gt, data, ext, scale, note, compare):
    p = os.path.join(OUT, name); os.makedirs(p, exist_ok=True)
    gt.save(os.path.join(p, "gt.png"))
    with open(os.path.join(p, f"input.{ext}"), "wb") as f:
        f.write(data)
    inp = Image.open(io.BytesIO(data))
    return {"name": name, "input": f"input.{ext}", "scale": scale, "note": note, "compare": compare,
            "input_size": list(inp.size), "gt_size": list(gt.size)}


def png(im):
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


cases = []
astro = Image.fromarray(d.astronaut())
coffee = Image.fromarray(d.coffee())
chelsea = Image.fromarray(d.chelsea())
rocket = Image.fromarray(d.rocket())

# 1. 500 x 500 compressed portrait: 256 px JPEG q35 of the astronaut, 2x back to 512
cases.append(save("portrait_compressed", astro, jpeg(astro.resize((256, 256), Image.BICUBIC), 35), "jpg", 2,
                  "256 px portrait, JPEG quality 35, enhanced 2x", "gt"))
# 2. slightly blurry photo at full size: gaussian blur 1.6 px + JPEG q80, enhanced 2x, scored at original size
cases.append(save("blurry_photo", coffee, jpeg(coffee.filter(ImageFilter.GaussianBlur(1.6)), 80), "jpg", 2,
                  "full size photo with 1.6 px blur and JPEG 80, enhanced 2x, scored at original size", "downscale"))
# 3. low resolution landscape: quarter size, JPEG q85, 4x
cases.append(save("lowres_landscape", rocket, jpeg(rocket.resize((160, 107), Image.BICUBIC), 85), "jpg", 4,
                  "160 px landscape, JPEG 85, enhanced 4x", "gt"))
# 4. cropped social media screenshot: shared at 40% as JPEG q60, shown 1.6x on a phone, captured as PNG
shared = Image.open(io.BytesIO(jpeg(chelsea.resize((180, 120), Image.BICUBIC), 60))).convert("RGB")
shot = shared.resize((288, 192), Image.BILINEAR)
cases.append(save("social_screenshot", chelsea, png(shot), "png", 2,
                  "JPEG 60 at 40% size, enlarged 1.6x by a phone and screenshotted, enhanced 2x then compared at original size", "resize"))
# 5. noisy, compressed: half size, gaussian noise 8, JPEG q45, 2x
half = np.asarray(rocket.resize((320, 214), Image.BICUBIC), np.float32) + rng.normal(0, 8, (214, 320, 3))
cases.append(save("noisy_compressed", rocket.resize((640, 428), Image.BICUBIC), jpeg(Image.fromarray(np.clip(half, 0, 255).astype(np.uint8)), 45), "jpg", 2,
                  "half size, noise sigma 8, JPEG 45, enhanced 2x", "gt"))
# 6. genuinely out of focus: disk blur radius 5 at full size; analysis must flag it
cases.append(save("out_of_focus", astro, png(disk_blur(astro, 5)), "png", 2,
                  "full size, defocus (disk radius 5 px), enhanced 2x, scored at original size", "downscale"))

json.dump(cases, open(os.path.join(OUT, "cases.json"), "w"), indent=1)
for c in cases:
    print(c["name"], c["input_size"], "->", c["scale"], "x")

# raw RGBA copies for the Node harness (no image codecs needed there)
for c in cases:
    p = os.path.join(OUT, c["name"])
    for src, dst in ((c["input"], "input.rgba"), ("gt.png", "gt.rgba")):
        with open(os.path.join(p, dst), "wb") as f:
            f.write(np.asarray(Image.open(os.path.join(p, src)).convert("RGBA")).tobytes())
