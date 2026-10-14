"""Scores benchmark outputs against the clean references and builds comparison crops.

Metrics (luminance):
  PSNR, SSIM   fidelity to the reference; they reward smooth guesses, so they
               are read together with the detail metrics, never alone
  LPIPS        perceptual distance, when the `lpips` package (PyTorch) is installed
  detail       gradient energy of the result (at reference size) relative to the
               reference; 1.0 is reference-level edge detail, below 1 is blurrier,
               far above 1 is oversharpened or noisy
  blur         Crete-Roffet no-reference blur estimate (0 sharp .. 1 blurry)
Identity distance (from run.json) is face-api's 128-d descriptor distance to the
reference face; below 0.6 is treated as the same person by that model.

Writes bench/out/report.json, bench/out/report.md and crops in bench/out/crops/.
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CASES = os.path.join(OUT, "cases")

try:
    import lpips, torch  # type: ignore
    _lp = lpips.LPIPS(net="alex", verbose=False)
except Exception:
    _lp = None


def Y(a):
    a = np.asarray(a, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def grad_energy(y):
    gx, gy = np.diff(y, axis=1)[:-1], np.diff(y, axis=0)[:, :-1]
    return float(np.mean(np.hypot(gx, gy)))


def blur_metric(y):
    from scipy.ndimage import uniform_filter1d
    out = []
    for ax in (0, 1):
        b = uniform_filter1d(y, 9, axis=ax)
        d = np.abs(np.diff(y, axis=ax)); db = np.abs(np.diff(b, axis=ax))
        sd = d.sum(); out.append((sd - np.maximum(0, d - db).sum()) / sd if sd else 1)
    return float(max(out))


def to_compare(out_im, gt, how):
    if out_im.size == gt.size:
        return out_im
    return out_im.resize(gt.size, Image.BOX if how == "downscale" else Image.LANCZOS)


def lp(a, b):
    if _lp is None:
        return None
    t = lambda im: torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]  # noqa: E731
    with torch.no_grad():
        return float(_lp(t(a), t(b)))


def main():
    cases = {c["name"]: c for c in json.load(open(os.path.join(CASES, "cases.json")))}
    run = json.load(open(os.path.join(OUT, "run.json")))
    report = []
    os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
    for r in run:
        c = cases[r["case"]]
        d = os.path.join(CASES, c["name"])
        gt = Image.open(os.path.join(d, "gt.png")).convert("RGB")
        gy = Y(gt)
        ge = grad_energy(gy)
        rows = []
        for m in r["methods"]:
            if "error" in m:
                rows.append({"id": m["id"], "label": m["label"], "error": m["error"]}); continue
            im = Image.open(os.path.join(d, f"{m['id']}.png")).convert("RGB")
            cmp = to_compare(im, gt, c["compare"])
            y = Y(cmp)
            rows.append({
                "id": m["id"], "label": m["label"],
                "psnr": round(psnr(gy, y, data_range=255), 2), "ssim": round(ssim(gy, y, data_range=255), 4),
                "lpips": (round(lp(cmp, gt), 4) if _lp else None),
                "detail": round(grad_energy(y) / ge, 3),
                "blur": round(blur_metric(Y(im)), 3),
                "identity": None if m.get("identityDistance") is None else round(m["identityDistance"], 3),
                "seconds": round(m["ms"] / 1000, 1), "models": m.get("models", []),
            })
        report.append({"case": c["name"], "note": c["note"], "analysis": r["analysis"], "faces": r["faces"], "rows": rows,
                       "gt_blur": round(blur_metric(gy), 3)})
        crops(c, d, gt, [x for x in rows if "error" not in x])
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    md = ["| Case | Method | PSNR | SSIM | LPIPS | Detail | Blur | Identity | Time |", "|---|---|---|---|---|---|---|---|---|"]
    for r in report:
        for x in r["rows"]:
            if "error" in x:
                md.append(f"| {r['case']} | {x['label']} | failed: {x['error']} |||||||"); continue
            md.append(f"| {r['case']} | {x['label']} | {x['psnr']} | {x['ssim']} | {x['lpips'] if x['lpips'] is not None else 'n/a'} | {x['detail']} | {x['blur']} | {x['identity'] if x['identity'] is not None else ''} | {x['seconds']} s |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("LPIPS:", "computed" if _lp else "not available (install torch and lpips to enable)")


CROPS = {  # region of the reference to show, in reference pixels
    "portrait_compressed": (175, 45, 290, 165),
    "blurry_photo": (180, 60, 380, 200),
    "lowres_landscape": (230, 120, 420, 300),
    "social_screenshot": (130, 40, 330, 180),
    "noisy_compressed": (230, 120, 420, 300),
    "out_of_focus": (175, 45, 290, 165),
}


def crops(c, d, gt, rows, z=3):
    box = CROPS[c["name"]]
    tiles = []
    inp = Image.open(os.path.join(d, c["input"])).convert("RGB")
    sx = inp.width / gt.width
    show = [("Input", inp.crop(tuple(int(v * sx) for v in box)))]
    for x in rows:
        if x["id"] in ("bicubic", "v1_auto", "faithful", "deep", "max", "deep_gfpgan"):
            im = Image.open(os.path.join(d, f"{x['id']}.png")).convert("RGB")
            k = im.width / gt.width
            show.append((x["label"].replace("v1.1 ", "").replace("Deep Restoration", "Deep").replace(" (server)", ""), im.crop(tuple(int(v * k) for v in box))))
    show.append(("Reference", gt.crop(box)))
    W, H = (box[2] - box[0]) * z // 2, (box[3] - box[1]) * z // 2
    for label, im in show:
        t = Image.new("RGB", (W, H + 20), "white")
        t.paste(im.resize((W, H), Image.LANCZOS if im.width > W else Image.NEAREST), (0, 20))
        ImageDraw.Draw(t).text((5, 4), label, fill="black")
        tiles.append(t)
    cols = 4
    rows_n = (len(tiles) + cols - 1) // cols
    M = Image.new("RGB", (W * cols + 6 * (cols - 1), (H + 20) * rows_n + 6 * (rows_n - 1)), "white")
    for i, t in enumerate(tiles):
        M.paste(t, ((i % cols) * (W + 6), (i // cols) * (H + 26)))
    M.save(os.path.join(OUT, "crops", f"{c['name']}.png"))


if __name__ == "__main__":
    main()
