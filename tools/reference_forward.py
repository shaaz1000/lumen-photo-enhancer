"""Reference SRVGGNetCompact forward pass in NumPy using the ORIGINAL PyTorch
weight layout and torch.pixel_shuffle semantics. Used to prove the browser
engine (TF.js, reordered channels) produces the same numbers.

Writes tests/fixtures/srvgg_ref.json with input and expected output.
"""
import json, sys
import numpy as np
sys.path.insert(0, "tools")
from convert_weights import load_pth


def conv(x, w, b):  # x [C,H,W], w [O,C,3,3]
    C, H, W = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((w.shape[0], H, W), np.float64)
    for i in range(3):
        for j in range(3):
            out += np.einsum("oc,chw->ohw", w[:, :, i, j], p[:, i:i + H, j:j + W])
    return out + b[:, None, None]


def prelu(x, a):
    return np.where(x >= 0, x, x * a[:, None, None])


def pixel_shuffle(x, r):
    C, H, W = x.shape
    c = C // (r * r)
    return x.reshape(c, r, r, H, W).transpose(0, 3, 1, 4, 2).reshape(c, H * r, W * r)


def forward(sd, x):
    keys = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.")})
    y = x.astype(np.float64)
    for i in keys:
        w = sd[f"body.{i}.weight"].astype(np.float64)
        y = conv(y, w, sd[f"body.{i}.bias"].astype(np.float64)) if w.ndim == 4 else prelu(y, w)
    y = pixel_shuffle(y, 4)
    base = x.repeat(4, axis=1).repeat(4, axis=2)
    return y + base


if __name__ == "__main__":
    sd = load_pth(sys.argv[1])
    # round weights through fp16 like the shipped file does
    sd = {k: v.astype(np.float16).astype(np.float32) for k, v in sd.items()}
    rng = np.random.default_rng(7)
    H, W = 9, 11
    x = rng.random((3, H, W)).astype(np.float32)
    y = forward(sd, x)
    json.dump({"h": H, "w": W,
               "input_hwc": x.transpose(1, 2, 0).ravel().tolist(),
               "output_hwc": y.transpose(1, 2, 0).ravel().tolist()},
              open(sys.argv[2], "w"))
    print("ok", y.shape, float(y.mean()))
