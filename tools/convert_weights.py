"""Convert official Real-ESRGAN .pth checkpoints into compact fp16 files the
browser engine loads with TensorFlow.js. Needs only NumPy (no PyTorch).

  python3 tools/convert_weights.py realesr-general-x4v3.pth public/models/general-x4v3.bin
  python3 tools/convert_weights.py RealESRGAN_x4plus.pth public/models/deep-x4plus.bin --split 12

File layout (little endian): magic "RESR", u32 version, u32 json_len, json
header, padding to 2 bytes, then fp16 data in header order.

Version 1 (SRVGGNetCompact): tensors in network order with TF layouts
  conv [kh, kw, in, out], bias [out], prelu [ch]. The last conv's output
  channels are permuted so tf.depthToSpace (NHWC, index (i*r + j)*C + c)
  matches torch.pixel_shuffle (c*r*r + i*r + j).
Version 2 (RRDBNet): tensors carry their PyTorch names; convs are [kh, kw, in, out].

--split N writes N MB parts named <stem>.partK.wasm (artifact hosting only
serves web file types; the loader fetches the parts and concatenates them).
"""
import argparse, io, json, os, pickle, struct, zipfile
from collections import OrderedDict
import numpy as np

DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16}


def load_pth(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, *a):
                    arr = storage[offset:]
                    if not size:
                        return arr[:1].reshape(())
                    st = [s * arr.itemsize for s in stride]
                    return np.lib.stride_tricks.as_strided(arr, size, st).copy()
                return rebuild
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, *a: data
            if mod == "torch" and name in DTYPES:
                return name
            if mod == "collections" and name == "OrderedDict":
                return OrderedDict
            raise pickle.UnpicklingError(f"refusing to load {mod}.{name}")

        def persistent_load(self, pid):
            _, dtype, key, _loc, _n = pid
            return np.frombuffer(z.read(f"{prefix}/data/{key}"), dtype=DTYPES[dtype])

    obj = U(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()
    for k in ("params_ema", "params"):
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return obj


def srvgg_tensors(sd, scale=4):
    keys = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.")})
    convs = [i for i in keys if sd[f"body.{i}.weight"].ndim == 4]
    out = []
    for i in keys:
        w = sd[f"body.{i}.weight"]
        if w.ndim == 4:
            b = sd[f"body.{i}.bias"]
            if i == convs[-1]:
                c_out = w.shape[0] // (scale * scale)
                perm = [c * scale * scale + p for p in range(scale * scale) for c in range(c_out)]
                w, b = w[perm], b[perm]
            out += [("conv", None, w.transpose(2, 3, 1, 0)), ("bias", None, b)]
        else:
            out.append(("prelu", None, w.reshape(-1)))
    return out


def rrdb_tensors(sd):
    out = []
    for k, v in sd.items():
        v = np.asarray(v, np.float32)
        out.append(("conv" if v.ndim == 4 else "bias", k, v.transpose(2, 3, 1, 0) if v.ndim == 4 else v))
    return out


def write(tensors, path, arch, version, scale=4):
    header = {"arch": arch, "scale": scale, "tensors": []}
    blobs = []
    for kind, name, t in tensors:
        entry = {"kind": kind, "shape": list(t.shape)}
        if name:
            entry["name"] = name
        header["tensors"].append(entry)
        f16 = t.astype(np.float16)
        if not np.isfinite(f16).all():
            raise ValueError(f"{name or kind} overflows fp16")
        blobs.append(f16.tobytes())
    hj = json.dumps(header).encode()
    buf = io.BytesIO()
    buf.write(b"RESR" + struct.pack("<II", version, len(hj)) + hj)
    buf.write(b"\0" * ((-(12 + len(hj))) % 2))
    for b in blobs:
        buf.write(b)
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return data


def split(data, path, mb):
    stem = os.path.splitext(path)[0]
    size = mb * 1024 * 1024
    parts = [data[i:i + size] for i in range(0, len(data), size)]
    for k, p in enumerate(parts):
        with open(f"{stem}.part{k}.wasm", "wb") as f:
            f.write(p)
    return len(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--split", type=int, default=0, help="also write parts of this many MB")
    a = ap.parse_args()
    sd = {k: np.asarray(v, np.float32) for k, v in load_pth(a.src).items()}
    if any(k.startswith("conv_first") for k in sd):
        t, arch, ver = rrdb_tensors(sd), "RRDBNet", 2
    else:
        t, arch, ver = srvgg_tensors(sd), "SRVGGNetCompact", 1
    data = write(t, a.dst, arch, ver)
    n = sum(x.size for _, _, x in t)
    msg = f"{a.dst}: {arch}, {len(t)} tensors, {n:,} params, {len(data) / 1e6:.1f} MB"
    if a.split:
        msg += f", {split(data, a.dst, a.split)} parts"
    print(msg)
