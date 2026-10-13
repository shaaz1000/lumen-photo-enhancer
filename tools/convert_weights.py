"""Convert Real-ESRGAN realesr-general-x4v3 (.pth) weights to a compact binary
that the browser engine loads with TensorFlow.js.

No PyTorch needed: the .pth zip is unpickled with a minimal loader.

Output layout (little endian):
  magic "RESR", u32 version=1, u32 json_len, json header, then fp16 data.
Header lists tensors in order with TF layouts:
  conv kernels  [kh, kw, in, out]   (PyTorch [out, in, kh, kw] transposed)
  biases        [out]
  prelu alphas  [ch]
The final conv's output channels are permuted so tf.depthToSpace (NHWC,
channel index (i*r + j)*C + c) matches torch.pixel_shuffle (c*r*r + i*r + j).
"""
import io, json, pickle, struct, sys, zipfile
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
            if mod == "torch" and name in DTYPES:
                return name
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, req, hooks: data
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, dtype, key, _loc, _n = pid
            raw = z.read(f"{prefix}/data/{key}")
            return np.frombuffer(raw, dtype=DTYPES[dtype])

    obj = U(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()
    for k in ("params_ema", "params"):
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return obj


def to_tf(sd, scale=4):
    keys = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.")})
    tensors = []
    convs = [i for i in keys if f"body.{i}.weight" in sd and sd[f"body.{i}.weight"].ndim == 4]
    last = convs[-1]
    for i in keys:
        w = sd[f"body.{i}.weight"]
        if w.ndim == 4:
            if i == last:
                out = w.shape[0]
                c_out = out // (scale * scale)
                perm = [c * scale * scale + p for p in range(scale * scale) for c in range(c_out)]
                w = w[perm]
                b = sd[f"body.{i}.bias"][perm]
            else:
                b = sd[f"body.{i}.bias"]
            tensors.append(("conv", w.transpose(2, 3, 1, 0)))
            tensors.append(("bias", b))
        else:
            tensors.append(("prelu", w.reshape(-1)))
    return tensors


def write_bin(tensors, path, scale=4):
    header = {"arch": "SRVGGNetCompact", "scale": scale, "tensors": []}
    blobs = []
    for kind, t in tensors:
        header["tensors"].append({"kind": kind, "shape": list(t.shape)})
        blobs.append(t.astype(np.float16).tobytes())
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(b"RESR" + struct.pack("<II", 1, len(hj)) + hj)
        pad = (-(12 + len(hj))) % 2
        f.write(b"\0" * pad)
        for b in blobs:
            f.write(b)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    sd = load_pth(src)
    t = to_tf(sd)
    write_bin(t, dst)
    n = sum(x.size for _, x in t)
    print(f"{dst}: {len(t)} tensors, {n} params")
