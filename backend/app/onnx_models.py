"""Real-ESRGAN networks built as ONNX graphs straight from the official .pth
checkpoints, then run with ONNX Runtime (CPU, CUDA or CoreML providers).

No PyTorch or basicsr is needed at runtime: the checkpoint is unpickled with
a minimal loader and the graph is assembled with onnx.helper.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from collections import OrderedDict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16}


def load_pth(path: str) -> dict[str, np.ndarray]:
    """Reads a PyTorch zip checkpoint into numpy arrays without importing torch."""
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]

    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, *_):
                    arr = storage[offset:]
                    if not size:
                        return arr[:1].reshape(())
                    st = [s * arr.itemsize for s in stride]
                    return np.lib.stride_tricks.as_strided(arr, size, st).copy()
                return rebuild
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, *_: data
            if mod == "torch" and name in _DTYPES:
                return name
            if mod == "collections" and name == "OrderedDict":
                return OrderedDict
            raise pickle.UnpicklingError(f"refusing to load {mod}.{name}")

        def persistent_load(self, pid):
            _, dtype, key, _loc, _n = pid
            return np.frombuffer(z.read(f"{prefix}/data/{key}"), dtype=_DTYPES[dtype])

    obj = _U(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()
    for k in ("params_ema", "params"):
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
            break
    return {k: np.asarray(v, np.float32) for k, v in obj.items()}


def blend(a: dict, b: dict, alpha: float) -> dict:
    """Deep network interpolation, as in Real-ESRGAN's --denoise_strength."""
    return {k: alpha * a[k] + (1 - alpha) * b[k] for k in a}


class _G:
    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0

    def name(self, p="t"):
        self.n += 1
        return f"{p}{self.n}"

    def const(self, arr, p="w"):
        nm = self.name(p)
        self.inits.append(numpy_helper.from_array(np.asarray(arr, np.float32), nm))
        return nm

    def op(self, typ, inputs, **attrs):
        out = self.name(typ.lower())
        self.nodes.append(helper.make_node(typ, inputs, [out], **attrs))
        return out

    def conv(self, x, w, b):
        return self.op("Conv", [x, self.const(w), self.const(b)], pads=[1, 1, 1, 1], kernel_shape=[3, 3])

    def model(self, inp, out):
        g = helper.make_graph(
            self.nodes, "realesrgan",
            [helper.make_tensor_value_info(inp, TensorProto.FLOAT, [1, 3, "h", "w"])],
            [helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, 3, "H", "W"])],
            self.inits,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8
        onnx.checker.check_model(m)
        return m


def build_srvgg(sd: dict, scale: int = 4) -> onnx.ModelProto:
    """SRVGGNetCompact (realesr-general-x4v3)."""
    g = _G()
    x = "input"
    keys = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.")})
    y = x
    for i in keys:
        w = sd[f"body.{i}.weight"]
        if w.ndim == 4:
            y = g.conv(y, w, sd[f"body.{i}.bias"])
        else:
            y = g.op("PRelu", [y, g.const(w.reshape(-1, 1, 1))])
    y = g.op("DepthToSpace", [y], blocksize=scale, mode="CRD")  # == torch.pixel_shuffle
    scales = g.const(np.array([1, 1, scale, scale], np.float32), "s")
    base = g.op("Resize", [x, "", scales], mode="nearest", nearest_mode="floor", coordinate_transformation_mode="asymmetric")
    out = g.op("Add", [y, base])
    return g.model(x, out)


def build_rrdb(sd: dict) -> onnx.ModelProto:
    """RRDBNet (RealESRGAN_x4plus): 23 residual-in-residual dense blocks."""
    g = _G()
    x = "input"

    def lrelu(t):
        return g.op("LeakyRelu", [t], alpha=0.2)

    def c(t, k):
        return g.conv(t, sd[f"{k}.weight"], sd[f"{k}.bias"])

    def rdb(t, k):
        feats = [t]
        for j in range(1, 5):
            inp = feats[0] if len(feats) == 1 else g.op("Concat", feats, axis=1)
            feats.append(lrelu(c(inp, f"{k}.conv{j}")))
        x5 = c(g.op("Concat", feats, axis=1), f"{k}.conv5")
        return g.op("Add", [g.op("Mul", [x5, g.const(np.float32(0.2))]), t])

    feat = c(x, "conv_first")
    t = feat
    n = len({k.split(".")[1] for k in sd if k.startswith("body.")})
    for i in range(n):
        o = rdb(rdb(rdb(t, f"body.{i}.rdb1"), f"body.{i}.rdb2"), f"body.{i}.rdb3")
        t = g.op("Add", [g.op("Mul", [o, g.const(np.float32(0.2))]), t])
    feat = g.op("Add", [feat, c(t, "conv_body")])
    s2 = g.const(np.array([1, 1, 2, 2], np.float32), "s")
    for k in ("conv_up1", "conv_up2"):
        up = g.op("Resize", [feat, "", s2], mode="nearest", nearest_mode="floor", coordinate_transformation_mode="asymmetric")
        feat = lrelu(c(up, k))
    out = c(lrelu(c(feat, "conv_hr")), "conv_last")
    return g.model(x, out)
