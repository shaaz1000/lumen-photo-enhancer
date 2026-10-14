"""Exports the Real-ESRGAN graphs the benchmark runs through onnxruntime-node.

The graphs come from backend/app/onnx_models.py (verified against a NumPy port
of the PyTorch networks) with weights rounded through fp16, exactly like the
browser files, so benchmark numbers match what the browser computes.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
from app.onnx_models import blend, build_rrdb, build_srvgg, load_pth  # noqa: E402

W = os.path.join(HERE, "..", "backend", "weights")
OUT = os.path.join(HERE, ".models"); os.makedirs(OUT, exist_ok=True)
f16 = lambda sd: {k: v.astype(np.float16).astype(np.float32) for k, v in sd.items()}  # noqa: E731
g, wdn = f16(load_pth(f"{W}/realesr-general-x4v3.pth")), f16(load_pth(f"{W}/realesr-general-wdn-x4v3.pth"))
for i in range(21):
    dni = i / 20
    path = f"{OUT}/general_{i:02d}.onnx"
    if not os.path.exists(path):
        open(path, "wb").write(build_srvgg(blend(g, wdn, dni)).SerializeToString())
path = f"{OUT}/deep.onnx"
if not os.path.exists(path):
    open(path, "wb").write(build_rrdb(f16(load_pth(f"{W}/RealESRGAN_x4plus.pth"))).SerializeToString())
print("models in", OUT)
