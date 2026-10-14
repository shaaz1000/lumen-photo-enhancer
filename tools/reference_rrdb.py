"""Reference output of RealESRGAN_x4plus for the browser parity test.

Uses the ONNX graph from backend/app/onnx_models.py, which the backend tests
check against a direct NumPy port of the PyTorch network. Weights are rounded
through fp16 exactly like the shipped browser files.
"""
import json, sys
import numpy as np
import onnxruntime as ort
sys.path.insert(0, "backend")
from app.onnx_models import build_rrdb, load_pth

sd = {k: v.astype(np.float16).astype(np.float32) for k, v in load_pth(sys.argv[1]).items()}
rng = np.random.default_rng(11)
H, W = 10, 12
x = rng.random((3, H, W)).astype(np.float32)
sess = ort.InferenceSession(build_rrdb(sd).SerializeToString(), providers=["CPUExecutionProvider"])
y = sess.run(None, {"input": x[None]})[0][0]
json.dump({"h": H, "w": W, "input_hwc": x.transpose(1, 2, 0).ravel().tolist(),
           "output_hwc": y.transpose(1, 2, 0).ravel().round(6).tolist()}, open(sys.argv[2], "w"))
print("ok", y.shape)
