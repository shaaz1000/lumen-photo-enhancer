#!/usr/bin/env sh
# Downloads model files into backend/weights and checks their SHA-256.
#   Real-ESRGAN general v3 and x4plus: BSD 3-Clause (Xintao Wang)
#   GFPGAN v1.4 (--faces): Apache 2.0 (TencentARC), ONNX export published by the FaceFusion project
set -eu
cd "$(dirname "$0")/.."
mkdir -p weights
get() { # url file sha256
  if [ ! -f "weights/$2" ]; then curl -fL --retry 3 -o "weights/$2.part" "$1" && mv "weights/$2.part" "weights/$2"; fi
  if command -v sha256sum >/dev/null 2>&1; then
    echo "$3  weights/$2" | sha256sum -c - >/dev/null || { echo "checksum mismatch for $2" >&2; rm -f "weights/$2"; exit 1; }
  fi
}
R=https://github.com/xinntao/Real-ESRGAN/releases/download
get $R/v0.2.5.0/realesr-general-x4v3.pth realesr-general-x4v3.pth 8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292
get $R/v0.2.5.0/realesr-general-wdn-x4v3.pth realesr-general-wdn-x4v3.pth 1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca
get $R/v0.1.0/RealESRGAN_x4plus.pth RealESRGAN_x4plus.pth 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1
if [ "${1:-}" = "--faces" ]; then
  get https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx gfpgan_1.4.onnx accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385
fi
echo "weights ready in backend/weights"
