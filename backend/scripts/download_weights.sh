#!/usr/bin/env sh
# Downloads official checkpoints into backend/weights.
# Real-ESRGAN: BSD 3-Clause. GFPGAN: Apache 2.0 (optional, pass --faces).
set -eu
cd "$(dirname "$0")/.."
mkdir -p weights
get() { [ -f "weights/$2" ] || curl -fL --retry 3 -o "weights/$2" "$1"; }
R=https://github.com/xinntao/Real-ESRGAN/releases/download
get $R/v0.2.5.0/realesr-general-x4v3.pth realesr-general-x4v3.pth
get $R/v0.2.5.0/realesr-general-wdn-x4v3.pth realesr-general-wdn-x4v3.pth
get $R/v0.1.0/RealESRGAN_x4plus.pth RealESRGAN_x4plus.pth
if [ "${1:-}" = "--faces" ]; then
  get https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth GFPGANv1.4.pth
fi
echo "weights ready in backend/weights"
