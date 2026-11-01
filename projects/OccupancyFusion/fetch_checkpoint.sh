#!/usr/bin/env bash
# Official SparseOcc v1.1 24-epoch ResNet-50 8-frame weights.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=checkpoints/sparseocc
mkdir -p "$out"
name=sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth
url="https://github.com/MCG-NJU/SparseOcc/releases/download/v1.1/${name}"
if [[ ! -s "$out/$name" ]]; then
    wget -c -O "$out/$name" "$url"
fi
sha256sum "$out/$name"
