#!/usr/bin/env bash
# Download the official MapTRv2 nuScenes checkpoints (Google Drive links from the
# hustvl/MapTR maptrv2 README) into checkpoints/maptrv2/ and print their sha256.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=checkpoints/maptrv2
mkdir -p "$out"

declare -A ids=(
    [maptrv2_nusc_r50_24ep_w_centerline.pth]=1m02OKAKPhzMOaSu_4STVcepY8jbE7v3o
    [maptrv2_nusc_r50_24ep.pth]=1AmQ3fT-J-MM4B8kh_9Gm2G5guM92Agww
)
for name in "${!ids[@]}"; do
    [[ -s "$out/$name" ]] || uvx gdown "${ids[$name]}" -O "$out/$name"
done
sha256sum "$out"/*.pth
