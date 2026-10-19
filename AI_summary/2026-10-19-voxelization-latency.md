# Non-deterministic voxelization latency

## Change intent

Remove the largest BEVFusion inference bottleneck during Foxglove scene
playback by selecting the existing non-deterministic hard-voxelization CUDA
path.

Parent commit: `aa3ecd16`
(`feat(bevfusion): stream the six camera JPEGs to Foxglove`).

## Main change

- `projects/BEVFusion/configs/bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
  - Sets `deterministic=False` in `voxelize_cfg`.
  - Selects the already-compiled `nondisterministic_hard_voxelize_gpu` path;
    no CUDA extension rebuild is required.

## Verification

- Profiled nuScenes scene `n015-2018-07-24-11-22-45+0800` on an NVIDIA A10G.
- Used a 273,651-point keyframe-plus-nine-sweeps input, three warmup
  iterations, and ten measured iterations.
- Custom voxelization averaged **1.329 ms** (1.319–1.336 ms), down from the
  earlier deterministic measurement of about **100.7 ms**.
- Full GPU inference averaged **135.985 ms** (135.369–138.045 ms), down from
  about **235.6 ms**.
- Rebuilt all 39 Foxglove playback frames with non-empty prediction and
  ground-truth sets; the stream is available at `ws://127.0.0.1:8765`.

## Compatibility and follow-up

- Point ordering inside full voxels can vary, so predictions are not
  guaranteed to be bit-identical between runs.
- The next major targets are the camera backbone and depth lift (about 78 ms)
  and sparse LiDAR encoder (about 32 ms).
