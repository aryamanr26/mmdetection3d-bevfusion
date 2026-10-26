# Example daily activity log

Use this density. Do not copy the content into a real day's file.

```markdown
# Daily activity — 2026-09-05

## Summary
Extended the BEVFusion demo so a nuScenes sweep can be colorized and split into static vs dynamic BEV point clouds, then wrote the outputs under `demo/motion_out/`.

## Major tasks
- Added camera-to-LiDAR coloring and static/dynamic BEV exports in the inference demo path.
- Ran the demo on scene `n008-2018-08-30-15-16-55-0400` and inspected `bev.png`, `bev_static.png`, and `bev_dynamic.png`.

## Commands and code changes
- `python demo/mono_det_demo.py ...` — first attempt; wrong entry point for fusion inputs.
- `python demo/multi_modality_demo.py ${PCD} ${IMAGE} ${ANN} configs/bevfusion/... --out-dir demo/motion_out` — produced camera and BEV overlays.
- `mmdet3d/apis/inference.py` — point-cloud coloring and static/dynamic BEV split used by the demo.

## Configuration and environment
- Checkpoint: `checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.pth`
- Output dir: `demo/motion_out/` (PLY under `colorized_pc/`)

## Problems and troubleshooting
- **Problem:** `KeyError: 'points'` when the demo loaded camera-only results.
  - **Tried:** Re-ran with the lidar+camera config and a `.bin` point cloud.
  - **Solution:** Use the multi-modality demo with both PCD and image plus the fusion config.

## Decisions
- Write PLY next to the PNG overlays so the same frame can be opened in a point-cloud viewer.

## Completed
- Static/dynamic BEV PNGs and PLYs for the test scene.

## Discussed but not implemented
- Batch export over the full nuScenes val split.
- Motion arrows on the BEV image.

## Current state
Single-scene export works. Next step: add a `--split-static-dynamic` flag and document the exact demo command in `projects/BEVFusion/README.md`.
```
