# BEVFusion camera–LiDAR visualization

## Commit intent

Add an end-to-end, headless visualization workflow for BEVFusion detections on
nuScenes camera and LiDAR data. The workflow runs six-camera plus LiDAR
inference, camera-colorizes the LiDAR cloud without changing its coordinate
frame, classifies detections as dynamic or static, and exports BEV, camera, and
PLY views.

Parent branch baseline:
`4ff136163ee4e6436e7f5c44f73c4a4932aa7657`
(`research/camera-lidar`).

## Main changes

### Multi-camera inference

- `mmdet3d/apis/inference.py`
  - Adds `cam_type='all'` support to
    `inference_multi_modality_detector`.
  - Preserves every camera entry and calibration matrix from the annotation.
  - Resolves all image paths under the supplied image directory.
  - Keeps the existing single-camera path for KITTI/SUNRGBD-style inference.

### BEVFusion checkpoint and box-coordinate corrections

- `projects/BEVFusion/bevfusion/bevfusion.py`
  - Makes converted-checkpoint yaw/length-width remapping opt-in through
    `use_converted_checkpoint`.
  - Aliases official checkpoint keys from `view_transform.*` to the local
    `vtransform.*` module name while loading.
- `projects/BEVFusion/bevfusion/transfusion_head.py`
  - Removes a duplicate gravity-center-to-bottom-center z conversion.
  - Prevents predicted boxes from being shifted downward by half their height.

### Visualization and export tools

- `projects/BEVFusion/demo/multi_modality_demo.py`
  - Runs multimodal inference and writes detection metadata.
  - Produces combined, dynamic-only, and static-only BEV/PLY views.
  - Projects corrected boxes into all camera views.
  - Draws velocity arrows and supports optional radar motion evidence.
  - Keeps moving boxes red and groups non-moving detections by class:
    vehicles blue, pedestrians yellow, motorcycles/bicycles green, barriers
    orange, and traffic cones magenta.
- `projects/BEVFusion/demo/colorize_lidar.py`
  - Projects LiDAR points into all six cameras.
  - Uses a depth-tolerant image-space z-buffer and best-view selection.
  - Returns per-point RGB while preserving XYZ in the LiDAR frame.
- `projects/BEVFusion/demo/motion_utils.py`
  - Classifies motion from predicted planar velocity with optional radar
    evidence.
  - Defines consistent colors and line styles for renderers.
- `projects/BEVFusion/demo/radar_utils.py`
  - Loads nuScenes radar channels and transforms radar points and compensated
    velocity into the LiDAR frame.
- `projects/BEVFusion/demo/export_ply.py`
  - Rebuilds PLY and BEV outputs from saved detections without rerunning the
    model.
- `projects/BEVFusion/demo/batch_gt_vs_pred.py`
  - Generates batch ground-truth-versus-prediction BEV and camera comparisons.

### Repository hygiene

- `.gitignore`
  - Excludes generated top-level demo output directories matching
    `demo/*_out/`.
- `projects/BEVFusion/demo/export_multi_scene_ply.py` is deliberately excluded
  from this commit because it is an experiment driver with hard-coded
  checkpoint, device, sample count, and output paths.
- Generated PLY, PNG, JSON, work logs, datasets, and checkpoints are not part of
  the commit.

## Troubleshooting decisions

### Boxes below detected objects

The bbox coder already converts network output from gravity-center z to
MMDetection3D's bottom-center convention. `TransFusionHead` performed the same
half-height subtraction a second time. Removing the second conversion aligned
box bottoms and tops with the LiDAR returns and camera objects.

### Gray-looking colorized clouds

Unpainted points retain a neutral fallback color, while camera-visible points
receive image RGB. Saturation boosting and omission of unpainted points in the
colorized export make camera colors visible in PLY viewers that support vertex
RGB.

### Coordinate-frame safety

Colorization changes only RGB. Point XYZ and predicted box corners both remain
in the `LIDAR_TOP` frame; no camera- or ego-frame coordinates are written into
the PLY.

## Verification performed

- Ran full BEVFusion inference with the official camera-plus-LiDAR checkpoint.
- Regenerated the original visualization and five additional nuScenes scenes.
- Confirmed corrected box heights in both camera overlays and PLY point clouds.
- Confirmed all six camera views contribute to point coloring.
- Verified the grouped PLY palette by counting exact wireframe RGB values.
- Ran repository lint checks on the edited Python files.
- Ran `git diff --check`; no whitespace errors were reported.
- Searched the proposed source files for credentials and machine-specific
  absolute paths; none were found.

## Compatibility notes

- MIT-converted checkpoints must explicitly set
  `use_converted_checkpoint=True`; official OpenMMLab camera-plus-LiDAR weights
  use the default `False`.
- PLY vertex color display depends on the viewer. A PLY-aware 3D viewer such as
  MeshLab, CloudCompare, or a VS Code/Cursor PLY visualizer is recommended.
