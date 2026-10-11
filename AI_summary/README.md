# AI-assisted commit history

This directory records the intent, implementation, verification, and operational
notes for substantial AI-assisted commits in this fork. Add one Markdown file
per feature commit and keep the newest entry first.

The fork started this documented history from upstream commit
`4ff136163ee4e6436e7f5c44f73c4a4932aa7657`. The 1,362 earlier commits are the
inherited MMDetection3D history and remain available through `git log`; they are
not duplicated here.

## Commits

- [Foxglove multi-sweep stream](2026-10-11-foxglove-multisweep-stream.md)
  — live Foxglove point cloud and 3D boxes, with the keyframe plus nine past
  LiDAR sweeps.
- [BEVFusion camera–LiDAR visualization](2026-10-11-bevfusion-camera-lidar-visualization.md)
  — multi-camera inference, camera-colored LiDAR PLY export, motion and class
  visualization, radar helpers, and corrected 3D box coordinates.

## Adding future entries

Each entry should state:

1. the parent commit or branch baseline;
2. why the change was needed;
3. the files and behavior changed;
4. tests or manual validation performed;
5. generated data or local files deliberately excluded;
6. any compatibility or follow-up notes.
