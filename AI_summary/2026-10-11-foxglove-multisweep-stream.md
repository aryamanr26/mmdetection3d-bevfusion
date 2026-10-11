# Foxglove multi-sweep visualization stream

## Commit intent

Stream one BEVFusion frame to the Foxglove browser app without downloading PLY
files. The cloud is the keyframe plus the nine previous LiDAR sweeps used by
the detector, with camera color and the existing detection boxes.

Parent commit: `30ffb89a` (`feat(bevfusion): add camera-lidar visualization tools`).

## Main changes

- `projects/BEVFusion/demo/foxglove_stream.py`
  - Serves a Foxglove WebSocket on `127.0.0.1:8765`.
  - Publishes `/lidar` as a colored `PointCloud` and `/detections` as oriented
    translucent cubes plus wireframes.
  - Keeps dynamic boxes red and the grouped class colors for everything else.
  - Merges nine past sweeps into the keyframe LiDAR frame with the same
    `lidar2sensor` conversion as `LoadPointsFromMultiSweeps`.
  - If the requested frame is the start of a log and has no past sweeps, steps
    to the next keyframe in that log and runs BEVFusion there so the boxes
    match the dense cloud.
  - `--sweeps 0` keeps the earlier single-sweep path from a PCD file and
    `detections.json`.

## Verification

- Confirmed the original demo keyframe has zero past sweeps, and the next
  keyframe has ten.
- Accumulated that frame to 273,651 points. Road returns stay near
  `z = -1.7` m, so the sweeps land in the keyframe LiDAR frame.
- Ran detection: 50 boxes above score 0.2. Camera color covered 75.6% of the
  merged points.
- Opened the stream in Foxglove at `ws://localhost:8765`.

## Excluded

- `.cursor/`
- `projects/BEVFusion/demo/export_multi_scene_ply.py`, which still hard-codes
  the checkpoint, device, sample count, and output path.
- Generated PLY, PNG, JSON, and MCAP outputs.

## Notes

The browser connects to this machine. The point cloud is not uploaded to
Foxglove's servers. `z = 0` remains the LiDAR sensor, about 1.7 m above the
road. Restarting the server requires reconnecting the Foxglove panel.
