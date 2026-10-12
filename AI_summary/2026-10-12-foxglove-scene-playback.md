# Foxglove scene playback

## Commit intent

Play one complete nuScenes scene in timestamp order, with the point cloud,
predictions, and ground-truth boxes changing together. Foxglove's own timeline
controls pause, play, and seek.

Parent commit: `d2440243`
(`feat(bevfusion): stream multi-sweep detections to Foxglove`).

## Main changes

- `projects/BEVFusion/demo/foxglove_stream.py`
  - `--play-scene` infers every keyframe in the log, in timestamp order.
  - Each frame keeps the keyframe plus nine past LiDAR sweeps.
  - `/detections` remains the colored predictions.
  - `/ground_truth` adds white nuScenes wireframes labeled `GT <class>`.
  - The WebSocket advertises Foxglove playback control. Pause, play, and
    timeline seeks come from the Foxglove app.
  - Playback stops on the last frame instead of looping.

## Verification

- Scene `n015-2018-07-24-11-22-45+0800`: 39 keyframes spanning 19.1 seconds.
- Each later frame contained about 270,000 points, with roughly 74% colored
  by the six cameras.
- Predictions and ground-truth counts were published on the same log time.
- Reconnected to `ws://127.0.0.1:8765` and confirmed the timeline can pause
  and play the scene.

## Excluded

- `.cursor/`
- `projects/BEVFusion/demo/export_multi_scene_ply.py`
- Generated PLY, PNG, JSON, and MCAP outputs.
