# Training-free 3D tracking for BEVFusion

## Baseline

- Parent branch: `research/camera-lidar` at `ea2fa81f`.
- Detector: frozen camera-LiDAR BEVFusion checkpoint.
- Dataset: nuScenes `v1.0-mini`, with the 81-frame `mini_val` split used for
  all reported metrics.

## Implementation

- `projects/BEVFusion/demo/tracker.py` tracks detections in the global frame,
  removing ego motion before association.
- The filter uses class-dependent motion models: coordinated turn for larger
  vehicles, a higher-noise turn model for bicycles and motorcycles, and
  constant velocity for pedestrians.
- High-confidence detections are associated first. A second Hungarian pass
  recovers tracks from low-confidence detections. Class-specific center gates,
  center NMS, confirmation after two hits, and three-second coasting control
  track lifecycle.
- Each track exports a stable ID, filtered velocity, moving/static state with
  hysteresis, motion direction relative to box yaw, yaw rate, acceleration,
  two seconds of history, and a three-second CV/CTRV forecast.
- `projects/BEVFusion/demo/run_tracking.py` caches low-threshold detections,
  writes official nuScenes tracking JSON, runs `TrackingEval`, validates the
  output with a ground-truth identity oracle, and measures forecast ADE/FDE.
- `/tracks` in `foxglove_stream.py` displays class-colored boxes and ID labels,
  velocity arrows, history trails, dashed forecast paths, and coasted state.

## mini-val results

- AMOTA: 0.6640
- AMOTP: 0.8084 m
- MOTA: 0.6452
- MOTAR: 0.9277
- Recall: 0.7027
- Identity switches: 15
- Fragmentations: 15
- GT oracle: 0.9715 AMOTA, 0 identity switches
- Forecast ADE at 1/2/3-second checkpoints: 0.5243 m
- Forecast FDE: 0.3626 m at 1 s, 0.5348 m at 2 s, and 0.7476 m at 3 s

The forecast metric matches predicted tracks to ground truth within 2 m at the
current frame. It therefore measures motion extrapolation on successfully
localized tracks, not end-to-end detection recall.

## Verification

- Three tracker unit tests cover coordinate round trips, identity through
  updates and coasting, expiry, and class-safe association.
- All 81 mini-val frames ran through BEVFusion: 5,409 detections produced
  3,909 confirmed or coasted track outputs across two scenes.
- The official nuScenes evaluator completed for detector tracks and the GT
  oracle.
- Foxglove `SceneUpdate` messages containing boxes, labels, arrows, history,
  and forecasts encoded successfully. A 39-frame live playback built without
  errors and published `/tracks` alongside the existing topics.

## Usage

```bash
# Run inference, cache detections, track, forecast, and evaluate.
.venv/bin/python projects/BEVFusion/demo/run_tracking.py

# Re-run tracking/evaluation from the detection cache.
.venv/bin/python projects/BEVFusion/demo/run_tracking.py --skip-inference

# Visualize a tracked scene.
.venv/bin/python projects/BEVFusion/demo/foxglove_stream.py --play-scene
```

Generated artifacts are under `demo/eval_out/tracking/`.
