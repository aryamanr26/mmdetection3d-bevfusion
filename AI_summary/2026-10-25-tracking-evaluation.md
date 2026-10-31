# Tracking evaluation, baseline comparison, and latency

Branch `research/tracking`, on top of the
[training-free 3D tracking](2026-10-25-bevfusion-tracking.md) entry.
BEVFusion, its checkpoint, and its detections are unchanged; every tracker
reads the same cached detections.

## Result

The Poly-MOT-style tracker now starts tracks from any unmatched detection at
or above a 0.05 score, instead of only from detections at or above 0.4. The
0.4 split between its two association stages is unchanged. On the two
`mini_val` scenes, the only scenes the detector did not train on, this raises
AMOTA from 0.664 to 0.790 and improves AMOTP from 0.81 m to 0.52 m. It keeps
good identity continuity: 16 ID switches against 15 before. AMOTA improves in
all ten mini scenes.

The minimal constant-velocity (CV) baseline still scores slightly higher on
AMOTA: 0.808 on val and 0.800 on all scenes, against 0.790 and 0.765. The
Poly-style tracker has fewer identity errors: 16 vs 28 ID switches and 11 vs
29 fragmentations on val, and 52 vs 87 switches on all scenes. Its velocity
error on moving objects is lower, 0.51 vs 0.67 m/s, and its median track
lasts 5.0 s against 1.4 s.

GPU camera colorization brought the full Foxglove frame within the latency
targets: 422 ms mean and 449 ms p95, down from 529 ms and 567 ms. Perception
alone (preprocessing, inference, and tracking) takes 337 ms mean and 357 ms
p95. Tracking takes 9.2 ms mean and 10.7 ms p95 with about 76 tracks per
frame. Peak GPU use is 3.6 GB of 22.6 GB.

## Audit findings

- **Scene grouping bug.** `run_tracking.py` reset the tracker per log rather
  than per scene. nuScenes-mini has 10 scenes in 8 logs, so two training logs
  would have carried tracks across a scene boundary. Frames are now grouped by
  scene name. With the old birth rule, the earlier val result reproduces
  exactly (0.6640 AMOTA, 15 IDS).
- **Yaw-rate update bug, not fixed.** `Track.update` blends the yaw rate
  toward `(measured yaw - predicted yaw) / dt`, which is a residual rather than
  a rate, and nothing guards against the detector flipping heading by π.
  Swapping CTRV for plain CV gives the same AMOTA (0.666 vs 0.664 on val, old
  birth rule), so the turn models add nothing as implemented.
- **Train-scene leakage.** The checkpoint was trained on the full nuScenes
  train split, which contains all eight `mini_train` scenes. Train-scene
  numbers are optimistic and are reported separately.
- **Playback threshold.** Foxglove playback filters detections at the stream's
  0.2 score before tracking, so its effective birth threshold is 0.2. The
  evaluation and the latency benchmark use 0.05.

## Method

- **Detections:** `dets_val.pkl` (81 frames) and `dets_train.pkl` (323
  frames), written by `run_tracking.py` with a 0.05 score threshold. All 404
  mini keyframes are covered.
- **Official metrics:** nuScenes devkit 1.2.0 `TrackingEval` with
  `tracking_nips_2019`: 7 classes, 2 m center-distance matching, and the
  per-class 40/50 m ranges. Each scene and each aggregate (`val_scenes`,
  `train_scenes`, `all_scenes`) is a custom split in a temporary
  `splits.json`. The devkit's split-file lookup is patched to that file, so
  `data/` is untouched. The custom-split path filters the shared result file
  by sample token. Otherwise it is the standard evaluation, and `val_scenes`
  reproduces the standard `mini_val` result.
- **Supplementary metrics** (`evaluate_trackers.py`):
  - Matching: Hungarian on center distance within a class, at most 2 m.
  - Ground truth: the devkit class ranges, and at least one LiDAR or radar
    point.
  - Deviations from the devkit: no bike-rack filter, no interpolation, and
    every track output counts, with no best-MOTA score threshold.
  - Velocity error: the norm of the planar velocity difference in the global
    frame, in m/s. Ground-truth velocity is `box_velocity`; pairs where it is
    NaN are excluded, so the counts differ per row.
  - Moving and static: ground-truth speed at least 0.5 m/s is moving, at most
    0.2 m/s is static.
- **Baseline** (`ConstantVelocityTracker`, CenterPoint-style):
  - Inputs: the same class filter, score threshold, and class NMS as the
    Poly-style tracker.
  - Prediction: position plus the last detector velocity times dt, with no
    filtering.
  - Association: per class, detections in descending score order each take
    the nearest unclaimed predicted track within the class gate.
  - Lifecycle: every unmatched detection creates a track; a matched track
    copies the detection box and velocity; a track is deleted after 1.5 s
    unmatched; only tracks matched in the current frame are output.
- **Birth threshold:** `MultiObjectTracker(birth_score=0.05)` is now the
  default. Unmatched detections from either association stage start
  tentative tracks if they score at least `birth_score`. Tracks still need 2
  hits to be output. `--birth-score 0.4` restores the earlier behavior.

## Comparison (official)

| Split | Tracker | AMOTA | AMOTP (m) | MOTA | Recall | IDS | FRAG | MT | ML | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val (2 scenes) | Poly-style, birth ≥ 0.05 (default) | 0.790 | 0.523 | 0.737 | **0.826** | 16 | **11** | **150** | **27** | 587 | **450** |
| val (2 scenes) | Poly-style, earlier birth ≥ 0.4 | 0.664 | 0.808 | 0.645 | 0.703 | **15** | 15 | 98 | 38 | 443 | 883 |
| val (2 scenes) | CV baseline | **0.808** | 0.524 | **0.757** | 0.800 | 28 | 29 | 149 | 29 | **296** | 482 |
| train (8, seen) | Poly-style, birth ≥ 0.05 (default) | 0.770 | 0.548 | 0.754 | 0.854 | 36 | 35 | 341 | 48 | 835 | 894 |
| train (8, seen) | Poly-style, earlier birth ≥ 0.4 | 0.669 | 0.789 | 0.683 | 0.751 | 21 | 29 | 228 | 66 | 572 | 1687 |
| train (8, seen) | CV baseline | 0.805 | 0.506 | 0.809 | 0.866 | 59 | 44 | 342 | 46 | 350 | 829 |
| all 10 | Poly-style, birth ≥ 0.05 (default) | 0.765 | 0.550 | 0.740 | 0.846 | 52 | 46 | 498 | 67 | 1535 | 1275 |
| all 10 | Poly-style, earlier birth ≥ 0.4 | 0.667 | 0.793 | 0.664 | 0.733 | 32 | 41 | 322 | 111 | 1034 | 2633 |
| all 10 | CV baseline | 0.800 | 0.521 | 0.790 | 0.855 | 87 | 71 | 495 | 67 | 719 | 1246 |

Per scene, AMOTA / IDS:

| Scene | Split | Frames | Poly, birth ≥ 0.05 | Poly, birth ≥ 0.4 | CV baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| scene-0103 | val | 40 | 0.583 / 5 | 0.461 / 1 | 0.657 / 7 |
| scene-0916 | val | 41 | 0.730 / 11 | 0.584 / 13 | 0.764 / 20 |
| scene-0061 | train | 39 | 0.853 / 5 | 0.741 / 7 | 0.838 / 11 |
| scene-0553 | train | 41 | 0.584 / 1 | 0.556 / 1 | 0.606 / 2 |
| scene-0655 | train | 41 | 0.793 / 9 | 0.576 / 10 | 0.846 / 18 |
| scene-0757 | train | 41 | 0.704 / 2 | 0.530 / 1 | 0.809 / 7 |
| scene-0796 | train | 40 | 0.769 / 4 | 0.589 / 1 | 0.893 / 6 |
| scene-1077 | train | 41 | 0.931 / 2 | 0.752 / 0 | 0.977 / 1 |
| scene-1094 | train | 40 | 0.823 / 6 | 0.715 / 1 | 0.876 / 13 |
| scene-1100 | train | 40 | 0.915 / 1 | 0.847 / 0 | 0.937 / 6 |

Per class across all scenes, AMOTA / IDS:

| Class | Poly, birth ≥ 0.05 | Poly, birth ≥ 0.4 | CV baseline |
| --- | ---: | ---: | ---: |
| bicycle | 0.492 / 1 | 0.435 / 0 | 0.514 / 3 |
| bus | 0.872 / 0 | 0.824 / 0 | 0.899 / 0 |
| car | 0.806 / 37 | 0.667 / 21 | 0.842 / 46 |
| motorcycle | 0.765 / 2 | 0.659 / 1 | 0.806 / 5 |
| pedestrian | 0.868 / 12 | 0.763 / 10 | 0.901 / 30 |
| trailer | 0.925 / 0 | 0.850 / 0 | 0.975 / 0 |
| truck | 0.627 / 0 | 0.471 / 0 | 0.660 / 3 |

## Velocity, continuity, and association (all 10 scenes)

| Metric | Detector (raw) | Poly, birth ≥ 0.05 | Poly, birth ≥ 0.4 | CV baseline |
| --- | ---: | ---: | ---: | ---: |
| Velocity error, all (m/s, mean) | 0.331 (n=9341) | 0.272 (n=9070) | **0.254** (n=7599) | 0.331 (n=9297) |
| Velocity error, moving GT | 0.670 | 0.507 | **0.457** | 0.671 |
| Velocity error, static GT | **0.058** | 0.086 | 0.078 | **0.058** |
| Velocity error, 0–20 / 20–30 / 30–50 m | 0.26 / 0.33 / 0.45 | 0.21 / 0.27 / 0.36 | **0.21 / 0.26 / 0.34** | 0.26 / 0.32 / 0.45 |
| Heading of motion error, moving GT (deg, median) | n/a | 3.1 | **2.9** | 3.8 |
| GT frame coverage (mean per instance) | n/a | 0.863 | 0.668 | **0.916** |
| Track IDs per tracked GT instance | n/a | 1.19 | **1.14** | 1.33 |
| GT instances kept on one ID | n/a | 0.836 | **0.883** | 0.749 |
| Track duration, median (s) | n/a | 5.0 | **6.9** | 1.4 |
| Tracks lasting 2 frames or fewer | n/a | 61 | **24** | 760 |
| GT found by a detection ≥ 0.4 that is also tracked | n/a | 0.964 | 0.882 | **0.997** |
| Track output precision (all scores) | n/a | 0.639 | **0.783** | 0.770 |
| Moving GT labeled moving / static GT labeled static | n/a | 0.991 / 0.934 | 0.992 / 0.940 | 0.956 / 0.977 |

The velocity rows cover different numbers of matched pairs, because each
tracker outputs a different set of objects. The new default matches about
1,500 more pairs than the old rule, mostly harder low-score objects. Its
filtered velocity is still about 25% better than the raw detector on moving
objects.

## Lifecycle diagnostics (single factor, not tuned)

These start from the earlier rule (births at or above 0.4, 2 hits to output,
coasted tracks output).

| Poly-style variant | val AMOTA | val AMOTP | val IDS | all AMOTA | all IDS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier rule | 0.664 | 0.808 | 15 | 0.667 | 32 |
| Confirm on first hit | 0.715 | 0.703 | 21 | 0.713 | 53 |
| One association stage, births ≥ 0.05 | 0.790 | 0.525 | 17 | 0.765 | 53 |
| **Two stages, births ≥ 0.05 (new default)** | **0.790** | **0.523** | **16** | **0.765** | **52** |
| No coasted output | 0.659 | 0.841 | 15 | 0.663 | 34 |
| All three (one stage, first hit, no coast) | 0.807 | 0.518 | 25 | 0.814 | 91 |

## Latency (A10G, 81 val frames after 10 warm-up frames, score 0.05)

| Stage | Before: mean / p95 (ms) | After: mean (ms) | After: median | After: p95 | After: max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preprocess (sweeps, JPEG decode, model transforms) | 195.3 / 209.3 | 193.4 | 190.4 | 207.5 | 387.8 |
| Inference (`test_step` + device-to-host copy) | 134.8 / 137.6 | 134.9 | 134.9 | 138.0 | 142.3 |
| Tracking (motion labels, transform, `tracker.step`) | 8.9 / 10.7 | 9.2 | 9.5 | 10.7 | 16.2 |
| Colorize displayed cloud | 133.9 / 149.5 | **33.9** | 34.5 | **36.2** | 40.4 |
| Elevation grid + messages | 35.9 / 38.9 | 27.0 | 26.9 | 31.1 | 34.5 |
| Foxglove message construction | 19.0 / 35.9 | 22.1 | 22.9 | 26.3 | 39.9 |
| Protobuf encode | 1.1 / 1.3 | 1.2 | 1.2 | 1.4 | 1.5 |
| **Perception** (first three stages) | 339.0 / 357.8 | **337.4** | 336.0 | **356.6** | 530.6 |
| **Full frame** (all stages) | 528.9 / 567.2 | **421.7** | 419.4 | **448.6** | 610.7 |

"Before" is the earlier birth rule with CPU colorization. "After" is the new
birth default with GPU colorization; it outputs about 76 tracks per val
frame. The stages run back to back, and their sum equals the
measured wall time, so nothing is double-counted. Inference is bracketed by
`torch.cuda.synchronize`. One frame in 81 exceeded 500 ms because of a single
388 ms preprocessing spike. The elevation stage does not use the GPU, so its
drop from 36 to 27 ms is run-to-run variation. GPU memory peaks at 2.2 GB
allocated, 3.3 GB reserved by PyTorch, and 3.6 GB for the whole process
according to nvidia-smi, out of 22.6 GB.

Optimizations, each verified to leave outputs unchanged:

- **GPU colorization.** `colorize_lidar` runs on CUDA when it is available,
  using the same float64 projection, a `scatter_reduce` z-buffer, and the
  same scoring and saturation. `device='cpu'` keeps the NumPy path. Colors and
  masks were bit-identical to the CPU path on 27 val frames with border
  margins of 0 and 5 px, and a CUDA unit test covers this.
- **Tracker association.** The cost matrix is one vectorized expression, and
  class NMS uses a distance matrix. `tracker.step` dropped from 50 ms to 5–7
  ms per frame with the old birth rule, with bit-identical outputs.
- **Kalman update.** The update uses index slices instead of selection
  matrices and constant noise arrays. `tracker.step` runs with
  single-threaded BLAS (`threadpoolctl`; if it is missing, nothing changes),
  because multithreaded BLAS slows these tiny matrices. Track IDs and motion
  states are identical over 18,076 outputs, with boxes within 2e-13 m.
  `tracker.step` with the new default fell from 10.5 ms to 7.2 ms.
- **Display transform and camera images.** `tracks_to_lidar` is vectorized
  (maximum difference 2e-13 m). The decoded `uint8` camera images go straight
  to `ImageAug3D`, which casts to `uint8` first anyway; packed network inputs
  are bit-identical.

## Limitations

- Only 2 scenes (81 frames) are unseen by the detector, so val AMOTA moves by
  several points per scene. The new default was chosen on these same scenes,
  so its val result is not an independent held-out estimate. It also improves
  all eight train scenes.
- The supplementary metrics are this repository's approximations; their
  conventions and deviations are listed under Method. The official devkit
  metrics are the reference.
- The lower birth threshold adds about 500 false positives across all scenes
  (FP 1535 vs 1034). Track output precision across all scores falls from
  0.78 to 0.64. AMOTA accounts for this through its score thresholds, but
  consumers that ignore scores see more short, low-confidence tracks.
- Tracking p95 is 10.7 ms against the 10 ms target. The mean, 9.2 ms, meets
  it.
- Forecasts are evaluated only on tracks that currently match ground truth.
  On val, 3 s FDE is 0.73 m with the new default.

## Next steps

1. Fix the yaw-rate update, adding the innovation instead of blending toward
   it and guarding against π heading flips. Re-check the per-class value of
   CTRV.
2. Evaluate on more held-out scenes (full nuScenes val) before treating 0.05
   as final, because the remaining AMOTA gap to the CV baseline (0.79 vs 0.81)
   is within per-scene noise here.
3. Use the same tracking score threshold in Foxglove playback and in
   evaluation.

## Reproduce

```bash
python projects/BEVFusion/demo/run_tracking.py                     # val dets + eval
python projects/BEVFusion/demo/run_tracking.py \
    --infos data/nuscenes/nuscenes_infos_train.pkl --skip-eval     # train dets
python projects/BEVFusion/demo/evaluate_trackers.py                # comparison
python projects/BEVFusion/demo/evaluate_trackers.py \
    --out-dir demo/eval_out/tracking_ablation \
    --trackers poly_birth_0p4 poly_min_hits_1 poly_birth_all \
    poly_no_coast_output poly_all_three cv_baseline \
    --official-splits val_scenes all_scenes
python projects/BEVFusion/demo/benchmark_latency.py                # stop playback first
```

Outputs: `demo/eval_out/tracking_compare/` (`comparison.md`,
`comparison.json`, `latency.json`, and per-tracker submissions), plus
`demo/eval_out/tracking_ablation/` and `demo/eval_out/tracking_birth/`.

## Files changed

- `projects/BEVFusion/demo/tracker.py`:
  - `ConstantVelocityTracker`.
  - `birth_score`, defaulting to 0.05.
  - Vectorized association and class NMS.
  - Kalman update using index slices.
  - Single-threaded BLAS inside `step`.
  - An `output_coasted` option, on by default.
- `projects/BEVFusion/demo/run_tracking.py`:
  - `--tracker` and `--birth-score` options.
  - Per-scene grouping.
  - Per-split cache and output names.
  - Custom-split `official_eval`.
  - Per-frame `step_ms`, and more compact metrics.
- `projects/BEVFusion/demo/evaluate_trackers.py` (new): the comparison
  driver, supplementary metrics, and Markdown report.
- `projects/BEVFusion/demo/benchmark_latency.py` (new): the per-stage latency
  benchmark.
- `projects/BEVFusion/demo/colorize_lidar.py`: the CUDA colorization path and
  a `device` argument.
- `projects/BEVFusion/demo/foxglove_stream.py`: `uint8` camera images into the
  model transforms and a vectorized `tracks_to_lidar`.
- `tests/test_projects/test_bevfusion/test_tracker.py`: baseline lifecycle and
  birth-threshold tests.
- `tests/test_projects/test_bevfusion/test_colorize_lidar.py` (new): a test
  that CUDA and NumPy colorization give the same result.
