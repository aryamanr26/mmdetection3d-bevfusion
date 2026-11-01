# MapTRv2 accuracy, ground elevation, and a 2 Hz sidecar

Branch `research/lane-perception`, parent `ca018a26`. No training and no
change to BEVFusion weights, detection thresholds, tracker parameters, or
BEVFusion preprocessing. MapTRv2 stays in the `maptrv2:cu111` container.
The checkpoint is `checkpoints/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.pth`.

The 500 ms full-frame target is **not** met. With MapTRv2 pinned to one
vCPU and updating at 2.0 Hz, BEVFusion's full frame on the same A10G is
545 ms mean and 635 ms p95. Details are under Latency.

## Map expansion

nuScenes map expansion v1.3 is installed under `data/nuscenes/maps/`
(`expansion/` and `basemap/`), which is where the devkit looks. It does not
replace the existing semantic-prior PNGs. `boston-seaport.json` reports
`version` 1.3 and loads through `NuScenesMap`.

Held-out scenes only:

| Scene | Log location | Keyframes |
| --- | --- | --- |
| scene-0103 | boston-seaport | 40 |
| scene-0916 | singapore-queenstown | 41 |

The other eight mini scenes are in the nuScenes train split, which this
checkpoint was trained on. They are not used as accuracy evidence.

What each MapTRv2 class is compared to:

| Class | What the map actually stores | What the metric uses |
| --- | --- | --- |
| divider | Annotated `road_divider` and `lane_divider` polylines | Those lines, clipped to the patch. This is the only class whose ground truth is already a labeled polyline of the same kind. |
| centerline | Annotated arc-line paths on `lane` and `lane_connector` | Those paths discretized at 0.5 m, clipped, chained with the MapTRv2 converter's directed graph, and simplified at 0.2 m. The chain is derived. |
| boundary | `road_segment` and `lane` polygons. There is no boundary polyline. | Outline of the union, inset 0.2 m, matching the MapTRv2 converter. Derived. |
| ped_crossing | `ped_crossing` polygons | Outline of the union, expanded 0.2 m. Derived. scene-0916 has none inside the patch. |

The local patch is the MapTRv2 range: x ±15 m (right), y ±30 m (forward),
in LIDAR_TOP. The patch is centered on the LiDAR origin in the global map,
rotated by `quaternion_yaw` of `ego2global @ lidar2ego`, then translated so
the LiDAR origin is zero. Fragments shorter than 1 m are dropped as clipping
slivers. That cutoff was fixed before looking at scores and was not adjusted
afterward. Nothing in this path runs at inference time.

## Geometric accuracy

Predictions are the saved centerline-checkpoint outputs. Score threshold
0.3 is the Foxglove display threshold. It was not swept on these scenes.
Distance thresholds are MapTRv2's 0.5 m, 1.0 m, and 1.5 m. They were not
chosen from these two scenes.

Chamfer distance is the average of the two directed mean nearest-point
distances. A pair matches when that chamfer is within the threshold.
Matching is greedy and one-to-one, and it is done inside each keyframe.
Counts are then summed. A prediction cannot match a ground-truth line from
another frame. These numbers are not the paper's mAP.

At 1.0 m, summed over keyframes:

| Scene | Class | Precision | Recall | F1 | Matched chamfer (m) | Nearest median (m) | Pred / frame | GT / frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scene-0103 | divider | 0.258 | 0.332 | 0.291 | 0.75 | 1.45 | 11.2 | 8.7 |
| scene-0103 | ped_crossing | 0.205 | 0.224 | 0.214 | 0.87 | 1.20 | 2.1 | 1.9 |
| scene-0103 | boundary | 0.266 | 0.266 | 0.266 | 0.74 | 1.54 | 3.1 | 3.1 |
| scene-0103 | centerline | 0.112 | 0.125 | 0.118 | 0.83 | 2.42 | 11.2 | 10.0 |
| scene-0916 | divider | 0.092 | 0.118 | 0.103 | 0.84 | 1.87 | 2.4 | 1.9 |
| scene-0916 | ped_crossing | — | — | — | — | — | 0.0 | 0.0 |
| scene-0916 | boundary | 0.114 | 0.124 | 0.119 | 0.83 | 2.06 | 4.3 | 3.9 |
| scene-0916 | centerline | 0.006 | 0.008 | 0.007 | 0.99 | 2.34 | 4.2 | 3.2 |

Both scenes together at 1.0 m: divider F1 0.257, pedestrian crossing F1
0.214, boundary F1 0.181, centerline F1 0.089. At 0.5 m almost nothing
matches (divider F1 0.023). At 1.5 m, divider F1 rises to 0.484 and
centerline F1 to 0.265. The full tables are in
`demo/eval_out/lane_perception/accuracy/accuracy.md`.

When a pair does match at 1.0 m, its chamfer is about 0.7–1.0 m. The
nearest-median column is higher (1.2–2.4 m) because most predictions are
not inside 1.0 m of any same-class ground-truth instance.

Overlays (solid = ground truth, dashed = prediction, LIDAR frame, +y
forward) are in `demo/eval_out/lane_perception/accuracy/overlays/`.

- scene-0103 keyframe 1 is a straight road. The predicted dividers,
  centerlines, and boundaries sit on the map lines.
- scene-0103 keyframe 20 is an intersection. Centerlines fan through the
  junction and do not follow a single ground-truth chain. Short divider
  fragments appear beside the real lines. Pedestrian-crossing outlines and
  the predicted crossing strokes only partly overlap.
- scene-0916 keyframe 1 is a curve into a junction. The main curve is in
  the right corridor, with extra fragments and a side structure the model
  does not cover.
- scene-0916 keyframe 20 follows the bend. One boundary on the right is
  missed, and a short boundary fragment has no map counterpart.

Centerline F1 stays low even on frames that look aligned because the score
is one-to-one instance chamfer. A predicted path that shares a corridor
with a ground-truth chain still fails the match when the symmetric chamfer
exceeds 1 m, which is common where several chains share the junction.
On scene-0103 keyframe 20 the median centerline length is about 52 m on
both sides, and the median best symmetric chamfer is 2.75 m, so the gap is
geometric and not only a short-versus-long artifact.

## Ground elevation

`estimate_elevation_grid` in `projects/BEVFusion/demo/elevation_grid.py`
fits a ground surface from LiDAR returns. The grid is 50 m × 50 m, origin
(−25, −25) m, 0.2 m cells, 250 × 250. `ground[iy, ix]` is the fitted
surface height in the LiDAR frame, not an offset from a display plane.
Unknown cells are NaN. The ego disk around the sensor is left empty. The
grid does not cover MapTRv2's y reach of ±30 m; points with |x| or |y|
above 25 m have no cell.

`sample_ground` returns `(height, valid)`. `assign_elevation` writes
`points_xyz` as `[x, y, z or None]` and does not change XY. Missing cells
stay missing. The Foxglove line is split at those points instead of being
drawn at the old −1.8 m display plane. The −1.8 m constant is not added to
the grid height.

On the first LIDAR_TOP sweep of scene-0103, cells at y = 5 m are about
−1.61 m. The median finite height moves from −2.27 m in the band y ∈
[−20, −10] to −1.39 m in y ∈ [10, 20], so the same grid already carries
the uphill/downhill change. The sensor origin itself is NaN. Of the lane
points at score ≥ 0.3 on that keyframe, 81 received a height (median
−1.90 m) and 179 did not, 93 of those because they fall outside the 50 m
grid. A double application of −1.8 m would have put the road near −3.5 m.
It does not.

Playback publishes `/lane_map` and `/lane_graph` in frame `lidar`. Each
lane entity records `elevation_valid`. A `lane_status` entity records
`age_s`, `estimate_time`, `seq`, and `stale`.

## Asynchronous 2 Hz MapTRv2

`lane_worker.py` runs inside the existing container, one inference at a
time. `LatestFrameScheduler` accepts a frame every 0.5 s. If a job is
running, only the latest due frame is kept. A finished result is ignored
when a newer sequence has already been accepted. Worker exceptions are
written as an error payload and do not stop BEVFusion.

`LaneHold` keeps the last accepted polylines. XY is moved into the latest
LiDAR frame with z = 0 in the source frame, so the rigid transform does
not invent a road height. The destination elevation grid supplies Z.
Estimates older than 2 s, or with a negative age, are not returned as
current.

Foxglove playback uses the saved predictions, not a live container. Mini
keyframes are already 0.5 s apart, so the hold updates on every keyframe
and the age on those frames is about zero. Ego-motion compensation is what
runs between updates. It is covered by `test_ego_motion_compensation`.

The integrated benchmark is `projects/LaneMapping/benchmark_async.py`. It
starts the worker, then runs `benchmark_latency.py` unchanged (score 0.05,
9 sweeps, 81 val frames, 10 warm-up frames). The worker is pinned with
`--cpuset-cpus 0`. BEVFusion is pinned with `taskset -c 1-3`. An earlier
unpinned run is kept at
`demo/eval_out/lane_perception/async_benchmark/summary_unpinned.json`.

### Latency (A10G, 4 vCPU)

| Stage | Alone, 25 Oct, mean / p95 | Full-rate MapTRv2, mean / p95 | 2 Hz, unpinned, mean / p95 | 2 Hz, CPU-pinned, mean / p95 |
| --- | ---: | ---: | ---: | ---: |
| Preprocess | 193.4 / 207.5 | 262.5 / 302.6 | 299.1 / 391.1 | 248.9 / 303.3 |
| Inference | 134.9 / 138.0 | 195.4 / 213.3 | 192.1 / 246.3 | 161.8 / 216.2 |
| Tracking | 9.2 / 10.7 | 11.8 / 15.1 | 18.8 / 47.0 | 13.1 / 16.8 |
| Full frame | 421.7 / 448.6 | 579.6 / 643.9 | 658.6 / 807.7 | 544.9 / 634.9 |

Pinned MapTRv2, measured during that BEVFusion run (131 updates, 65.7 s):

- Update rate 1.994 Hz.
- Lane-estimate age 0.250 s mean, 0.495 s p95.
- Network 92.2 ms mean (p50 83.6, p95 141.3). Standalone network was 75.7 ms.
- Request total, including image decode, 317.7 ms mean (p95 369.0).
- CPU busy fraction 0.991.
- GPU utilization mean 26.8%. Device memory peak 6208 MiB of 22590 MiB.
  BEVFusion's own process still peaks at 3634 MiB.

2 Hz and lane freshness hold. The full-frame p95 does not go under 500 ms.
Pinning one core for MapTRv2 is better than letting both processes roam
(p95 635 ms versus 808 ms) and better on the mean than full-rate overlap
(545 ms versus 580 ms). The p95 is about the same as full-rate overlap
(635 ms versus 644 ms). Preprocess and colorize stay high because three
cores are still saturated, and inference stays above the solo 135 ms
because the GPU is shared for about 90 ms out of every 500 ms.

## Tests

`.venv/bin/python -m pytest -q tests/test_projects` — 15 passed.

That includes centerline connectivity, ego-motion compensation, stale and
out-of-order results, the one-in-flight scheduler, per-frame chamfer
matching, and elevation sampling on a flat grid, an uphill/downhill slope,
and a missing cell. The elevation cases construct a grid directly so they
do not depend on a particular nuScenes sweep. The scene-0103 sweep above
is the check against a real cloud.

`evaluate_map.py` completed on both val scenes. The pinned async benchmark
completed on all 81 val frames. Foxglove playback of scene-0103 was
restarted after a label crash (`Point3` has no `.x`) and a graph crash
(an arrow Z of `None`). Both are fixed: labels use the numeric point, and
an arrow whose endpoint has no elevation is omitted.

## Commands

```bash
# Accuracy on scene-0103 and scene-0916 only
.venv/bin/python projects/LaneMapping/evaluate_map.py

# Integrated latency. Stop Foxglove first so it is not a third GPU resident.
.venv/bin/python projects/LaneMapping/benchmark_async.py

# Playback with saved lanes, 2 Hz hold, and elevation-grid Z
.venv/bin/python projects/BEVFusion/demo/foxglove_stream.py --play-scene \
  --infos data/nuscenes/nuscenes_infos_val.pkl \
  --pcd data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin
```

Foxglove: `ws://127.0.0.1:8765`, topics `/lane_map` and `/lane_graph` in
frame `lidar`.

```text
https://app.foxglove.dev/~/view?ds=foxglove-websocket&ds.url=ws%3A%2F%2Flocalhost%3A8765
```

## Files

Added:

- `projects/LaneMapping/map_gt.py`, `map_metrics.py`, `evaluate_map.py`
- `projects/LaneMapping/lane_elevation.py`, `lane_motion.py`, `lane_schedule.py`
- `projects/LaneMapping/lane_worker.py`, `benchmark_async.py`
- `tests/test_projects/test_lane_mapping/test_lane_runtime.py`

Modified:

- `projects/BEVFusion/demo/elevation_grid.py` — `sample_ground`
- `projects/BEVFusion/demo/foxglove_stream.py` — 2 Hz hold, elevation, graph rebuild
- `projects/LaneMapping/lane_foxglove.py` — XYZ lines, age, no invented height

Generated, gitignored:

- `demo/eval_out/lane_perception/accuracy/`
- `demo/eval_out/lane_perception/async_benchmark/`
- `demo/eval_out/lane_perception/live/`

## Remaining limitations

- Full-frame p95 is 635 ms, not under 500 ms.
- Accuracy is two mini-val scenes. The checkpoint was trained on the other
  mini scenes, so those scenes are not a second test.
- Boundary and pedestrian-crossing scores compare polylines to polygon
  outlines.
- Centerline one-to-one chamfer is strict at junctions.
- The elevation grid is 50 m square and often empty, so many lane points
  have no Z and the drawn line breaks there.
- Live 2 Hz inference was benchmarked. Foxglove playback still uses the
  saved JSON rather than the Docker worker.

## Next steps

- Give BEVFusion the decode budget this host does not have: another vCPU,
  or a MapTRv2 preprocess that does not decode six JPEGs on the same cores.
- If the question is "does the prediction lie in the lane corridor," report
  a directed point-to-line distance in addition to one-to-one instance
  chamfer. Do that on a larger held-out set, not by retuning 0.5/1.0/1.5 m
  on these two scenes.
- Fill elevation only where the existing grid is confident. Do not add a
  second ground estimator to close the gaps.
