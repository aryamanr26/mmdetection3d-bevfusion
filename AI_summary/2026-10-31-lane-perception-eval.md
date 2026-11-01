# Pretrained mapless lane perception: MapTRv2 on nuScenes-mini

Branch `research/lane-perception`, on top of `research/tracking` at
`ca018a26`. No training or fine-tuning. This note covers model selection and
standalone inference. Map-expansion accuracy, ground elevation, and the 2 Hz
sidecar are in
[2026-10-31-maptr-accuracy-elevation-async.md](2026-10-31-maptr-accuracy-elevation-async.md).
That later work edits the Foxglove stream and the elevation grid. The
BEVFusion `.venv` is still separate from the MapTRv2 container.

## Result

MapTRv2 with centerlines (`maptrv2_nusc_r50_24ep_w_centerline`) is the
selected model. Its official nuScenes checkpoint runs unmodified on all 404
nuScenes-mini keyframes, using only the six cameras with no HD map. It
predicts four classes of polylines with confidences in the LIDAR_TOP frame:
dividers, pedestrian crossings, road boundaries, and directed lane
centerlines. Lane connectivity (forks and merges) comes from deterministic
post-processing of the centerlines.

On the A10G, standalone:

- **Network:** 75.7 ms mean and 79.3 ms p95.
- **Full standalone frame:** 278 ms mean and 297 ms p95, where JPEG decode and
  resizing take 198 ms.
- **GPU memory:** the process adds 2.5 GB to the device (674 MB allocated by
  PyTorch, 954 MB reserved).

Predictions line up with curbs and lane markings in the LiDAR BEV and in all
six cameras.

The main open issue is latency. Running it **concurrently** with BEVFusion
pushes BEVFusion's full frame from 422 to 580 ms mean (p95 449 to 644 ms),
which breaks the 500 ms target. The cause is mainly CPU contention on this
4-vCPU host. The model is ready for a sidecar integration, but not inside
the 500 ms BEVFusion frame without the changes under Next steps.

## Candidate comparison

| | LaneSegNet | MapTRv2 (selected) |
| --- | --- | --- |
| Repository | [OpenDriveLab/LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet) | [hustvl/MapTR](https://github.com/hustvl/MapTR), `maptrv2` branch |
| License | Apache 2.0 | MIT |
| Released checkpoints | One: `lanesegnet_r50_8x1_24e_olv2_subset_A` (182 MB, Hugging Face) | nuScenes R50 24 ep, with and without centerline (485 MB each, Google Drive); Argoverse 2 |
| Training data of checkpoint | OpenLane-V2 subset A = **Argoverse 2, 7 cameras** | **nuScenes, 6 cameras** |
| Outputs | Lane segments (centerline plus left and right boundaries, boundary types), pedestrian crossings, lane-to-lane topology matrix | Dividers, pedestrian crossings, boundaries, centerlines (20-point polylines and scores); no topology head |
| BEV range | 102.4 × 51.2 m | 30 × 60 m (x ±15, y ±30) |
| Inference entry point | `tools/test.py` with OpenLane-V2 `*_lanesegnet.pkl` infos; no single-frame demo (issue #47) | `tools/test.py`, `nusc_vis_pred.py`; needs map-expansion infos for GT |
| Temporal / CAN bus | BEVFormer encoder with `can_bus` | LSS (bevpool) encoder; `can_bus` unused, single frame |
| Stack | Py 3.8, torch 1.9.1 cu111, mmcv-full 1.5.2, mmdet 2.26, mmseg 0.29.1, mmdet3d 1.0.0rc6, openlanev2 2.1.0 | Py 3.8, torch 1.9.1 cu111, mmcv-full 1.4.0, mmdet 2.14, mmseg 0.14.1, vendored mmdet3d 0.17, GKT CUDA op |
| Paper numbers | 33.5 mAP, 25.4 TOP_lsls (subset A) | 61.4 mAP base, 54.3 mAP with centerline (nuScenes val) |

**Why MapTRv2.** LaneSegNet was the preferred candidate, but only if its
released checkpoint could produce lane geometry and connectivity without
retraining. It cannot do that on our data. The only checkpoint is trained
on Argoverse 2:

- **Cameras:** 7 ring cameras instead of nuScenes' 6.
- **Camera geometry:** different intrinsics and image size.
- **Ego frame:** a different ego frame.

There is no OpenLane-V2 subset B (nuScenes) checkpoint, and the map-element
bucket variant ships a config without weights. Running it on nuScenes would
be a cross-dataset transfer that needs fine-tuning, which is out of scope.

MapTRv2 is trained on exactly our six nuScenes cameras. Its encoder is the
same LSS/bevpool family as BEVFusion's camera branch. It produces lane
centerlines natively, so it gives lane geometry now, and connectivity can be
derived without training.

## What runs

- **Environment:** an isolated Docker image, `maptrv2:cu111`, built from
  `projects/LaneMapping/docker/Dockerfile`.
  - Base: `nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04`, Python 3.8, torch
    1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0, and mmseg 0.14.1.
  - MapTR is pinned to commit `e03f097`. Its vendored mmdet3d 0.17 is built
    with `bev_pool`, along with the GKT op, for sm_86.
  - The BEVFusion `.venv` (torch 2.0.1, mmcv 2.0.1, mmdet3d 1.1) cannot host
    this stack.
- **Inputs:** `maptr_infer.py` builds MapTR's `get_data_info` dictionary from
  nuScenes devkit records: the six cameras in the converter order, plus
  `lidar2img`, `camera2ego`, `camera_intrinsics`, and `lidar2ego`.
  - It then applies the upstream test pipeline unchanged: 0.5× resize to
    800×450, normalize, pad to 800×480. It calls the model exactly as
    `nusc_vis_pred.py` does, with fp16 enabled as in the config.
  - No map-expansion data, CAN bus expansion, or info-file regeneration is
    needed.
- **Checkpoint:** loaded with `strict=True`, so every key matches. Its meta
  records mmdet3d 0.17.2, epoch 24.
- **Coordinate frame:** LSS lifts camera features to ego and then to
  LIDAR_TOP (`get_geometry_v1`), and the decoder denormalizes to metres in
  that frame. The output therefore uses the same `lidar` frame as
  `foxglove_stream.py`, with +y forward and +x right. The overlays confirm
  this: boundaries sit on LiDAR curbs, and projected lines follow lane
  markings in all six cameras.

## Initial results (all 404 mini keyframes)

Elements per frame at score ≥ 0.3, from `run_centerline.log`:

| Scene | Split | Frames | Elements per frame |
| --- | --- | ---: | ---: |
| scene-0103 | val | 40 | 27.9 |
| scene-0916 | val | 41 | 10.6 (parking lot) |
| scene-0061 | train | 39 | 29.0 |
| scene-0553 | train | 41 | 28.1 |
| scene-0655 | train | 41 | 21.7 |
| scene-0757 | train | 41 | 22.4 |
| scene-0796 | train | 40 | 21.5 |
| scene-1077 | train | 41 | 19.0 |
| scene-1094 | train | 40 | 31.7 |
| scene-1100 | train | 40 | 23.6 |

Per-frame means at score ≥ 0.3:

| Split | Divider | Ped. crossing | Boundary | Centerline | Graph nodes | Forks | Merges | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val (81 frames) | 6.8 | 1.0 | 3.7 | 7.6 | 6.4 | 2.8 | 2.2 | 0.59 |
| train (323 frames) | 6.7 | 2.3 | 4.2 | 11.3 | 12.4 | 3.8 | 2.3 | 0.72 |

- The base checkpoint (no centerline class) gives 7 to 15 elements per frame
  and the same 75.7 ms network time (`base/summary.json`).
- **Leakage:** MapTRv2 was trained on the full nuScenes train split, which
  contains all eight mini-train scenes. Only scene-0103 and scene-0916 are
  unseen. The lower val mean score (0.59 vs 0.72) is consistent with that.
- **Connectivity:** successor links are 0 everywhere. This is expected for
  path-wise (LaneGAP) centerlines, because each prediction is a whole path
  through the view. Topology therefore shows up as forks (paths sharing a
  start) and merges (paths sharing an end), not as end-to-start chains.
- **No accuracy number:** no quantitative map mAP was computed. It needs the
  nuScenes map expansion (v1.3), which is not under `data/nuscenes/maps`.

## Latency and memory (A10G, 4 vCPU, 10 warm-up frames, 394 timed)

| Stage | Standalone mean / p50 / p95 / max (ms) | Concurrent with BEVFusion benchmark, mean / p95 (ms) |
| --- | --- | --- |
| Preprocess (6 JPEG decode, resize, normalize, pad, collate) | 198.1 / 195.0 / 214.9 / 262.9 | 212.9 / 240.5 |
| Network (`torch.cuda.synchronize` bracketed) | 75.7 / 75.2 / 79.3 / 85.1 | 92.1 / 140.4 |
| Post-process (decode, centerline graph) | 4.4 / 3.5 / 10.0 / 179.8 | — |
| **Total** | **278.2 / 274.9 / 297.1 / 470.1** | **308.0 / 350.1** |

Profiling one scene splits the preprocessing into:

- **JPEG decode:** 121 ms for six `mmcv.imread` calls at 1600×900.
- **Resize:** 38 ms.
- **Normalize:** 18 ms.
- **Pad and format:** 12 ms.

BEVFusion full frame (`benchmark_latency.py`, 81 val frames), alone versus
with MapTRv2 running at the same time:

| Stage | Alone, 2026-10-25 (mean / p95 ms) | With MapTRv2 (mean / p95 ms) |
| --- | --- | --- |
| Preprocess | 193.4 / 207.5 | 262.5 / 302.6 |
| Inference | 134.9 / 138.0 | 195.4 / 213.3 |
| Tracking | 9.2 / 10.7 | 11.8 / 15.1 |
| Perception | 337.4 / 356.6 | 469.6 / 529.4 |
| **Full frame** | **421.7 / 448.6** | **579.6 / 643.9** |

GPU memory:

- **Standalone:** MapTRv2 adds 2,567 MiB on the device, including the CUDA
  context.
- **All at once:** with the existing Foxglove playback, the BEVFusion
  benchmark, and MapTRv2 all running, the device peaked at 9,847 MiB of
  22,590 MiB.

Memory is not a constraint; CPU time and GPU time-sharing are.

A first measurement showed 532 ms of preprocessing and bimodal network times.
It was discarded. The cause was the memory sampler, which spawned
`nvidia-smi` every 50 ms on this 4-core host; it now reads one
`nvidia-smi -lms 100` process.

## Foxglove

`lane_foxglove.py` runs in the existing `.venv`. It replays one scene's
predictions in a loop on **port 8766**, so the BEVFusion stream on 8765 keeps
running, and it was running throughout this work. It publishes these topics
in the `lidar` frame:

- `/lane_map`: the map elements, colored by class, with alpha set by score
  and a score label.
- `/lane_graph`: centerline direction arrows and successor links.
- `/lane_lidar`: the keyframe LiDAR sweep.
- `/lane_CAM_FRONT`: the front camera image.

Verification:

- **Live server:** a WebSocket client subscribed to all four advertised
  topics and received messages on each.
- **MCAP:** `--mcap` writes one pass to a file; for scene-0103 that is 40
  frames and 19 MB.

## Installation and checkpoints

```bash
docker build -t maptrv2:cu111 projects/LaneMapping/docker     # ~15 min, 1st time
projects/LaneMapping/fetch_checkpoints.sh                     # gdown via uvx
```

| File | Size | sha256 |
| --- | ---: | --- |
| `checkpoints/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.pth` | 485 MB | `45001433ffe780a913cb609b83e8153e7754e2b04c80c2680e5cecb7a76cb93d` |
| `checkpoints/maptrv2/maptrv2_nusc_r50_24ep.pth` | 485 MB | `f143106dd60da13af9f1462aec0006ebd834fec6da9a2571b9a7a414e305a1dc` |

Fixes the Dockerfile applies to upstream MapTR:

- **mmdet3d install:** `python setup.py develop --no-deps`, because mmdet3d
  0.17 pins `numpy<1.20` and `numba==0.48`, which conflict with
  nuscenes-devkit.
- **numba import:** `numba.errors` is rewritten to `numba.core.errors` for
  numba 0.56.
- **GKT build:** the GKT `setup.py` GPU check is bypassed, because there is
  no GPU during `docker build`; `TORCH_CUDA_ARCH_LIST=8.6` is set.
- **Shapely:** pinned to 1.8.5 for nuscenes-devkit 1.1.10.
- **Tk:** `python3-tk` is installed, because the BEVFormer plugin imports
  `tkinter`.

## Reproduce

```bash
docker run --rm --gpus all --shm-size 8g -u $(id -u):$(id -g) \
    -e HOME=/tmp -e MPLCONFIGDIR=/tmp -v $PWD:/workspace -w /workspace \
    maptrv2:cu111 python projects/LaneMapping/maptr_infer.py   # --variant base
.venv/bin/python projects/LaneMapping/lane_foxglove.py --scene scene-0103
.venv/bin/python -m pytest tests/test_projects/test_lane_mapping
```

Outputs are in `demo/eval_out/lane_perception/`:

- **`centerline/` and `base/`:**
  - `preds/<scene>.json`: per frame, each element's `id`, `cls`, `score`, and
    `points` in metres in LIDAR_TOP, plus `centerline_graph` and `lidar2ego`.
  - `vis/*.png`: six camera views with projected elements, plus the BEV over
    LiDAR.
  - `summary.json`.
  - `latency.csv` (centerline run only).
- **`concurrent/`:** the contention run.

## Limitations

- No quantitative accuracy on mini: there is no map-expansion ground truth,
  and only 2 of the 10 scenes are unseen by the checkpoint.
- Camera overlays project elements at a fixed z = -1.8 m below LIDAR_TOP, so
  they drift on slopes. The BEV view is exact.
- The connectivity graph is post-processing over thresholded predictions.
  Forks and merges depend on `share_tol` (0.75 m) and `node_radius` (1.5 m),
  which were set by hand, not tuned.
- The 30 × 60 m perception range is much shorter than BEVFusion's detection
  range.
- Each timed frame decodes its JPEGs from the page cache. A live feed would
  share the decode with BEVFusion.

## Next steps (no training)

1. **Share preprocessing with BEVFusion.** Both models decode the same six
   JPEGs, and decode plus resize is 160 ms of MapTRv2's 198 ms. A shared
   decode, or a decoder that downsamples during decode (PIL `draft` or
   libjpeg-turbo at half scale), removes most of it. The half-scale decode
   changes the inputs slightly and must be checked against the current
   outputs.
2. **Run as a sidecar at a lower rate.** For example, map at 2 Hz, offset
   from BEVFusion frames, with ego-motion compensation of the last map.
   Static map elements tolerate that much latency better than detections do.
3. **Speed up the 76 ms network** with TensorRT or ONNX for the backbone and
   depth net. This is a deployment change, not training.
4. **Fuse into the existing Foxglove stream only after (1) and (2)** keep the
   full frame under 500 ms.
5. **Optionally download the nuScenes map expansion v1.3** to score chamfer
   mAP on the two val scenes.

## Files added

- `projects/LaneMapping/docker/Dockerfile`
- `projects/LaneMapping/fetch_checkpoints.sh`
- `projects/LaneMapping/maptr_infer.py`
- `projects/LaneMapping/centerline_graph.py`
- `projects/LaneMapping/lane_foxglove.py`
- `tests/test_projects/test_lane_mapping/test_centerline_graph.py`
- `AI_summary/2026-10-31-lane-perception-eval.md`
