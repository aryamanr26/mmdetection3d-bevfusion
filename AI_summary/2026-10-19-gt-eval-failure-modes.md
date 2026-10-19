# Ground-truth evaluation and failure modes

## Change intent

Add a repeatable offline evaluator that matches BEVFusion predictions to
nuScenes annotations by class and 3D IoU, then rank the dominant failure modes
with representative timestamps for the Foxglove playback scene.

Parent commit: `aa3ecd16`
(`feat(bevfusion): stream the six camera JPEGs to Foxglove`).

## Main changes

- `projects/BEVFusion/demo/eval_gt_vs_pred.py`
  - Infers the full log of `--pcd` with `--play-scene`.
  - Matches predictions to GT by class + greedy 3D IoU.
  - Reports precision, recall, FP/FN, localization error, confidence bins,
    and breakdowns by class and distance.
  - Writes `demo/eval_out/metrics.json`, `failures.json`, and `report.md`.
  - Remaps info `bbox_label_3d` through `dataset.metainfo['categories']` onto
    the detector class order (those two orders differ in MMDetection3D).
- `projects/BEVFusion/demo/foxglove_stream.py`
  - Applies the same GT label remap so `/ground_truth` class names match the
    detector vocabulary.

## Baseline metrics

Scene `n015-2018-07-24-11-22-45+0800`, 39 keyframes, score threshold 0.2,
primary IoU 0.5, official camera+LiDAR checkpoint:

| Metric | Value |
| --- | --- |
| Precision | 0.709 |
| Recall | 0.623 |
| F1 | 0.663 |
| TP / FP / FN | 2103 / 862 / 1274 |
| Mean IoU (TP) | 0.670 |
| Mean center error | 0.157 m |
| Mean yaw error | 17.43 deg |
| Mean TP score | 0.574 |
| Mean FP score | 0.380 |

IoU sensitivity: at 0.25, F1 rises to 0.834; at 0.7 it falls to 0.244.

Strongest classes at IoU 0.5: truck (F1 0.829), car (0.825). Weakest:
bicycle (0.168), bus (0.414). Distance collapse: 0–20 m F1 0.836 versus
50 m+ F1 0.255.

## Ranked failure modes

| Rank | Mode | Count | Example timestamp | Example |
| --- | --- | --- | --- | --- |
| 1 | FN overall | 1274 | 1532402927.647951 | pedestrian @ 62.3 m |
| 2 | FN + sparse LiDAR (≤5 pts) | 1086 | 1532402927.647951 | pedestrian @ 62.3 m |
| 3 | FN + distant (≥30 m) | 862 | 1532402927.647951 | pedestrian @ 62.3 m |
| 4 | FP overall | 862 | 1532402927.647951 | barrier @ 11.2 m |
| 5 | FN + crowded scene (≥80 GT) | 838 | 1532402930.648325 | traffic_cone @ 40.9 m |
| 6 | FN + small footprint (≤1 m²) | 568 | 1532402927.647951 | pedestrian @ 62.3 m |
| 7 | FP + crowded scene | 554 | 1532402930.648325 | pedestrian @ 27.1 m |
| 8 | FP + distant | 480 | 1532402927.647951 | pedestrian @ 43.2 m |
| 9 | FP + low confidence (<0.3) | 344 | 1532402927.647951 | pedestrian @ 41.5 m |
| 10 | Poor yaw among matches | 258 | 1532402927.647951 | traffic_cone @ 16.8 m |

True class swaps at IoU ≥ 0.5 were rare (one `bus->truck` at
`1532402936.198962`).

## How to reproduce

```bash
.venv/bin/python projects/BEVFusion/demo/eval_gt_vs_pred.py --play-scene \
  --out-dir demo/eval_out
```

## Excluded

- Generated `demo/eval_out/` artifacts (gitignored via `demo/*_out/`).
- Official nuScenes center-distance mAP/NDS (`tools/test.py` + `NuScenesMetric`).
