"""Chamfer matching for one class of polylines.

The distance is the average of the two directed mean nearest-point distances.
A pair matches when that distance is within a threshold. Matching is greedy
and one to one. The thresholds 0.5 m, 1.0 m, and 1.5 m are the ones MapTRv2
uses for average precision. They are fixed here and are not chosen from the
validation scenes.
"""
import numpy as np
from scipy.spatial.distance import cdist


def chamfer(pred, gt):
    dist = cdist(np.asarray(pred, float), np.asarray(gt, float))
    return 0.5 * (float(dist.min(axis=1).mean()) + float(dist.min(axis=0).mean()))


def match(preds, gts, threshold):
    """Return true-positive count and the chamfer of each matched pair."""
    if not preds or not gts:
        return 0, []
    pairs = []
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            dist = chamfer(pred, gt)
            if dist <= threshold:
                pairs.append((dist, i, j))
    pairs.sort()
    used_pred, used_gt, matched = set(), set(), []
    for dist, i, j in pairs:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matched.append(dist)
    return len(matched), matched


def scores(preds, gts, thresholds):
    """Precision, recall, and F1 at each distance threshold."""
    rows = []
    for threshold in thresholds:
        tp, matched = match(preds, gts, threshold)
        pred_n, gt_n = len(preds), len(gts)
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / gt_n if gt_n else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        rows.append(dict(
            threshold_m=threshold, tp=tp, pred=pred_n, gt=gt_n,
            precision=precision, recall=recall, f1=f1,
            matched_m=matched,
            chamfer_matched_m=(float(np.mean(matched)) if matched else None)))
    nearest = []
    for pred in preds:
        if gts:
            nearest.append(min(chamfer(pred, gt) for gt in gts))
    return dict(
        thresholds=rows,
        nearest_m=nearest,
        nearest_chamfer_mean_m=(float(np.mean(nearest)) if nearest else None),
        nearest_chamfer_median_m=(float(np.median(nearest)) if nearest else None))


def _rates(tp, pred_n, gt_n, matched):
    precision = tp / pred_n if pred_n else 0.0
    recall = tp / gt_n if gt_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return dict(
        tp=tp, pred=pred_n, gt=gt_n, precision=precision, recall=recall, f1=f1,
        chamfer_matched_m=(float(np.mean(matched)) if matched else None))


def micro_scores(frames, thresholds):
    """Match inside each frame, then sum counts across frames.

    Predictions are not allowed to match ground truth from another frame.
    ``frames`` is a list of ``(predictions, ground_truth)`` polyline lists.
    """
    nearest = []
    buckets = {threshold: dict(tp=0, pred=0, gt=0, matched=[])
               for threshold in thresholds}
    for preds, gts in frames:
        result = scores(preds, gts, thresholds)
        nearest.extend(result['nearest_m'])
        for row in result['thresholds']:
            bucket = buckets[row['threshold_m']]
            bucket['tp'] += row['tp']
            bucket['pred'] += row['pred']
            bucket['gt'] += row['gt']
            bucket['matched'].extend(row['matched_m'])
    rows = []
    for threshold in thresholds:
        bucket = buckets[threshold]
        rows.append(dict(
            threshold_m=threshold,
            **_rates(bucket['tp'], bucket['pred'], bucket['gt'], bucket['matched'])))
    return dict(
        thresholds=rows,
        nearest_chamfer_mean_m=(float(np.mean(nearest)) if nearest else None),
        nearest_chamfer_median_m=(float(np.median(nearest)) if nearest else None))
