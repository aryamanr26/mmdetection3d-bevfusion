# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluate BEVFusion predictions against nuScenes ground truth.

Matches predictions to annotations by class and 3D IoU, then reports
precision, recall, FP/FN counts, localization error, confidence behavior,
and breakdowns by class and distance. Also writes a ranked failure-mode
catalog with scene timestamps for offline review.

Example:
    python projects/BEVFusion/demo/eval_gt_vs_pred.py --play-scene
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from foxglove_stream import (  # noqa: E402
    CHECKPOINT, CLASS_NAMES, CONFIG, INFOS, _SensorCache, load_detector,
    scene_rows)
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmdet3d.structures import LiDARInstance3DBoxes  # noqa: E402

DEFAULT_PCD = (
    'demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__'
    '1532402927647951.pcd.bin')
IOU_THRESHOLDS = (0.25, 0.5, 0.7)
PRIMARY_IOU = 0.5
DISTANCE_BINS = (
    ('0-20m', 0.0, 20.0),
    ('20-30m', 20.0, 30.0),
    ('30-50m', 30.0, 50.0),
    ('50m+', 50.0, float('inf')),
)
SMALL_FOOTPRINT_M2 = 1.0
SPARSE_LIDAR_PTS = 5
CROWDED_GT = 80
POOR_CENTER_M = 1.0
POOR_YAW_DEG = 30.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcd', default=DEFAULT_PCD)
    parser.add_argument('--infos', default=INFOS)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--score-thr', type=float, default=0.2)
    parser.add_argument('--iou-thr', type=float, default=PRIMARY_IOU)
    parser.add_argument('--sweeps', type=int, default=9)
    parser.add_argument(
        '--play-scene', action='store_true',
        help='Evaluate every keyframe in the log of --pcd.')
    parser.add_argument(
        '--out-dir', default='demo/eval_out',
        help='Directory for metrics JSON/Markdown and failure catalog.')
    return parser.parse_args()


def _yaw_diff_rad(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _distance(box):
    return float(math.hypot(box[0], box[1]))


def _footprint(box):
    return float(abs(box[3]) * abs(box[4]))


def _distance_bin(dist):
    for name, lo, hi in DISTANCE_BINS:
        if lo <= dist < hi:
            return name
    return DISTANCE_BINS[-1][0]


def info_label_to_model(class_names, metainfo=None):
    """Map nuScenes info ``bbox_label_3d`` ids onto the model class order.

    MMDetection3D infos store labels via ``metainfo['categories']``, which is
    not the same order as ``metainfo['classes']`` / the detector head.
    """
    categories = (metainfo or {}).get('categories') or {}
    if categories:
        id_to_name = {int(v): str(k) for k, v in categories.items()}
    else:
        id_to_name = {i: name for i, name in enumerate(class_names)}
    name_to_model = {name: i for i, name in enumerate(class_names)}
    mapping = {}
    for info_id, name in id_to_name.items():
        if name in name_to_model:
            mapping[info_id] = (name_to_model[name], name)
    return mapping


def gt_records_from_info(info, class_names, metainfo=None):
    """Valid GT boxes in LiDAR bottom-center convention, with metadata."""
    label_map = info_label_to_model(class_names, metainfo)
    rows, meta = [], []
    for obj in info.get('instances', []):
        if not obj.get('bbox_3d_isvalid', True):
            continue
        raw_label = int(obj.get('bbox_label_3d', -1))
        mapped = label_map.get(raw_label)
        if mapped is None:
            continue
        label, name = mapped
        rows.append(obj['bbox_3d'][:7])
        meta.append({
            'label': label,
            'class_name': name,
            'info_label': raw_label,
            'num_lidar_pts': int(obj.get('num_lidar_pts', 0)),
            'num_radar_pts': int(obj.get('num_radar_pts', 0)),
            'velocity': list(obj.get('velocity', [0.0, 0.0])),
        })
    if not rows:
        return []
    converted = LiDARInstance3DBoxes(
        np.asarray(rows, dtype=np.float32),
        box_dim=7,
        origin=(0.5, 0.5, 0.5),
    ).tensor.detach().cpu().numpy()
    records = []
    for item, box in zip(meta, converted):
        bbox = box[:7].astype(np.float64)
        records.append({
            **item,
            'bbox_3d': bbox,
            'distance': _distance(bbox),
            'footprint': _footprint(bbox),
            'distance_bin': _distance_bin(_distance(bbox)),
        })
    return records


def infer_predictions(model, dataset, index, score_thr, class_names,
                      sweeps_num, cache):
    """Run one sample and keep scores for confidence analysis."""
    info = dataset.get_data_info(index)
    from foxglove_stream import (
        _load_cameras, _load_sweep_points, _pack_model_inputs)

    points = _load_sweep_points(info, sweeps_num, cache)
    cameras = _load_cameras(info, cache)
    packed = _pack_model_inputs(dataset, info, points, cameras, cache)
    with torch.inference_mode():
        result = model.test_step(pseudo_collate([packed]))[0]
        pred = result.pred_instances_3d
        keep = pred.scores_3d >= score_thr
        labels = pred.labels_3d[keep].detach().cpu().numpy().astype(int)
        scores = pred.scores_3d[keep].detach().cpu().numpy().astype(np.float64)
        bboxes = pred.bboxes_3d.tensor[keep].detach().cpu().numpy().astype(
            np.float64)
    records = []
    for label, score, bbox in zip(labels, scores, bboxes):
        box = bbox[:7]
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        records.append({
            'label': int(label),
            'class_name': name,
            'score': float(score),
            'bbox_3d': box,
            'distance': _distance(box),
            'footprint': _footprint(box),
            'distance_bin': _distance_bin(_distance(box)),
        })
    return records


def iou_matrix(pred_boxes, gt_boxes):
    if not pred_boxes or not gt_boxes:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    preds = LiDARInstance3DBoxes(
        torch.as_tensor(np.stack(pred_boxes), dtype=torch.float32), box_dim=7)
    gts = LiDARInstance3DBoxes(
        torch.as_tensor(np.stack(gt_boxes), dtype=torch.float32), box_dim=7)
    return LiDARInstance3DBoxes.overlaps(preds, gts).detach().cpu().numpy()


def match_frame(preds, gts, iou_thr):
    """Greedy class-aware matching by descending IoU."""
    matches = []
    unmatched_pred = set(range(len(preds)))
    unmatched_gt = set(range(len(gts)))
    if not preds or not gts:
        return matches, sorted(unmatched_pred), sorted(unmatched_gt)

    ious = iou_matrix([p['bbox_3d'] for p in preds],
                      [g['bbox_3d'] for g in gts])
    candidates = []
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            if pred['label'] != gt['label']:
                continue
            iou = float(ious[i, j])
            if iou >= iou_thr:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    for iou, i, j in candidates:
        if i not in unmatched_pred or j not in unmatched_gt:
            continue
        unmatched_pred.remove(i)
        unmatched_gt.remove(j)
        pred = preds[i]
        gt = gts[j]
        pb, gb = pred['bbox_3d'], gt['bbox_3d']
        matches.append({
            'pred_index': i,
            'gt_index': j,
            'class_name': pred['class_name'],
            'iou': iou,
            'score': pred['score'],
            'distance': gt['distance'],
            'distance_bin': gt['distance_bin'],
            'center_error_m': float(np.linalg.norm(pb[:3] - gb[:3])),
            'xy_error_m': float(np.hypot(pb[0] - gb[0], pb[1] - gb[1])),
            'z_error_m': float(abs(pb[2] - gb[2])),
            'size_error_m': float(np.linalg.norm(pb[3:6] - gb[3:6])),
            'yaw_error_deg': math.degrees(_yaw_diff_rad(pb[6], gb[6])),
            'pred': pred,
            'gt': gt,
        })
    return matches, sorted(unmatched_pred), sorted(unmatched_gt)


def wrong_class_pairs(preds, gts, unmatched_pred, unmatched_gt, iou_thr):
    """Find high-IoU pairs among leftovers that disagree on class."""
    if not unmatched_pred or not unmatched_gt:
        return []
    sub_preds = [preds[i] for i in unmatched_pred]
    sub_gts = [gts[j] for j in unmatched_gt]
    ious = iou_matrix([p['bbox_3d'] for p in sub_preds],
                      [g['bbox_3d'] for g in sub_gts])
    pairs = []
    used_p, used_g = set(), set()
    candidates = []
    for a, i in enumerate(unmatched_pred):
        for b, j in enumerate(unmatched_gt):
            iou = float(ious[a, b])
            if iou >= iou_thr and preds[i]['label'] != gts[j]['label']:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    for iou, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append({
            'iou': iou,
            'pred_class': preds[i]['class_name'],
            'gt_class': gts[j]['class_name'],
            'score': preds[i]['score'],
            'distance': gts[j]['distance'],
            'pred': preds[i],
            'gt': gts[j],
        })
    return pairs


def empty_counters():
    return {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'gt': 0,
        'pred': 0,
        'scores_tp': [],
        'scores_fp': [],
        'center_errors': [],
        'xy_errors': [],
        'yaw_errors': [],
        'ious': [],
    }


def accumulate(bucket, matches, fp_preds, fn_gts):
    bucket['tp'] += len(matches)
    bucket['fp'] += len(fp_preds)
    bucket['fn'] += len(fn_gts)
    bucket['gt'] += len(matches) + len(fn_gts)
    bucket['pred'] += len(matches) + len(fp_preds)
    bucket['scores_tp'].extend(m['score'] for m in matches)
    bucket['scores_fp'].extend(p['score'] for p in fp_preds)
    bucket['center_errors'].extend(m['center_error_m'] for m in matches)
    bucket['xy_errors'].extend(m['xy_error_m'] for m in matches)
    bucket['yaw_errors'].extend(m['yaw_error_deg'] for m in matches)
    bucket['ious'].extend(m['iou'] for m in matches)


def finalize_bucket(bucket):
    tp, fp, fn = bucket['tp'], bucket['fp'], bucket['fn']
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall /
          (precision + recall)) if (precision + recall) else 0.0

    def mean(values):
        return float(np.mean(values)) if values else None

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'gt': bucket['gt'],
        'pred': bucket['pred'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean(bucket['ious']),
        'mean_center_error_m': mean(bucket['center_errors']),
        'mean_xy_error_m': mean(bucket['xy_errors']),
        'mean_yaw_error_deg': mean(bucket['yaw_errors']),
        'mean_score_tp': mean(bucket['scores_tp']),
        'mean_score_fp': mean(bucket['scores_fp']),
    }


def classify_failures(frame_meta, matches, fp_preds, fn_gts, wrong_class,
                      score_thr):
    """Tag false negatives/positives and poor matches for the catalog."""
    events = []
    crowded = frame_meta['gt_count'] >= CROWDED_GT
    for gt in fn_gts:
        tags = ['false_negative']
        if gt['distance'] >= 30.0:
            tags.append('distant')
        if gt['footprint'] <= SMALL_FOOTPRINT_M2:
            tags.append('small_object')
        if gt['num_lidar_pts'] <= SPARSE_LIDAR_PTS:
            tags.append('sparse_lidar')
        if crowded:
            tags.append('crowded_scene')
        events.append({
            'kind': 'false_negative',
            'tags': tags,
            'class_name': gt['class_name'],
            'distance_m': gt['distance'],
            'distance_bin': gt['distance_bin'],
            'num_lidar_pts': gt['num_lidar_pts'],
            'footprint_m2': gt['footprint'],
            'score': None,
            'iou': None,
            **frame_meta,
        })
    for pred in fp_preds:
        tags = ['false_positive']
        if pred['distance'] >= 30.0:
            tags.append('distant')
        if pred['footprint'] <= SMALL_FOOTPRINT_M2:
            tags.append('small_object')
        if pred['score'] < score_thr + 0.1:
            tags.append('low_confidence')
        if crowded:
            tags.append('crowded_scene')
        events.append({
            'kind': 'false_positive',
            'tags': tags,
            'class_name': pred['class_name'],
            'distance_m': pred['distance'],
            'distance_bin': pred['distance_bin'],
            'num_lidar_pts': None,
            'footprint_m2': pred['footprint'],
            'score': pred['score'],
            'iou': None,
            **frame_meta,
        })
    for pair in wrong_class:
        events.append({
            'kind': 'wrong_class',
            'tags': ['wrong_class', 'camera_lidar_disagreement_candidate'],
            'class_name': f"{pair['pred_class']}->{pair['gt_class']}",
            'distance_m': pair['distance'],
            'distance_bin': _distance_bin(pair['distance']),
            'num_lidar_pts': pair['gt']['num_lidar_pts'],
            'footprint_m2': pair['gt']['footprint'],
            'score': pair['score'],
            'iou': pair['iou'],
            **frame_meta,
        })
    for match in matches:
        poor = []
        if match['center_error_m'] >= POOR_CENTER_M:
            poor.append('poor_localization')
        if match['yaw_error_deg'] >= POOR_YAW_DEG:
            poor.append('poor_yaw')
        if match['iou'] < 0.7 and match['center_error_m'] >= 0.5:
            poor.append('loose_match')
        if not poor:
            continue
        events.append({
            'kind': 'poor_localization',
            'tags': poor,
            'class_name': match['class_name'],
            'distance_m': match['distance'],
            'distance_bin': match['distance_bin'],
            'num_lidar_pts': match['gt']['num_lidar_pts'],
            'footprint_m2': match['gt']['footprint'],
            'score': match['score'],
            'iou': match['iou'],
            'center_error_m': match['center_error_m'],
            'yaw_error_deg': match['yaw_error_deg'],
            **frame_meta,
        })
    return events


def rank_failure_modes(events):
    counter = Counter()
    examples = defaultdict(list)
    for event in events:
        for tag in event['tags']:
            if tag in ('false_negative', 'false_positive'):
                continue
            key = f"{event['kind']}:{tag}"
            counter[key] += 1
            if len(examples[key]) < 5:
                examples[key].append({
                    'timestamp': event['timestamp'],
                    'lidar_name': event['lidar_name'],
                    'class_name': event['class_name'],
                    'distance_m': event['distance_m'],
                    'score': event.get('score'),
                    'iou': event.get('iou'),
                    'num_lidar_pts': event.get('num_lidar_pts'),
                })
    # Also rank pure FN/FP volume.
    for kind in ('false_negative', 'false_positive', 'wrong_class',
                 'poor_localization'):
        count = sum(1 for e in events if e['kind'] == kind)
        if count:
            counter[f'{kind}:all'] += count
            for event in events:
                if event['kind'] != kind:
                    continue
                key = f'{kind}:all'
                if len(examples[key]) < 5:
                    examples[key].append({
                        'timestamp': event['timestamp'],
                        'lidar_name': event['lidar_name'],
                        'class_name': event['class_name'],
                        'distance_m': event['distance_m'],
                        'score': event.get('score'),
                        'iou': event.get('iou'),
                        'num_lidar_pts': event.get('num_lidar_pts'),
                    })
    ranked = []
    for key, count in counter.most_common():
        ranked.append({
            'mode': key,
            'count': count,
            'examples': examples[key],
        })
    return ranked


def confidence_table(matches, fp_preds, bins=None):
    bins = bins or [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        tp = sum(1 for m in matches if lo <= m['score'] < hi)
        fp = sum(1 for p in fp_preds if lo <= p['score'] < hi)
        rows.append({
            'score_bin': f'{lo:.1f}-{hi:.1f}',
            'tp': tp,
            'fp': fp,
            'precision': tp / (tp + fp) if (tp + fp) else None,
        })
    return rows


def select_rows(data_list, lidar_name, play_scene):
    if play_scene:
        return scene_rows(data_list, lidar_name)
    log, rows = scene_rows(data_list, lidar_name)
    base = os.path.basename(lidar_name)
    for index, info, name in rows:
        if name == base:
            return log, [(index, info, name)]
    raise FileNotFoundError(f'Keyframe not found: {base}')


def render_markdown(report):
    overall = report['overall']
    lines = [
        '# BEVFusion GT vs prediction evaluation',
        '',
        f"- Scene: `{report['scene']}`",
        f"- Frames: {report['num_frames']}",
        f"- Score threshold: {report['score_thr']}",
        f"- Primary IoU threshold: {report['iou_thr']}",
        '',
        '## Overall (primary IoU)',
        '',
        f"| metric | value |",
        f"| --- | --- |",
        f"| TP | {overall['tp']} |",
        f"| FP | {overall['fp']} |",
        f"| FN | {overall['fn']} |",
        f"| Precision | {overall['precision']:.3f} |",
        f"| Recall | {overall['recall']:.3f} |",
        f"| F1 | {overall['f1']:.3f} |",
        (
            f"| Mean IoU (TP) | {overall['mean_iou']:.3f} |"
            if overall['mean_iou'] is not None else
            '| Mean IoU (TP) | n/a |'
        ),
        (
            f"| Mean center error (m) | {overall['mean_center_error_m']:.3f} |"
            if overall['mean_center_error_m'] is not None else
            '| Mean center error (m) | n/a |'
        ),
        (
            f"| Mean yaw error (deg) | {overall['mean_yaw_error_deg']:.2f} |"
            if overall['mean_yaw_error_deg'] is not None else
            '| Mean yaw error (deg) | n/a |'
        ),
        (
            f"| Mean TP score | {overall['mean_score_tp']:.3f} |"
            if overall['mean_score_tp'] is not None else
            '| Mean TP score | n/a |'
        ),
        (
            f"| Mean FP score | {overall['mean_score_fp']:.3f} |"
            if overall['mean_score_fp'] is not None else
            '| Mean FP score | n/a |'
        ),
        '',
        '## By IoU threshold',
        '',
        '| IoU | Precision | Recall | F1 | TP | FP | FN |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for thr, metrics in report['by_iou'].items():
        lines.append(
            f"| {thr} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['tp']} | {metrics['fp']} | "
            f"{metrics['fn']} |")
    lines.extend([
        '',
        '## By class (primary IoU)',
        '',
        '| Class | GT | Pred | P | R | F1 | TP | FP | FN | mean center err (m) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ])
    for name, metrics in report['by_class'].items():
        center = (f"{metrics['mean_center_error_m']:.3f}"
                  if metrics['mean_center_error_m'] is not None else 'n/a')
        lines.append(
            f"| {name} | {metrics['gt']} | {metrics['pred']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['tp']} | {metrics['fp']} | "
            f"{metrics['fn']} | {center} |")
    lines.extend([
        '',
        '## By distance (primary IoU)',
        '',
        '| Distance | GT | Pred | P | R | F1 | TP | FP | FN |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ])
    for name, metrics in report['by_distance'].items():
        lines.append(
            f"| {name} | {metrics['gt']} | {metrics['pred']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['tp']} | {metrics['fp']} | "
            f"{metrics['fn']} |")
    lines.extend([
        '',
        '## Confidence bins (primary IoU)',
        '',
        '| Score bin | TP | FP | Precision |',
        '| --- | --- | --- | --- |',
    ])
    for row in report['confidence_bins']:
        prec = ('n/a' if row['precision'] is None
                else f"{row['precision']:.3f}")
        lines.append(
            f"| {row['score_bin']} | {row['tp']} | {row['fp']} | {prec} |")
    lines.extend([
        '',
        '## Ranked failure modes',
        '',
        '| Rank | Mode | Count | Example timestamp | Example class | Dist (m) |',
        '| --- | --- | --- | --- | --- | --- |',
    ])
    for rank, item in enumerate(report['failure_modes'], 1):
        example = item['examples'][0] if item['examples'] else {}
        lines.append(
            f"| {rank} | `{item['mode']}` | {item['count']} | "
            f"{example.get('timestamp', 'n/a')} | "
            f"{example.get('class_name', 'n/a')} | "
            f"{example.get('distance_m', float('nan')):.1f} |")
    lines.append('')
    return '\n'.join(lines)


def evaluate(args):
    with open(args.infos, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    lidar_name = os.path.basename(args.pcd)
    log, rows = select_rows(data_list, lidar_name, args.play_scene)
    print(f'Scene {log}: evaluating {len(rows)} keyframe(s)', flush=True)

    model, dataset, class_names = load_detector(
        args.infos, args.config, args.checkpoint)
    cache = _SensorCache()

    overall_raw = empty_counters()
    by_class_raw = defaultdict(empty_counters)
    by_distance_raw = defaultdict(empty_counters)
    by_iou_raw = {thr: empty_counters() for thr in IOU_THRESHOLDS}
    all_matches = []
    all_fp = []
    all_events = []
    frame_summaries = []

    for number, (index, info, name) in enumerate(rows, 1):
        preds = infer_predictions(
            model, dataset, index, args.score_thr, class_names, args.sweeps,
            cache)
        gts = gt_records_from_info(
            info, class_names, metainfo=getattr(dataset, 'metainfo', None))
        matches, unmatched_pred, unmatched_gt = match_frame(
            preds, gts, args.iou_thr)
        fp_preds = [preds[i] for i in unmatched_pred]
        fn_gts = [gts[j] for j in unmatched_gt]
        wrong = wrong_class_pairs(
            preds, gts, unmatched_pred, unmatched_gt, args.iou_thr)

        accumulate(overall_raw, matches, fp_preds, fn_gts)
        for match in matches:
            accumulate(by_class_raw[match['class_name']], [match], [], [])
            accumulate(by_distance_raw[match['distance_bin']], [match], [], [])
        for pred in fp_preds:
            accumulate(by_class_raw[pred['class_name']], [], [pred], [])
            accumulate(by_distance_raw[pred['distance_bin']], [], [pred], [])
        for gt in fn_gts:
            accumulate(by_class_raw[gt['class_name']], [], [], [gt])
            accumulate(by_distance_raw[gt['distance_bin']], [], [], [gt])

        for thr in IOU_THRESHOLDS:
            m, up, ug = match_frame(preds, gts, thr)
            accumulate(
                by_iou_raw[thr], m, [preds[i] for i in up],
                [gts[j] for j in ug])

        frame_meta = {
            'frame_index': number - 1,
            'infos_index': index,
            'lidar_name': name,
            'timestamp': float(info['timestamp']),
            'gt_count': len(gts),
            'pred_count': len(preds),
        }
        events = classify_failures(
            frame_meta, matches, fp_preds, fn_gts, wrong, args.score_thr)
        all_events.extend(events)
        all_matches.extend(matches)
        all_fp.extend(fp_preds)
        summary = {
            **frame_meta,
            'tp': len(matches),
            'fp': len(fp_preds),
            'fn': len(fn_gts),
            'wrong_class': len(wrong),
            'precision': len(matches) / len(preds) if preds else 0.0,
            'recall': len(matches) / len(gts) if gts else 0.0,
        }
        frame_summaries.append(summary)
        print(
            f'[{number}/{len(rows)}] {name} GT={len(gts)} pred={len(preds)} '
            f'TP={len(matches)} FP={len(fp_preds)} FN={len(fn_gts)} '
            f'P={summary["precision"]:.3f} R={summary["recall"]:.3f}',
            flush=True)

    class_names_present = sorted(
        set(by_class_raw) | set(CLASS_NAMES),
        key=lambda n: CLASS_NAMES.index(n) if n in CLASS_NAMES else 999)
    report = {
        'scene': log,
        'num_frames': len(rows),
        'score_thr': args.score_thr,
        'iou_thr': args.iou_thr,
        'iou_thresholds': list(IOU_THRESHOLDS),
        'overall': finalize_bucket(overall_raw),
        'by_iou': {
            str(thr): finalize_bucket(by_iou_raw[thr]) for thr in IOU_THRESHOLDS
        },
        'by_class': {
            name: finalize_bucket(by_class_raw[name])
            for name in class_names_present if by_class_raw[name]['gt']
            or by_class_raw[name]['pred']
        },
        'by_distance': {
            name: finalize_bucket(by_distance_raw[name])
            for name, _, _ in DISTANCE_BINS
            if by_distance_raw[name]['gt'] or by_distance_raw[name]['pred']
        },
        'confidence_bins': confidence_table(all_matches, all_fp),
        'failure_modes': rank_failure_modes(all_events),
        'frames': frame_summaries,
    }
    return report, all_events


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report, events = evaluate(args)

    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    failures_path = os.path.join(args.out_dir, 'failures.json')
    markdown_path = os.path.join(args.out_dir, 'report.md')

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with open(failures_path, 'w', encoding='utf-8') as f:
        json.dump({
            'failure_modes': report['failure_modes'],
            'events': events,
        }, f, indent=2)
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(render_markdown(report))

    overall = report['overall']
    print('\n=== Baseline metrics ===', flush=True)
    print(
        f"P={overall['precision']:.3f} R={overall['recall']:.3f} "
        f"F1={overall['f1']:.3f} TP={overall['tp']} FP={overall['fp']} "
        f"FN={overall['fn']}",
        flush=True)
    print('Top failure modes:', flush=True)
    for item in report['failure_modes'][:8]:
        example = item['examples'][0] if item['examples'] else {}
        print(
            f"  {item['count']:4d}  {item['mode']}  "
            f"e.g. t={example.get('timestamp', 'n/a')} "
            f"{example.get('class_name', '')}",
            flush=True)
    print(f'Wrote {metrics_path}', flush=True)
    print(f'Wrote {failures_path}', flush=True)
    print(f'Wrote {markdown_path}', flush=True)


if __name__ == '__main__':
    main()
