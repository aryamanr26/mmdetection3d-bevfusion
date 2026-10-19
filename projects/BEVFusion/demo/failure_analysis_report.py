# Copyright (c) OpenMMLab. All rights reserved.
"""Per-object failure analysis for BEVFusion on a nuScenes scene.

Produces unique ground-truth diagnostics, multi-label failure tags with
co-occurrence statistics, and confidence-threshold curves. Tags are allowed
to overlap; unique false-negative counts are always reported separately.

Example:
    python projects/BEVFusion/demo/failure_analysis_report.py --play-scene
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_gt_vs_pred import (  # noqa: E402
    DISTANCE_BINS, IOU_THRESHOLDS, PRIMARY_IOU, SMALL_FOOTPRINT_M2,
    SPARSE_LIDAR_PTS, CROWDED_GT, POOR_CENTER_M, POOR_YAW_DEG, DEFAULT_PCD,
    _distance, _distance_bin, _footprint, _yaw_diff_rad, finalize_bucket,
    empty_counters, accumulate, gt_records_from_info, infer_predictions,
    iou_matrix, match_frame, select_rows)
from foxglove_stream import (  # noqa: E402
    CHECKPOINT, CONFIG, INFOS, _SensorCache, load_detector)

SCORE_SWEEP = [round(x, 2) for x in np.arange(0.05, 0.95, 0.05)]
CANDIDATE_SCORE_THR = 0.01
SPARSE_BINS = (
    ('0-2pts', 0, 2),
    ('3-5pts', 3, 5),
    ('6+pts', 6, 10**9),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcd', default=DEFAULT_PCD)
    parser.add_argument('--infos', default=INFOS)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--score-thr', type=float, default=0.2,
                        help='Operating-point score used for FP/FN tags.')
    parser.add_argument('--iou-thr', type=float, default=PRIMARY_IOU)
    parser.add_argument('--sweeps', type=int, default=9)
    parser.add_argument('--play-scene', action='store_true')
    parser.add_argument('--out-dir', default='demo/eval_out')
    return parser.parse_args()


def sparse_bin(num_pts):
    for name, lo, hi in SPARSE_BINS:
        if lo <= num_pts <= hi:
            return name
    return SPARSE_BINS[-1][0]


def lidar2img_matrices(info):
    mats = {}
    shapes = {}
    for cam_id, cam_item in info.get('images', {}).items():
        intrinsic = np.asarray(cam_item['cam2img'], dtype=np.float64)
        if intrinsic.shape == (3, 3):
            k4 = np.eye(4)
            k4[:3, :3] = intrinsic
            intrinsic = k4
        elif intrinsic.shape == (3, 4):
            k4 = np.eye(4)
            k4[:3, :] = intrinsic
            intrinsic = k4
        extrinsic = np.asarray(cam_item['lidar2cam'], dtype=np.float64)
        mats[cam_id] = intrinsic @ extrinsic
        # Default nuScenes camera size if unknown; overridden when available.
        shapes[cam_id] = (900, 1600)
    return mats, shapes


def camera_visibility(info, box):
    """Project the GT box center into each camera; report visible cameras."""
    mats, shapes = lidar2img_matrices(info)
    # Gravity-ish center from bottom-center box: z + h/2.
    center = np.array(
        [box[0], box[1], box[2] + 0.5 * box[5], 1.0], dtype=np.float64)
    visible = []
    depths = {}
    for cam_id, mat in mats.items():
        uvw = mat @ center
        depth = float(uvw[2])
        depths[cam_id] = depth
        if depth <= 0.1:
            continue
        u = float(uvw[0] / depth)
        v = float(uvw[1] / depth)
        h, w = shapes[cam_id]
        if 0 <= u < w and 0 <= v < h:
            visible.append(cam_id)
    return {
        'visible_cameras': visible,
        'num_visible_cameras': len(visible),
        'camera_visible': bool(visible),
        'min_positive_depth': min(
            (d for d in depths.values() if d > 0.1), default=None),
    }


def attach_gt_ids(gts, info, frame_index):
    token = str(info.get('token', f'frame{frame_index}'))
    timestamp = float(info['timestamp'])
    lidar_name = os.path.basename(info['lidar_points']['lidar_path'])
    crowded = len(gts) >= CROWDED_GT
    for local_idx, gt in enumerate(gts):
        gt['object_id'] = f'{token}#{local_idx}'
        gt['instance_index'] = local_idx
        gt['sample_token'] = token
        gt['timestamp'] = timestamp
        gt['lidar_name'] = lidar_name
        gt['frame_index'] = frame_index
        gt['crowded_scene'] = crowded
        gt['sparse_bin'] = sparse_bin(gt['num_lidar_pts'])
        gt.update(camera_visibility(info, gt['bbox_3d']))
    return gts


def best_same_class_pred(gt, preds, ious_col, min_iou=0.05):
    """Best same-class prediction for one GT, even below the operating score.

    Predictions with IoU below ``min_iou`` are ignored so a high-score box
    elsewhere in the scene is not reported as a near-miss.
    """
    best = None
    for i, pred in enumerate(preds):
        if pred['label'] != gt['label']:
            continue
        iou = float(ious_col[i])
        if iou < min_iou:
            continue
        pb = pred['bbox_3d']
        gb = gt['bbox_3d']
        cand = {
            'pred_index': i,
            'score': pred['score'],
            'class_name': pred['class_name'],
            'iou': iou,
            'center_error_m': float(np.linalg.norm(pb[:3] - gb[:3])),
            'xy_error_m': float(np.hypot(pb[0] - gb[0], pb[1] - gb[1])),
            'yaw_error_deg': math.degrees(_yaw_diff_rad(pb[6], gb[6])),
            'distance_m': pred['distance'],
        }
        if best is None or cand['iou'] > best['iou'] or (
                cand['iou'] == best['iou'] and cand['score'] > best['score']):
            best = cand
    return best


def failure_mechanism(status, best, score_thr, iou_thr):
    """Classify why a GT was missed or how a TP was achieved."""
    if status == 'tp':
        return 'matched'
    if best is None or best['iou'] < 0.05:
        return 'no_usable_prediction'
    if best['score'] < score_thr and best['iou'] >= iou_thr:
        return 'low_confidence'
    if best['score'] >= score_thr and best['iou'] < iou_thr:
        return 'inaccurate_box'
    if best['score'] < score_thr and best['iou'] < iou_thr:
        if best['iou'] >= 0.25:
            return 'low_confidence_and_weak_localization'
        return 'weak_or_no_localization'
    return 'other'


def tag_gt(gt, status, match=None):
    tags = []
    if status == 'fn':
        tags.append('false_negative')
        if gt['num_lidar_pts'] <= SPARSE_LIDAR_PTS:
            tags.append('sparse_lidar')
        if gt['distance'] >= 30.0:
            tags.append('distant')
        if gt['crowded_scene']:
            tags.append('crowded_scene')
        if gt['footprint'] <= SMALL_FOOTPRINT_M2:
            tags.append('small_object')
        if not gt['camera_visible']:
            tags.append('camera_invisible')
    elif status == 'tp' and match is not None:
        if match['yaw_error_deg'] >= POOR_YAW_DEG:
            tags.append('poor_yaw')
        if match['center_error_m'] >= POOR_CENTER_M:
            tags.append('poor_localization')
    return tags


def metrics_at_threshold(all_preds, all_gts, score_thr, iou_thr):
    raw = empty_counters()
    for preds, gts in zip(all_preds, all_gts):
        kept = [p for p in preds if p['score'] >= score_thr]
        matches, up, ug = match_frame(kept, gts, iou_thr)
        accumulate(
            raw, matches, [kept[i] for i in up], [gts[j] for j in ug])
    return finalize_bucket(raw)


def cooccurrence(tag_sets, focus_tags):
    """Count unique objects per tag and pairwise overlaps among focus tags."""
    unique = {tag: 0 for tag in focus_tags}
    pairs = {frozenset(pair): 0 for pair in combinations(focus_tags, 2)}
    triple = 0
    for tags in tag_sets:
        present = [t for t in focus_tags if t in tags]
        for tag in present:
            unique[tag] += 1
        for pair in combinations(present, 2):
            pairs[frozenset(pair)] += 1
        if len(present) >= 3:
            triple += 1
    return {
        'unique_per_tag': unique,
        'pairwise_overlap': {
            ' + '.join(sorted(k)): v for k, v in pairs.items() if v
        },
        'objects_with_ge3_focus_tags': triple,
    }


def summarize_scope(objects, matches_by_class_dist):
    by_class = Counter(o['class_name'] for o in objects)
    by_dist = Counter(o['distance_bin'] for o in objects)
    return {
        'num_frames': len({o['sample_token'] for o in objects}),
        'num_gt_objects': len(objects),
        'gt_by_class': dict(by_class),
        'gt_by_distance': dict(by_dist),
        'matches_by_class_distance': matches_by_class_dist,
        'note': (
            'This report covers one 39-keyframe log only. It is for debugging '
            'and failure triage, not for claiming general model performance. '
            'Use tools/test.py + NuScenesMetric on the full val set for mAP/NDS.'
        ),
    }


def render_markdown(report):
    ver = report['verification']
    scope = report['scope']
    ov = report['operating_point']
    mech = report['failure_mechanisms']
    sparse = report['sparse_lidar_breakdown']
    co = report['fn_tag_cooccurrence']
    lines = [
        '# Failure-analysis report',
        '',
        '## Evaluation scope',
        '',
        f"- Scene: `{report['scene']}`",
        f"- Frames: {scope['num_frames']}",
        f"- Unique GT objects: {scope['num_gt_objects']}",
        f"- Operating point: score ≥ {report['score_thr']}, "
        f"IoU ≥ {report['iou_thr']}",
        f"- Candidate predictions kept down to score "
        f"{report['candidate_score_thr']} for low-confidence diagnosis",
        f"- {scope['note']}",
        '',
        '### GT objects by class',
        '',
        '| Class | GT count |',
        '| --- | ---: |',
    ]
    for name, count in sorted(scope['gt_by_class'].items(),
                              key=lambda kv: (-kv[1], kv[0])):
        lines.append(f'| {name} | {count} |')
    lines.extend([
        '',
        '### GT objects by distance',
        '',
        '| Distance | GT count |',
        '| --- | ---: |',
    ])
    for name, _, _ in DISTANCE_BINS:
        lines.append(f'| {name} | {scope["gt_by_distance"].get(name, 0)} |')
    lines.extend([
        '',
        '## Matching verification',
        '',
        f"- Coordinate frame: {ver['coordinate_frame']}",
        f"- GT box convention: {ver['gt_box_convention']}",
        f"- Pred box convention: {ver['pred_box_convention']}",
        f"- Class rule: {ver['class_rule']}",
        f"- Matcher: {ver['matcher']}",
        f"- One-to-one check: every TP uses a unique pred index and unique "
        f"GT object_id within each frame "
        f"({ver['one_to_one_ok']})",
        f"- `num_lidar_pts` source: {ver['num_lidar_pts_source']}",
        f"- Sparse threshold: ≤ {SPARSE_LIDAR_PTS} annotated LiDAR points",
        f"- Tagging policy: {ver['tagging_policy']}",
        '',
        '## Operating-point summary',
        '',
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Unique TP | {ov['tp']} |",
        f"| Unique FP | {ov['fp']} |",
        f"| Unique FN | {ov['fn']} |",
        f"| Precision | {ov['precision']:.3f} |",
        f"| Recall | {ov['recall']:.3f} |",
        f"| F1 | {ov['f1']:.3f} |",
        '',
        '## Unique FN tags (no double-counting of objects)',
        '',
        'Each missed GT object has one `object_id`. Tags are multi-label, so '
        'tag counts can sum to more than the unique FN total.',
        '',
        f"| Tag | Unique FN objects | Share of {ov['fn']} FN |",
        '| --- | ---: | ---: |',
    ])
    for tag, count in co['unique_per_tag'].items():
        share = count / ov['fn'] if ov['fn'] else 0.0
        lines.append(f'| `{tag}` | {count} | {share:.1%} |')
    lines.extend([
        '',
        f"Objects with ≥3 focus tags: **{co['objects_with_ge3_focus_tags']}**",
        '',
        '### Pairwise tag overlap (unique objects)',
        '',
        '| Tags | Unique objects |',
        '| --- | ---: |',
    ])
    for key, count in sorted(co['pairwise_overlap'].items(),
                             key=lambda kv: -kv[1]):
        lines.append(f'| `{key}` | {count} |')
    lines.extend([
        '',
        '## Why were objects missed?',
        '',
        'For every FN at the operating point, inspect the best same-class '
        'prediction among candidates with score ≥ '
        f"{report['candidate_score_thr']}.",
        '',
        '| Mechanism | Unique FN | Meaning |',
        '| --- | ---: | --- |',
    ])
    meanings = {
        'no_usable_prediction': 'No same-class box with IoU ≥ 0.05',
        'low_confidence': f'Same-class IoU ≥ {report["iou_thr"]} but score < {report["score_thr"]}',
        'inaccurate_box': f'Score ≥ {report["score_thr"]} but IoU < {report["iou_thr"]}',
        'low_confidence_and_weak_localization': 'Score low and IoU in [0.25, thr)',
        'weak_or_no_localization': 'Best same-class IoU < 0.25',
        'other': 'Residual',
    }
    for key, count in mech.items():
        lines.append(f'| `{key}` | {count} | {meanings.get(key, "")} |')
    lines.extend([
        '',
        '## Sparse-LiDAR false negatives',
        '',
        '`num_lidar_pts` is the nuScenes annotation field (points inside the '
        'GT box on the keyframe cloud), not a recount of multi-sweep points.',
        '',
        '| Point bin | Unique FN | Camera-visible FN | Mean distance (m) | '
        'No usable pred | Low confidence | Inaccurate box |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for name, row in sparse.items():
        lines.append(
            f"| {name} | {row['unique_fn']} | {row['camera_visible']} | "
            f"{row['mean_distance_m']:.1f} | {row['no_usable_prediction']} | "
            f"{row['low_confidence']} | {row['inaccurate_box']} |")
    lines.extend([
        '',
        '## Confidence-threshold sweep (IoU '
        f"{report['iou_thr']})",
        '',
        '| Score thr | Precision | Recall | F1 | TP | FP | FN |',
        '| --- | --- | --- | --- | ---: | ---: | ---: |',
    ])
    for row in report['confidence_curves']:
        lines.append(
            f"| {row['score_thr']:.2f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} |")
    lines.extend([
        '',
        '## Example objects to inspect first',
        '',
    ])
    for title, rows in report['examples'].items():
        lines.append(f'### {title}')
        lines.append('')
        lines.append(
            '| object_id | timestamp | class | dist | lidar_pts | '
            'cams | mechanism | best score | best IoU |')
        lines.append('| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |')
        for row in rows:
            best = row.get('best_prediction') or {}
            score = best.get('score')
            iou = best.get('iou')
            score_s = 'n/a' if score is None else f'{score:.3f}'
            iou_s = 'n/a' if iou is None else f'{iou:.3f}'
            lines.append(
                f"| `{row['object_id']}` | {row['timestamp']:.6f} | "
                f"{row['class_name']} | {row['distance_m']:.1f} | "
                f"{row['num_lidar_pts']} | {row['num_visible_cameras']} | "
                f"{row['failure_mechanism']} | {score_s} | {iou_s} |")
        lines.append('')
    lines.extend([
        '',
        '## Objects and matches by class × distance',
        '',
        '| Class | Distance | GT | TP | FN |',
        '| --- | --- | ---: | ---: | ---: |',
    ])
    for _, row in report['scope']['matches_by_class_distance']:
        lines.append(
            f"| {row['class_name']} | {row['distance_bin']} | {row['gt']} | "
            f"{row['tp']} | {row['fn']} |")
    lines.append('')
    return '\n'.join(lines)


def analyze(args):
    with open(args.infos, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    lidar_name = os.path.basename(args.pcd)
    log, rows = select_rows(data_list, lidar_name, args.play_scene)
    print(f'Scene {log}: analyzing {len(rows)} keyframe(s)', flush=True)

    model, dataset, class_names = load_detector(
        args.infos, args.config, args.checkpoint)
    metainfo = getattr(dataset, 'metainfo', None)
    cache = _SensorCache()

    all_preds = []
    all_gts = []
    object_rows = []
    one_to_one_ok = True
    class_dist = defaultdict(lambda: {'gt': 0, 'tp': 0, 'fn': 0})

    for frame_index, (index, info, name) in enumerate(rows):
        preds = infer_predictions(
            model, dataset, index, CANDIDATE_SCORE_THR, class_names,
            args.sweeps, cache)
        gts = attach_gt_ids(
            gt_records_from_info(info, class_names, metainfo=metainfo),
            info, frame_index)
        all_preds.append(preds)
        all_gts.append(gts)

        kept = [p for p in preds if p['score'] >= args.score_thr]
        matches, unmatched_pred, unmatched_gt = match_frame(
            kept, gts, args.iou_thr)
        matched_pred = {m['pred_index'] for m in matches}
        matched_gt = {m['gt_index'] for m in matches}
        if len(matched_pred) != len(matches) or len(matched_gt) != len(matches):
            one_to_one_ok = False

        ious = iou_matrix(
            [p['bbox_3d'] for p in preds],
            [g['bbox_3d'] for g in gts]) if preds and gts else np.zeros(
                (len(preds), len(gts)), dtype=np.float32)
        match_by_gt = {m['gt_index']: m for m in matches}

        for j, gt in enumerate(gts):
            status = 'tp' if j in match_by_gt else 'fn'
            match = match_by_gt.get(j)
            best = best_same_class_pred(
                gt, preds, ious[:, j] if ious.size else np.zeros(len(preds)))
            mechanism = failure_mechanism(
                status, best, args.score_thr, args.iou_thr)
            tags = tag_gt(gt, status, match)
            key = (gt['class_name'], gt['distance_bin'])
            class_dist[key]['gt'] += 1
            class_dist[key]['tp' if status == 'tp' else 'fn'] += 1
            object_rows.append({
                'object_id': gt['object_id'],
                'sample_token': gt['sample_token'],
                'timestamp': gt['timestamp'],
                'lidar_name': gt['lidar_name'],
                'frame_index': gt['frame_index'],
                'class_name': gt['class_name'],
                'distance_m': gt['distance'],
                'distance_bin': gt['distance_bin'],
                'num_lidar_pts': gt['num_lidar_pts'],
                'sparse_bin': gt['sparse_bin'],
                'footprint_m2': gt['footprint'],
                'bbox_3d': gt['bbox_3d'].tolist(),
                'camera_visible': gt['camera_visible'],
                'visible_cameras': gt['visible_cameras'],
                'num_visible_cameras': gt['num_visible_cameras'],
                'crowded_scene': gt['crowded_scene'],
                'status': status,
                'failure_tags': tags,
                'failure_mechanism': mechanism,
                'best_prediction': best,
                'match': None if match is None else {
                    'iou': match['iou'],
                    'score': match['score'],
                    'center_error_m': match['center_error_m'],
                    'yaw_error_deg': match['yaw_error_deg'],
                },
            })
        print(
            f'[{frame_index + 1}/{len(rows)}] {name}: '
            f'GT={len(gts)} cand={len(preds)} '
            f'opTP={len(matches)} opFP={len(unmatched_pred)} '
            f'opFN={len(unmatched_gt)}',
            flush=True)

    op = metrics_at_threshold(all_preds, all_gts, args.score_thr, args.iou_thr)
    fn_rows = [r for r in object_rows if r['status'] == 'fn']
    focus_tags = [
        'sparse_lidar', 'distant', 'crowded_scene', 'small_object',
        'camera_invisible',
    ]
    co = cooccurrence([r['failure_tags'] for r in fn_rows], focus_tags)

    mech_counts = Counter(r['failure_mechanism'] for r in fn_rows)
    sparse_breakdown = {}
    for bin_name, lo, hi in SPARSE_BINS:
        subset = [r for r in fn_rows if lo <= r['num_lidar_pts'] <= hi]
        sparse_breakdown[bin_name] = {
            'unique_fn': len(subset),
            'camera_visible': sum(1 for r in subset if r['camera_visible']),
            'mean_distance_m': float(np.mean([r['distance_m'] for r in subset]))
            if subset else 0.0,
            'no_usable_prediction': sum(
                1 for r in subset
                if r['failure_mechanism'] == 'no_usable_prediction'),
            'low_confidence': sum(
                1 for r in subset
                if r['failure_mechanism'] == 'low_confidence'),
            'inaccurate_box': sum(
                1 for r in subset
                if r['failure_mechanism'] == 'inaccurate_box'),
        }

    curves = []
    for thr in SCORE_SWEEP:
        metrics = metrics_at_threshold(all_preds, all_gts, thr, args.iou_thr)
        curves.append({'score_thr': thr, **metrics})

    def pick_examples(predicate, limit=5):
        rows = [r for r in fn_rows if predicate(r)]
        rows.sort(key=lambda r: (-r['distance_m'], r['num_lidar_pts']))
        return rows[:limit]

    examples = {
        'Sparse LiDAR FN (≤5 pts)': pick_examples(
            lambda r: 'sparse_lidar' in r['failure_tags']),
        'Distant FN (≥30 m)': pick_examples(
            lambda r: 'distant' in r['failure_tags']),
        'Low-confidence misses': pick_examples(
            lambda r: r['failure_mechanism'] == 'low_confidence'),
        'No usable prediction': pick_examples(
            lambda r: r['failure_mechanism'] == 'no_usable_prediction'),
        'Inaccurate box (score OK, IoU low)': pick_examples(
            lambda r: r['failure_mechanism'] == 'inaccurate_box'),
    }

    # Named examples from the previous report.
    highlighted = []
    for row in fn_rows:
        if (abs(row['timestamp'] - 1532402927.647951) < 1e-3
                and row['class_name'] == 'pedestrian'
                and row['distance_m'] > 60):
            highlighted.append(row)
        if (abs(row['timestamp'] - 1532402930.648325) < 1e-3
                and row['class_name'] == 'traffic_cone'
                and 40 < row['distance_m'] < 42):
            highlighted.append(row)
    examples['Highlighted prior examples'] = highlighted[:5]

    matches_table = []
    for (cls_name, dist_bin), stats in sorted(
            class_dist.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        matches_table.append((
            f'{cls_name}|{dist_bin}',
            {
                'class_name': cls_name,
                'distance_bin': dist_bin,
                **stats,
            },
        ))

    poor_yaw = [
        r for r in object_rows
        if r['status'] == 'tp' and 'poor_yaw' in r['failure_tags']
    ]
    yaw_by_class = Counter(r['class_name'] for r in poor_yaw)

    report = {
        'scene': log,
        'score_thr': args.score_thr,
        'iou_thr': args.iou_thr,
        'candidate_score_thr': CANDIDATE_SCORE_THR,
        'verification': {
            'coordinate_frame': 'LIDAR_TOP / model LiDAR frame',
            'gt_box_convention': (
                'nuScenes center origin converted with '
                'LiDARInstance3DBoxes(..., origin=(0.5,0.5,0.5)) to '
                'bottom-center'),
            'pred_box_convention': (
                'detector pred_instances_3d boxes already bottom-center'),
            'class_rule': (
                'same class only after remapping info categories -> model '
                'classes'),
            'matcher': (
                'greedy one-to-one by descending 3D IoU; each pred and each '
                'GT used at most once per frame'),
            'one_to_one_ok': one_to_one_ok,
            'num_lidar_pts_source': (
                'nuScenes info instances[].num_lidar_pts on the keyframe'),
            'tagging_policy': (
                'multi-label on unique object_id; tag totals may overlap and '
                'must not be summed as disjoint counts'),
        },
        'scope': summarize_scope(object_rows, matches_table),
        'operating_point': op,
        'fn_tag_cooccurrence': co,
        'failure_mechanisms': dict(mech_counts),
        'sparse_lidar_breakdown': sparse_breakdown,
        'confidence_curves': [
            {
                'score_thr': row['score_thr'],
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1'],
                'tp': row['tp'],
                'fp': row['fp'],
                'fn': row['fn'],
            }
            for row in curves
        ],
        'poor_yaw': {
            'unique_tp_with_poor_yaw': len(poor_yaw),
            'by_class': dict(yaw_by_class),
        },
        'examples': {
            title: [{
                'object_id': r['object_id'],
                'timestamp': r['timestamp'],
                'class_name': r['class_name'],
                'distance_m': r['distance_m'],
                'num_lidar_pts': r['num_lidar_pts'],
                'num_visible_cameras': r['num_visible_cameras'],
                'visible_cameras': r['visible_cameras'],
                'failure_mechanism': r['failure_mechanism'],
                'failure_tags': r['failure_tags'],
                'best_prediction': r['best_prediction'],
            } for r in rows]
            for title, rows in examples.items()
        },
        'iou_thresholds_checked': list(IOU_THRESHOLDS),
    }
    return report, object_rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    report, object_rows = analyze(args)

    objects_path = os.path.join(args.out_dir, 'objects.json')
    report_json = os.path.join(args.out_dir, 'failure_analysis.json')
    report_md = os.path.join(args.out_dir, 'failure_analysis.md')

    with open(objects_path, 'w', encoding='utf-8') as f:
        json.dump(object_rows, f, indent=2)
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write(render_markdown(report))

    print('\n=== Failure analysis ===', flush=True)
    print(
        f"Unique FN={report['operating_point']['fn']} "
        f"FP={report['operating_point']['fp']} "
        f"TP={report['operating_point']['tp']}",
        flush=True)
    print('Unique FN tags:', report['fn_tag_cooccurrence']['unique_per_tag'],
          flush=True)
    print('Mechanisms:', report['failure_mechanisms'], flush=True)
    print(f'Wrote {objects_path}', flush=True)
    print(f'Wrote {report_json}', flush=True)
    print(f'Wrote {report_md}', flush=True)


if __name__ == '__main__':
    main()
