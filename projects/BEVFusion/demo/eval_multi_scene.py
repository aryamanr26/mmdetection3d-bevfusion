# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluate BEVFusion on every local nuScenes log.

The checkout only contains a handful of logs (train plus val). This script
scores each one with the same class-aware 3D-IoU matcher as
``eval_gt_vs_pred.py`` and writes a pooled comparison.

Example:
    python projects/BEVFusion/demo/eval_multi_scene.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_gt_vs_pred import (  # noqa: E402
    CLASS_NAMES, DISTANCE_BINS, IOU_THRESHOLDS, PRIMARY_IOU, accumulate,
    empty_counters, finalize_bucket, gt_records_from_info, infer_predictions,
    match_frame)
from foxglove_stream import (  # noqa: E402
    CHECKPOINT, CONFIG, _SensorCache, load_detector, scene_rows)

SPLITS = (
    ('train', 'data/nuscenes/nuscenes_infos_train.pkl'),
    ('val', 'data/nuscenes/nuscenes_infos_val.pkl'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--score-thr', type=float, default=0.2)
    parser.add_argument('--iou-thr', type=float, default=PRIMARY_IOU)
    parser.add_argument('--sweeps', type=int, default=9)
    parser.add_argument('--out-dir', default='demo/eval_out')
    return parser.parse_args()


def logs_in(data_list):
    """Return ``(log, first_lidar_name, frame_count)`` in timestamp order."""
    groups = defaultdict(list)
    for info in data_list:
        name = os.path.basename(info['lidar_points']['lidar_path'])
        groups[name.split('__')[0]].append(info)
    rows = []
    for log, infos in groups.items():
        infos.sort(key=lambda item: float(item['timestamp']))
        name = os.path.basename(infos[0]['lidar_points']['lidar_path'])
        rows.append((log, name, len(infos)))
    rows.sort(key=lambda item: item[0])
    return rows


def score_scene(model, dataset, class_names, rows, score_thr, iou_thr,
                sweeps, cache):
    overall = empty_counters()
    by_class = defaultdict(empty_counters)
    by_distance = defaultdict(empty_counters)
    metainfo = getattr(dataset, 'metainfo', None)
    for index, info, _name in rows:
        preds = infer_predictions(
            model, dataset, index, score_thr, class_names, sweeps, cache)
        gts = gt_records_from_info(info, class_names, metainfo=metainfo)
        matches, unmatched_pred, unmatched_gt = match_frame(
            preds, gts, iou_thr)
        fp_preds = [preds[i] for i in unmatched_pred]
        fn_gts = [gts[j] for j in unmatched_gt]
        accumulate(overall, matches, fp_preds, fn_gts)
        for match in matches:
            accumulate(by_class[match['class_name']], [match], [], [])
            accumulate(by_distance[match['distance_bin']], [match], [], [])
        for pred in fp_preds:
            accumulate(by_class[pred['class_name']], [], [pred], [])
            accumulate(by_distance[pred['distance_bin']], [], [pred], [])
        for gt in fn_gts:
            accumulate(by_class[gt['class_name']], [], [], [gt])
            accumulate(by_distance[gt['distance_bin']], [], [], [gt])
    return overall, by_class, by_distance


def merge_into(dst, src):
    for key in ('tp', 'fp', 'fn', 'gt', 'pred'):
        dst[key] += src[key]
    for key in ('scores_tp', 'scores_fp', 'center_errors', 'xy_errors',
                'yaw_errors', 'ious'):
        dst[key].extend(src[key])


def clock_from_log(log):
    """Local clock embedded in a nuScenes log name, if present."""
    parts = log.split('-')
    if len(parts) >= 6 and parts[4].isdigit() and parts[5].isdigit():
        return f'{parts[4]}:{parts[5]}'
    return 'unknown'


def render(report):
    lines = [
        '# Multi-scene BEVFusion evaluation',
        '',
        f"- Score threshold: {report['score_thr']}",
        f"- IoU threshold: {report['iou_thr']}",
        f"- Scenes: {report['num_scenes']} "
        f"({report['num_frames']} keyframes)",
        '- Scope: every log present in the local train and val info files. '
        'This is broader than one playback scene and still not the full '
        'nuScenes val set.',
        '',
        '## Per scene',
        '',
        '| Split | Log | Local time | Frames | GT | P | R | F1 | '
        'center err (m) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for scene in report['scenes']:
        metrics = scene['overall']
        center = (f"{metrics['mean_center_error_m']:.3f}"
                  if metrics['mean_center_error_m'] is not None else 'n/a')
        lines.append(
            f"| {scene['split']} | `{scene['log']}` | {scene['local_time']} | "
            f"{scene['num_frames']} | {metrics['gt']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {center} |")
    pooled = report['pooled']
    lines.extend([
        '',
        '## Pooled',
        '',
        f"| Precision | {pooled['precision']:.3f} |",
        f"| Recall | {pooled['recall']:.3f} |",
        f"| F1 | {pooled['f1']:.3f} |",
        f"| TP / FP / FN | {pooled['tp']} / {pooled['fp']} / {pooled['fn']} |",
        (
            f"| Mean center error (m) | {pooled['mean_center_error_m']:.3f} |"
            if pooled['mean_center_error_m'] is not None else
            '| Mean center error (m) | n/a |'
        ),
        '',
        '## Pooled by class',
        '',
        '| Class | GT | P | R | F1 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for name, metrics in report['by_class'].items():
        lines.append(
            f"| {name} | {metrics['gt']} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} |")
    lines.extend([
        '',
        '## Pooled by distance',
        '',
        '| Distance | GT | P | R | F1 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for name, metrics in report['by_distance'].items():
        lines.append(
            f"| {name} | {metrics['gt']} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} |")
    f1s = [scene['overall']['f1'] for scene in report['scenes']]
    lines.extend([
        '',
        '## Spread',
        '',
        f"- Scene F1 min / max: {min(f1s):.3f} / {max(f1s):.3f}",
        f"- Playback reference scene "
        f"`n015-2018-07-24-11-22-45+0800` is included in the train split.",
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scenes = []
    pooled = empty_counters()
    by_class = defaultdict(empty_counters)
    by_distance = defaultdict(empty_counters)

    for split, infos_path in SPLITS:
        if not os.path.exists(infos_path):
            print(f'Skipping missing {infos_path}', flush=True)
            continue
        with open(infos_path, 'rb') as handle:
            data_list = pickle.load(handle)['data_list']
        catalog = logs_in(data_list)
        print(f'{split}: {len(catalog)} logs, {len(data_list)} frames',
              flush=True)
        model, dataset, class_names = load_detector(
            infos_path, args.config, args.checkpoint)
        for log, first_name, _count in catalog:
            _log, rows = scene_rows(data_list, first_name)
            cache = _SensorCache()
            print(f'Evaluating {split} {log} ({len(rows)} frames)', flush=True)
            overall, scene_class, scene_distance = score_scene(
                model, dataset, class_names, rows, args.score_thr,
                args.iou_thr, args.sweeps, cache)
            metrics = finalize_bucket(overall)
            scenes.append({
                'split': split,
                'log': log,
                'local_time': clock_from_log(log),
                'num_frames': len(rows),
                'overall': metrics,
            })
            merge_into(pooled, overall)
            for name, bucket in scene_class.items():
                merge_into(by_class[name], bucket)
            for name, bucket in scene_distance.items():
                merge_into(by_distance[name], bucket)
            print(
                f'  P={metrics["precision"]:.3f} R={metrics["recall"]:.3f} '
                f'F1={metrics["f1"]:.3f} GT={metrics["gt"]}',
                flush=True)
            del cache

    class_order = [
        name for name in CLASS_NAMES
        if by_class[name]['gt'] or by_class[name]['pred']
    ]
    report = {
        'score_thr': args.score_thr,
        'iou_thr': args.iou_thr,
        'iou_thresholds_available': list(IOU_THRESHOLDS),
        'num_scenes': len(scenes),
        'num_frames': sum(scene['num_frames'] for scene in scenes),
        'scenes': scenes,
        'pooled': finalize_bucket(pooled),
        'by_class': {name: finalize_bucket(by_class[name]) for name in class_order},
        'by_distance': {
            name: finalize_bucket(by_distance[name])
            for name, _, _ in DISTANCE_BINS
            if by_distance[name]['gt'] or by_distance[name]['pred']
        },
    }
    json_path = os.path.join(args.out_dir, 'multi_scene_metrics.json')
    md_path = os.path.join(args.out_dir, 'multi_scene_report.md')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write(render(report))
    print('\n=== Pooled ===', flush=True)
    pooled_metrics = report['pooled']
    print(
        f"scenes={report['num_scenes']} frames={report['num_frames']} "
        f"P={pooled_metrics['precision']:.3f} "
        f"R={pooled_metrics['recall']:.3f} "
        f"F1={pooled_metrics['f1']:.3f}",
        flush=True)
    print(f'Wrote {json_path}', flush=True)
    print(f'Wrote {md_path}', flush=True)


if __name__ == '__main__':
    main()
