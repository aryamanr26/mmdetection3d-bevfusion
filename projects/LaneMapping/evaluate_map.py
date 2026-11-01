"""Geometric accuracy of the saved MapTRv2 predictions on mini val.

Uses the two scenes the checkpoint did not train on, scene-0103 and
scene-0916. Ground truth comes from nuScenes map expansion v1.3. See
``map_gt.py`` for which classes are annotations and which are derived.

    .venv/bin/python projects/LaneMapping/evaluate_map.py
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_val
from shapely.geometry import LineString, box

from map_gt import CLASSES, GT_SOURCES, PC_RANGE, MapGroundTruth
from map_metrics import micro_scores

THRESHOLDS = (0.5, 1.0, 1.5)
SCORE_THR = 0.3
RANGE = box(*PC_RANGE)


def _clip(points):
    line = LineString(np.asarray(points, float)[:, :2])
    clipped = line.intersection(RANGE)
    if clipped.is_empty:
        return []
    parts = [clipped] if clipped.geom_type == 'LineString' else list(
        getattr(clipped, 'geoms', []))
    return [np.round(np.asarray(part.coords)[:, :2], 3) for part in parts
            if part.geom_type == 'LineString' and part.length >= 1.0]


def _lidar2global(nusc, sample):
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    from pyquaternion import Quaternion
    def pose(rec):
        m = np.eye(4)
        m[:3, :3] = Quaternion(rec['rotation']).rotation_matrix
        m[:3, 3] = rec['translation']
        return m
    lidar2ego = pose(nusc.get('calibrated_sensor', sd['calibrated_sensor_token']))
    ego2global = pose(nusc.get('ego_pose', sd['ego_pose_token']))
    return ego2global @ lidar2ego


def _load_preds(path, score_thr):
    with open(path) as handle:
        scene = json.load(handle)
    by_token = {}
    for frame in scene['frames']:
        grouped = {name: [] for name in CLASSES}
        for element in frame['elements']:
            if element['score'] < score_thr or element['cls'] not in grouped:
                continue
            grouped[element['cls']].extend(_clip(element['points']))
        by_token[frame['sample_token']] = grouped
    return by_token


def _accumulate(frames):
    """Per-frame matching, summed over keyframes. Not one match across the scene."""
    return {
        name: micro_scores(
            [(frame['pred'][name], frame['gt'][name]) for frame in frames],
            THRESHOLDS)
        for name in CLASSES
    }


def _overlay(path, pred, gt, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {
        'divider': '#ff9900', 'ped_crossing': '#1f77ff',
        'boundary': '#e8102e', 'centerline': '#22bb33',
    }
    fig, ax = plt.subplots(figsize=(5.2, 9))
    for name in CLASSES:
        for line in gt[name]:
            ax.plot(line[:, 0], line[:, 1], color=colors[name], lw=2.0, alpha=0.45)
        for line in pred[name]:
            ax.plot(line[:, 0], line[:, 1], color=colors[name], lw=1.2, ls='--')
    ax.plot(0, 0, 'k+', ms=8)
    ax.set_xlim(-16, 16)
    ax.set_ylim(-32, 32)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m, right)')
    ax.set_ylabel('y (m, forward)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt(row):
    chamfer = row['chamfer_matched_m']
    chamfer_s = 'n/a' if chamfer is None else f'{chamfer:.2f}'
    return (f"| {row['threshold_m']:.1f} | {row['tp']} | {row['pred']} | "
            f"{row['gt']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {chamfer_s} |")


def _markdown(report):
    lines = [
        '# MapTRv2 geometric accuracy on nuScenes-mini val',
        '',
        'Checkpoint `maptrv2_nusc_r50_24ep_w_centerline`, predictions already '
        f"produced for every mini keyframe. Score threshold {report['score_thr']} "
        '(the Foxglove display threshold; it was not swept on these scenes). '
        'Distance thresholds 0.5 m, 1.0 m, and 1.5 m are the MapTRv2 average-'
        'precision thresholds. Only scene-0103 and scene-0916 are held out of '
        'nuScenes train.',
        '',
        'Divider ground truth is the annotated divider lines. Centerlines are '
        'chained from annotated lane arc-line paths. Boundaries and pedestrian '
        'crossings are polygon outlines, so they are derived representations, '
        'not labeled polylines.',
        '',
        'Each keyframe is matched on its own. Counts are then summed, so a '
        'prediction cannot match a ground-truth line from another frame. '
        'Chamfer is the average of the two directed mean nearest-point '
        'distances, in metres, in the LIDAR_TOP frame, inside x ±15 m and '
        'y ±30 m. Solid lines in the overlays are ground truth and dashed '
        'lines are predictions.',
        '',
    ]
    for scene in report['scenes']:
        lines.append(f"## {scene['scene']}")
        lines.append('')
        lines.append(f"{scene['frames']} keyframes. "
                     f"Per-frame mean counts at the score threshold:")
        lines.append('')
        lines.append('| Class | Pred / frame | GT / frame |')
        lines.append('| --- | ---: | ---: |')
        for name, counts in scene['counts'].items():
            lines.append(f"| {name} | {counts['pred_per_frame']:.1f} | "
                         f"{counts['gt_per_frame']:.1f} |")
        lines.append('')
        lines.append('| Class | Thr (m) | TP | Pred | GT | Precision | Recall | F1 | Matched chamfer (m) |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for name, result in scene['metrics'].items():
            for row in result['thresholds']:
                lines.append(f"| {name} {_fmt(row)[1:]}")
            nearest = result['nearest_chamfer_median_m']
            nearest_s = 'n/a' if nearest is None else f'{nearest:.2f}'
            lines.append(f"| {name} | nearest median |  |  |  |  |  |  | {nearest_s} |")
        lines.append('')
    lines.append('## Both val scenes')
    lines.append('')
    lines.append('| Class | Thr (m) | TP | Pred | GT | Precision | Recall | F1 | Matched chamfer (m) |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for name, result in report['pooled'].items():
        for row in result['thresholds']:
            lines.append(f"| {name} {_fmt(row)[1:]}")
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--pred-dir',
                        default='demo/eval_out/lane_perception/centerline/preds')
    parser.add_argument('--out-dir',
                        default='demo/eval_out/lane_perception/accuracy')
    parser.add_argument('--score-thr', type=float, default=SCORE_THR)
    parser.add_argument('--scenes', nargs='*', default=list(mini_val))
    args = parser.parse_args()
    os.makedirs(osp.join(args.out_dir, 'overlays'), exist_ok=True)

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    maps = {}
    scene_reports = []
    all_frames = []
    for name in args.scenes:
        record = next(scene for scene in nusc.scene if scene['name'] == name)
        log = nusc.get('log', record['log_token'])
        maps.setdefault(log['location'], MapGroundTruth(args.dataroot, log['location']))
        gt_map = maps[log['location']]
        preds = _load_preds(osp.join(args.pred_dir, name + '.json'), args.score_thr)
        token = record['first_sample_token']
        frames = []
        while token:
            sample = nusc.get('sample', token)
            pred = preds[sample['token']]
            gt = gt_map.frame(_lidar2global(nusc, sample))
            frame = {'token': sample['token'], 'pred': pred,
                     'gt': {cls: [np.asarray(line) for line in gt[cls]]
                            for cls in CLASSES}}
            frames.append(frame)
            if len(frames) in (1, 20):
                _overlay(
                    osp.join(args.out_dir, 'overlays',
                             f'{name}_{len(frames):02d}.png'),
                    pred, frame['gt'],
                    f'{name} keyframe {len(frames)}  solid=GT dashed=pred')
            token = sample['next']
        metrics = _accumulate(frames)
        counts = {}
        for cls in CLASSES:
            counts[cls] = {
                'pred_per_frame': float(np.mean([len(f['pred'][cls]) for f in frames])),
                'gt_per_frame': float(np.mean([len(f['gt'][cls]) for f in frames])),
            }
        scene_reports.append(dict(
            scene=name, location=log['location'], frames=len(frames),
            counts=counts, metrics=metrics))
        all_frames.extend(frames)
        print(name, log['location'], len(frames), 'frames', flush=True)
        for cls in CLASSES:
            row = metrics[cls]['thresholds'][1]
            print(f"  {cls:14} @1.0m  P {row['precision']:.3f}  "
                  f"R {row['recall']:.3f}  F1 {row['f1']:.3f}  "
                  f"pred {counts[cls]['pred_per_frame']:.1f}  "
                  f"gt {counts[cls]['gt_per_frame']:.1f}", flush=True)

    report = dict(
        score_thr=args.score_thr, thresholds_m=list(THRESHOLDS),
        range_m={'x': [-15, 15], 'y': [-30, 30]},
        scenes=scene_reports, pooled=_accumulate(all_frames),
        gt_sources=GT_SOURCES)
    with open(osp.join(args.out_dir, 'metrics.json'), 'w') as handle:
        json.dump(report, handle, indent=2)
    text = _markdown(report)
    with open(osp.join(args.out_dir, 'accuracy.md'), 'w') as handle:
        handle.write(text)
    print('wrote', args.out_dir)


if __name__ == '__main__':
    main()
