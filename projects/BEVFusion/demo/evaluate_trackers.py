"""Compare trackers on identical cached BEVFusion detections.

Every tracker reads the same ``dets_<split>.pkl`` caches written by
``run_tracking.py``, so the detector is fixed and only tracking changes.

Official metrics come from the nuScenes devkit ``TrackingEval`` with config
``tracking_nips_2019``, evaluated as custom splits on each scene, on the
``mini_val`` scenes (``val_scenes``), the ``mini_train`` scenes
(``train_scenes``), and all ten mini scenes (``all_scenes``). Custom split
names make the devkit filter the shared result file by sample token.

Supplementary metrics are computed here because the devkit does not report
them. They follow the devkit conventions where possible: 2 m center-distance
matching within a class, the per-class ego range from the tracking config, and
ground truth with at least one LiDAR or radar point. They do not apply the
bike-rack filter or the devkit's track interpolation, and they use every track
output instead of the best-MOTA score threshold.

Example:
    python projects/BEVFusion/demo/run_tracking.py   # caches mini_val dets
    python projects/BEVFusion/demo/run_tracking.py \
        --infos data/nuscenes/nuscenes_infos_train.pkl --skip-eval
    python projects/BEVFusion/demo/evaluate_trackers.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_tracking import (  # noqa: E402
    DEFAULT_DATA_ROOT, compact_metrics, official_eval, run_tracker,
    tracks_to_submission)
from tracker import TRACKING_CLASSES, transform_detection  # noqa: E402

TRACKERS = {
    'poly': dict(tracker='poly', motion_model='poly'),
    'poly_birth_0p4': dict(
        tracker='poly', motion_model='poly', birth_score=0.4),
    'cv_baseline': dict(tracker='cv_baseline', motion_model='cv'),
    # Single-factor diagnostics, run on request. Except for poly_birth_all,
    # they start from the earlier 0.4 birth threshold.
    'poly_cv_model': dict(
        tracker='poly', motion_model='cv', birth_score=0.4),
    'poly_min_hits_1': dict(
        tracker='poly', motion_model='poly', min_hits=1, birth_score=0.4),
    'poly_birth_all': dict(
        tracker='poly', motion_model='poly', high_score=0.05,
        birth_score=0.05),
    'poly_no_coast_output': dict(
        tracker='poly', motion_model='poly', output_coasted=False,
        birth_score=0.4),
    'poly_all_three': dict(
        tracker='poly', motion_model='poly', min_hits=1, high_score=0.05,
        birth_score=0.05, output_coasted=False),
}
TRACKER_LABELS = {
    'poly': 'Poly-MOT-style (birth >= 0.05)',
    'poly_birth_0p4': 'Poly-MOT-style, earlier birth >= 0.4',
    'cv_baseline': 'Constant-velocity baseline',
    'poly_cv_model': 'Poly, birth >= 0.4, CV motion model',
    'poly_min_hits_1': 'Poly, birth >= 0.4, confirm on first hit',
    'poly_birth_all': 'Poly, one stage, birth >= 0.05',
    'poly_no_coast_output': 'Poly, birth >= 0.4, no coasted output',
    'poly_all_three': 'Poly, all three changes',
}
MATCH_DIST_M = 2.0
DISTANCE_BANDS = (('0-20m', 0.0, 20.0), ('20-30m', 20.0, 30.0),
                  ('30-50m', 30.0, 50.0))
MOVING_MPS = 0.5
STATIC_MPS = 0.2
GOOD_DET_SCORE = 0.4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--det-dir', default='demo/eval_out/tracking')
    parser.add_argument('--out-dir', default='demo/eval_out/tracking_compare')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--splits', nargs='+', default=['val', 'train'])
    parser.add_argument(
        '--trackers', nargs='+', choices=list(TRACKERS),
        default=['poly', 'poly_birth_0p4', 'cv_baseline'])
    parser.add_argument(
        '--official-splits', nargs='+', default=None,
        help='custom splits for the devkit eval (default: every scene and '
             'the val_scenes, train_scenes, all_scenes aggregates)')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--high-score', type=float, default=0.4)
    parser.add_argument(
        '--skip-official', action='store_true',
        help='only compute the supplementary metrics')
    return parser.parse_args()


def load_frames(det_dir, splits):
    frames = []
    for split in splits:
        path = os.path.join(det_dir, f'dets_{split}.pkl')
        with open(path, 'rb') as handle:
            payload = pickle.load(handle)
        for frame in payload['frames']:
            frame['split'] = split
        frames.extend(payload['frames'])
        print(f'{path}: {len(payload["frames"])} frames', flush=True)
    return {'frames': frames}


def load_ground_truth(nusc, tokens, class_range):
    """Per sample: ego xy, scene, timestamp, and filtered GT boxes."""
    from nuscenes.eval.tracking.utils import category_to_tracking_name

    samples = {}
    for token in tokens:
        sample = nusc.get('sample', token)
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego = np.asarray(
            nusc.get('ego_pose', lidar['ego_pose_token'])['translation'][:2])
        boxes = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            name = category_to_tracking_name(ann['category_name'])
            if name not in TRACKING_CLASSES:
                continue
            if ann['num_lidar_pts'] + ann['num_radar_pts'] == 0:
                continue
            xy = np.asarray(ann['translation'][:2], dtype=np.float64)
            distance = float(np.linalg.norm(xy - ego))
            if distance > class_range[name]:
                continue
            velocity = nusc.box_velocity(ann_token)[:2]
            boxes.append({
                'instance': ann['instance_token'],
                'class_name': name,
                'xy': xy,
                'distance': distance,
                'velocity': (np.asarray(velocity, dtype=np.float64)
                             if np.isfinite(velocity).all() else None),
            })
        samples[token] = {
            'scene': nusc.get('scene', sample['scene_token'])['name'],
            'timestamp': sample['timestamp'] / 1e6,
            'ego': ego,
            'gt': boxes,
        }
    return samples


def match(preds, gts):
    """Class-aware Hungarian matching on BEV center distance (<= 2 m)."""
    pairs = []
    for name in TRACKING_CLASSES:
        p_idx = [i for i, p in enumerate(preds) if p['class_name'] == name]
        g_idx = [j for j, g in enumerate(gts) if g['class_name'] == name]
        if not p_idx or not g_idx:
            continue
        cost = np.array([
            [np.linalg.norm(preds[i]['xy'] - gts[j]['xy']) for j in g_idx]
            for i in p_idx])
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if cost[row, col] <= MATCH_DIST_M:
                pairs.append((p_idx[row], g_idx[col]))
    return pairs


def in_range(xy, ego, name, class_range):
    return float(np.linalg.norm(xy - ego)) <= class_range[name]


def distance_band(distance):
    for name, low, high in DISTANCE_BANDS:
        if low <= distance < high:
            return name
    return DISTANCE_BANDS[-1][0]


def summarize(values):
    if not values:
        return {'count': 0}
    values = np.asarray(values, dtype=np.float64)
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
    }


def velocity_bucket():
    return defaultdict(list)


def supplementary_metrics(tracked, det_frames, samples, class_range):
    """Velocity, continuity, and association metrics for one tracker."""
    vel = defaultdict(velocity_bucket)
    det_vel = defaultdict(velocity_bucket)
    motion = defaultdict(int)
    gt_ids = defaultdict(set)
    gt_frames = defaultdict(int)
    gt_matched = defaultdict(int)
    gt_switches = defaultdict(int)
    last_id = {}
    track_frames = defaultdict(list)
    track_gt = defaultdict(lambda: defaultdict(int))
    good_det_gt = 0
    good_det_tracked = 0
    outputs = 0
    outputs_matched = 0

    dets_by_token = {frame['token']: frame for frame in det_frames}
    for frame in tracked:
        sample = samples[frame['token']]
        gts = sample['gt']
        ego = sample['ego']
        preds = []
        for track in frame['tracks']:
            box = np.asarray(track['bbox_3d'], dtype=np.float64)
            if not in_range(box[:2], ego, track['class_name'], class_range):
                continue
            preds.append({
                'class_name': track['class_name'],
                'xy': box[:2],
                'velocity': box[7:9],
                'moving': bool(track['moving']),
                'id': f"{sample['scene']}/{track['tracking_id']}",
            })
        outputs += len(preds)
        for gt in gts:
            gt_frames[gt['instance']] += 1
        for pred in preds:
            track_frames[pred['id']].append(sample['timestamp'])

        pairs = match(preds, gts)
        outputs_matched += len(pairs)
        tracked_instances = set()
        for p, g in pairs:
            pred, gt = preds[p], gts[g]
            instance = gt['instance']
            tracked_instances.add(instance)
            gt_matched[instance] += 1
            gt_ids[instance].add(pred['id'])
            if instance in last_id and last_id[instance] != pred['id']:
                gt_switches[instance] += 1
            last_id[instance] = pred['id']
            track_gt[pred['id']][instance] += 1
            if gt['velocity'] is None:
                continue
            error = float(np.linalg.norm(pred['velocity'] - gt['velocity']))
            gt_speed = float(np.linalg.norm(gt['velocity']))
            state = ('moving' if gt_speed >= MOVING_MPS else
                     'static' if gt_speed <= STATIC_MPS else 'slow')
            for key in ('all', gt['class_name'], distance_band(
                    gt['distance']), f'gt_{state}'):
                vel[key]['error'].append(error)
            if state == 'moving':
                angle = abs(math.atan2(
                    pred['velocity'][1], pred['velocity'][0])
                    - math.atan2(gt['velocity'][1], gt['velocity'][0]))
                angle = min(angle, 2.0 * math.pi - angle)
                vel['all']['heading_deg'].append(math.degrees(angle))
            if state != 'slow':
                label = 'moving' if pred['moving'] else 'static'
                motion[f'{state}_as_{label}'] += 1

        det_frame = dets_by_token[frame['token']]
        matrix = np.asarray(det_frame['lidar2global'], dtype=np.float64)
        dets = []
        for det in det_frame['detections']:
            if det['class_name'] not in TRACKING_CLASSES:
                continue
            box = transform_detection(det, matrix)['bbox_3d']
            if not in_range(box[:2], ego, det['class_name'], class_range):
                continue
            dets.append({
                'class_name': det['class_name'],
                'xy': np.asarray(box[:2]),
                'velocity': np.asarray(box[7:9]),
                'score': float(det['score']),
            })
        for d, g in match(dets, gts):
            det, gt = dets[d], gts[g]
            if det['score'] >= GOOD_DET_SCORE:
                good_det_gt += 1
                good_det_tracked += gt['instance'] in tracked_instances
            if gt['velocity'] is None:
                continue
            error = float(np.linalg.norm(det['velocity'] - gt['velocity']))
            gt_speed = float(np.linalg.norm(gt['velocity']))
            state = ('moving' if gt_speed >= MOVING_MPS else
                     'static' if gt_speed <= STATIC_MPS else 'slow')
            for key in ('all', gt['class_name'], distance_band(
                    gt['distance']), f'gt_{state}'):
                det_vel[key]['error'].append(error)

    durations = []
    purities = []
    short_tracks = 0
    for track_id, stamps in track_frames.items():
        durations.append(max(stamps) - min(stamps))
        short_tracks += len(stamps) <= 2
        matched = track_gt.get(track_id)
        if matched:
            purities.append(max(matched.values()) / sum(matched.values()))
    seen = [i for i in gt_frames if gt_matched.get(i)]
    coverage = [gt_matched[i] / gt_frames[i] for i in gt_frames]
    ids_per_gt = [len(gt_ids[i]) for i in seen]
    moving_total = motion['moving_as_moving'] + motion['moving_as_static']
    static_total = motion['static_as_static'] + motion['static_as_moving']
    return {
        'velocity_track': {
            key: {name: summarize(values) for name, values in bucket.items()}
            for key, bucket in vel.items()},
        'velocity_detector': {
            key: {name: summarize(values) for name, values in bucket.items()}
            for key, bucket in det_vel.items()},
        'motion_classification': {
            'moving_recall': (motion['moving_as_moving'] / moving_total
                              if moving_total else None),
            'static_recall': (motion['static_as_static'] / static_total
                              if static_total else None),
            'moving_gt': moving_total,
            'static_gt': static_total,
        },
        'continuity': {
            'gt_instances': len(gt_frames),
            'gt_instances_tracked': len(seen),
            'gt_frame_coverage_mean': float(np.mean(coverage)),
            'track_ids_per_tracked_gt_mean': float(np.mean(ids_per_gt)),
            'gt_with_one_id_fraction': float(np.mean(
                [n == 1 for n in ids_per_gt])),
            'id_changes_on_matched_gt': int(sum(gt_switches.values())),
            'output_tracks': len(track_frames),
            'track_duration_s': summarize(durations),
            'tracks_two_frames_or_fewer': short_tracks,
            'track_purity_mean': float(np.mean(purities)),
        },
        'association': {
            'good_detection_gt_pairs': good_det_gt,
            'good_detection_also_tracked': (
                good_det_tracked / good_det_gt if good_det_gt else None),
            'track_outputs': outputs,
            'track_output_precision': (
                outputs_matched / outputs if outputs else None),
        },
        'step_ms': summarize([frame['step_ms'] for frame in tracked]),
        'step_ms_p95': float(np.percentile(
            [frame['step_ms'] for frame in tracked], 95)),
    }


def write_splits(path, samples):
    splits = defaultdict(set)
    for item in samples.values():
        scene = item['scene']
        splits[scene].add(scene)
        splits['all_scenes'].add(scene)
        splits[f"{item['split']}_scenes"].add(scene)
    splits = {name: sorted(scenes) for name, scenes in splits.items()}
    with open(path, 'w') as handle:
        json.dump(splits, handle, indent=2)
    return splits


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def render(report):
    names = report['trackers']
    lines = [
        '# Tracker comparison on nuScenes-mini',
        '',
        'Same cached BEVFusion detections for every tracker (score >= '
        f"{report['score_thr']}). Official metrics: nuScenes "
        '`TrackingEval`, `tracking_nips_2019`. `mini_val` is the only split '
        'the detector was not trained on; `mini_train` scenes are part of '
        "the checkpoint's training set, so their numbers are optimistic.",
        '',
        '## Aggregate (official)',
        '',
        '| Split | Tracker | AMOTA | AMOTP (m) | MOTA | Recall | IDS | FRAG '
        '| MT | ML | FP | FN |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: '
        '| ---: | ---: |',
    ]
    for split in ('val_scenes', 'train_scenes', 'all_scenes'):
        for name in names:
            m = report['official'].get(name, {}).get(split)
            if not m:
                continue
            lines.append(
                f"| {split} | {TRACKER_LABELS[name]} | {fmt(m['amota'])} | "
                f"{fmt(m['amotp'])} | {fmt(m['mota'])} | {fmt(m['recall'])} | "
                f"{int(m['ids'])} | {int(m['frag'])} | {int(m['mt'])} | "
                f"{int(m['ml'])} | {int(m['fp'])} | {int(m['fn'])} |")
    lines += [
        '',
        '## Per scene (official AMOTA / IDS)',
        '',
        '| Scene | Split | Frames | ' + ' | '.join(
            TRACKER_LABELS[n] for n in names) + ' |',
        '| --- | --- | ---: | ' + ' | '.join('---:' for _ in names) + ' |',
    ]
    for scene, info in sorted(report['scenes'].items()):
        cells = []
        for name in names:
            m = report['official'].get(name, {}).get(scene)
            cells.append(f"{fmt(m['amota'])} / {int(m['ids'])}"
                         if m else 'n/a')
        lines.append(f"| {scene} | {info['split']} | {info['frames']} | "
                     + ' | '.join(cells) + ' |')
    lines += [
        '',
        '## Per class, all scenes (official AMOTA / IDS)',
        '',
        '| Class | ' + ' | '.join(TRACKER_LABELS[n] for n in names) + ' |',
        '| --- | ' + ' | '.join('---:' for _ in names) + ' |',
    ]
    for cls in sorted(TRACKING_CLASSES):
        cells = []
        for name in names:
            labels = report['official_labels'].get(name, {})
            amota = labels.get('amota', {}).get(cls)
            ids = labels.get('ids', {}).get(cls)
            cells.append(f"{fmt(amota)} / {fmt(ids, 0)}")
        lines.append(f'| {cls} | ' + ' | '.join(cells) + ' |')

    sup = report['supplementary']
    lines += [
        '',
        '## Velocity error, all scenes (supplementary)',
        '',
        'Magnitude of the global-frame planar velocity difference, m/s, on '
        'class-matched pairs within 2 m. Ground truth velocity is the devkit '
        '`box_velocity`; pairs where it is undefined (NaN) are excluded. '
        '"Detector" is the raw BEVFusion velocity of the matched detection, '
        'independent of tracking.',
        '',
        '| Subset | Source | N | Mean | Median | p90 |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    first = names[0]
    subsets = (['all', 'gt_moving', 'gt_static']
               + [band for band, _, _ in DISTANCE_BANDS]
               + sorted(TRACKING_CLASSES))
    for subset in subsets:
        det = sup[first]['velocity_detector'].get(subset, {}).get('error')
        if det and det['count']:
            lines.append(
                f"| {subset} | Detector (raw) | {det['count']} | "
                f"{fmt(det['mean'])} | {fmt(det['median'])} | "
                f"{fmt(det['p90'])} |")
        for name in names:
            err = sup[name]['velocity_track'].get(subset, {}).get('error')
            if err and err['count']:
                lines.append(
                    f"| {subset} | {TRACKER_LABELS[name]} | {err['count']} | "
                    f"{fmt(err['mean'])} | {fmt(err['median'])} | "
                    f"{fmt(err['p90'])} |")
    lines += [
        '',
        '## Continuity, association, and motion state (supplementary)',
        '',
        '| Metric | ' + ' | '.join(TRACKER_LABELS[n] for n in names) + ' |',
        '| --- | ' + ' | '.join('---:' for _ in names) + ' |',
    ]
    rows = [
        ('GT instances tracked at least once',
         lambda s: f"{s['continuity']['gt_instances_tracked']} / "
                   f"{s['continuity']['gt_instances']}"),
        ('Mean GT frame coverage',
         lambda s: fmt(s['continuity']['gt_frame_coverage_mean'])),
        ('Track IDs per tracked GT instance',
         lambda s: fmt(s['continuity']['track_ids_per_tracked_gt_mean'])),
        ('GT instances kept on one ID',
         lambda s: fmt(s['continuity']['gt_with_one_id_fraction'])),
        ('ID changes on matched GT',
         lambda s: str(s['continuity']['id_changes_on_matched_gt'])),
        ('Output track IDs',
         lambda s: str(s['continuity']['output_tracks'])),
        ('Track duration median / p90 (s)',
         lambda s: f"{fmt(s['continuity']['track_duration_s']['median'], 1)}"
                   f" / {fmt(s['continuity']['track_duration_s']['p90'], 1)}"),
        ('Tracks with <= 2 frames',
         lambda s: str(s['continuity']['tracks_two_frames_or_fewer'])),
        ('Track purity (dominant GT share)',
         lambda s: fmt(s['continuity']['track_purity_mean'])),
        (f'GT found by a detection >= {GOOD_DET_SCORE} also tracked',
         lambda s: fmt(s['association']['good_detection_also_tracked'])),
        ('Track output precision (all scores)',
         lambda s: fmt(s['association']['track_output_precision'])),
        ('Moving GT classified moving',
         lambda s: fmt(s['motion_classification']['moving_recall'])),
        ('Static GT classified static',
         lambda s: fmt(s['motion_classification']['static_recall'])),
        ('Tracker step mean / p95 (ms)',
         lambda s: f"{fmt(s['step_ms']['mean'], 2)} / "
                   f"{fmt(s['step_ms_p95'], 2)}"),
    ]
    for label, getter in rows:
        lines.append(f'| {label} | ' + ' | '.join(
            getter(sup[n]) for n in names) + ' |')
    lines += [
        '',
        f'Moving GT: speed >= {MOVING_MPS} m/s; static: <= {STATIC_MPS} m/s; '
        'speeds in between are excluded from motion-state scoring.',
        '',
    ]
    return '\n'.join(lines)


def main():
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    payload = load_frames(args.det_dir, args.splits)
    nusc = NuScenes(version='v1.0-mini', dataroot=args.data_root,
                    verbose=False)
    class_range = config_factory('tracking_nips_2019').class_range
    tokens = [frame['token'] for frame in payload['frames']]
    samples = load_ground_truth(nusc, tokens, class_range)
    split_of = {frame['token']: frame['split'] for frame in payload['frames']}
    for token, item in samples.items():
        item['split'] = split_of[token]
    scene_of = {token: item['scene'] for token, item in samples.items()}
    splits_path = os.path.abspath(os.path.join(args.out_dir, 'splits.json'))
    splits = write_splits(splits_path, samples)
    scenes = {}
    for item in samples.values():
        entry = scenes.setdefault(
            item['scene'], {'split': item['split'], 'frames': 0})
        entry['frames'] += 1

    report = {
        'score_thr': args.score_thr,
        'trackers': args.trackers,
        'scenes': scenes,
        'official': {},
        'official_labels': {},
        'supplementary': {},
    }
    for name in args.trackers:
        options = dict(score_thr=args.score_thr, high_score=args.high_score,
                       data_root=args.data_root)
        options.update(TRACKERS[name])
        options = SimpleNamespace(**options)
        tracked = run_tracker(payload, options, scene_of=scene_of,
                              write=False)
        tracker_dir = os.path.join(args.out_dir, name)
        os.makedirs(tracker_dir, exist_ok=True)
        with open(os.path.join(tracker_dir, 'tracks.pkl'), 'wb') as handle:
            pickle.dump(tracked, handle)
        result_path = os.path.join(tracker_dir, 'tracking_result.json')
        tracks_to_submission(tracked, result_path)
        report['supplementary'][name] = supplementary_metrics(
            tracked, payload['frames'], samples, class_range)
        if args.skip_official:
            continue
        report['official'][name] = {}
        for split in sorted(args.official_splits or splits):
            summary = official_eval(
                result_path, os.path.join(tracker_dir, 'eval', split),
                args.data_root, eval_set=split, splits_path=splits_path,
                verbose=False)
            report['official'][name][split] = compact_metrics(summary)
            if split == 'all_scenes':
                report['official_labels'][name] = summary['label_metrics']
            print(f"{name} {split}: AMOTA "
                  f"{report['official'][name][split]['amota']:.3f}",
                  flush=True)

    json_path = os.path.join(args.out_dir, 'comparison.json')
    with open(json_path, 'w') as handle:
        json.dump(report, handle, indent=2, default=float)
    md_path = os.path.join(args.out_dir, 'comparison.md')
    if not args.skip_official:
        with open(md_path, 'w') as handle:
            handle.write(render(report))
    print(f'Wrote {json_path}', flush=True)
    if not args.skip_official:
        print(f'Wrote {md_path}', flush=True)


if __name__ == '__main__':
    main()
