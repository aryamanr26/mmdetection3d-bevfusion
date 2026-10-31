"""Run and evaluate training-free 3D tracking on nuScenes mini.

Examples:
    # Cache BEVFusion detections, track them, and run official mini-val eval.
    python projects/BEVFusion/demo/run_tracking.py

    # Re-tune the tracker without running the detector again.
    python projects/BEVFusion/demo/run_tracking.py --skip-inference
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from foxglove_stream import (  # noqa: E402
    CHECKPOINT, CONFIG, _SensorCache, infer_loaded, load_detector)
from tracker import (  # noqa: E402
    TRACKING_CLASSES, ConstantVelocityTracker, MultiObjectTracker,
    transform_detection, transform_matrix)


DEFAULT_INFOS = 'data/nuscenes/nuscenes_infos_val.pkl'
DEFAULT_OUT = 'demo/eval_out/tracking'
DEFAULT_DATA_ROOT = 'data/nuscenes'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infos', default=DEFAULT_INFOS)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--high-score', type=float, default=0.4)
    parser.add_argument(
        '--birth-score', type=float, default=0.05,
        help='minimum score for an unmatched detection to start a track')
    parser.add_argument('--sweeps', type=int, default=9)
    parser.add_argument(
        '--motion-model', choices=('poly', 'cv'), default='poly')
    parser.add_argument(
        '--tracker', choices=('poly', 'cv_baseline'), default='poly',
        help='poly: Kalman MultiObjectTracker; cv_baseline: '
             'ConstantVelocityTracker')
    parser.add_argument('--skip-inference', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--oracle-only', action='store_true')
    return parser.parse_args()


def scene_key(info: dict) -> str:
    """Scene/log key encoded in the nuScenes LiDAR filename."""
    name = os.path.basename(info['lidar_points']['lidar_path'])
    return name.split('__')[0]


def split_name(infos_path: str) -> str:
    """``nuscenes_infos_val.pkl`` -> ``val``."""
    stem = os.path.splitext(os.path.basename(infos_path))[0]
    return stem.rsplit('_', 1)[-1]


def scene_names_by_token(data_root: str) -> dict:
    """Map sample token to nuScenes scene name.

    A log can hold several scenes, so the log prefix in the LiDAR filename is
    not a scene identifier.
    """
    from nuscenes import NuScenes

    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)
    return {
        sample['token']: nusc.get('scene', sample['scene_token'])['name']
        for sample in nusc.sample}


def make_tracker(args):
    if args.tracker == 'cv_baseline':
        return ConstantVelocityTracker(low_score=args.score_thr)
    return MultiObjectTracker(
        high_score=args.high_score,
        low_score=args.score_thr,
        min_hits=getattr(args, 'min_hits', 2),
        motion_model=args.motion_model,
        output_coasted=getattr(args, 'output_coasted', True),
        birth_score=getattr(args, 'birth_score', None) or 0.05)


def _jsonable_box(box: dict) -> dict:
    result = dict(box)
    result['bbox_3d'] = np.asarray(box['bbox_3d']).tolist()
    return result


def cache_detections(args) -> tuple[dict, list[str]]:
    """Run BEVFusion once and persist low-threshold per-frame detections."""
    cache_path = os.path.join(
        args.out_dir, f'dets_{split_name(args.infos)}.pkl')
    if args.skip_inference:
        with open(cache_path, 'rb') as handle:
            payload = pickle.load(handle)
        return payload, payload['class_names']

    model, dataset, class_names = load_detector(
        args.infos, args.config, args.checkpoint)
    groups = defaultdict(list)
    # BaseDataset serializes data_list by default, leaving the public list
    # empty. get_data_info() works in both serialized and unserialized modes.
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        groups[scene_key(info)].append((index, info))
    for rows in groups.values():
        rows.sort(key=lambda item: float(item[1]['timestamp']))

    frames = []
    sensor_cache = _SensorCache()
    total = sum(map(len, groups.values()))
    number = 0
    for scene, rows in sorted(groups.items()):
        sensor_cache.release_points()
        for index, raw_info in rows:
            number += 1
            started = time.perf_counter()
            boxes, _images, _meta, _xyz = infer_loaded(
                model, dataset, index, args.score_thr, class_names,
                args.sweeps, sensor_cache)
            info = dataset.get_data_info(index)
            frames.append({
                'scene': scene,
                'token': str(info['token']),
                'timestamp': float(info['timestamp']),
                'lidar2global': transform_matrix(info).tolist(),
                'detections': [_jsonable_box(box) for box in boxes],
            })
            print(
                f'[{number}/{total}] {scene} {len(boxes)} detections '
                f'{time.perf_counter() - started:.2f}s', flush=True)
    payload = {
        'infos': os.path.abspath(args.infos),
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint),
        'score_thr': args.score_thr,
        'class_names': class_names,
        'frames': frames,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(cache_path, 'wb') as handle:
        pickle.dump(payload, handle)
    print(f'Wrote {cache_path}', flush=True)
    return payload, class_names


def run_tracker(payload: dict, args, scene_of=None,
                write=True) -> list[dict]:
    """Track cached detections independently in each nuScenes scene."""
    if scene_of is None:
        scene_of = scene_names_by_token(args.data_root)
    frames_by_scene = defaultdict(list)
    for frame in payload['frames']:
        frames_by_scene[scene_of[frame['token']]].append(frame)
    tracked_frames = []
    for scene, frames in sorted(frames_by_scene.items()):
        frames.sort(key=lambda frame: frame['timestamp'])
        tracker = make_tracker(args)
        for frame in frames:
            matrix = np.asarray(frame['lidar2global'], dtype=np.float64)
            global_dets = [
                transform_detection(det, matrix)
                for det in frame['detections']]
            started = time.perf_counter()
            tracks = tracker.step(global_dets, frame['timestamp'])
            step_ms = (time.perf_counter() - started) * 1e3
            tracked_frames.append({
                'scene': scene,
                'token': frame['token'],
                'timestamp': frame['timestamp'],
                'lidar2global': frame['lidar2global'],
                'num_detections': len(frame['detections']),
                'step_ms': step_ms,
                'tracks': tracks,
            })
    tracked_frames.sort(key=lambda frame: (frame['scene'], frame['timestamp']))
    if not write:
        return tracked_frames
    path = os.path.join(
        args.out_dir, f'tracks_{split_name(args.infos)}.pkl')
    with open(path, 'wb') as handle:
        pickle.dump(tracked_frames, handle)
    print(f'Wrote {path}', flush=True)
    return tracked_frames


def _yaw_quaternion(yaw: float) -> list[float]:
    return [
        float(math.cos(yaw / 2.0)), 0.0, 0.0,
        float(math.sin(yaw / 2.0))]


def tracks_to_submission(frames: list[dict], path: str) -> dict:
    """Write tracked global boxes in the official nuScenes JSON schema."""
    results = {}
    for frame in frames:
        rows = []
        for track in frame['tracks']:
            box = np.asarray(track['bbox_3d'], dtype=np.float64)
            rows.append({
                'sample_token': frame['token'],
                'translation': [
                    float(box[0]), float(box[1]),
                    float(box[2] + box[5] / 2.0)],
                'size': [float(box[4]), float(box[3]), float(box[5])],
                'rotation': _yaw_quaternion(float(box[6])),
                'velocity': [float(box[7]), float(box[8])],
                'tracking_id': str(track['tracking_id']),
                'tracking_name': track['class_name'],
                'tracking_score': float(np.clip(track['score'], 0.0, 1.0)),
            })
        results[frame['token']] = rows
    payload = {
        'meta': {
            'use_camera': True,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        },
        'results': results,
    }
    with open(path, 'w') as handle:
        json.dump(payload, handle)
    print(f'Wrote {path}', flush=True)
    return payload


def make_oracle_submission(data_root: str, sample_tokens: set[str],
                           path: str) -> None:
    """Write an identity-preserving GT submission to validate the pipeline."""
    from nuscenes import NuScenes
    from nuscenes.eval.tracking.utils import category_to_tracking_name

    nusc = NuScenes(
        version='v1.0-mini', dataroot=data_root, verbose=False)
    results = {}
    for token in sample_tokens:
        sample = nusc.get('sample', token)
        rows = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            name = category_to_tracking_name(ann['category_name'])
            if name not in TRACKING_CLASSES:
                continue
            velocity = np.nan_to_num(nusc.box_velocity(ann_token)[:2])
            rows.append({
                'sample_token': token,
                'translation': list(map(float, ann['translation'])),
                'size': list(map(float, ann['size'])),
                'rotation': list(map(float, ann['rotation'])),
                'velocity': list(map(float, velocity)),
                'tracking_id': str(ann['instance_token']),
                'tracking_name': name,
                'tracking_score': 1.0,
            })
        results[token] = rows
    with open(path, 'w') as handle:
        json.dump({
            'meta': {
                'use_camera': False, 'use_lidar': True, 'use_radar': False,
                'use_map': False, 'use_external': False,
            },
            'results': results,
        }, handle)
    print(f'Wrote {path}', flush=True)


def official_eval(result_path: str, output_dir: str,
                  data_root: str, eval_set: str = 'mini_val',
                  splits_path: str = None, verbose: bool = True) -> dict:
    """Run official nuScenes tracking metrics and return their summary.

    ``eval_set`` may be a custom split named in ``splits_path``, a JSON map of
    split name to scene names. The devkit reads custom splits from
    ``<dataroot>/<version>/splits.json``; patching its path lookup keeps the
    dataset directory untouched.
    """
    from unittest import mock

    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval

    with mock.patch(
            'nuscenes.utils.splits._get_custom_splits_file_path',
            return_value=splits_path):
        evaluator = TrackingEval(
            config=config_factory('tracking_nips_2019'),
            result_path=result_path,
            eval_set=eval_set,
            output_dir=output_dir,
            nusc_version='v1.0-mini',
            nusc_dataroot=data_root,
            verbose=verbose,
        )
        evaluator.main(render_curves=False)
    with open(os.path.join(output_dir, 'metrics_summary.json')) as handle:
        return json.load(handle)


def _quat_yaw(rotation: list[float]) -> float:
    w, x, y, z = rotation
    return math.atan2(
        2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def evaluate_forecasts(frames: list[dict], data_root: str) -> dict:
    """Evaluate matched track forecasts against future GT instance centers."""
    from nuscenes import NuScenes
    from nuscenes.eval.tracking.utils import category_to_tracking_name

    nusc = NuScenes(
        version='v1.0-mini', dataroot=data_root, verbose=False)
    errors = defaultdict(list)
    for frame in frames:
        sample = nusc.get('sample', frame['token'])
        gt = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            name = category_to_tracking_name(ann['category_name'])
            if name in TRACKING_CLASSES:
                gt.append((ann, name))
        used = set()
        for track in sorted(
                frame['tracks'], key=lambda item: item['score'], reverse=True):
            box = np.asarray(track['bbox_3d'])
            candidates = [
                (np.linalg.norm(box[:2] - np.asarray(ann['translation'][:2])),
                 idx, ann)
                for idx, (ann, name) in enumerate(gt)
                if idx not in used and name == track['class_name']]
            if not candidates:
                continue
            distance, idx, ann = min(candidates, key=lambda item: item[0])
            if distance > 2.0:
                continue
            used.add(idx)
            future = {}
            token = ann['next']
            current_time = frame['timestamp']
            while token:
                item = nusc.get('sample_annotation', token)
                future_sample = nusc.get('sample', item['sample_token'])
                dt = future_sample['timestamp'] / 1e6 - current_time
                if dt > 3.25:
                    break
                future[round(dt * 2.0) / 2.0] = np.asarray(
                    item['translation'][:2], dtype=np.float64)
                token = item['next']
            pred = {
                round(point[0] * 2.0) / 2.0: np.asarray(point[1:3])
                for point in track['forecast']}
            track_errors = []
            for horizon in (1.0, 2.0, 3.0):
                if horizon in pred and horizon in future:
                    error = float(np.linalg.norm(
                        pred[horizon] - future[horizon]))
                    errors[f'fde_{int(horizon)}s'].append(error)
                    track_errors.append(error)
            if track_errors:
                errors['ade'].append(float(np.mean(track_errors)))
    summary = {
        key: {
            'mean_m': float(np.mean(values)),
            'count': len(values),
        }
        for key, values in errors.items()
    }
    return summary


def compact_metrics(summary: dict) -> dict:
    keys = ('amota', 'amotp', 'mota', 'motar', 'recall', 'ids', 'frag',
            'mt', 'ml', 'fp', 'fn', 'tp', 'gt', 'tid', 'lgd', 'faf')
    return {key: summary.get(key) for key in keys if key in summary}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.infos, 'rb') as handle:
        infos = pickle.load(handle)['data_list']
    sample_tokens = {str(info['token']) for info in infos}
    split = split_name(args.infos)
    # Keep the original val file names so earlier outputs stay comparable.
    suffix = '' if split == 'val' else f'_{split}'
    eval_set = f'mini_{split}'

    report = {}
    if not args.oracle_only:
        payload, _class_names = cache_detections(args)
        frames = run_tracker(payload, args)
        result_path = os.path.join(
            args.out_dir, f'tracking_result{suffix}.json')
        tracks_to_submission(frames, result_path)
        forecast = evaluate_forecasts(frames, args.data_root)
        report['forecast'] = forecast
        if not args.skip_eval:
            metrics = official_eval(
                result_path, os.path.join(args.out_dir, f'eval{suffix}'),
                args.data_root, eval_set=eval_set)
            report['tracking'] = compact_metrics(metrics)

    oracle_path = os.path.join(args.out_dir, f'tracking_oracle{suffix}.json')
    make_oracle_submission(args.data_root, sample_tokens, oracle_path)
    if not args.skip_eval:
        oracle = official_eval(
            oracle_path, os.path.join(args.out_dir, f'oracle_eval{suffix}'),
            args.data_root, eval_set=eval_set)
        report['oracle'] = compact_metrics(oracle)

    report_path = os.path.join(args.out_dir, f'summary{suffix}.json')
    with open(report_path, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f'Wrote {report_path}', flush=True)


if __name__ == '__main__':
    main()
