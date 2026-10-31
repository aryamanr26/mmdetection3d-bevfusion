"""Per-stage latency of the BEVFusion + tracking + Foxglove frame path.

Runs the same work as ``foxglove_stream.build_playback_frame`` one keyframe
at a time, in scene order, with a fresh tracker and sensor cache per scene.
Stages are timed back to back with ``time.perf_counter`` so they do not
overlap, and ``torch.cuda.synchronize`` brackets inference so queued GPU work
is charged to inference only:

* preprocess: info lookup, LiDAR sweep merge, JPEG decode, model transforms
* inference: ``model.test_step`` plus the device-to-host copy of the boxes
* tracking: detection motion labels, global transform, ``tracker.step``
* colorize: camera colors for the displayed cloud (visualization only)
* elevation: ground elevation grid and its Grid messages (display only)
* messages: tracks back to the LiDAR frame, then Foxglove point cloud,
  scene, and image message construction
* encode: protobuf serialization of every message (done at publish time)

``perception`` is preprocess + inference + tracking. ``full`` adds the four
visualization stages. The first ``--warmup`` frames are run and discarded.
LiDAR and image files are read from the OS page cache after warmup, which
matches an in-memory sensor feed better than a cold disk.

Example:
    python projects/BEVFusion/demo/benchmark_latency.py
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from foxglove_stream import (  # noqa: E402
    CHECKPOINT, CONFIG, DEFAULT_LIDAR_RGB, _cached_points, _colorize_points,
    _load_cameras, _load_sweep_points, _pack_model_inputs, _SensorCache,
    _subsample, classify_detections, estimate_elevation_grid,
    gt_boxes_from_info, load_detector, make_camera_images,
    make_elevation_grids, make_ego_scene, make_gt_scene, make_pointcloud,
    make_scene, make_track_scene, tracks_to_lidar)
from run_tracking import scene_names_by_token  # noqa: E402
from tracker import (  # noqa: E402
    MultiObjectTracker, transform_detection, transform_matrix)

STAGES = ('preprocess', 'inference', 'tracking', 'colorize', 'elevation',
          'messages', 'encode')
PERCEPTION = ('preprocess', 'inference', 'tracking')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--infos', default='data/nuscenes/nuscenes_infos_val.pkl')
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--sweeps', type=int, default=9)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--max-points', type=int, default=350000)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument(
        '--out', default='demo/eval_out/tracking_compare/latency.json')
    return parser.parse_args()


def run_frame(model, dataset, class_names, index, sweeps, score_thr,
              max_points, cache, tracker):
    import torch
    from mmengine.dataset import pseudo_collate

    times = {}
    started = time.perf_counter()
    info = dataset.get_data_info(index)
    points = _load_sweep_points(info, sweeps, cache)
    cameras = _load_cameras(info, cache)
    packed = _pack_model_inputs(dataset, info, points, cameras, cache)
    torch.cuda.synchronize()
    mark = time.perf_counter()
    times['preprocess'] = mark - started

    with torch.inference_mode():
        result = model.test_step(pseudo_collate([packed]))[0]
        pred = result.pred_instances_3d
        keep = pred.scores_3d >= score_thr
        labels = pred.labels_3d[keep].cpu().numpy().astype(int)
        scores = pred.scores_3d[keep].cpu().numpy()
        bboxes = pred.bboxes_3d.tensor[keep].cpu().numpy()
    torch.cuda.synchronize()
    now = time.perf_counter()
    times['inference'], mark = now - mark, now

    motions = classify_detections(labels, bboxes, class_names)
    boxes = [{
        'label': int(label),
        'class_name': motion['class_name'],
        'motion': motion['motion'],
        'score': float(score),
        'bbox_3d': bbox.tolist(),
    } for label, score, bbox, motion in zip(labels, scores, bboxes, motions)]
    lidar2global = transform_matrix(info)
    global_tracks = tracker.step(
        [transform_detection(box, lidar2global) for box in boxes],
        float(info['timestamp']))
    now = time.perf_counter()
    times['tracking'], mark = now - mark, now

    xyz = np.ascontiguousarray(points[:, :3])
    meta = {
        'lidar2img': cameras['lidar2img'],
        'img_path': cameras['paths'],
        'camera_names': cameras['names'],
        'camera_jpeg': cameras['jpeg'],
    }
    rgb = _colorize_points(xyz, cameras['images'], meta)
    if rgb is None:
        rgb = np.repeat(np.asarray(DEFAULT_LIDAR_RGB, dtype=np.uint8)[None],
                        len(xyz), axis=0)
    xyz, rgb = _subsample(xyz, rgb, max_points)
    now = time.perf_counter()
    times['colorize'], mark = now - mark, now

    time_ns = int(float(info['timestamp']) * 1_000_000_000)
    key = _cached_points(info['lidar_points']['lidar_path'], cache)
    elevation = make_elevation_grids(
        estimate_elevation_grid(key[:, :3]), time_ns)
    now = time.perf_counter()
    times['elevation'], mark = now - mark, now

    tracks = tracks_to_lidar(global_tracks, lidar2global)
    gt_boxes = gt_boxes_from_info(
        info, class_names=class_names,
        metainfo=getattr(dataset, 'metainfo', None))
    messages = [
        make_pointcloud(xyz, rgb),
        make_scene(boxes, replace=True),
        make_track_scene(tracks, replace=True),
        make_gt_scene(gt_boxes, replace=True),
        make_ego_scene(replace=True),
    ]
    messages += list(make_camera_images(
        meta['camera_names'], meta['camera_jpeg']).values())
    now = time.perf_counter()
    times['messages'], mark = now - mark, now

    encoded = sum(len(message.encode())
                  for message in messages + list(elevation.values()))
    now = time.perf_counter()
    times['encode'] = now - mark
    times['wall'] = now - started
    return {name: value * 1e3 for name, value in times.items()}, {
        'detections': len(boxes),
        'tracks': len(tracks),
        'points': int(len(points)),
        'encoded_bytes': int(encoded),
    }


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'max': float(values.max()),
    }


def gpu_memory_mib():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.strip().splitlines():
        pid, used = (part.strip() for part in line.split(','))
        if int(pid) == os.getpid():
            return int(used)
    return None


def main():
    import torch

    args = parse_args()
    model, dataset, class_names = load_detector(
        args.infos, args.config, args.checkpoint)
    with open(args.infos, 'rb') as handle:
        data_list = pickle.load(handle)['data_list']
    scene_of = scene_names_by_token(args.data_root)
    scenes = {}
    for index, info in enumerate(data_list):
        scenes.setdefault(scene_of[info['token']], []).append(
            (float(info['timestamp']), index))
    order = [[index for _, index in sorted(rows)]
             for _, rows in sorted(scenes.items())]

    cache = _SensorCache()
    tracker = MultiObjectTracker(low_score=args.score_thr)
    for index in order[0][:args.warmup]:
        run_frame(model, dataset, class_names, index, args.sweeps,
                  args.score_thr, args.max_points, cache, tracker)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for indices in order:
        cache = _SensorCache()
        tracker = MultiObjectTracker(low_score=args.score_thr)
        for index in indices:
            times, counts = run_frame(
                model, dataset, class_names, index, args.sweeps,
                args.score_thr, args.max_points, cache, tracker)
            times['perception'] = sum(times[name] for name in PERCEPTION)
            times['full'] = sum(times[name] for name in STAGES)
            rows.append({**times, **counts})
        cache.release_points()

    report = {
        'frames': len(rows),
        'warmup_frames': args.warmup,
        'score_thr': args.score_thr,
        'sweeps': args.sweeps,
        'device': torch.cuda.get_device_name(0),
        'stages_ms': {
            name: stats([row[name] for row in rows])
            for name in STAGES + ('perception', 'full', 'wall')},
        'perception_over_500ms': int(sum(
            row['perception'] > 500.0 for row in rows)),
        'full_over_500ms': int(sum(row['full'] > 500.0 for row in rows)),
        'workload': {
            name: stats([row[name] for row in rows])
            for name in ('detections', 'tracks', 'points', 'encoded_bytes')},
        'gpu_memory_mib': {
            'torch_peak_allocated': torch.cuda.max_memory_allocated() / 2**20,
            'torch_peak_reserved': torch.cuda.max_memory_reserved() / 2**20,
            'process_nvidia_smi': gpu_memory_mib(),
            'device_total': torch.cuda.get_device_properties(0).total_memory
            / 2**20,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f'{"stage":<12}{"mean":>9}{"median":>9}{"p95":>9}{"max":>9}  ms')
    for name, value in report['stages_ms'].items():
        print(f'{name:<12}{value["mean"]:9.1f}{value["median"]:9.1f}'
              f'{value["p95"]:9.1f}{value["max"]:9.1f}')
    print(json.dumps(report['gpu_memory_mib'], indent=2))
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
