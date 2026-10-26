# Copyright (c) OpenMMLab. All rights reserved.
"""Build a local elevation grid for the nuScenes keyframes on disk.

The grid is LiDAR-only. Pictures overlay nuScenes ground-truth boxes in the
LiDAR frame. ``--detections`` also runs BEVFusion on those pictured frames.

Example:
    python projects/BEVFusion/demo/elevation_grid_demo.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation_grid import (  # noqa: E402
    RESOLUTION, SIZE, X_MIN, Y_MIN, estimate_elevation_grid, self_check)
from eval_gt_vs_pred import CLASS_NAMES, gt_records_from_info  # noqa: E402
from foxglove_stream import (  # noqa: E402
    ELEVATION_TOPICS, _elevation_channels, make_elevation_grids)
from multi_modality_demo import lidar_box_corners, load_lidar_xyz  # noqa: E402

LIDAR_DIR = 'data/nuscenes/samples/LIDAR_TOP'
INFO_FILES = (
    'data/nuscenes/nuscenes_infos_train.pkl',
    'data/nuscenes/nuscenes_infos_val.pkl',
)
DEFAULT_FRAMES = (
    'n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin',
    'n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin',
)
BEV_FACE = [0, 3, 7, 4]


def _resolve(name):
    for path in (
            os.path.join(LIDAR_DIR, name),
            os.path.join('demo/data/nuscenes', name),
            name):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(name)


def _info_index():
    index = {}
    for path in INFO_FILES:
        with open(path, 'rb') as handle:
            blob = pickle.load(handle)
        meta = blob.get('metainfo') or {}
        for info in blob['data_list']:
            name = os.path.basename(info['lidar_points']['lidar_path'])
            index[name] = (info, meta, path)
    return index


def _overlay(ax, records, color):
    if not records:
        return
    boxes = np.stack([record['bbox_3d'][:7] for record in records])
    corners = lidar_box_corners(boxes)
    for corner in corners:
        footprint = corner[BEV_FACE, :2]
        ax.add_patch(Polygon(
            footprint, closed=True, fill=False, edgecolor=color,
            linewidth=0.8))


def _panel(ax, values, title, cmap, vmin, vmax):
    masked = np.ma.array(values, mask=~np.isfinite(values))
    span = SIZE * RESOLUTION
    image = ax.imshow(
        masked, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[X_MIN, X_MIN + span, Y_MIN, Y_MIN + span],
        interpolation='nearest')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(X_MIN, X_MIN + span)
    ax.set_ylim(Y_MIN, Y_MIN + span)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    return image


def _limits(values, default_max):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, default_max
    return float(finite.min()), float(max(finite.max(), default_max))


def render_frame(grid, gt_records, pred_records, out_path):
    """Save one figure with ground, slope, and obstacle heatmaps."""
    ground_finite = grid.ground[grid.valid]
    if ground_finite.size:
        gmin, gmax = np.percentile(ground_finite, [5, 95])
        if gmax - gmin < 0.05:
            gmax = gmin + 0.05
    else:
        gmin, gmax = -2.0, -1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = (
        (grid.ground, 'Ground height (m)', 'terrain', gmin, gmax),
        (grid.slope, 'Slope (deg)', 'viridis', 0.0, 15.0),
        (grid.obstacle, 'Obstacle evidence', 'inferno', 0.0,
         max(_limits(grid.obstacle, 1.0)[1], 1.0)),
    )
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, panels):
        image = _panel(ax, values, title, cmap, vmin, vmax)
        _overlay(ax, gt_records, 'white')
        _overlay(ax, pred_records, '#ff3030')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('White: ground truth. Red: detections, when requested.')
    fig.tight_layout()
    started = time.perf_counter()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return (time.perf_counter() - started) * 1e3


def _summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def _write_timing(path, rows, pictures):
    lines = [
        '# Elevation grid timing',
        '',
        f'- Keyframes: {len(rows)}',
        '- Stages are measured separately. Detector time is not included.',
        '',
        '| Stage | Median (ms) | p95 (ms) |',
        '| --- | ---: | ---: |',
    ]
    for stage in ('preprocess', 'ground_fit', 'grid_update'):
        median, p95 = _summarize([row[stage] for row in rows])
        lines.append(f'| {stage} | {median:.2f} | {p95:.2f} |')
    if pictures:
        lines.extend([
            '',
            '## Pictured frames',
            '',
            '| Frame | Known cells | Median ground (m) | Median slope (deg) | PNG (ms) | Foxglove pack (ms) |',
            '| --- | ---: | ---: | ---: | ---: | ---: |',
        ])
        for item in pictures:
            lines.append(
                f'| `{item["name"]}` | {item["known"]:.1%} | '
                f'{item["ground"]:.2f} | {item["slope"]:.2f} | '
                f'{item["png_ms"]:.1f} | {item["pack_ms"]:.1f} |')
    text = '\n'.join(lines) + '\n'
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def _predictions(index, names, config, checkpoint, score_thr):
    """Run BEVFusion on the pictured frames only."""
    from collections import defaultdict

    from eval_gt_vs_pred import infer_predictions
    from foxglove_stream import CONFIG, _SensorCache, load_detector

    grouped = defaultdict(list)
    for name in names:
        _info, _meta, info_path = index[name]
        grouped[info_path].append(name)
    found = {}
    for info_path, group in grouped.items():
        model, dataset, class_names = load_detector(
            info_path, config or CONFIG, checkpoint)
        with open(info_path, 'rb') as handle:
            data_list = pickle.load(handle)['data_list']
        lookup = {
            os.path.basename(info['lidar_points']['lidar_path']): i
            for i, info in enumerate(data_list)
        }
        cache = _SensorCache()
        for name in group:
            found[name] = infer_predictions(
                model, dataset, lookup[name], score_thr, class_names, 9, cache)
    return found


def _check_pack(grid):
    packed = make_elevation_grids(grid)
    if set(packed) != {'ground', 'slope', 'obstacle'}:
        raise AssertionError(sorted(packed))
    for message in packed.values():
        if len(message.encode()) < grid.size * grid.size:
            raise AssertionError('elevation grid message is too small')
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='demo/eval_out/elevation')
    parser.add_argument('--frames', nargs='*', default=list(DEFAULT_FRAMES))
    parser.add_argument(
        '--self-check-only', action='store_true',
        help='run the synthetic ramp, curb, and box check, then exit')
    parser.add_argument(
        '--detections', action='store_true',
        help='overlay BEVFusion boxes on the pictured frames')
    parser.add_argument(
        '--config',
        default=('projects/BEVFusion/configs/'
                 'bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'))
    parser.add_argument(
        '--checkpoint', default='checkpoints/bevfusion_lidar-cam.pth')
    parser.add_argument('--score-thr', type=float, default=0.2)
    args = parser.parse_args()

    grid = self_check()
    _check_pack(grid)
    print('Synthetic ramp, curb, and box check passed.', flush=True)
    if args.self_check_only:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    index = _info_index()
    names = sorted(name for name in os.listdir(LIDAR_DIR) if name.endswith('.pcd.bin'))
    rows = []
    cached = {}
    for number, name in enumerate(names, 1):
        xyz = load_lidar_xyz(os.path.join(LIDAR_DIR, name))
        grid = estimate_elevation_grid(xyz)
        rows.append(grid.timings_ms)
        if name in args.frames:
            cached[name] = grid
        if number % 100 == 0 or number == len(names):
            print(f'Timed {number}/{len(names)} keyframes', flush=True)

    predictions = {}
    if args.detections:
        predictions = _predictions(
            index, args.frames, args.config, args.checkpoint, args.score_thr)

    pictures = []
    packed_frames = []
    for name in args.frames:
        grid = cached.get(name)
        if grid is None:
            grid = estimate_elevation_grid(load_lidar_xyz(_resolve(name)))
        info, meta, _path = index[name]
        gt_records = gt_records_from_info(info, CLASS_NAMES, meta)
        stem = name.replace('.pcd.bin', '')
        png_ms = render_frame(
            grid, gt_records, predictions.get(name, []),
            os.path.join(args.out_dir, f'{stem}.png'))
        np.savez_compressed(
            os.path.join(args.out_dir, f'{stem}.npz'),
            ground=grid.ground, variance=grid.variance, slope=grid.slope,
            roughness=grid.roughness, obstacle=grid.obstacle,
            resolution=np.float32(RESOLUTION),
            x_min=np.float32(X_MIN), y_min=np.float32(Y_MIN))
        started = time.perf_counter()
        packed = make_elevation_grids(grid)
        pack_ms = (time.perf_counter() - started) * 1e3
        packed_frames.append(packed)
        finite = grid.ground[grid.valid]
        slope = grid.slope[grid.valid]
        pictures.append({
            'name': name,
            'known': float(grid.valid.mean()),
            'ground': float(np.median(finite)) if finite.size else float('nan'),
            'slope': float(np.median(slope)) if slope.size else float('nan'),
            'png_ms': png_ms,
            'pack_ms': pack_ms,
        })
        print(f'Wrote {stem}.png', flush=True)

    import foxglove
    mcap_path = os.path.join(args.out_dir, 'elevation.mcap')
    if os.path.exists(mcap_path):
        os.remove(mcap_path)
    channels = _elevation_channels()
    with foxglove.open_mcap(mcap_path):
        for packed in packed_frames:
            for kind, _topic, _offset in ELEVATION_TOPICS:
                channels[kind].log(packed[kind])
    print(f'Wrote {mcap_path}', flush=True)

    timing_path = os.path.join(args.out_dir, 'timing.md')
    _write_timing(timing_path, rows, pictures)
    print(f'Wrote {timing_path}', flush=True)


if __name__ == '__main__':
    main()
