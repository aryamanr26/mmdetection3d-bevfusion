# Copyright (c) OpenMMLab. All rights reserved.
"""Batch GT-vs-prediction visualizer for BEVFusion on nuScenes (headless).

Builds the val dataset from the config, runs the model on each frame, and
saves a BEV plot and a 6-camera mosaic per frame with ground-truth boxes
(green) overlaid on the model predictions (per-class colors).

Example:
    python projects/BEVFusion/demo/batch_gt_vs_pred.py \
        projects/BEVFusion/configs/bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
        checkpoints/bevfusion_lidar-cam.pth \
        --score-thr 0.3 --out-dir demo/mini_out
"""
import argparse
import os
import os.path as osp
import pickle
import sys

import matplotlib
import mmcv
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402

from mmdet3d.apis import init_model  # noqa: E402
from mmdet3d.registry import DATASETS  # noqa: E402
from mmdet3d.structures import LiDARInstance3DBoxes  # noqa: E402

# Allow running as a script: make sibling modules importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_utils import (  # noqa: E402
    DEFAULT_SPEED_THRESH, classify_detections, motion_color, motion_linestyle)
from multi_modality_demo import (  # noqa: E402
    DEFAULT_ARROW_HORIZON, velocity_arrow_endpoints)

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
CLASS_COLORS = [
    (0.10, 0.45, 0.95), (0.10, 0.45, 0.95), (0.10, 0.45, 0.95),
    (0.10, 0.45, 0.95), (0.10, 0.45, 0.95), (1.00, 0.45, 0.05),
    (0.10, 0.75, 0.30), (0.10, 0.75, 0.30), (1.00, 0.85, 0.00),
    (0.95, 0.15, 0.75)
]
GT_COLOR = (0.0, 0.85, 0.0)
# Order of img_paths / lidar2img, i.e. the nuScenes info 'images' dict order.
CAM_SOURCE_ORDER = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
    'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]
# Edges over the 8 corners (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, ...).
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
# Bottom (ground) face, all at z0 -> correct BEV footprint.
BEV_FACE = [0, 3, 7, 4]


def _to_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _l2i_for_cam(lidar2img, cam_idx):
    mat = np.asarray(lidar2img, dtype=np.float64)
    if mat.ndim == 3:
        mat = mat[cam_idx]
    if mat.shape == (3, 3):
        out = np.eye(4)
        out[:3, :3] = mat
        return out
    if mat.shape == (3, 4):
        out = np.eye(4)
        out[:3, :] = mat
        return out
    return mat


def _draw_bev(ax, corners, color, lw):
    for c in corners:
        poly = c[BEV_FACE, :2]
        ax.add_patch(
            Polygon(poly, closed=True, fill=False, edgecolor=color,
                    linewidth=lw))


def _draw_cam(ax, corners, color, lidar2img, w, h, lw):
    for c in corners:
        pts = np.concatenate([c, np.ones((8, 1))], axis=1) @ lidar2img.T
        depth = pts[:, 2]
        if (depth > 0).sum() == 0:
            continue
        uv = pts[:, :2] / np.clip(pts[:, 2:3], 1e-5, None)
        infront = depth > 0
        if ((uv[infront, 0] < 0) | (uv[infront, 0] >= w)
                | (uv[infront, 1] < 0) | (uv[infront, 1] >= h)).all():
            continue
        for a, b in BOX_EDGES:
            # Per-edge near-plane clip: skip edges crossing depth<=0.
            if depth[a] <= 0 or depth[b] <= 0:
                continue
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]],
                    color=color, linewidth=lw)


def _gt_from_info(info):
    """Build (corners[N,8,3], labels[N]) of valid GT boxes from a raw info."""
    rows, labels = [], []
    for obj in info.get('instances', []):
        if not obj.get('bbox_3d_isvalid', True):
            continue
        lab = obj['bbox_label_3d']
        if lab < 0:
            continue
        rows.append(obj['bbox_3d'][:7])
        labels.append(lab)
    if not rows:
        return np.zeros((0, 8, 3)), np.zeros((0,), dtype=int)
    arr = np.asarray(rows, dtype=np.float32)
    boxes = LiDARInstance3DBoxes(arr, box_dim=7, origin=(0.5, 0.5, 0.5))
    return _to_numpy(boxes.corners), np.asarray(labels, dtype=int)


def _radar_evidence_for_frame(token, p_bboxes, dataroot, version, nsweeps,
                              margin):
    """Load radar for a frame's sample token and associate to predictions."""
    from motion_utils import associate_radar_to_boxes
    if not token:
        return None
    try:
        from radar_utils import load_radar_for_info
        xyz, _vcomp, speed = load_radar_for_info(
            dataroot, token, version=version, nsweeps=nsweeps)
        if len(xyz) == 0:
            return None
        return associate_radar_to_boxes(
            p_bboxes, xyz, speed, bev_margin=margin)
    except Exception as exc:  # noqa: BLE001
        print(f'  [radar] disabled ({type(exc).__name__}: {exc})')
        return None


def _passes_filter(motion, motion_filter):
    if motion_filter == 'all':
        return True
    return motion == motion_filter


def _render_bev(points, p_corners, p_bboxes, p_labels, p_scores, p_motions,
                g_corners, out_path, title, motion_filter, color_by,
                arrow_horizon, draw_arrows):
    """One BEV panel: GT green + predictions colored by motion state."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points[:, 0], points[:, 1], s=0.15, c='#b0b0b0', linewidths=0)
    _draw_bev(ax, g_corners, GT_COLOR, 2.0)
    n_drawn = 0
    for i, c in enumerate(p_corners):
        m = p_motions[i]
        if not _passes_filter(m['motion'], motion_filter):
            continue
        n_drawn += 1
        class_color = CLASS_COLORS[p_labels[i] % len(CLASS_COLORS)]
        color = motion_color(m['motion'], class_color, color_by)
        ls = motion_linestyle(m['motion'])
        poly = c[BEV_FACE, :2]
        ax.add_patch(
            Polygon(poly, closed=True, fill=False, edgecolor=color,
                    linewidth=1.4, linestyle=ls))
        if draw_arrows and m['motion'] == 'dynamic':
            endpoints = velocity_arrow_endpoints(
                p_bboxes[i], m, arrow_horizon)
            if endpoints is not None:
                start, tip = endpoints
                ax.annotate(
                    '', xy=tip[:2], xytext=start[:2],
                    arrowprops=dict(
                        arrowstyle='-|>', color=color, linewidth=1.8,
                        mutation_scale=10))
        speed = m['speed_mps']
        speed_txt = f' {speed:.1f}m/s' if speed is not None else ''
        tag = 'dyn' if m['motion'] == 'dynamic' else m['motion'][:4]
        ax.text(poly[:, 0].mean(), poly[:, 1].mean(),
                f"{m['class_name']} {p_scores[i]:.2f} {tag}{speed_txt}",
                color=color, fontsize=6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{title} ({n_drawn} pred)')
    ax.grid(True, alpha=0.2)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def render_frame(points, p_corners, p_bboxes, p_labels, p_scores, p_motions,
                 g_corners, g_labels, lidar2img, img_paths, out_dir, score_thr,
                 motion_filter='all', color_by='motion',
                 arrow_horizon=DEFAULT_ARROW_HORIZON, draw_arrows=True):
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    # ---- BEV panels: combined + (when unfiltered) dynamic / static ----
    panels = [('bev.png', f'BEV GT (green) vs pred (score >= {score_thr})',
               motion_filter)]
    if motion_filter == 'all':
        panels += [
            ('bev_dynamic.png', 'BEV dynamic pred', 'dynamic'),
            ('bev_static.png', 'BEV static pred', 'static'),
        ]
    for fname, title, filt in panels:
        path = osp.join(out_dir, fname)
        _render_bev(points, p_corners, p_bboxes, p_labels, p_scores,
                    p_motions, g_corners, path, title, filt, color_by,
                    arrow_horizon, draw_arrows)
        saved.append(path)

    # ---- Cameras ----
    # Source order of img_paths / lidar2img (matches the pkl images dict).
    n = len(img_paths)
    # Desired mosaic layout: front row (left, center, right) on top,
    # back row (left, center, right) on the bottom.
    layout = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    def _find_cam(name):
        # Prefer matching the camera token in the file path (robust to order);
        # fall back to the known source order.
        for idx, p in enumerate(img_paths):
            if osp.sep + name + osp.sep in p or name in osp.basename(p):
                return idx
        if name in CAM_SOURCE_ORDER:
            return CAM_SOURCE_ORDER.index(name)
        return -1

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for pos, ax in enumerate(axes.ravel()):
        cam_idx = _find_cam(layout[pos]) if pos < len(layout) else pos
        if cam_idx < 0 or cam_idx >= n:
            ax.set_axis_off()
            continue
        img = mmcv.imconvert(mmcv.imread(img_paths[cam_idx]), 'bgr', 'rgb')
        h, w = img.shape[:2]
        ax.imshow(img)
        l2i = _l2i_for_cam(lidar2img, cam_idx)
        _draw_cam(ax, g_corners, GT_COLOR, l2i, w, h, 2.0)
        for i, c in enumerate(p_corners):
            m = p_motions[i]
            if not _passes_filter(m['motion'], motion_filter):
                continue
            class_color = CLASS_COLORS[p_labels[i] % len(CLASS_COLORS)]
            color = motion_color(m['motion'], class_color, color_by)
            _draw_cam(ax, c[None], color, l2i, w, h, 1.3)
            if draw_arrows and m['motion'] == 'dynamic':
                endpoints = velocity_arrow_endpoints(
                    p_bboxes[i], m, arrow_horizon)
                if endpoints is not None:
                    arrow_xyz = np.stack(endpoints)
                    arrow_h = np.concatenate(
                        [arrow_xyz, np.ones((2, 1))], axis=1)
                    projected = arrow_h @ l2i.T
                    if np.all(projected[:, 2] > 0):
                        arrow_uv = projected[:, :2] / projected[:, 2:3]
                        ax.annotate(
                            '', xy=arrow_uv[1], xytext=arrow_uv[0],
                            arrowprops=dict(
                                arrowstyle='-|>', color=color,
                                linewidth=1.8, mutation_scale=10))
        ax.set_title(layout[pos], fontsize=9)
        ax.set_axis_off()
    fig.suptitle('GT (green) vs pred (dynamic=red, static=gray)', fontsize=12)
    fig.tight_layout()
    cams_path = osp.join(out_dir, 'cameras.png')
    fig.savefig(cams_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    saved.append(cams_path)
    return saved


def parse_args():
    p = argparse.ArgumentParser(description='Batch GT-vs-pred BEVFusion viz')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--score-thr', type=float, default=0.3)
    p.add_argument('--speed-thresh', type=float, default=DEFAULT_SPEED_THRESH,
                   help='planar speed (m/s) above which a box is dynamic')
    p.add_argument('--motion-filter', choices=['all', 'dynamic', 'static'],
                   default='all', help='which motion class to render')
    p.add_argument('--color-by', choices=['motion', 'class'], default='motion',
                   help=('moving predictions red and others by grouped class; '
                         'use class to disable the dynamic-red override'))
    p.add_argument('--arrow-horizon', type=float,
                   default=DEFAULT_ARROW_HORIZON,
                   help='seconds of constant velocity represented by arrows')
    p.add_argument('--no-arrows', action='store_true',
                   help='disable velocity arrows on dynamic boxes')
    # Phase 2: radar velocity fusion.
    p.add_argument('--use-radar', action='store_true',
                   help='fuse nuScenes radar compensated velocity')
    p.add_argument('--nusc-version', default='v1.0-mini',
                   help='nuScenes version for radar loading')
    p.add_argument('--radar-fusion', choices=['radar_primary', 'or'],
                   default='radar_primary',
                   help='how to combine radar and box speed')
    p.add_argument('--min-radar-pts', type=int, default=3,
                   help='min in-box radar points to trust radar speed')
    p.add_argument('--radar-margin', type=float, default=0.5,
                   help='BEV margin (m) around each box for radar association')
    p.add_argument('--radar-nsweeps', type=int, default=1,
                   help='radar sweeps to accumulate per channel')
    p.add_argument('--max-frames', type=int, default=0,
                   help='0 = all val frames')
    p.add_argument('--out-dir', default='demo/mini_out')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    model = init_model(args.config, args.checkpoint, device=args.device)

    ds_cfg = cfg.test_dataloader.dataset
    dataset = DATASETS.build(ds_cfg)

    val_pkl = osp.join(cfg.data_root, ds_cfg['ann_file'])
    with open(val_pkl, 'rb') as f:
        data_list = pickle.load(f)['data_list']

    n = len(dataset)
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    print(f'Rendering {n} frames to {args.out_dir}')

    for i in range(n):
        data = dataset[i]
        batch = pseudo_collate([data])
        with torch.no_grad():
            result = model.test_step(batch)[0]

        pred = result.pred_instances_3d
        keep = pred.scores_3d >= args.score_thr
        pred = pred[keep].to('cpu')
        p_corners = _to_numpy(pred.bboxes_3d.corners)
        p_labels = _to_numpy(pred.labels_3d).astype(int)
        p_scores = _to_numpy(pred.scores_3d)
        p_bboxes = _to_numpy(pred.bboxes_3d.tensor)  # 9-D: velocity in 7,8

        # Phase 2: optionally associate radar to predictions for this frame.
        radar_evidence = None
        if args.use_radar:
            radar_evidence = _radar_evidence_for_frame(
                data_list[i].get('token'), p_bboxes, cfg.data_root,
                args.nusc_version, args.radar_nsweeps, args.radar_margin)

        # Classify predictions static/dynamic (box velocity + optional radar).
        p_motions = classify_detections(
            p_labels, p_bboxes, CLASS_NAMES, speed_thresh=args.speed_thresh,
            radar_evidence=radar_evidence, radar_fusion=args.radar_fusion,
            min_radar_pts=args.min_radar_pts)
        n_dyn = sum(1 for m in p_motions if m['motion'] == 'dynamic')
        n_stat = sum(1 for m in p_motions if m['motion'] == 'static')

        g_corners, g_labels = _gt_from_info(data_list[i])

        points = _to_numpy(data['inputs']['points'])
        meta = result.metainfo
        lidar2img = meta.get('ori_lidar2img', meta.get('lidar2img'))
        img_paths = meta['img_path']
        if not isinstance(img_paths, list):
            img_paths = [img_paths]

        out_dir = osp.join(args.out_dir, f'frame_{i:03d}')
        render_frame(
            points, p_corners, p_bboxes, p_labels, p_scores, p_motions,
            g_corners, g_labels, lidar2img, img_paths, out_dir,
            args.score_thr, motion_filter=args.motion_filter,
            color_by=args.color_by, arrow_horizon=args.arrow_horizon,
            draw_arrows=not args.no_arrows)
        print(f'[{i + 1}/{n}] {out_dir}: {len(p_corners)} pred '
              f'({n_dyn} dyn, {n_stat} static), {len(g_corners)} GT')


if __name__ == '__main__':
    main()
