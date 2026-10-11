# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import sys
from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet3d.apis import inference_multi_modality_detector, init_model
from mmdet3d.registry import VISUALIZERS

# Allow running as a script: make sibling modules importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_utils import (  # noqa: E402
    DEFAULT_SPEED_THRESH, classify_detections, motion_color, motion_linestyle)
from colorize_lidar import colorize_lidar  # noqa: E402

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
CLASS_COLORS = [
    (0.10, 0.45, 0.95),  # car: blue
    (0.10, 0.45, 0.95),  # truck: blue
    (0.10, 0.45, 0.95),  # construction vehicle: blue
    (0.10, 0.45, 0.95),  # bus: blue
    (0.10, 0.45, 0.95),  # trailer: blue
    (1.00, 0.45, 0.05),  # barrier: orange
    (0.10, 0.75, 0.30),  # motorcycle: green
    (0.10, 0.75, 0.30),  # bicycle: green
    (1.00, 0.85, 0.00),  # pedestrian: yellow
    (0.95, 0.15, 0.75),  # traffic cone: magenta
]
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def _to_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _lidar2img_matrix(meta, cam_idx=None):
    lidar2img = meta.get('ori_lidar2img', meta.get('lidar2img'))
    if lidar2img is None:
        return None
    mat = np.asarray(lidar2img, dtype=np.float64)
    if mat.ndim == 3:
        mat = mat[0 if cam_idx is None else cam_idx]
    if mat.shape == (3, 3):
        out = np.eye(4)
        out[:3, :3] = mat
        return out
    if mat.shape == (3, 4):
        out = np.eye(4)
        out[:3, :] = mat
        return out
    return mat


def _lidar2img_list(meta, num_cams):
    """Return a list of per-camera 4x4 lidar2img matrices (index-aligned)."""
    mats = []
    for i in range(num_cams):
        mats.append(_lidar2img_matrix(meta, i))
    return mats


def _colorize_points(points, imgs, meta):
    """Camera-colorize the lidar cloud; returns (N, 3) uint8 or None.

    XYZ is never moved out of the LiDAR frame -- only the RGB is sampled from
    the camera images via ``lidar2img``, so boxes still align.
    """
    xyz = np.asarray(points)[:, :3]
    img_list = imgs if isinstance(imgs, list) else [imgs]
    img_list = [_to_numpy(im) for im in img_list]
    lidar2img = _lidar2img_list(meta, len(img_list))
    if all(m is None for m in lidar2img):
        print('  [colorize] no lidar2img in metainfo; skipping colorization')
        return None
    rgb, colored = colorize_lidar(xyz, img_list, lidar2img)
    pct = 100.0 * colored.sum() / max(1, len(colored))
    print(f'  [colorize] {colored.sum():,}/{len(colored):,} points colored '
          f'({pct:.1f}%) from {len(img_list)} camera(s)')
    return rgb


def _passes_filter(motion, motion_filter):
    """Whether a box with the given motion label should be drawn."""
    if motion_filter == 'all':
        return True
    return motion == motion_filter


# Gray used for the LiDAR cloud — matches the BEV scatter in _draw_bev_panel.
_LIDAR_RGB = np.array([176, 176, 176], dtype=np.uint8)
# Extra samples per metre along each box edge so wireframes read as lines.
_PLY_EDGE_PTS_PER_M = 30.0
DEFAULT_ARROW_HORIZON = 1.0


def _rgb_u8(color):
    """Convert a matplotlib 0-1 RGB tuple (or 0-255) to uint8."""
    c = np.asarray(color, dtype=np.float64).ravel()[:3]
    if c.max() <= 1.0:
        c = c * 255.0
    return np.clip(np.round(c), 0, 255).astype(np.uint8)


def lidar_box_corners(bboxes):
    """LiDAR-box corners ``(N, 8, 3)``, matching ``LiDARInstance3DBoxes``.

    Input boxes are ``[x, y, z, x_size, y_size, z_size, yaw, ...]`` with the
    mmdet3d LiDAR origin ``(0.5, 0.5, 0)`` (bottom-center) and yaw about +z.
    """
    bboxes = np.asarray(bboxes, dtype=np.float64)
    if bboxes.ndim == 1:
        bboxes = bboxes[None]
    if bboxes.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)
    dims = bboxes[:, 3:6]
    corners_norm = np.stack(
        np.unravel_index(np.arange(8), [2, 2, 2]), axis=1).astype(np.float64)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.0])
    corners = dims[:, None, :] * corners_norm[None, :, :]
    yaw = bboxes[:, 6]
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z = corners[..., 0], corners[..., 1], corners[..., 2]
    xr = c[:, None] * x - s[:, None] * y
    yr = s[:, None] * x + c[:, None] * y
    corners = np.stack([xr, yr, z], axis=-1)
    corners += bboxes[:, None, :3]
    return corners


def _sample_box_wireframe(corners, color, dashed=False, pts_per_m=_PLY_EDGE_PTS_PER_M):
    """Densely sample the 12 box edges so they show up as lines in a PLY."""
    rgb = _rgb_u8(color)
    pts, cols = [], []
    for a, b in BOX_EDGES:
        p0 = np.asarray(corners[a], dtype=np.float64)
        p1 = np.asarray(corners[b], dtype=np.float64)
        length = float(np.linalg.norm(p1 - p0))
        n = max(2, int(np.ceil(length * pts_per_m)))
        t = np.linspace(0.0, 1.0, n)
        if dashed:
            dist = t * length
            # 20 cm dash / 15 cm gap — same idea as the BEV dashed stroke.
            t = t[(dist % 0.35) < 0.20]
        if t.size == 0:
            continue
        xyz = p0[None, :] + t[:, None] * (p1 - p0)
        pts.append(xyz)
        cols.append(np.repeat(rgb[None, :], len(xyz), axis=0))
    if not pts:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8))
    return np.vstack(pts).astype(np.float32), np.vstack(cols)


def velocity_arrow_endpoints(bbox, motion=None,
                             horizon=DEFAULT_ARROW_HORIZON):
    """Return a box-center velocity arrow in LiDAR XYZ, or ``None``.

    The model's velocity vector is used rather than box yaw. ``motion`` may
    carry persisted ``vx_mps``/``vy_mps`` values; the 9-D box is the fallback.
    """
    bbox = np.asarray(bbox, dtype=np.float64).ravel()
    vx = motion.get('vx_mps') if motion is not None else None
    vy = motion.get('vy_mps') if motion is not None else None
    if vx is None or vy is None:
        if bbox.size < 9:
            return None
        vx, vy = bbox[7], bbox[8]
    if not np.isfinite(vx) or not np.isfinite(vy) or horizon <= 0:
        return None
    if float(np.hypot(vx, vy)) <= 1e-6:
        return None
    # Boxes use bottom-center origin, so put the arrow halfway up the object.
    start = bbox[:3].copy()
    if bbox.size >= 6:
        start[2] += bbox[5] * 0.5
    tip = start + np.array([vx * horizon, vy * horizon, 0.0])
    return start, tip


def _sample_velocity_arrow(start, tip, color,
                           pts_per_m=_PLY_EDGE_PTS_PER_M):
    """Sample a shaft and two arrowhead strokes for a vertex-only PLY."""
    start = np.asarray(start, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64)
    delta = tip - start
    length = float(np.linalg.norm(delta[:2]))
    if length <= 1e-6:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8))
    direction = delta[:2] / length
    normal = np.array([-direction[1], direction[0]])
    head_len = min(1.0, max(0.35, 0.2 * length))
    head_width = 0.5 * head_len
    base_xy = tip[:2] - head_len * direction
    head_a = tip.copy()
    head_b = tip.copy()
    head_a[:2] = base_xy + head_width * normal
    head_b[:2] = base_xy - head_width * normal

    rgb = _rgb_u8(color)
    points, colors = [], []
    for p0, p1 in ((start, tip), (tip, head_a), (tip, head_b)):
        seg_len = float(np.linalg.norm(p1 - p0))
        n = max(2, int(np.ceil(seg_len * pts_per_m)))
        t = np.linspace(0.0, 1.0, n)
        xyz = p0[None, :] + t[:, None] * (p1 - p0)
        points.append(xyz)
        colors.append(np.repeat(rgb[None, :], n, axis=0))
    return np.vstack(points).astype(np.float32), np.vstack(colors)


def _inbox_indices(points, bboxes):
    """Return the box index of each lidar point, or -1 if outside all boxes."""
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    bboxes = np.asarray(bboxes, dtype=np.float64)
    idx = np.full(len(xyz), -1, dtype=np.int32)
    if len(bboxes) == 0 or len(xyz) == 0:
        return idx
    px, py, pz = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for i, b in enumerate(bboxes):
        cx, cy, cz = b[0], b[1], b[2]
        length, width, height = b[3], b[4], b[5]
        yaw = b[6]
        dx, dy = px - cx, py - cy
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        local_x = dx * cos_y + dy * sin_y
        local_y = -dx * sin_y + dy * cos_y
        inside = ((np.abs(local_x) <= length / 2.0)
                  & (np.abs(local_y) <= width / 2.0)
                  & (pz >= cz) & (pz <= cz + height))
        idx[inside] = i
    return idx


def write_ply_xyzrgb(path, xyz, rgb, ascii=True):
    """Write a PLY with ``x y z red green blue``.

    ASCII is the default because many viewers (and the Cursor/VS Code 3D
    preview) silently drop vertex colors on binary PLY. The colors are the
    same either way.
    """
    xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32))
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f'xyz must be (N, 3+), got {xyz.shape}')
    if rgb.shape != (len(xyz), 3):
        raise ValueError(f'rgb must be (N, 3), got {rgb.shape}')
    n = len(xyz)
    fmt = 'ascii 1.0' if ascii else 'binary_little_endian 1.0'
    header = (
        'ply\n'
        f'format {fmt}\n'
        'comment camera-colorized lidar (vertex red/green/blue)\n'
        f'element vertex {n}\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'property uchar red\n'
        'property uchar green\n'
        'property uchar blue\n'
        'end_header\n'
    )
    if ascii:
        data = np.column_stack([xyz[:, :3], rgb.astype(np.uint16)])
        with open(path, 'w', encoding='ascii') as f:
            f.write(header)
            np.savetxt(f, data, fmt='%.6f %.6f %.6f %d %d %d')
        return path
    # Packed 15-byte records: do not use an aligned dtype (itemsize 16).
    rec = np.empty(
        n,
        dtype=np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1')], align=False))
    rec['x'] = xyz[:, 0]
    rec['y'] = xyz[:, 1]
    rec['z'] = xyz[:, 2]
    rec['r'] = rgb[:, 0]
    rec['g'] = rgb[:, 1]
    rec['b'] = rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        rec.tofile(f)
    return path


def load_lidar_xyz(pcd_path):
    """Load nuScenes ``.pcd.bin`` / ``.bin`` lidar as ``(N, 3)`` XYZ."""
    raw = np.fromfile(pcd_path, dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if raw.size % 5 == 0:
        pts = raw.reshape(-1, 5)
    elif raw.size % 4 == 0:
        pts = raw.reshape(-1, 4)
    elif raw.size % 3 == 0:
        pts = raw.reshape(-1, 3)
    else:
        raise ValueError(
            f'Cannot reshape {raw.size} floats from {pcd_path} as xyz[i]')
    return pts[:, :3].astype(np.float32)


def _ground_z_offset(points, bboxes=None):
    """Return ``dz`` so the local road sits at ``z=0`` (sensor-plane origin).

    nuScenes LiDAR ``z=0`` is the sensor, ~1.6–2.0 m above the pavement.
    Box bottoms therefore sit at ``z ≈ -2``. Adding this offset is viz-only:
    points and boxes move together so they stay aligned.
    """
    if bboxes is not None:
        bboxes = np.asarray(bboxes, dtype=np.float64)
        if len(bboxes):
            return float(-np.median(bboxes[:, 2]))
    z = np.asarray(points, dtype=np.float64)[:, 2]
    road = z[(z > -3.5) & (z < -0.4)]
    if road.size:
        return float(-np.median(road))
    if z.size:
        return float(-np.percentile(z, 20))
    return 0.0


def _draw_ply_panel(points, corners, labels, motions, bboxes, out_path,
                    motion_filter, color_by, point_rgb=None,
                    arrow_horizon=DEFAULT_ARROW_HORIZON, draw_arrows=True):
    """Write one colored PLY: base lidar cloud + motion-colored 3D wireframes.

    ``point_rgb`` (N, 3 uint8), when provided, is the per-point camera color
    used as the base cloud (from :func:`colorize_lidar`). Otherwise the cloud
    falls back to flat gray. Camera-colored clouds get only a very light in-box
    tint so the real color stays visible; the wireframe carries the motion cue.
    """
    xyz = np.asarray(points, dtype=np.float32)[:, :3].copy()
    corners = np.asarray(corners, dtype=np.float64).copy()
    if bboxes is not None:
        bboxes = np.asarray(bboxes, dtype=np.float64).copy()
    dz = _ground_z_offset(xyz, bboxes)
    xyz[:, 2] += dz
    if corners.size:
        corners[..., 2] += dz
    if bboxes is not None and len(bboxes):
        bboxes[:, 2] += dz

    if point_rgb is not None:
        rgb = np.asarray(point_rgb, dtype=np.uint8).reshape(-1, 3)[:len(xyz)].copy()
        if len(rgb) < len(xyz):  # pad defensively
            pad = np.repeat(_LIDAR_RGB[None, :], len(xyz) - len(rgb), axis=0)
            rgb = np.vstack([rgb, pad])
        # Drop lidar returns no camera painted (same as --no-uncolored).
        # Leaving them as #b0b0b0 makes the whole cloud look uncolored.
        painted = ~np.all(rgb == _LIDAR_RGB, axis=1)
        xyz = xyz[painted]
        rgb = rgb[painted]
        # Do not wash camera RGB toward the box color.
        tint_alpha = 0.0
    else:
        rgb = np.repeat(_LIDAR_RGB[None, :], len(xyz), axis=0)
        tint_alpha = 0.65

    keep = [i for i, m in enumerate(motions)
            if _passes_filter(m['motion'], motion_filter)]
    if keep:
        if tint_alpha > 0:
            kept_boxes = np.asarray(bboxes, dtype=np.float64)[keep]
            inbox = _inbox_indices(xyz, kept_boxes)
            for local_i, src_i in enumerate(keep):
                class_color = CLASS_COLORS[int(labels[src_i]) % len(CLASS_COLORS)]
                color = motion_color(motions[src_i]['motion'], class_color, color_by)
                tint = _rgb_u8(color)
                sel = inbox == local_i
                if sel.any():
                    rgb[sel] = np.clip(
                        ((1.0 - tint_alpha) * rgb[sel].astype(np.float32)
                         + tint_alpha * tint.astype(np.float32)),
                        0, 255).astype(np.uint8)

        extra_xyz, extra_rgb = [], []
        for src_i in keep:
            class_color = CLASS_COLORS[int(labels[src_i]) % len(CLASS_COLORS)]
            color = motion_color(motions[src_i]['motion'], class_color, color_by)
            dashed = motion_linestyle(motions[src_i]['motion']) == '--'
            w_xyz, w_rgb = _sample_box_wireframe(
                corners[src_i], color, dashed=dashed)
            if len(w_xyz):
                extra_xyz.append(w_xyz)
                extra_rgb.append(w_rgb)
            if draw_arrows and motions[src_i]['motion'] == 'dynamic':
                endpoints = velocity_arrow_endpoints(
                    bboxes[src_i], motions[src_i], arrow_horizon)
                if endpoints is not None:
                    start, tip = endpoints
                    a_xyz, a_rgb = _sample_velocity_arrow(
                        start, tip, color)
                    extra_xyz.append(a_xyz)
                    extra_rgb.append(a_rgb)
        if extra_xyz:
            xyz = np.vstack([xyz, np.vstack(extra_xyz)])
            rgb = np.vstack([rgb, np.vstack(extra_rgb)])

    return write_ply_xyzrgb(out_path, xyz, rgb)


def save_ply_vis(points, corners, labels, motions, bboxes, out_dir,
                 motion_filter='all', color_by='motion', point_rgb=None,
                 arrow_horizon=DEFAULT_ARROW_HORIZON, draw_arrows=True):
    """Write BEV-matching PLY scenes (combined / dynamic-only / static-only)."""
    os.makedirs(out_dir, exist_ok=True)
    panels = [('bev.ply', motion_filter)]
    if motion_filter == 'all':
        panels += [
            ('bev_dynamic.ply', 'dynamic'),
            ('bev_static.ply', 'static'),
        ]
    saved = []
    for fname, filt in panels:
        path = os.path.join(out_dir, fname)
        _draw_ply_panel(points, corners, labels, motions, bboxes, path, filt,
                        color_by, point_rgb=point_rgb,
                        arrow_horizon=arrow_horizon,
                        draw_arrows=draw_arrows)
        saved.append(path)
    return saved


def _lidar2img_from_ann(ann_path):
    """Build ``(img_paths, lidar2img_list)`` from a BEVFusion ann pkl.

    Mirrors ``BEVLoadMultiViewImageFromFiles`` in
    ``projects/BEVFusion/bevfusion/loading.py``:
    ``lidar2img = cam2img(4x4) @ lidar2cam(4x4)`` per camera.
    """
    import pickle
    with open(ann_path, 'rb') as f:
        ann = pickle.load(f)
    info = ann['data_list'][0]
    images = info.get('images', {})
    img_paths, mats = [], []
    for _, cam_item in images.items():
        img_paths.append(cam_item.get('img_path'))
        lidar2cam = np.asarray(cam_item['lidar2cam'], dtype=np.float64)
        cam2img = np.eye(4, dtype=np.float64)
        k = np.asarray(cam_item['cam2img'], dtype=np.float64)
        cam2img[:3, :3] = k[:3, :3]
        mats.append(cam2img @ lidar2cam)
    return img_paths, mats


def export_ply_from_files(pcd_path, detections_path, out_dir,
                          motion_filter='all', color_by='motion',
                          ann_path=None, img_dir=None, write_bev_png=True,
                          arrow_horizon=DEFAULT_ARROW_HORIZON,
                          draw_arrows=True):
    """Rebuild PLY scenes from a lidar file + ``detections.json`` (no model).

    If ``ann_path`` (a BEVFusion ann pkl) is provided, the lidar cloud is
    camera-colorized via ``lidar2img`` built from the pkl. Images are resolved
    from the pkl's ``img_path`` entries, optionally rebased under ``img_dir``.
    When ``write_bev_png`` is set, matching BEV PNG panels are also written so
    the 2D panels stay consistent with the 3D PLYs.
    """
    from colorize_lidar import load_images_rgb

    with open(detections_path, 'r') as f:
        payload = json.load(f)
    boxes = payload.get('boxes', [])
    labels = [int(b['label']) for b in boxes]
    bboxes = [b['bbox_3d'] for b in boxes]
    scores = [float(b.get('score', 0.0)) for b in boxes]
    motions = boxes
    corners = lidar_box_corners(bboxes) if bboxes else np.zeros((0, 8, 3))
    points = load_lidar_xyz(pcd_path)

    point_rgb = None
    if ann_path is not None:
        try:
            img_paths, lidar2img = _lidar2img_from_ann(ann_path)
            resolved = []
            for p in img_paths:
                if p is None:
                    resolved.append(None)
                    continue
                cand = p
                if img_dir is not None and not os.path.isabs(p):
                    cand = os.path.join(img_dir, os.path.basename(p))
                if img_dir is not None and not os.path.exists(cand):
                    cand = os.path.join(img_dir, os.path.basename(p))
                resolved.append(cand if os.path.exists(cand) else p)
            images = load_images_rgb(resolved)
            rgb, colored = colorize_lidar(points[:, :3], images, lidar2img)
            pct = 100.0 * colored.sum() / max(1, len(colored))
            print(f'  [colorize] {colored.sum():,}/{len(colored):,} points '
                  f'colored ({pct:.1f}%) from {len(images)} camera(s)')
            point_rgb = rgb
        except Exception as exc:  # noqa: BLE001
            print(f'  [colorize] disabled ({type(exc).__name__}: {exc})')

    saved = save_ply_vis(
        points, corners, labels, motions, bboxes, out_dir,
        motion_filter=motion_filter, color_by=color_by, point_rgb=point_rgb,
        arrow_horizon=arrow_horizon, draw_arrows=draw_arrows)

    if write_bev_png:
        import matplotlib
        matplotlib.use('Agg')
        os.makedirs(out_dir, exist_ok=True)
        class_names = list(CLASS_NAMES)
        panels = [('bev.png', 'BEV detections', motion_filter)]
        if motion_filter == 'all':
            panels += [
                ('bev_dynamic.png', 'BEV dynamic', 'dynamic'),
                ('bev_static.png', 'BEV static', 'static'),
            ]
        for fname, title, filt in panels:
            path = os.path.join(out_dir, fname)
            _draw_bev_panel(points, corners, bboxes, labels, scores, motions,
                            class_names, path, title, filt, color_by,
                            point_rgb=point_rgb,
                            arrow_horizon=arrow_horizon,
                            draw_arrows=draw_arrows)
            saved.append(path)

    return saved


def _draw_bev_panel(points, corners, bboxes, labels, scores, motions,
                    class_names, out_path, title, motion_filter, color_by,
                    point_rgb=None, arrow_horizon=DEFAULT_ARROW_HORIZON,
                    draw_arrows=True):
    """Render one BEV panel, optionally filtered by motion state.

    ``point_rgb`` (N, 3 uint8), when given, colors the scatter with the
    per-point camera color so the PNG matches the colorized 3D PLY.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(10, 10))
    if point_rgb is not None:
        c = np.asarray(point_rgb, dtype=np.float32).reshape(-1, 3)[:len(points)]
        c = np.clip(c / 255.0, 0.0, 1.0)
        ax.scatter(points[:, 0], points[:, 1], s=0.15, c=c, linewidths=0)
    else:
        ax.scatter(points[:, 0], points[:, 1], s=0.15, c='#b0b0b0',
                   linewidths=0)
    n_drawn = 0
    for i, box_corners in enumerate(corners):
        m = motions[i]
        if not _passes_filter(m['motion'], motion_filter):
            continue
        n_drawn += 1
        class_color = CLASS_COLORS[labels[i] % len(CLASS_COLORS)]
        color = motion_color(m['motion'], class_color, color_by)
        ls = motion_linestyle(m['motion'])
        # Ground rectangle (all z=z0): corners 0,3,7,4. Using 0,1,2,3 picks a
        # vertical face, which collapses the BEV footprint to a line.
        bev = box_corners[[0, 3, 7, 4], :2]
        ax.add_patch(
            Polygon(bev, closed=True, fill=False, edgecolor=color,
                    linewidth=1.4, linestyle=ls))
        if draw_arrows and m['motion'] == 'dynamic':
            endpoints = velocity_arrow_endpoints(
                bboxes[i], m, arrow_horizon)
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
        ax.text(
            bev[:, 0].mean(),
            bev[:, 1].mean(),
            f"{m['class_name']} {scores[i]:.2f} {tag}{speed_txt}",
            color=color,
            fontsize=6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{title} ({n_drawn} boxes)')
    ax.grid(True, alpha=0.2)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_offline_vis(result, data, imgs, out_dir, score_thr, class_names,
                     speed_thresh=DEFAULT_SPEED_THRESH, motion_filter='all',
                     color_by='motion', radar_fusion='radar_primary',
                     motions=None, colorize=True,
                     arrow_horizon=DEFAULT_ARROW_HORIZON,
                     draw_arrows=True):
    """Save BEV and camera overlays without Open3D / a display.

    Boxes are classified static/dynamic via ``motion_utils`` and colored by
    motion state (dynamic=red solid, static=gray dashed). Three BEV panels are
    written: combined, dynamic-only, static-only.

    If ``motions`` (a list of per-box classification dicts aligned with the
    score-thresholded predictions) is provided, it is used directly so the
    visualization matches ``detections.json`` (including radar fusion).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    pred = result.pred_instances_3d
    if len(pred) == 0:
        print('No detections to visualize.')
        return []

    pred = pred[pred.scores_3d >= score_thr].to('cpu')
    scores = _to_numpy(pred.scores_3d)
    labels = _to_numpy(pred.labels_3d).astype(int)
    corners = _to_numpy(pred.bboxes_3d.corners)
    # Full box tensor carries velocity in columns 7,8 for 9-D boxes.
    box_tensor = _to_numpy(pred.bboxes_3d.tensor)
    points = _to_numpy(data['inputs']['points'])
    meta = result.metainfo

    if motions is None:
        motions = classify_detections(
            labels, box_tensor, class_names, speed_thresh=speed_thresh,
            radar_fusion=radar_fusion)
        n_dyn = sum(1 for m in motions if m['motion'] == 'dynamic')
        n_stat = sum(1 for m in motions if m['motion'] == 'static')
        print(f'  motion: {n_dyn} dynamic, {n_stat} static '
              f'(speed_thresh={speed_thresh} m/s)')

    saved = []

    # ---- Camera-colorize the lidar cloud (XYZ stays in the LiDAR frame) ----
    point_rgb = None
    if colorize:
        point_rgb = _colorize_points(points, imgs, meta)

    # ---- BEV panels: combined, dynamic-only, static-only ----
    panels = [('bev.png', 'BEV detections', motion_filter)]
    if motion_filter == 'all':
        panels += [
            ('bev_dynamic.png', 'BEV dynamic', 'dynamic'),
            ('bev_static.png', 'BEV static', 'static'),
        ]
    for fname, title, filt in panels:
        path = os.path.join(out_dir, fname)
        _draw_bev_panel(points, corners, box_tensor, labels, scores, motions,
                        class_names,
                        path, title, filt, color_by, point_rgb=point_rgb,
                        arrow_horizon=arrow_horizon,
                        draw_arrows=draw_arrows)
        saved.append(path)

    # 3D PLY scenes matching the BEV panels (colorized lidar + colored boxes).
    saved.extend(
        save_ply_vis(
            points, corners, labels, motions, box_tensor, out_dir,
            motion_filter=motion_filter, color_by=color_by,
            point_rgb=point_rgb, arrow_horizon=arrow_horizon,
            draw_arrows=draw_arrows))

    if not isinstance(imgs, list):
        imgs = [imgs]
    cam_names = list(meta.get('images', {}).keys()) if isinstance(
        meta.get('images'), dict) else [f'cam_{i}' for i in range(len(imgs))]
    if len(cam_names) < len(imgs):
        cam_names = [f'cam_{i}' for i in range(len(imgs))]

    cam_paths = []
    for cam_idx, img in enumerate(imgs):
        img_np = _to_numpy(img)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        h, w = img_np.shape[:2]
        lidar2img = _lidar2img_matrix(meta, cam_idx)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(img_np)
        if lidar2img is not None:
            for i, box_corners in enumerate(corners):
                m = motions[i]
                if not _passes_filter(m['motion'], motion_filter):
                    continue
                pts = np.concatenate(
                    [box_corners, np.ones((8, 1))], axis=1) @ lidar2img.T
                depth = pts[:, 2]
                # Skip a box only if it is entirely behind the camera.
                if (depth > 0).sum() == 0:
                    continue
                uv = pts[:, :2] / np.clip(pts[:, 2:3], 1e-5, None)
                # Cull boxes fully outside the image frame.
                infront = depth > 0
                if ((uv[infront, 0] < 0) | (uv[infront, 0] >= w)
                        | (uv[infront, 1] < 0) | (uv[infront, 1] >= h)).all():
                    continue
                class_color = CLASS_COLORS[labels[i] % len(CLASS_COLORS)]
                color = motion_color(m['motion'], class_color, color_by)
                for a, b in BOX_EDGES:
                    # Per-edge near-plane clip: drawing an edge that crosses
                    # depth<=0 produces huge warped lines, so skip it.
                    if depth[a] <= 0 or depth[b] <= 0:
                        continue
                    ax.plot(
                        [uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]],
                        color=color,
                        linewidth=1.2)
                if draw_arrows and m['motion'] == 'dynamic':
                    endpoints = velocity_arrow_endpoints(
                        box_tensor[i], m, arrow_horizon)
                    if endpoints is not None:
                        arrow_xyz = np.stack(endpoints)
                        arrow_h = np.concatenate(
                            [arrow_xyz, np.ones((2, 1))], axis=1)
                        projected = arrow_h @ lidar2img.T
                        if np.all(projected[:, 2] > 0):
                            arrow_uv = (
                                projected[:, :2] / projected[:, 2:3])
                            ax.annotate(
                                '', xy=arrow_uv[1], xytext=arrow_uv[0],
                                arrowprops=dict(
                                    arrowstyle='-|>', color=color,
                                    linewidth=1.8, mutation_scale=10))
        ax.set_axis_off()
        ax.set_title(cam_names[cam_idx])
        cam_path = os.path.join(out_dir, f'{cam_names[cam_idx]}.png')
        fig.savefig(cam_path, dpi=120, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        cam_paths.append(cam_path)
        saved.append(cam_path)

    if cam_paths:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, path, name in zip(axes.ravel(), cam_paths, cam_names):
            ax.imshow(plt.imread(path))
            ax.set_title(name, fontsize=8)
            ax.set_axis_off()
        mosaic = os.path.join(out_dir, 'cameras.png')
        fig.tight_layout()
        fig.savefig(mosaic, dpi=120, bbox_inches='tight')
        plt.close(fig)
        saved.append(mosaic)

    return saved


def _radar_evidence_for_ann(ann_path, kept_bboxes, dataroot, version, nsweeps,
                            margin):
    """Load radar for the ann file's sample and associate to kept boxes.

    Returns a list of ``(radar_speed, radar_pts)`` aligned with ``kept_bboxes``,
    or ``None`` if the sample token / radar data is unavailable.
    """
    import pickle

    from motion_utils import associate_radar_to_boxes
    try:
        with open(ann_path, 'rb') as f:
            ann = pickle.load(f)
        token = ann['data_list'][0].get('token')
        if not token:
            print('  [radar] no token in ann file; skipping radar fusion')
            return None
        from radar_utils import load_radar_for_info
        xyz, _vcomp, speed = load_radar_for_info(
            dataroot, token, version=version, nsweeps=nsweeps)
        print(f'  [radar] loaded {len(xyz)} radar points '
              f'(dataroot={dataroot}, version={version})')
        if len(xyz) == 0:
            return None
        return associate_radar_to_boxes(
            np.asarray(kept_bboxes, dtype=np.float64), xyz, speed,
            bev_margin=margin)
    except Exception as exc:  # noqa: BLE001
        print(f'  [radar] disabled ({type(exc).__name__}: {exc})')
        return None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('img', help='Image file or multi-view image directory')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM_FRONT',
        help='camera type to infer; use "all" for BEVFusion multi-view')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--speed-thresh',
        type=float,
        default=DEFAULT_SPEED_THRESH,
        help='planar speed (m/s) above which a box is dynamic')
    parser.add_argument(
        '--motion-filter',
        choices=['all', 'dynamic', 'static'],
        default='all',
        help='which motion class to render')
    parser.add_argument(
        '--color-by',
        choices=['motion', 'class'],
        default='motion',
        help=('moving boxes red and others by grouped class color; '
              'use class to disable the dynamic-red override'))
    parser.add_argument(
        '--arrow-horizon',
        type=float,
        default=DEFAULT_ARROW_HORIZON,
        help='seconds of constant velocity represented by each arrow')
    parser.add_argument(
        '--no-arrows',
        action='store_true',
        help='disable velocity arrows on dynamic boxes')
    # Phase 2: radar velocity fusion (nuScenes devkit).
    parser.add_argument(
        '--use-radar',
        action='store_true',
        help='fuse nuScenes radar compensated velocity into motion labels')
    parser.add_argument(
        '--nusc-dataroot',
        type=str,
        default='data/nuscenes',
        help='nuScenes dataset root (for radar loading)')
    parser.add_argument(
        '--nusc-version',
        type=str,
        default='v1.0-mini',
        help='nuScenes version for radar loading')
    parser.add_argument(
        '--radar-fusion',
        choices=['radar_primary', 'or'],
        default='radar_primary',
        help='how to combine radar and box speed')
    parser.add_argument(
        '--min-radar-pts',
        type=int,
        default=3,
        help='min in-box radar points to trust radar speed')
    parser.add_argument(
        '--radar-margin',
        type=float,
        default=0.5,
        help='BEV margin (m) around each box for radar association')
    parser.add_argument(
        '--radar-nsweeps',
        type=int,
        default=1,
        help='number of radar sweeps to accumulate per channel')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args


def main(args):
    model = init_model(args.config, args.checkpoint, device=args.device)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    result, data = inference_multi_modality_detector(model, args.pcd, args.img,
                                                     args.ann, args.cam_type)
    points = data['inputs']['points']
    if isinstance(getattr(result, 'img_path', None), list):
        img = []
        for img_path in result.img_path:
            single_img = mmcv.imread(img_path)
            single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')
            img.append(single_img)
    else:
        img_path = getattr(result, 'img_path', args.img)
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
    data_input = dict(points=points, img=img)

    pred = result.pred_instances_3d
    scores = pred.scores_3d.detach().cpu().numpy() if len(pred) else []
    labels = pred.labels_3d.detach().cpu().numpy() if len(pred) else []
    bboxes = pred.bboxes_3d.tensor.detach().cpu().numpy() if len(pred) else []
    keep = [i for i, s in enumerate(scores) if s >= args.score_thr]
    print(f'Detections: {len(pred)} boxes, {len(keep)} above score {args.score_thr}')
    if len(keep) > 0:
        print(f'  scores: min={scores[keep].min():.3f} max={scores[keep].max():.3f}')
        print(f'  labels: {labels[keep][:20].tolist()}')

    class_names = list(
        (model.dataset_meta or {}).get('classes', CLASS_NAMES))

    kept_labels = [int(labels[i]) for i in keep]
    kept_bboxes = [bboxes[i] for i in keep]

    # Phase 2: optionally load radar and associate to boxes for fusion.
    radar_evidence = None
    if args.use_radar:
        radar_evidence = _radar_evidence_for_ann(
            args.ann, kept_bboxes, args.nusc_dataroot, args.nusc_version,
            args.radar_nsweeps, args.radar_margin)

    motions = classify_detections(
        kept_labels, kept_bboxes, class_names, speed_thresh=args.speed_thresh,
        radar_evidence=radar_evidence, radar_fusion=args.radar_fusion,
        min_radar_pts=args.min_radar_pts)
    n_dyn = sum(1 for m in motions if m['motion'] == 'dynamic')
    n_stat = sum(1 for m in motions if m['motion'] == 'static')
    n_radar = sum(1 for m in motions if m['source'] == 'radar')
    print(f'  motion: {n_dyn} dynamic, {n_stat} static '
          f'(speed_thresh={args.speed_thresh} m/s)'
          + (f'; {n_radar} via radar' if args.use_radar else ''))

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, 'detections.json')
    with open(out_json, 'w') as f:
        json.dump(
            {
                'score_thr': args.score_thr,
                'speed_thresh': args.speed_thresh,
                'num_raw': int(len(pred)),
                'num_kept': len(keep),
                'num_dynamic': n_dyn,
                'num_static': n_stat,
                'boxes': [{
                    'score': float(scores[i]),
                    'label': int(labels[i]),
                    'class_name': m['class_name'],
                    'motion': m['motion'],
                    'motion_source': m['source'],
                    'speed_mps': m['speed_mps'],
                    'box_speed_mps': m['box_speed_mps'],
                    'vx_mps': m['vx_mps'],
                    'vy_mps': m['vy_mps'],
                    'radar_speed_mps': m['radar_speed_mps'],
                    'radar_pts': m['radar_pts'],
                    'bbox_3d': bboxes[i].tolist(),
                } for i, m in zip(keep, motions)],
            },
            f,
            indent=2)
    print(f'Wrote {out_json}')

    vis_paths = save_offline_vis(
        result, data, img, args.out_dir, args.score_thr, class_names,
        speed_thresh=args.speed_thresh, motion_filter=args.motion_filter,
        color_by=args.color_by, motions=motions,
        arrow_horizon=args.arrow_horizon, draw_arrows=not args.no_arrows)
    for path in vis_paths:
        print(f'Wrote {path}')

    try:
        visualizer.add_datasample(
            'result',
            data_input,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=-1,
            out_file=args.out_dir,
            pred_score_thr=args.score_thr,
            vis_task='multi-modality_det')
    except Exception as exc:
        print(f'Skipped Open3D visualization ({exc}). Detections were saved.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
