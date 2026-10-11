# Copyright (c) OpenMMLab. All rights reserved.
"""Camera-colorize a LiDAR point cloud using the demo's ``lidar2img`` matrices.

This ports the projection + multi-camera best-view fusion from the first-party
``nuscenes-sensor-fusion`` repo
(``scripts/nuscenes_lidar_viz/colorize_pointcloud.py`` and
``src/colorization.py``) into a single reusable function that works with the
data the BEVFusion demo already has on disk:

* ``points_lidar``   (N, 3+) LiDAR points, XYZ in the LIDAR_TOP frame
* ``images``         list of ``(H, W, 3)`` uint8 RGB arrays (one per camera)
* ``lidar2img``      one ``4x4`` projection matrix per camera
                     (``cam2img @ lidar2cam``, same as ``meta['lidar2img']``)

Crucially, the returned RGB is aligned with the *input* ``points_lidar`` and the
XYZ is never moved out of the LiDAR frame. That keeps the colorized cloud in the
same coordinate system as the BEVFusion 3D boxes, so wireframes still line up
(the frame mismatch that the upstream smoke test had by writing camera-frame
XYZ).

Algorithm (per camera), following the upstream repo:

1. Project points with ``lidar2img`` -> ``(u, v, depth)``.
2. Keep points with ``depth > min_depth`` that land inside the image (with an
   optional border margin).
3. Image-space z-buffer: for each pixel keep only the nearest point, so
   background returns do not steal a foreground pixel's color.
4. Multi-camera fusion: when several cameras see the same point, keep the view
   with the best quality score (most centred + closest), matching the
   "best-view selection" in the source repo.

Points seen by no camera keep ``default_rgb`` (gray, matching the BEV scatter).
"""
from typing import List, Optional, Sequence

import numpy as np

# Default per-point color for LiDAR returns that no camera sees. Matches the
# ``#b0b0b0`` gray used by the BEV scatter in ``multi_modality_demo.py``.
DEFAULT_LIDAR_RGB = (176, 176, 176)


def _as_4x4(mat) -> Optional[np.ndarray]:
    """Coerce a projection matrix to 4x4, or return ``None`` if unavailable."""
    if mat is None:
        return None
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4)
        out[:3, :] = mat
        return out
    if mat.shape == (3, 3):
        out = np.eye(4)
        out[:3, :3] = mat
        return out
    return None


def _project(points_xyz: np.ndarray, mat4: np.ndarray):
    """Project ``(N, 3)`` LiDAR points through a ``4x4`` lidar2img matrix.

    Returns ``(u, v, depth)`` where ``u, v`` are pixel coordinates and ``depth``
    is the camera-frame z (positive in front of the camera).
    """
    n = len(points_xyz)
    pts_h = np.concatenate(
        [points_xyz[:, :3], np.ones((n, 1), dtype=np.float64)], axis=1)
    proj = pts_h @ mat4.T  # (N, 4)
    depth = proj[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    u = proj[:, 0] / safe
    v = proj[:, 1] / safe
    return u, v, depth


def _zbuffer_keep(u: np.ndarray, v: np.ndarray, depth: np.ndarray,
                  width: int, height: int,
                  depth_tol_m: float = 1.0,
                  depth_tol_ratio: float = 1.3) -> np.ndarray:
    """Depth-tolerant z-buffer: keep the front surface at each pixel.

    A strict "nearest point per pixel" z-buffer caps the colored set at one
    point per pixel, which tanks coverage for dense multi-sweep clouds (only
    ~1/10 of points survive). Instead, for each integer pixel we find the
    nearest depth and keep every point whose depth is within a tolerance of it::

        depth <= min_depth_at_pixel * depth_tol_ratio   OR
        depth <= min_depth_at_pixel + depth_tol_m

    This colors the whole visible front surface densely while still rejecting
    far background returns that fall behind a near foreground edge (the bleed /
    ghosting the source repo's shadow filter targets). Set both tolerances to 0
    to recover the strict nearest-only behaviour.
    """
    ui = np.clip(np.round(u).astype(np.int64), 0, width - 1)
    vi = np.clip(np.round(v).astype(np.int64), 0, height - 1)
    keys = vi * width + ui

    # Per-pixel nearest depth via sort + reduceat over unique keys.
    order = np.argsort(keys, kind='stable')
    keys_s = keys[order]
    depth_s = depth[order]
    uniq_keys, group_start = np.unique(keys_s, return_index=True)
    # Minimum depth within each pixel group.
    min_per_group = np.minimum.reduceat(depth_s, group_start)
    # Map each point back to its pixel's min depth.
    group_of_point = np.searchsorted(uniq_keys, keys)
    min_depth_at_pixel = min_per_group[group_of_point]

    thresh = np.maximum(min_depth_at_pixel * depth_tol_ratio,
                        min_depth_at_pixel + depth_tol_m)
    return depth <= thresh


def _quality_score(u: np.ndarray, v: np.ndarray, depth: np.ndarray,
                   width: int, height: int) -> np.ndarray:
    """Best-view score: more centred + closer is better (0..1-ish).

    Mirrors ``_quality_score`` in ``src/colorization.py`` (center + depth
    terms); the border term is folded into the center distance here.
    """
    cx, cy = width * 0.5, height * 0.5
    dx = (u - cx) / (width * 0.5)
    dy = (v - cy) / (height * 0.5)
    center_score = 1.0 - np.clip(np.sqrt(dx * dx + dy * dy), 0.0, 1.0)
    depth_score = 1.0 / (1.0 + np.maximum(depth, 0.0))
    return 0.8 * center_score + 0.2 * depth_score


def boost_saturation(rgb: np.ndarray, factor: float = 2.2) -> np.ndarray:
    """Push camera RGB away from gray so vertex colors read in 3D viewers.

    nuScenes dusk/asphalt pixels are often R≈G≈B, so a faithful sample looks
    like an uncolored cloud. Expand chroma around each pixel's mean without
    changing brightness much.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    x = rgb.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    out = mean + (x - mean) * factor
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def colorize_lidar(points_lidar: np.ndarray,
                   images: Sequence[np.ndarray],
                   lidar2img: Sequence[np.ndarray],
                   min_depth: float = 1.0,
                   border_margin_px: int = 0,
                   default_rgb: Sequence[int] = DEFAULT_LIDAR_RGB):
    """Return per-point RGB by sampling camera images; XYZ stays LiDAR-frame.

    Args:
        points_lidar: ``(N, 3+)`` LiDAR points (only XYZ is used).
        images: list of ``(H, W, 3)`` uint8 RGB images, one per camera.
        lidar2img: list/array of per-camera ``4x4`` (or ``3x4``/``3x3``)
            projection matrices, index-aligned with ``images``.
        min_depth: minimum camera-frame depth (m) to accept a projection.
        border_margin_px: reject pixels within this many px of the image edge.
        default_rgb: color for points seen by no camera.

    Returns:
        ``(rgb, colored_mask)`` where ``rgb`` is ``(N, 3)`` uint8 aligned with
        ``points_lidar`` and ``colored_mask`` is a boolean ``(N,)`` array marking
        points that received a camera color.
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    n = len(points_lidar)
    rgb = np.tile(np.asarray(default_rgb, dtype=np.uint8), (n, 1))
    best_score = np.full(n, -np.inf, dtype=np.float64)
    colored_mask = np.zeros(n, dtype=bool)
    if n == 0:
        return rgb, colored_mask

    for cam_idx, img in enumerate(images):
        if cam_idx >= len(lidar2img):
            break
        mat4 = _as_4x4(lidar2img[cam_idx])
        if mat4 is None or img is None:
            continue
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[2] < 3:
            continue
        if img.dtype != np.uint8 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        h, w = img.shape[:2]

        u, v, depth = _project(points_lidar, mat4)
        in_front = depth > min_depth
        in_bounds = ((u >= border_margin_px) & (u < w - border_margin_px)
                     & (v >= border_margin_px) & (v < h - border_margin_px))
        valid = in_front & in_bounds
        if not valid.any():
            continue

        valid_idx = np.flatnonzero(valid)
        keep_local = _zbuffer_keep(
            u[valid_idx], v[valid_idx], depth[valid_idx], w, h)
        kept_idx = valid_idx[keep_local]
        if kept_idx.size == 0:
            continue

        scores = _quality_score(u[kept_idx], v[kept_idx], depth[kept_idx], w, h)
        better = scores > best_score[kept_idx]
        upd = kept_idx[better]
        if upd.size == 0:
            continue

        ui = np.clip(np.round(u[upd]).astype(np.int32), 0, w - 1)
        vi = np.clip(np.round(v[upd]).astype(np.int32), 0, h - 1)
        rgb[upd] = img[vi, ui, :3].astype(np.uint8)
        best_score[upd] = scores[better]
        colored_mask[upd] = True

    if colored_mask.any():
        rgb[colored_mask] = boost_saturation(rgb[colored_mask])
    return rgb, colored_mask


def load_images_rgb(paths: List[str]):
    """Load a list of image paths as ``(H, W, 3)`` uint8 RGB arrays.

    Uses ``mmcv`` if available (matches the demo), else falls back to PIL.
    """
    imgs = []
    try:
        import mmcv
        for p in paths:
            im = mmcv.imread(p)
            imgs.append(mmcv.imconvert(im, 'bgr', 'rgb'))
        return imgs
    except Exception:  # noqa: BLE001
        from PIL import Image
        for p in paths:
            imgs.append(np.asarray(Image.open(p).convert('RGB')))
        return imgs
