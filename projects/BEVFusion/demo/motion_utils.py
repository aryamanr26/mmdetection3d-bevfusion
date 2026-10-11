# Copyright (c) OpenMMLab. All rights reserved.
"""Static vs dynamic motion classification for BEVFusion detections.

Phase 1 (implemented): single-frame classification from the model's predicted
per-box velocity ``(vx, vy)`` plus a class prior. This matches the nuScenes
detection eval convention (``mmdet3d/evaluation/metrics/nuscenes_metric.py``),
which infers ``vehicle.moving`` / ``vehicle.parked`` from a 0.2 m/s speed
threshold.

Phase 2 (stubbed): fuse per-box radar compensated velocity. The public API is
built around :class:`MotionEvidence` so the demo scripts do not need to change
when radar association lands -- only ``radar_speed`` / ``radar_pts`` get
populated. See ``associate_radar_to_boxes`` below.
"""
from dataclasses import dataclass

import numpy as np

# nuScenes detection classes that are static infrastructure by definition.
# nuScenes assigns them a "void" attribute and ignores velocity/attribute
# errors for them during eval, so we never treat them as dynamic.
STATIC_CLASSES = {'barrier', 'traffic_cone'}

# Default planar-speed threshold (m/s). Matches nuScenes attribute inference.
DEFAULT_SPEED_THRESH = 0.2

# Minimum in-box radar points before trusting radar speed (Phase 2).
MIN_RADAR_PTS = 3

# Optional per-class speed-threshold overrides for viz tuning. Empty by
# default so behavior matches the nuScenes convention out of the box.
PER_CLASS_SPEED_THRESH = {
    # 'pedestrian': 0.1,
    # 'bicycle': 0.1,
    # 'motorcycle': 0.1,
    # 'trailer': 0.3,
    # 'construction_vehicle': 0.3,
}

# Motion -> RGB color for visualization (matplotlib 0-1 floats).
MOTION_COLORS = {
    'dynamic': (0.85, 0.10, 0.10),  # red
    'static': (0.10, 0.45, 0.95),   # blue fallback
    'unknown': (0.20, 0.40, 0.90),  # blue
}


@dataclass
class MotionEvidence:
    """Motion evidence collected for a single detection box.

    Attributes:
        class_name: nuScenes class name (e.g. ``'car'``).
        box_speed: Planar speed (m/s) from the predicted box velocity, or
            ``None`` if the box has no velocity channels.
        radar_speed: Planar speed (m/s) aggregated from associated radar
            points (Phase 2), or ``None`` if unavailable.
        radar_pts: Number of radar points associated with the box (Phase 2).
    """

    class_name: str
    box_speed: float | None = None
    radar_speed: float | None = None
    radar_pts: int = 0


def box_speed_from_bbox(bbox_3d) -> float | None:
    """Planar speed from a 9-D box ``[x,y,z,l,w,h,yaw,vx,vy]``.

    Returns ``None`` for 7-D boxes that carry no velocity.
    """
    bbox_3d = np.asarray(bbox_3d, dtype=np.float64).ravel()
    if bbox_3d.shape[0] >= 9:
        vx, vy = bbox_3d[7], bbox_3d[8]
        if np.isnan(vx) or np.isnan(vy):
            return None
        return float(np.hypot(vx, vy))
    return None


def box_velocity_xy(bbox_3d) -> tuple[float, float] | None:
    """Planar velocity ``(vx, vy)`` from a 9-D prediction box."""
    bbox_3d = np.asarray(bbox_3d, dtype=np.float64).ravel()
    if bbox_3d.shape[0] < 9:
        return None
    vx, vy = float(bbox_3d[7]), float(bbox_3d[8])
    if not np.isfinite(vx) or not np.isfinite(vy):
        return None
    return vx, vy


def speed_thresh_for(class_name: str, speed_thresh: float) -> float:
    """Resolve the effective threshold for a class (per-class override)."""
    return PER_CLASS_SPEED_THRESH.get(class_name, speed_thresh)


def classify_motion(evidence: MotionEvidence,
                    speed_thresh: float = DEFAULT_SPEED_THRESH,
                    radar_fusion: str = 'radar_primary',
                    min_radar_pts: int = MIN_RADAR_PTS) -> tuple:
    """Classify a single detection as static/dynamic/unknown.

    Args:
        evidence: Collected :class:`MotionEvidence` for the box.
        speed_thresh: Base planar-speed threshold in m/s.
        radar_fusion: How to combine radar and box speed when both exist
            (Phase 2). One of:
            - ``'radar_primary'``: trust radar when enough in-box points,
              otherwise fall back to box velocity.
            - ``'or'``: dynamic if *either* signal exceeds the threshold
              (catches occluded movers with near-zero predicted velocity).
        min_radar_pts: Minimum in-box radar points before trusting radar.

    Returns:
        ``(motion, source)`` where ``motion`` is
        ``'static' | 'dynamic' | 'unknown'`` and ``source`` is
        ``'class' | 'box' | 'radar' | 'fused'``.
    """
    if evidence.class_name in STATIC_CLASSES:
        return 'static', 'class'

    thr = speed_thresh_for(evidence.class_name, speed_thresh)
    has_radar = (evidence.radar_pts >= min_radar_pts
                 and evidence.radar_speed is not None)
    has_box = evidence.box_speed is not None

    if radar_fusion == 'or' and (has_radar or has_box):
        box_dyn = has_box and evidence.box_speed >= thr
        radar_dyn = has_radar and evidence.radar_speed >= thr
        motion = 'dynamic' if (box_dyn or radar_dyn) else 'static'
        source = 'fused' if (has_radar and has_box) else (
            'radar' if has_radar else 'box')
        return motion, source

    # Default 'radar_primary': prefer the direct radar measurement.
    if has_radar:
        speed, source = evidence.radar_speed, 'radar'
    elif has_box:
        speed, source = evidence.box_speed, 'box'
    else:
        return 'unknown', 'none'

    return ('dynamic' if speed >= thr else 'static'), source


def evidence_from_prediction(class_name: str,
                             bbox_3d,
                             radar_speed: float | None = None,
                             radar_pts: int = 0) -> MotionEvidence:
    """Build :class:`MotionEvidence` from a predicted box (+ optional radar)."""
    return MotionEvidence(
        class_name=class_name,
        box_speed=box_speed_from_bbox(bbox_3d),
        radar_speed=radar_speed,
        radar_pts=radar_pts,
    )


def classify_detections(labels,
                        bboxes,
                        class_names,
                        speed_thresh: float = DEFAULT_SPEED_THRESH,
                        radar_evidence=None,
                        radar_fusion: str = 'radar_primary',
                        min_radar_pts: int = MIN_RADAR_PTS) -> list:
    """Classify a batch of detections.

    Args:
        labels: Iterable of int class indices, length N.
        bboxes: Array-like of shape (N, >=7); columns 7,8 are vx,vy if present.
        class_names: List mapping label index -> class name.
        speed_thresh: Base planar-speed threshold (m/s).
        radar_evidence: Optional list of ``(radar_speed, radar_pts)`` tuples
            per box (Phase 2). ``None`` => box velocity only.
        radar_fusion: Fusion policy passed to :func:`classify_motion`.

    Returns:
        List of dicts with keys: ``motion``, ``source``, ``class_name``,
        ``speed_mps``, ``box_speed_mps``, ``radar_speed_mps``, ``radar_pts``.
    """
    bboxes = np.asarray(bboxes, dtype=np.float64)
    out = []
    for i, lab in enumerate(labels):
        lab = int(lab)
        name = class_names[lab] if 0 <= lab < len(class_names) else str(lab)
        bbox = bboxes[i] if i < len(bboxes) else np.zeros(7)
        r_speed, r_pts = (None, 0)
        if radar_evidence is not None and i < len(radar_evidence):
            r_speed, r_pts = radar_evidence[i]
        ev = evidence_from_prediction(name, bbox, r_speed, r_pts)
        velocity = box_velocity_xy(bbox)
        motion, source = classify_motion(
            ev, speed_thresh, radar_fusion, min_radar_pts)
        # Report the speed that drove the decision so viz labels are honest.
        if source == 'radar':
            used_speed = ev.radar_speed
        elif source == 'fused':
            used_speed = max(
                v for v in (ev.box_speed, ev.radar_speed) if v is not None)
        else:
            used_speed = ev.box_speed
        out.append({
            'motion': motion,
            'source': source,
            'class_name': name,
            'speed_mps': used_speed,
            'box_speed_mps': ev.box_speed,
            'vx_mps': velocity[0] if velocity is not None else None,
            'vy_mps': velocity[1] if velocity is not None else None,
            'radar_speed_mps': ev.radar_speed,
            'radar_pts': ev.radar_pts,
        })
    return out


def motion_color(motion: str, class_color=None, color_by: str = 'motion'):
    """Pick a draw color given the motion label.

    Args:
        motion: ``'static' | 'dynamic' | 'unknown'``.
        class_color: Grouped class color for non-dynamic detections.
        color_by: ``'motion'`` keeps moving objects red and uses grouped class
            colors for all others; ``'class'`` always uses the class color.
    """
    if color_by == 'class' and class_color is not None:
        return class_color
    if motion != 'dynamic' and class_color is not None:
        return class_color
    return MOTION_COLORS.get(motion, MOTION_COLORS['unknown'])


def motion_linestyle(motion: str) -> str:
    """Solid for dynamic, dashed for static/unknown."""
    return '-' if motion == 'dynamic' else '--'


# ---------------------------------------------------------------------------
# Phase 2 (radar): associate radar points to detection boxes.
# ---------------------------------------------------------------------------
def associate_radar_to_boxes(boxes,
                             radar_xyz,
                             radar_speed,
                             bev_margin: float = 0.5,
                             agg: str = 'median'):
    """Associate radar points to boxes and aggregate compensated speed.

    Association is done in BEV using each box's rotated footprint. A radar
    point belongs to a box if, in the box-local frame, it lies within the box
    extent expanded by ``bev_margin`` on each side (radar returns are sparse
    and often land just off the exact footprint).

    Args:
        boxes: (N, >=7) array of boxes ``[x, y, z, x_size, y_size, z_size,
            yaw, ...]`` in the LiDAR frame (mmdet3d convention: ``x_size`` is
            length along heading, ``y_size`` is width, ``yaw`` about +z).
        radar_xyz: (M, 3) radar positions in the same LiDAR frame.
        radar_speed: (M,) planar compensated speed per radar point (m/s).
        bev_margin: Extra half-extent (m) added around each box footprint.
        agg: Aggregation over in-box point speeds: ``'median'`` (robust,
            default) or ``'max'``.

    Returns:
        List of ``(radar_speed, radar_pts)`` per box (length N). ``radar_speed``
        is ``None`` when no radar points are associated. Ready to pass to
        :func:`classify_detections` as ``radar_evidence``.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    radar_xyz = np.asarray(radar_xyz, dtype=np.float64)
    radar_speed = np.asarray(radar_speed, dtype=np.float64).ravel()

    n = len(boxes)
    if n == 0 or radar_xyz.shape[0] == 0:
        return [(None, 0)] * n

    rx = radar_xyz[:, 0]
    ry = radar_xyz[:, 1]
    out = []
    for b in boxes:
        cx, cy = b[0], b[1]
        l, w = b[3], b[4]
        yaw = b[6]
        dx = rx - cx
        dy = ry - cy
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        # Rotate world offsets into the box-local frame (rotate by -yaw).
        local_x = dx * cos_y + dy * sin_y
        local_y = -dx * sin_y + dy * cos_y
        inside = ((np.abs(local_x) <= l / 2.0 + bev_margin)
                  & (np.abs(local_y) <= w / 2.0 + bev_margin))
        cnt = int(inside.sum())
        if cnt == 0:
            out.append((None, 0))
            continue
        speeds = radar_speed[inside]
        val = float(np.median(speeds)) if agg == 'median' else float(
            speeds.max())
        out.append((val, cnt))
    return out
