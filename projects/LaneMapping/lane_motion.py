"""Hold the latest lane estimate and move it into a newer LiDAR frame.

MapTRv2 does not have to run on every BEVFusion frame. Between updates the
last polylines are rigidly transformed with the LiDAR poses. The transform
uses z = 0 in the source LiDAR frame, so it carries the BEV coordinates and
does not invent a road height. Display height comes from the elevation grid
in the destination frame. An estimate older than ``stale_s`` is not returned
as current.
"""
import numpy as np


def transform_xy(points, src_lidar2global, dst_lidar2global):
    """XY of ``points`` expressed in the destination LiDAR frame."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    hom = np.c_[pts, np.zeros(len(pts)), np.ones(len(pts))]
    world = np.asarray(src_lidar2global, dtype=float) @ hom.T
    local = np.linalg.inv(np.asarray(dst_lidar2global, dtype=float)) @ world
    return local[:2].T


class LaneHold:
    """Latest accepted lane estimate, rejected when a newer one exists."""

    def __init__(self, stale_s=2.0):
        self.stale_s = stale_s
        self.record = None

    def update(self, seq, timestamp, lidar2global, elements):
        """Accept ``elements`` unless an equal or newer sequence is stored."""
        if self.record is not None and seq <= self.record['seq']:
            return False
        self.record = dict(
            seq=int(seq), time=float(timestamp),
            pose=np.asarray(lidar2global, dtype=float),
            elements=elements)
        return True

    def query(self, lidar2global, now):
        """Lanes in the current LiDAR frame, or a stale marker."""
        if self.record is None:
            return None
        age = float(now) - self.record['time']
        base = dict(seq=self.record['seq'], estimate_time=self.record['time'],
                    age_s=age)
        if age > self.stale_s or age < -0.05:
            base.update(stale=True, elements=[])
            return base
        moved = []
        for element in self.record['elements']:
            copy = dict(element)
            copy['points'] = np.round(transform_xy(
                element['points'], self.record['pose'], lidar2global), 3).tolist()
            moved.append(copy)
        base.update(stale=False, elements=moved)
        return base
