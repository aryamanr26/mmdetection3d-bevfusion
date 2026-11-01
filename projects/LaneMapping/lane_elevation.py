"""Attach LiDAR-frame ground height to MapTRv2 polylines.

``sample(xy)`` returns ``(height, valid)`` from the existing elevation grid.
Heights that the grid does not provide stay missing. The XY coordinates are
not changed.
"""
import numpy as np


def assign_elevation(elements, sample):
    """Return copies of ``elements`` with ``points_xyz`` of ``[x, y, z or None]``."""
    lifted = []
    for element in elements:
        xy = np.asarray(element['points'], dtype=float).reshape(-1, 2)
        height, valid = sample(xy)
        points_xyz = []
        for (x, y), z, ok in zip(xy, height, valid):
            points_xyz.append([
                round(float(x), 3), round(float(y), 3),
                None if not ok else round(float(z), 3)])
        copy = dict(element)
        copy['points'] = [[row[0], row[1]] for row in points_xyz]
        copy['points_xyz'] = points_xyz
        copy['elevation_valid'] = int(np.count_nonzero(valid))
        copy['elevation_points'] = int(len(valid))
        lifted.append(copy)
    return lifted
