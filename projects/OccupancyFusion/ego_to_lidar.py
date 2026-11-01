"""Ego-frame Occ3D voxels to the Foxglove ``lidar`` (LIDAR_TOP) frame.

SparseOcc predicts a dense 200x200x16 grid over
``[-40, -40, -1]`` to ``[40, 40, 5.4]`` metres in the nuScenes ego frame.
Foxglove in this checkout uses LIDAR_TOP, so voxel centres are transformed
with that sample's ``ego2lidar`` (the inverse of the LIDAR_TOP
calibrated-sensor pose).
"""
import numpy as np

PC_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], dtype=np.float64)
OCC_SIZE = (200, 200, 16)
FREE_CLASS = 17
OCC_CLASS_NAMES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
)


def voxel_resolution(occ_size=OCC_SIZE, pc_range=PC_RANGE):
    size = np.asarray(occ_size, dtype=np.float64)
    lo = np.asarray(pc_range[:3], dtype=np.float64)
    hi = np.asarray(pc_range[3:], dtype=np.float64)
    return (hi - lo) / size


def voxel_centers_from_indices(indices, occ_size=OCC_SIZE, pc_range=PC_RANGE):
    """World-frame centres of voxel indices ``(i, j, k)`` in the ego frame."""
    indices = np.asarray(indices, dtype=np.float64).reshape(-1, 3)
    lo = np.asarray(pc_range[:3], dtype=np.float64)
    res = voxel_resolution(occ_size, pc_range)
    return lo + (indices + 0.5) * res


def dense_from_sparse(occ_loc, sem_pred, occ_size=OCC_SIZE,
                      free_class=FREE_CLASS):
    """Scatter SparseOcc's sparse ``occ_loc`` / ``sem_pred`` into a dense grid."""
    dense = np.full(occ_size, free_class, dtype=np.uint8)
    loc = np.asarray(occ_loc).reshape(-1, 3).astype(np.int64)
    pred = np.asarray(sem_pred).reshape(-1).astype(np.uint8)
    if loc.size:
        dense[loc[:, 0], loc[:, 1], loc[:, 2]] = pred
    return dense


def transform_ego_to_lidar(points_ego, ego2lidar):
    """Apply a 4x4 ``ego2lidar`` matrix to ``(N, 3)`` ego points."""
    points = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(ego2lidar, dtype=np.float64).reshape(4, 4)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def occupied_lidar_points(sem_grid, ego2lidar, free_class=FREE_CLASS,
                          occ_size=OCC_SIZE, pc_range=PC_RANGE):
    """Occupied voxel centres in LIDAR_TOP, dropping the free class."""
    grid = np.asarray(sem_grid)
    occupied = grid != free_class
    indices = np.argwhere(occupied)
    labels = grid[occupied].astype(np.uint8)
    centres = voxel_centers_from_indices(indices, occ_size, pc_range)
    xyz = transform_ego_to_lidar(centres, ego2lidar).astype(np.float32)
    return xyz, labels
