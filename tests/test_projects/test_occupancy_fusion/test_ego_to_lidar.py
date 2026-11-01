import importlib.util
from pathlib import Path

import numpy as np

_PATH = (Path(__file__).resolve().parents[3] / 'projects' / 'OccupancyFusion'
         / 'ego_to_lidar.py')
_spec = importlib.util.spec_from_file_location('ego_to_lidar', _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_identity_round_trip():
    grid = np.full(mod.OCC_SIZE, mod.FREE_CLASS, dtype=np.uint8)
    grid[100, 100, 0] = 4
    xyz, labels = mod.occupied_lidar_points(grid, np.eye(4))
    assert labels.tolist() == [4]
    expected = mod.voxel_centers_from_indices([[100, 100, 0]])
    np.testing.assert_allclose(xyz, expected, atol=1e-6)


def test_lidar_is_1_8m_above_ego():
    """LIDAR_TOP sits 1.8 m above the ego origin; road voxels drop in z."""
    lidar2ego = np.eye(4)
    lidar2ego[2, 3] = 1.8
    ego2lidar = np.linalg.inv(lidar2ego)
    grid = np.full(mod.OCC_SIZE, mod.FREE_CLASS, dtype=np.uint8)
    grid[100, 100, 0] = 11
    xyz, labels = mod.occupied_lidar_points(grid, ego2lidar)
    assert labels.tolist() == [11]
    ego_xyz = mod.voxel_centers_from_indices([[100, 100, 0]])[0]
    np.testing.assert_allclose(xyz[0], [ego_xyz[0], ego_xyz[1], ego_xyz[2] - 1.8],
                               atol=1e-6)


def test_dense_from_sparse_and_drop_free():
    loc = np.array([[10, 20, 3], [11, 20, 3]], dtype=np.int64)
    pred = np.array([4, mod.FREE_CLASS], dtype=np.uint8)
    grid = mod.dense_from_sparse(loc, pred)
    assert grid.shape == mod.OCC_SIZE
    assert grid[10, 20, 3] == 4
    assert grid[11, 20, 3] == mod.FREE_CLASS
    xyz, labels = mod.occupied_lidar_points(grid, np.eye(4))
    assert labels.tolist() == [4]
    assert len(xyz) == 1
