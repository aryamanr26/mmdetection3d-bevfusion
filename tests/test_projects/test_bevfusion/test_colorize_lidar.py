import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'projects', 'BEVFusion',
        'demo'))

from colorize_lidar import colorize_lidar  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_colorize_matches_numpy():
    rng = np.random.default_rng(0)
    points = rng.uniform([-30, -30, -2], [30, 30, 3], size=(20000, 3))
    images, lidar2img = [], []
    for yaw in (0.0, np.pi / 2, np.pi):
        images.append(rng.integers(0, 256, (90, 160, 3), dtype=np.uint8))
        c, s = np.cos(yaw), np.sin(yaw)
        # LiDAR x-forward to camera z-forward, rotated by yaw.
        lidar2cam = np.eye(4)
        lidar2cam[:3, :3] = np.array(
            [[s, -c, 0.0], [0.0, 0.0, -1.0], [c, s, 0.0]])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = [[100.0, 0, 80], [0, 100.0, 45], [0, 0, 1]]
        lidar2img.append(intrinsic @ lidar2cam)

    for margin in (0, 4):
        ref, ref_mask = colorize_lidar(
            points, images, lidar2img, border_margin_px=margin, device='cpu')
        out, mask = colorize_lidar(
            points, images, lidar2img, border_margin_px=margin, device='cuda')
        assert ref_mask.any()
        np.testing.assert_array_equal(mask, ref_mask)
        np.testing.assert_array_equal(out, ref)
