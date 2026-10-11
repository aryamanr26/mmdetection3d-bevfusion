# Copyright (c) OpenMMLab. All rights reserved.
"""Phase 2 helper: load nuScenes radar points in the LiDAR frame.

BEVFusion outputs 3D boxes in the LIDAR_TOP frame. nuScenes provides 5 radar
sensors, each point carrying an ego-motion *compensated* velocity. We use the
nuScenes devkit to load all radar channels and transform both positions and
compensated velocities into the LIDAR_TOP frame (the same frame the detector's
boxes live in), so radar speed can be associated to each detection.

The devkit (`nuscenes`) is already a project dependency (see
``tools/dataset_converters/nuscenes_converter.py``).
"""
import numpy as np

# Cache NuScenes handles per (dataroot, version) -- construction is slow (~2 s)
# and the batch visualizer calls this once per frame.
_NUSC_CACHE = {}

RADAR_CHANNELS = (
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
)


def get_nusc(dataroot: str, version: str = 'v1.0-mini'):
    """Return a cached ``NuScenes`` handle for ``(dataroot, version)``."""
    key = (dataroot, version)
    if key not in _NUSC_CACHE:
        from nuscenes.nuscenes import NuScenes
        _NUSC_CACHE[key] = NuScenes(
            version=version, dataroot=dataroot, verbose=False)
    return _NUSC_CACHE[key]


def load_radar_lidar_frame(nusc, sample_token: str, nsweeps: int = 1):
    """Load all radar channels for a sample, in the LIDAR_TOP frame.

    Args:
        nusc: A ``NuScenes`` handle.
        sample_token: nuScenes sample token (present in the info pkl as
            ``info['token']``).
        nsweeps: Number of radar sweeps to accumulate per channel (1 = the
            keyframe only).

    Returns:
        ``(xyz, vcomp, speed)`` where
        - ``xyz`` is (M, 3) radar point positions in the LIDAR_TOP frame,
        - ``vcomp`` is (M, 2) ego-motion compensated velocity (vx, vy) rotated
          into the LIDAR_TOP frame,
        - ``speed`` is (M,) planar speed ``hypot(vx, vy)``.
        Returns empty arrays if no radar points are found.
    """
    from nuscenes.utils.data_classes import RadarPointCloud

    sample = nusc.get('sample', sample_token)
    cols = []
    for chan in RADAR_CHANNELS:
        if chan not in sample['data']:
            continue
        # from_file_multisweep transforms positions AND compensated velocities
        # (rows 8:10) into the ref_chan frame.
        pc, _ = RadarPointCloud.from_file_multisweep(
            nusc, sample, chan, ref_chan='LIDAR_TOP', nsweeps=nsweeps)
        if pc.points.shape[1] > 0:
            cols.append(pc.points)

    if not cols:
        return (np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, )))

    pts = np.concatenate(cols, axis=1)  # (18, M)
    xyz = pts[:3, :].T.astype(np.float64)
    vcomp = pts[8:10, :].T.astype(np.float64)  # vx_comp, vy_comp
    speed = np.hypot(vcomp[:, 0], vcomp[:, 1])
    return xyz, vcomp, speed


def load_radar_for_info(dataroot: str,
                        sample_token: str,
                        version: str = 'v1.0-mini',
                        nsweeps: int = 1):
    """Convenience wrapper: build/cache NuScenes and load radar for a token."""
    nusc = get_nusc(dataroot, version)
    return load_radar_lidar_frame(nusc, sample_token, nsweeps=nsweeps)
