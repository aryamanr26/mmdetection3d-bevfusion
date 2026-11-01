import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3] / 'projects' / 'LaneMapping'
DEMO = Path(__file__).resolve().parents[3] / 'projects' / 'BEVFusion' / 'demo'


def _load(name, directory):
    spec = importlib.util.spec_from_file_location(name, directory / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


motion = _load('lane_motion', ROOT)
schedule = _load('lane_schedule', ROOT)
elevation = _load('lane_elevation', ROOT)
grid_mod = _load('elevation_grid', DEMO)


def _pose(x, y, yaw):
    m = np.eye(4)
    m[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    m[0, 3], m[1, 3] = x, y
    return m


def _grid(fn, holes=()):
    size = grid_mod.SIZE
    res = grid_mod.RESOLUTION
    fy, fx = np.mgrid[0:size, 0:size]
    x = grid_mod.X_MIN + (fx + 0.5) * res
    y = grid_mod.Y_MIN + (fy + 0.5) * res
    ground = fn(x, y).astype(np.float32)
    for ix, iy in holes:
        ground[iy, ix] = np.nan
    nan = np.full_like(ground, np.nan)
    return grid_mod.ElevationGrid(
        ground=ground, variance=nan, slope=nan, roughness=nan, obstacle=nan,
        resolution=res, x_min=grid_mod.X_MIN, y_min=grid_mod.Y_MIN,
        size=size, timings_ms={})


def test_ego_motion_compensation():
    src = _pose(0, 0, 0)
    dst = _pose(10, 0, 0)
    moved = motion.transform_xy([[5.0, 1.0]], src, dst)
    assert np.allclose(moved, [[-5.0, 1.0]])
    hold = motion.LaneHold(stale_s=2.0)
    assert hold.update(1, 10.0, src, [{'points': [[5.0, 1.0]], 'score': 0.9}])
    view = hold.query(dst, 10.4)
    assert not view['stale']
    assert np.allclose(view['elements'][0]['points'], [[-5.0, 1.0]])
    assert abs(view['age_s'] - 0.4) < 1e-9


def test_stale_and_old_result_are_rejected():
    hold = motion.LaneHold(stale_s=1.0)
    pose = np.eye(4)
    hold.update(2, 5.0, pose, [{'points': [[0.0, 1.0]]}])
    assert hold.update(1, 9.0, pose, [{'points': [[0.0, 2.0]]}]) is False
    assert hold.record['seq'] == 2
    assert hold.query(pose, 6.5)['stale']
    assert hold.query(pose, 6.5)['elements'] == []


def test_scheduler_drops_frames_and_keeps_one_inflight():
    sched = schedule.LatestFrameScheduler(period_s=0.5)
    job, scheduled = sched.submit({'seq': 1}, 0.0)
    assert scheduled and job['seq'] == 1
    job, scheduled = sched.submit({'seq': 2}, 0.2)
    assert job is None and not scheduled
    job, scheduled = sched.submit({'seq': 3}, 0.5)
    assert job is None and scheduled
    assert sched.pending['seq'] == 3
    nxt = sched.complete(1)
    assert nxt['seq'] == 3
    assert sched.complete(1) is None
    assert sched.accept_result(3)
    assert sched.accept_result(3) is False
    assert sched.accept_result(2) is False


def test_micro_scores_do_not_match_across_frames():
    metrics = _load('map_metrics', ROOT)
    pred_a = [[0.4, 0.0], [0.4, 10.0]]
    gt_a = [[0.0, 0.0], [0.0, 10.0]]
    pred_b = [[0.6, 0.0], [0.6, 10.0]]
    gt_b = [[0.5, 0.0], [0.5, 10.0]]
    result = metrics.micro_scores(
        [([pred_a], [gt_a]), ([pred_b], [gt_b])], (0.45,))
    assert result['thresholds'][0]['tp'] == 2
    stolen = metrics.scores([pred_a, pred_b], [gt_a, gt_b], (0.45,))
    assert stolen['thresholds'][0]['tp'] == 1


def test_elevation_flat_slope_and_missing():
    flat = _grid(lambda x, y: np.full(x.shape, -1.7))
    z, valid = grid_mod.sample_ground(flat, [[0.0, 0.0], [40.0, 0.0]])
    assert valid.tolist() == [True, False]
    assert abs(z[0] + 1.7) < 1e-6

    uphill = _grid(lambda x, y: 0.1 * y)
    z, valid = grid_mod.sample_ground(uphill, [[0.0, 10.0], [0.0, -10.0]])
    assert valid.all()
    assert z[0] > 0.9 and z[1] < -0.9

    hole = _grid(lambda x, y: np.zeros(x.shape), holes=[(125, 125)])
    ix = int(np.floor((0.0 - grid_mod.X_MIN) / grid_mod.RESOLUTION))
    iy = int(np.floor((0.0 - grid_mod.Y_MIN) / grid_mod.RESOLUTION))
    hole.ground[iy, ix] = np.nan
    z, valid = grid_mod.sample_ground(hole, [[0.0, 0.0], [1.0, 0.0]])
    assert valid.tolist() == [False, True]
    lifted = elevation.assign_elevation(
        [{'points': [[0.0, 0.0], [1.0, 0.0]], 'score': 0.5}],
        lambda xy: grid_mod.sample_ground(hole, xy))
    assert lifted[0]['points_xyz'][0][2] is None
    assert lifted[0]['points_xyz'][1][2] == 0.0
    assert lifted[0]['points'][0][:2] == [0.0, 0.0]
