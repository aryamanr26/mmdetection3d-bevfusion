import math
import os
import sys

import numpy as np


DEMO_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'projects', 'BEVFusion', 'demo'))
sys.path.insert(0, DEMO_DIR)

from tracker import (  # noqa: E402
    ConstantVelocityTracker, MultiObjectTracker, inverse_transform_box,
    transform_detection, transform_matrix)


def _detection(x, y=0.0, score=0.9, name='car', yaw=0.0):
    return {
        'class_name': name,
        'score': score,
        'bbox_3d': [x, y, 0.0, 4.0, 2.0, 1.5, yaw, 2.0, 0.0],
    }


def test_global_transform_round_trip():
    yaw = math.pi / 2.0
    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw), 0.0, 10.0],
        [math.sin(yaw), math.cos(yaw), 0.0, 20.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    info = {
        'ego2global': rotation.tolist(),
        'lidar_points': {'lidar2ego': np.eye(4).tolist()},
    }
    matrix = transform_matrix(info)
    source = _detection(2.0, 1.0)
    global_det = transform_detection(source, matrix)
    restored = inverse_transform_box(global_det['bbox_3d'], matrix)
    assert np.allclose(restored, source['bbox_3d'])


def test_identity_survives_detection_and_coast():
    tracker = MultiObjectTracker(max_age_s=1.0)
    assert tracker.step([_detection(0.0)], 0.0) == []
    tracks = tracker.step([_detection(1.0)], 0.5)
    assert len(tracks) == 1
    assert tracks[0]['tracking_id'] == '1'
    assert tracks[0]['moving']
    coasted = tracker.step([], 1.0)
    assert coasted[0]['tracking_id'] == '1'
    assert coasted[0]['coasted']
    assert tracker.step([], 2.0) == []


def test_classes_are_never_cross_associated():
    tracker = MultiObjectTracker(min_hits=1)
    first = tracker.step([
        _detection(0.0, name='car'),
        _detection(0.0, name='pedestrian'),
    ], 0.0)
    assert {track['class_name'] for track in first} == {'car', 'pedestrian'}
    second = tracker.step([
        _detection(0.5, name='pedestrian'),
        _detection(0.5, name='car'),
    ], 0.5)
    ids = {track['class_name']: track['tracking_id'] for track in second}
    assert ids == {'car': '1', 'pedestrian': '2'}


def test_birth_score_separate_from_association_split():
    low = _detection(0.0, score=0.2)
    default = MultiObjectTracker(min_hits=1)
    assert len(default.step([low], 0.0)) == 1
    strict = MultiObjectTracker(min_hits=1, birth_score=0.4)
    assert strict.step([low], 0.0) == []
    # A low-score detection beside a matched track must not spawn a duplicate.
    tracker = MultiObjectTracker(min_hits=1)
    tracker.step([_detection(0.0)], 0.0)
    tracks = tracker.step([_detection(1.0), _detection(1.2, score=0.2)], 0.5)
    assert [track['tracking_id'] for track in tracks] == ['1']


def test_constant_velocity_baseline_lifecycle():
    tracker = ConstantVelocityTracker(max_age_s=1.0)
    first = tracker.step([_detection(0.0)], 0.0)
    assert [track['tracking_id'] for track in first] == ['1']
    # The detection velocity is 2 m/s, so the prediction at 0.5 s is x=1.
    second = tracker.step([_detection(1.1)], 0.5)
    assert [track['tracking_id'] for track in second] == ['1']
    assert tracker.step([], 1.0) == []
    assert len(tracker.tracks) == 1
    assert tracker.step([], 2.0) == []
    assert tracker.tracks == []
    third = tracker.step([_detection(5.0)], 2.5)
    assert [track['tracking_id'] for track in third] == ['2']
