import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np

_PATH = (Path(__file__).resolve().parents[3] / 'projects' / 'OccupancyFusion'
         / 'occupancy_foxglove.py')


def _load():
    spec = importlib.util.spec_from_file_location('occupancy_foxglove', _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_attach_occupancy_skips_missing():
    mod = _load()
    frame = {}
    assert mod.attach_occupancy(frame, '', '/tmp') is False
    assert mod.attach_occupancy(frame, 'missing-token', '/tmp') is False
    assert 'occupancy' not in frame


def test_attach_occupancy_joins_sample_token():
    mod = _load()
    token = '3e8750f331d7499e9b5123e9eb70f2e2'
    xyz = np.array([[1.0, 2.0, -1.5], [4.0, 0.0, 0.5]], dtype=np.float32)
    label = np.array([4, 11], dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(os.path.join(tmp, token + '.npz'), xyz=xyz, label=label)
        frame = {}
        assert mod.attach_occupancy(frame, token, tmp) is True
        assert frame['occupancy_count'] == 2
        assert 'occupancy' in frame
