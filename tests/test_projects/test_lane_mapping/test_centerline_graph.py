import importlib.util
from pathlib import Path

import numpy as np

_PATH = (Path(__file__).resolve().parents[3] / 'projects' / 'LaneMapping' /
         'centerline_graph.py')
_spec = importlib.util.spec_from_file_location('centerline_graph', _PATH)
centerline_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(centerline_graph)
build = centerline_graph.build_centerline_graph


def _line(x0, y0, x1, y1, n=10):
    return np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], axis=1)


def test_successor_chain():
    g = build([_line(0, 0, 0, 10), _line(0, 10.5, 0, 20)])
    assert g['successors'] == [[0, 1]]
    assert len(g['nodes']) == 3
    assert g['edges'][0]['to'] == g['edges'][1]['from']


def test_opposite_heading_is_not_successor():
    g = build([_line(0, 0, 0, 10), _line(0, 10.5, 0, 1)])
    assert g['successors'] == []


def test_fork_and_merge():
    straight = _line(0, 0, 0, 20, n=21)
    right = np.concatenate([_line(0, 0, 0, 8, n=9), _line(1, 9, 8, 20, n=12)])
    g = build([straight, right])
    assert g['forks'] == [[0, 1]]
    assert g['merges'] == []
    g = build([straight[::-1], right[::-1]])
    assert g['merges'] == [[0, 1]]


def test_empty():
    assert build([])['edges'] == []
