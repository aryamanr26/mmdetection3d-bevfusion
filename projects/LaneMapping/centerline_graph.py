"""Lane connectivity from MapTRv2 path-wise centerlines, without training.

MapTRv2 (LaneGAP-style) predicts each centerline as a directed path. Paths that
start together and then separate are forks, paths that end together are
merges, and a path whose end meets another path's start is its predecessor.
Endpoints are clustered into graph nodes so every centerline becomes one
directed edge between two nodes.
"""
import numpy as np


def _cluster(points, radius):
    """Greedy clustering; returns a label per point and the cluster means."""
    labels = -np.ones(len(points), dtype=int)
    centers = []
    for i, p in enumerate(points):
        if labels[i] >= 0:
            continue
        members = np.where((labels < 0) &
                           (np.linalg.norm(points - p, axis=1) <= radius))[0]
        labels[members] = len(centers)
        centers.append(points[members].mean(axis=0))
    return labels, np.asarray(centers).reshape(-1, points.shape[1])


def _heading(a, b):
    v = np.asarray(b, float) - np.asarray(a, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _shared_length(p, q, tol):
    """Number of leading points of ``p`` within ``tol`` of polyline ``q``."""
    n = 0
    for x in p:
        if np.min(np.linalg.norm(q - x, axis=1)) > tol:
            break
        n += 1
    return n


def build_centerline_graph(centerlines, node_radius=1.5, min_cos=0.5,
                           share_tol=0.75, min_shared=3):
    """Build a directed lane graph from centerline polylines.

    Args:
        centerlines (list[array (N, 2)]): ordered points, start to end.
        node_radius (float): endpoints closer than this (m) share a node.
        min_cos (float): heading agreement required to link i -> j.
        share_tol (float): distance (m) under which points count as shared.
        min_shared (int): shared leading/trailing points for a fork/merge.

    Returns:
        dict with ``nodes`` [[x, y]], ``edges`` [{lane, from, to}],
        ``successors`` [[i, j]], ``forks`` [[i, j]] and ``merges`` [[i, j]],
        where i, j index ``centerlines``.
    """
    lines = [np.asarray(c, dtype=float) for c in centerlines]
    lines = [c for c in lines if len(c) >= 2]
    if len(lines) != len(centerlines):
        raise ValueError('every centerline needs at least two points')
    if not lines:
        return dict(nodes=[], edges=[], successors=[], forks=[], merges=[])

    ends = np.concatenate([np.stack([c[0], c[-1]]) for c in lines])
    labels, nodes = _cluster(ends, node_radius)
    start_node, end_node = labels[0::2], labels[1::2]
    edges = [dict(lane=i, **{'from': int(start_node[i]), 'to': int(end_node[i])})
             for i in range(len(lines))]

    successors, forks, merges = [], [], []
    for i, a in enumerate(lines):
        out_dir = _heading(a[-2], a[-1])
        for j, b in enumerate(lines):
            if i == j:
                continue
            if (end_node[i] == start_node[j]
                    and float(out_dir @ _heading(b[0], b[1])) >= min_cos):
                successors.append([i, j])
            if j > i:
                head = _shared_length(a, b, share_tol)
                tail = _shared_length(a[::-1], b[::-1], share_tol)
                if head >= min_shared and head < min(len(a), len(b)):
                    forks.append([i, j])
                if tail >= min_shared and tail < min(len(a), len(b)):
                    merges.append([i, j])
    return dict(nodes=nodes.round(3).tolist(), edges=edges,
                successors=successors, forks=forks, merges=merges)
