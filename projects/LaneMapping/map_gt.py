"""Ground-truth map polylines in the LIDAR_TOP frame for MapTRv2's range.

This follows the MapTRv2 nuScenes converter (patch of 30 m in x and 60 m in
y, centered on the LiDAR pose) but only reads the map. Nothing here is used
at inference time.

What each class is:

* ``divider`` is the annotated ``road_divider`` and ``lane_divider`` lines,
  merged into the one class MapTRv2 predicts.
* ``centerline`` starts from the annotated arc-line path of each ``lane`` and
  ``lane_connector`` (devkit discretization at 0.5 m). Connected paths inside
  the patch are then chained. The chain is derived; the arc-line path is the
  annotation.
* ``boundary`` is the outline of the union of ``road_segment`` and ``lane``
  polygons. nuScenes does not label a boundary polyline, so this outline is
  derived.
* ``ped_crossing`` is the outline of the annotated ``ped_crossing`` polygon,
  not a centerline through the crosswalk.

Fragments shorter than ``MIN_LENGTH_M`` are dropped. That length is a
geometric cutoff for clipping slivers, not a score threshold.
"""
import numpy as np
import networkx as nx
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely import affinity, ops
from shapely.geometry import (
    LineString, MultiLineString, MultiPolygon, box)

PC_RANGE = (-15.0, -30.0, 15.0, 30.0)  # xmin, ymin, xmax, ymax in LIDAR_TOP
PATCH_H = 60.0
PATCH_W = 30.0
MIN_LENGTH_M = 1.0
CLASSES = ('divider', 'ped_crossing', 'boundary', 'centerline')
GT_SOURCES = {
    'divider': 'annotated road_divider and lane_divider lines',
    'ped_crossing': 'outline of annotated ped_crossing polygons',
    'boundary': 'outline of the union of road_segment and lane polygons',
    'centerline': 'chained 0.5 m discretization of annotated lane arc-line paths',
}


def _lines(geom):
    if geom is None or geom.is_empty:
        return
    kind = geom.geom_type
    if kind == 'LineString':
        yield geom
    elif kind == 'MultiLineString':
        yield from geom.geoms
    elif kind == 'GeometryCollection':
        for part in geom.geoms:
            yield from _lines(part)


def _keep(line):
    return line.length >= MIN_LENGTH_M and len(line.coords) >= 2


def _to_patch(geom, angle_deg, origin_x, origin_y):
    geom = affinity.rotate(geom, -angle_deg, origin=(origin_x, origin_y),
                           use_radians=False)
    return affinity.affine_transform(
        geom, [1.0, 0.0, 0.0, 1.0, -origin_x, -origin_y])


def _coords(line):
    return np.round(np.asarray(line.coords, dtype=float)[:, :2], 3)


class MapGroundTruth:
    """Local map polylines for one nuScenes map location."""

    def __init__(self, dataroot, location):
        self.location = location
        self.map = NuScenesMap(dataroot=dataroot, map_name=location)
        self.explorer = self.map.explorer

    def frame(self, lidar2global):
        """Polylines for one LIDAR_TOP pose, keyed by MapTRv2 class."""
        pose = np.asarray(lidar2global, dtype=float)
        origin_x, origin_y = pose[0, 3], pose[1, 3]
        yaw_deg = float(np.degrees(quaternion_yaw(Quaternion(matrix=pose[:3, :3]))))
        patch_box = (origin_x, origin_y, PATCH_H, PATCH_W)
        dividers = []
        for layer in ('road_divider', 'lane_divider'):
            dividers.extend(self._divider(patch_box, yaw_deg, layer))
        return {
            'divider': [_coords(line) for line in dividers if _keep(line)],
            'ped_crossing': self._ped(patch_box, yaw_deg),
            'boundary': self._boundary(patch_box, yaw_deg),
            'centerline': self._centerlines(patch_box, yaw_deg),
            'source': dict(GT_SOURCES),
        }

    def _patch(self, patch_box, angle_deg):
        return self.explorer.get_patch_coord(patch_box, angle_deg)

    def _divider(self, patch_box, angle_deg, layer):
        patch = self._patch(patch_box, angle_deg)
        origin_x, origin_y = patch_box[0], patch_box[1]
        lines = []
        for record in getattr(self.map, layer):
            line = self.map.extract_line(record['line_token'])
            if line.is_empty:
                continue
            clipped = line.intersection(patch)
            if clipped.is_empty:
                continue
            lines.extend(_lines(_to_patch(clipped, angle_deg, origin_x, origin_y)))
        return lines

    def _polygons(self, patch_box, angle_deg, layer):
        patch = self._patch(patch_box, angle_deg)
        origin_x, origin_y = patch_box[0], patch_box[1]
        polygons = []
        for record in getattr(self.map, layer):
            token = record.get('polygon_token')
            if not token:
                continue
            polygon = self.map.extract_polygon(token)
            if not polygon.is_valid:
                continue
            clipped = polygon.intersection(patch)
            if clipped.is_empty:
                continue
            moved = _to_patch(clipped, angle_deg, origin_x, origin_y)
            if moved.geom_type == 'Polygon':
                moved = MultiPolygon([moved])
            elif moved.geom_type == 'GeometryCollection':
                moved = MultiPolygon(
                    [part for part in moved.geoms if part.geom_type == 'Polygon'])
            if not moved.is_empty:
                polygons.append(moved)
        return polygons

    def _outlines(self, polygons, inset):
        if not polygons:
            return []
        merged = ops.unary_union(polygons)
        if merged.is_empty:
            return []
        if merged.geom_type != 'MultiPolygon':
            if merged.geom_type != 'Polygon':
                return []
            merged = MultiPolygon([merged])
        limit = box(-PATCH_W / 2 + inset, -PATCH_H / 2 + inset,
                    PATCH_W / 2 - inset, PATCH_H / 2 - inset)
        lines = []
        for poly in merged.geoms:
            rings = [poly.exterior, *poly.interiors]
            for ring in rings:
                clipped = ring.intersection(limit)
                if isinstance(clipped, MultiLineString):
                    clipped = ops.linemerge(clipped)
                lines.extend(line for line in _lines(clipped) if _keep(line))
        return [_coords(line) for line in lines]

    def _ped(self, patch_box, angle_deg):
        return self._outlines(self._polygons(patch_box, angle_deg, 'ped_crossing'),
                              inset=-0.2)

    def _boundary(self, patch_box, angle_deg):
        roads = self._polygons(patch_box, angle_deg, 'road_segment')
        lanes = self._polygons(patch_box, angle_deg, 'lane')
        return self._outlines(roads + lanes, inset=0.2)

    def _centerlines(self, patch_box, angle_deg):
        records = {}
        for layer in ('lane', 'lane_connector'):
            records.update(self._centerline_records(patch_box, angle_deg, layer))
        if not records:
            return []
        graph = nx.DiGraph()
        for value in records.values():
            parts = list(_lines(value['centerline']))
            if not parts:
                continue
            start = tuple(np.round(parts[0].coords[0], 3))
            end = tuple(np.round(parts[-1].coords[-1], 3))
            for part in parts:
                pts = np.round(np.asarray(part.coords), 3)
                for a, b in zip(pts[:-1], pts[1:]):
                    graph.add_edge(tuple(a), tuple(b))
            for pred in value['incoming']:
                other = records.get(pred)
                if other is None:
                    continue
                pred_parts = list(_lines(other['centerline']))
                if pred_parts:
                    pred_end = tuple(np.round(pred_parts[-1].coords[-1], 3))
                    graph.add_edge(pred_end, start)
            for succ in value['outgoing']:
                other = records.get(succ)
                if other is None:
                    continue
                succ_parts = list(_lines(other['centerline']))
                if succ_parts:
                    succ_start = tuple(np.round(succ_parts[0].coords[0], 3))
                    graph.add_edge(end, succ_start)
        if graph.number_of_edges() == 0:
            return []
        roots = [node for node, degree in graph.in_degree() if degree == 0]
        leaves = [node for node, degree in graph.out_degree() if degree == 0]
        paths = []
        for root in roots:
            paths.extend(nx.all_simple_paths(graph, root, leaves))
        lines = []
        for path in paths:
            if len(path) < 2:
                continue
            line = LineString(path).simplify(0.2, preserve_topology=True)
            if _keep(line):
                lines.append(_coords(line))
        return lines

    def _centerline_records(self, patch_box, angle_deg, layer):
        patch = self._patch(patch_box, angle_deg)
        origin_x, origin_y = patch_box[0], patch_box[1]
        found = {}
        for record in getattr(self.map, layer):
            if not record.get('polygon_token'):
                continue
            polygon = self.map.extract_polygon(record['polygon_token'])
            if not polygon.is_valid or polygon.intersection(patch).is_empty:
                continue
            points = self.map.discretize_lanes([record['token']], 0.5)
            raw = points.get(record['token']) or []
            if len(raw) < 2:
                continue
            center = LineString(np.round(np.asarray(raw)[:, :2], 3))
            clipped = center.intersection(patch)
            if clipped.is_empty:
                continue
            moved = _to_patch(clipped, angle_deg, origin_x, origin_y)
            found[record['token']] = {
                'centerline': moved,
                'incoming': self.map.get_incoming_lane_ids(record['token']),
                'outgoing': self.map.get_outgoing_lane_ids(record['token']),
            }
        return found
