"""nuScenes HD map in the LiDAR frame, for Foxglove only.

This reads map expansion. It is not an input to BEVFusion or MapTRv2.
Geometries are clipped to a square around the LiDAR pose and rotated into
the same ``lidar`` frame as the point cloud.

Lane polygons are not drawn. Lanes are the painted ``lane_divider`` and
``road_divider`` lines. Intersections are the centerlines of
``lane_connector`` records. Traffic lights are their short map lines.
"""
import json
import os.path as osp

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import LineString

from foxglove.messages import (
    Color, KeyValuePair, LinePrimitive, LinePrimitiveLineType, Point3, Pose,
    Quaternion, SceneEntity, SceneEntityDeletion, SceneEntityDeletionType,
    SceneUpdate, Timestamp, Vector3)

FRAME_ID = 'lidar'
HALF_EXTENT_M = 50.0
# Official nuScenes hues, with lane markings bright enough to read on LiDAR.
COLORS = {
    'lane_divider': (1.00, 0.92, 0.20),
    'road_divider': (0.80, 0.70, 0.95),
    'lane_connector': (0.20, 0.85, 0.45),
    'traffic_light': (1.00, 0.45, 0.10),
}
THICKNESS = {
    'lane_divider': 0.14,
    'road_divider': 0.16,
    'lane_connector': 0.12,
    'traffic_light': 0.35,
}
LINE_LAYERS = ('lane_divider', 'road_divider', 'traffic_light')
_IDENTITY = Pose(position=Vector3(x=0.0, y=0.0, z=0.0),
                 orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))


def map_location(dataroot, logfile):
    """nuScenes location name for a log filename such as ``n008-...``."""
    path = osp.join(dataroot, 'v1.0-mini', 'log.json')
    with open(path) as handle:
        logs = json.load(handle)
    for record in logs:
        if record.get('logfile') == logfile:
            return record['location']
    raise KeyError(f'no map location for log {logfile}')


def load_map(dataroot, location):
    return NuScenesMap(dataroot=dataroot, map_name=location)


def _lines_of(geom):
    if geom is None or geom.is_empty:
        return
    kind = geom.geom_type
    if kind == 'LineString':
        yield geom
    elif kind == 'LinearRing':
        yield LineString(geom.coords)
    elif kind == 'Polygon':
        yield LineString(geom.exterior.coords)
        for ring in geom.interiors:
            yield LineString(ring.coords)
    elif kind in ('MultiLineString', 'MultiPolygon', 'GeometryCollection'):
        for part in geom.geoms:
            yield from _lines_of(part)


def _to_lidar(geom, yaw_deg, origin_x, origin_y):
    from shapely import affinity
    moved = affinity.rotate(geom, -yaw_deg, origin=(origin_x, origin_y),
                            use_radians=False)
    return affinity.affine_transform(
        moved, [1.0, 0.0, 0.0, 1.0, -origin_x, -origin_y])


def _connector_centerlines(nusc_map, patch_box, yaw_deg):
    """Centerlines through an intersection, not the connector polygons."""
    explorer = nusc_map.explorer
    patch = explorer.get_patch_coord(patch_box, yaw_deg)
    origin_x, origin_y = patch_box[0], patch_box[1]
    tokens = []
    for record in nusc_map.lane_connector:
        polygon = nusc_map.extract_polygon(record['polygon_token'])
        if not polygon.is_valid or polygon.intersection(patch).is_empty:
            continue
        tokens.append(record['token'])
    if not tokens:
        return []
    paths = nusc_map.discretize_lanes(tokens, 0.5)
    geoms = []
    for token in tokens:
        raw = paths.get(token) or []
        if len(raw) < 2:
            continue
        clipped = LineString(np.asarray(raw)[:, :2]).intersection(patch)
        if clipped.is_empty:
            continue
        geoms.append(_to_lidar(clipped, yaw_deg, origin_x, origin_y))
    return geoms


def _local_geoms(nusc_map, layer, patch_box, yaw_deg):
    """Layer geometries already expressed with the LiDAR at the origin."""
    explorer = nusc_map.explorer
    if layer == 'lane_connector':
        return _connector_centerlines(nusc_map, patch_box, yaw_deg)
    if layer == 'traffic_light':
        patch = explorer.get_patch_coord(patch_box, yaw_deg)
        origin_x, origin_y = patch_box[0], patch_box[1]
        geoms = []
        for record in nusc_map.traffic_light:
            line = nusc_map.extract_line(record['line_token'])
            if line.is_empty:
                continue
            clipped = line.intersection(patch)
            if clipped.is_empty:
                continue
            geoms.append(_to_lidar(clipped, yaw_deg, origin_x, origin_y))
        return geoms
    found = explorer.get_map_geom(patch_box, yaw_deg, [layer])
    if not found:
        return []
    return found[0][1]


def local_map(nusc_map, lidar2global, half_extent=HALF_EXTENT_M):
    """Polylines per layer, in the LiDAR frame, within ``half_extent`` metres."""
    pose = np.asarray(lidar2global, dtype=float)
    origin_x, origin_y = float(pose[0, 3]), float(pose[1, 3])
    # Same yaw as nuScenes quaternion_yaw: heading of the local +x axis.
    ahead = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    yaw_deg = float(np.degrees(np.arctan2(ahead[1], ahead[0])))
    side = 2.0 * float(half_extent)
    patch_box = (origin_x, origin_y, side, side)
    layers = {}
    for name in ('lane_connector',) + LINE_LAYERS:
        lines = []
        for geom in _local_geoms(nusc_map, name, patch_box, yaw_deg):
            for line in _lines_of(geom):
                if line.length < 0.5 or len(line.coords) < 2:
                    continue
                simple = line.simplify(0.25, preserve_topology=False)
                coords = np.asarray(simple.coords, dtype=float)[:, :2]
                if len(coords) >= 2:
                    lines.append(np.round(coords, 3))
        layers[name] = lines
    return layers


def _stamp(seconds):
    ns = int(round(float(seconds) * 1e9))
    return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000)


def _lift(xy, sample, fallback):
    height, valid = sample(xy)
    z = np.where(valid, height, fallback)
    return [Point3(x=float(x), y=float(y), z=float(zz))
            for (x, y), zz in zip(xy, z)]


def make_hd_map(timestamp, layers, sample, fallback_z):
    """One scene update. ``sample(xy)`` returns ``(height, valid)``."""
    stamp = _stamp(timestamp)
    entities = []
    for name, lines in layers.items():
        if not lines:
            continue
        color = COLORS.get(name, (1.0, 1.0, 1.0))
        thickness = THICKNESS.get(name, 0.08)
        primitives = []
        for line in lines:
            points = _lift(line, sample, fallback_z)
            if len(points) < 2:
                continue
            primitives.append(LinePrimitive(
                type=LinePrimitiveLineType.LineStrip, pose=_IDENTITY,
                thickness=thickness, scale_invariant=False, points=points,
                color=Color(r=color[0], g=color[1], b=color[2], a=0.95),
                colors=[], indices=[]))
        if not primitives:
            continue
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=FRAME_ID, id=f'hd_{name}',
            metadata=[KeyValuePair(key='layer', value=name),
                      KeyValuePair(key='count', value=str(len(primitives)))],
            lines=primitives))
    return SceneUpdate(
        deletions=[SceneEntityDeletion(
            timestamp=stamp, type=SceneEntityDeletionType.All)],
        entities=entities)
