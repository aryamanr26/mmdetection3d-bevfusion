"""Publish MapTRv2 lane predictions to Foxglove, separately from BEVFusion.

Reads ``preds/<scene>.json`` written by ``maptr_infer.py`` and replays the
scene in a loop at its keyframe timestamps. It uses its own port (8766), so
the BEVFusion stream on 8765 is unaffected. Run it with the BEVFusion venv::

    .venv/bin/python projects/LaneMapping/lane_foxglove.py --scene scene-0103

Topics, all in the ``lidar`` (LIDAR_TOP) frame:
    /lane_map        map elements; color by class, alpha by score
    /lane_graph      centerline directions and successor links
    /lane_lidar      the keyframe LiDAR sweep, for context
    /lane_CAM_FRONT  the front camera image
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np

import foxglove
from foxglove.channels import (
    CompressedImageChannel, PointCloudChannel, SceneUpdateChannel)
from foxglove.messages import (
    ArrowPrimitive, Color, CompressedImage, KeyValuePair, LinePrimitive,
    LinePrimitiveLineType, PackedElementField, PackedElementFieldNumericType,
    Point3, PointCloud, Pose, Quaternion, SceneEntity, SceneEntityDeletion,
    SceneEntityDeletionType, SceneUpdate, TextPrimitive, Timestamp, Vector3)

FRAME_ID = 'lidar'
# Road surface below LIDAR_TOP; MapTRv2 predicts 2D map elements.
GROUND_Z = -1.8
COLORS = {'divider': (1.0, 0.6, 0.0), 'ped_crossing': (0.12, 0.47, 1.0),
          'boundary': (0.91, 0.06, 0.18), 'centerline': (0.13, 0.73, 0.2)}
THICKNESS = {'divider': 0.15, 'ped_crossing': 0.15, 'boundary': 0.2,
             'centerline': 0.1}
_IDENTITY = Pose(position=Vector3(x=0.0, y=0.0, z=0.0),
                 orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))


def _time(seconds):
    ns = int(round(seconds * 1e9))
    return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000)


def _points3(xy, z=GROUND_Z):
    return [Point3(x=float(x), y=float(y), z=z) for x, y in xy]


def _label_xyz(element):
    """A point for the class label. Missing heights do not fall back to a plane."""
    xyz = element.get('points_xyz')
    if xyz:
        valid = [row for row in xyz if row[2] is not None]
        if not valid:
            return None
        return valid[len(valid) // 2]
    x, y = element['points'][len(element['points']) // 2]
    return [x, y, GROUND_Z]


def _runs(element):
    """3D polylines. Missing elevation splits a line instead of inventing a height."""
    xyz = element.get('points_xyz')
    if xyz is None:
        return [_points3(element['points'])]
    runs, current = [], []
    for x, y, z in xyz:
        if z is None:
            if len(current) >= 2:
                runs.append(current)
            current = []
            continue
        current.append(Point3(x=float(x), y=float(y), z=float(z)))
    if len(current) >= 2:
        runs.append(current)
    return runs


def _yaw_quaternion(yaw):
    return Quaternion(x=0.0, y=0.0, z=float(np.sin(yaw / 2)),
                      w=float(np.cos(yaw / 2)))


def make_lane_map(frame, score_thr):
    stamp = _time(frame['timestamp'])
    entities = []
    for e in frame['elements']:
        if e['score'] < score_thr:
            continue
        r, g, b = COLORS[e['cls']]
        runs = _runs(e)
        label = _label_xyz(e)
        if not runs or label is None:
            continue
        coverage = e.get('elevation_valid')
        total = e.get('elevation_points')
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=FRAME_ID, id=f'{e["cls"]}_{e["id"]}',
            metadata=[KeyValuePair(key='class', value=e['cls']),
                      KeyValuePair(key='score', value=f'{e["score"]:.3f}')]
            + ([KeyValuePair(key='elevation_valid',
                             value=f'{coverage}/{total}')]
               if coverage is not None else []),
            lines=[LinePrimitive(
                type=LinePrimitiveLineType.LineStrip, pose=_IDENTITY,
                thickness=THICKNESS[e['cls']], scale_invariant=False,
                points=run,
                color=Color(r=r, g=g, b=b, a=0.35 + 0.65 * e['score']),
                colors=[], indices=[]) for run in runs],
            texts=[TextPrimitive(
                pose=Pose(position=Vector3(x=label[0], y=label[1], z=label[2] + 0.3),
                          orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
                billboard=True, font_size=10, scale_invariant=True,
                color=Color(r=r, g=g, b=b, a=1.0),
                text=f'{e["cls"][:4]} {e["score"]:.2f}')],
        ))
    if 'age_s' in frame and frame['age_s'] is not None:
        entities.append(SceneEntity(
            timestamp=stamp, frame_id=FRAME_ID, id='lane_status',
            metadata=[
                KeyValuePair(key='age_s', value=f'{frame["age_s"]:.3f}'),
                KeyValuePair(key='estimate_time',
                             value=str(frame.get('estimate_time'))),
                KeyValuePair(key='seq', value=str(frame.get('seq'))),
                KeyValuePair(key='stale', value=str(bool(frame.get('stale')))),
            ]))
    return SceneUpdate(
        deletions=[SceneEntityDeletion(timestamp=stamp,
                                       type=SceneEntityDeletionType.All)],
        entities=entities)


def make_lane_graph(frame):
    stamp = _time(frame['timestamp'])
    graph = frame.get('centerline_graph') or {}
    by_id = {e['id']: e for e in frame['elements']}
    arrows, links = [], []

    def _end_z(element, index, fallback):
        xyz = element.get('points_xyz')
        if not xyz:
            return fallback
        if xyz[index][2] is None:
            return None
        return float(xyz[index][2])

    for edge in graph.get('edges', []):
        element = by_id[edge['lane']]
        pts = np.asarray(element['points'])
        tail, head = pts[-3], pts[-1]
        d = head - tail
        z = _end_z(element, -1, GROUND_Z + 0.05)
        if z is None:
            continue
        arrows.append(ArrowPrimitive(
            pose=Pose(position=Vector3(x=float(tail[0]), y=float(tail[1]), z=z),
                      orientation=_yaw_quaternion(np.arctan2(d[1], d[0]))),
            shaft_length=float(max(np.linalg.norm(d) - 0.6, 0.2)),
            shaft_diameter=0.12, head_length=0.6, head_diameter=0.5,
            color=Color(r=0.13, g=0.73, b=0.2, a=1.0)))
    for i, j in graph.get('successors', []):
        a, b = by_id[i], by_id[j]
        za, zb = _end_z(a, -1, GROUND_Z + 0.1), _end_z(b, 0, GROUND_Z + 0.1)
        if za is None or zb is None:
            continue
        links.extend(_points3([a['points'][-1], b['points'][0]], (za + zb) / 2))
    entity = SceneEntity(
        timestamp=stamp, frame_id=FRAME_ID, id='lane_graph',
        metadata=[KeyValuePair(key=k, value=str(len(graph.get(k, []))))
                  for k in ('edges', 'successors', 'forks', 'merges')],
        arrows=arrows,
        lines=[LinePrimitive(
            type=LinePrimitiveLineType.LineList, pose=_IDENTITY,
            thickness=0.25, scale_invariant=False, points=links,
            color=Color(r=1.0, g=1.0, b=1.0, a=0.9), colors=[], indices=[])]
        if links else [])
    return SceneUpdate(
        deletions=[SceneEntityDeletion(timestamp=stamp,
                                       type=SceneEntityDeletionType.All)],
        entities=[entity])


def make_cloud(path, stamp):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
    pts = pts[np.linalg.norm(pts[:, :2], axis=1) < 40]
    numeric = PackedElementFieldNumericType
    return PointCloud(
        timestamp=_time(stamp), frame_id=FRAME_ID, pose=_IDENTITY,
        point_stride=12,
        fields=[PackedElementField(name=n, offset=4 * i, type=numeric.Float32)
                for i, n in enumerate('xyz')],
        data=np.ascontiguousarray(pts, dtype=np.float32).tobytes())


def load_predictions(pred_dir):
    """Map each nuScenes sample token to its saved lane frame."""
    index = {}
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith('.json'):
            continue
        with open(osp.join(pred_dir, name)) as handle:
            scene = json.load(handle)
        for frame in scene['frames']:
            index[frame['sample_token']] = frame
    return index


def clear_scene():
    """Drop every entity published earlier on a lane topic."""
    return SceneUpdate(
        deletions=[SceneEntityDeletion(type=SceneEntityDeletionType.All)],
        entities=[])


def load_frames(pred_path, dataroot, version):
    from nuscenes.nuscenes import NuScenes
    with open(pred_path) as f:
        scene = json.load(f)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    for frame in scene['frames']:
        sample = nusc.get('sample', frame['sample_token'])
        frame['lidar_path'] = osp.join(dataroot, nusc.get(
            'sample_data', sample['data']['LIDAR_TOP'])['filename'])
        frame['cam_path'] = osp.join(dataroot, nusc.get(
            'sample_data', sample['data']['CAM_FRONT'])['filename'])
    return scene


def build_messages(frame, score_thr):
    with open(frame['cam_path'], 'rb') as f:
        jpeg = f.read()
    return dict(
        lane_map=make_lane_map(frame, score_thr),
        lane_graph=make_lane_graph(frame),
        cloud=make_cloud(frame['lidar_path'], frame['timestamp']),
        image=CompressedImage(timestamp=_time(frame['timestamp']),
                              frame_id='CAM_FRONT', data=jpeg, format='jpeg'))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default='scene-0103')
    parser.add_argument('--pred-dir',
                        default='demo/eval_out/lane_perception/centerline/preds')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--mcap', help='write one pass to this MCAP file and exit')
    args = parser.parse_args()

    scene = load_frames(osp.join(args.pred_dir, args.scene + '.json'),
                        args.dataroot, args.version)
    frames = scene['frames']
    channels = dict(lane_map=SceneUpdateChannel('/lane_map'),
                    lane_graph=SceneUpdateChannel('/lane_graph'),
                    cloud=PointCloudChannel('/lane_lidar'),
                    image=CompressedImageChannel('/lane_CAM_FRONT'))
    messages = [build_messages(fr, args.score_thr) for fr in frames]

    def publish(i, log_time):
        for key, channel in channels.items():
            channel.log(messages[i][key], log_time=log_time)

    if args.mcap:
        with foxglove.open_mcap(args.mcap, allow_overwrite=True):
            for i, fr in enumerate(frames):
                publish(i, int(fr['timestamp'] * 1e9))
        print(f'Wrote {len(frames)} frames of {args.scene} to {args.mcap}')
        return

    server = foxglove.start_server(host=args.host, port=args.port)
    print(f'{scene["model"]} lanes for {args.scene} ({len(frames)} frames) on '
          f'ws://{args.host}:{args.port}; topics /lane_map /lane_graph '
          f'/lane_lidar /lane_CAM_FRONT, fixed frame "{FRAME_ID}"', flush=True)
    t0 = frames[0]['timestamp']
    try:
        while True:
            start = time.perf_counter()
            for i, fr in enumerate(frames):
                delay = (fr['timestamp'] - t0) - (time.perf_counter() - start)
                if delay > 0:
                    time.sleep(delay)
                publish(i, time.time_ns())
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
