# Copyright (c) OpenMMLab. All rights reserved.
"""Stream one BEVFusion scene to Foxglove over a local WebSocket.

The Foxglove browser app connects to this process and renders the
camera-colored LiDAR cloud plus oriented detection boxes. Nothing is uploaded
to Foxglove's servers: the browser pulls the stream from this machine.

By default the cloud is the keyframe plus the nine previous LiDAR sweeps used
by BEVFusion, transformed into the keyframe LiDAR frame. The first sample of a
log has no past sweeps, so the stream steps forward to the next keyframe that
does and runs detection there.

Example:
    python projects/BEVFusion/demo/foxglove_stream.py

Then open https://app.foxglove.dev, choose Open connection, Foxglove WebSocket,
and accept ws://localhost:8765. In a 3D panel, set the display frame to
``lidar`` and enable ``/lidar`` and ``/detections``.
"""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colorize_lidar import (  # noqa: E402
    DEFAULT_LIDAR_RGB, colorize_lidar, load_images_rgb)
from motion_utils import classify_detections, motion_color  # noqa: E402
from multi_modality_demo import (  # noqa: E402
    BOX_EDGES, CLASS_COLORS, CLASS_NAMES, _colorize_points, _lidar2img_from_ann,
    lidar_box_corners, load_lidar_xyz)

import foxglove  # noqa: E402
from foxglove.channels import PointCloudChannel, SceneUpdateChannel  # noqa: E402
from foxglove.messages import (  # noqa: E402
    Color, CubePrimitive, LinePrimitive, LinePrimitiveLineType,
    PackedElementField, PackedElementFieldNumericType, Point3, PointCloud,
    Pose, Quaternion, SceneEntity, SceneUpdate, Vector3)


FRAME_ID = 'lidar'
CONFIG = (
    'projects/BEVFusion/configs/'
    'bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')
CHECKPOINT = 'checkpoints/bevfusion_lidar-cam.pth'
INFOS = 'data/nuscenes/nuscenes_infos_train.pkl'
_IDENTITY = Pose(
    position=Vector3(x=0.0, y=0.0, z=0.0),
    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
)


def _yaw_quaternion(yaw):
    half = 0.5 * float(yaw)
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def _resolve_images(ann_path, img_dir):
    img_paths, lidar2img = _lidar2img_from_ann(ann_path)
    resolved = []
    for path in img_paths:
        if not path:
            resolved.append(path)
            continue
        candidate = path
        if img_dir and not os.path.isabs(path):
            candidate = os.path.join(img_dir, os.path.basename(path))
        resolved.append(candidate if os.path.exists(candidate) else path)
    return resolved, lidar2img


def _load_bin_xyz(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 == 0:
        return raw.reshape(-1, 5)[:, :3]
    if raw.size % 4 == 0:
        return raw.reshape(-1, 4)[:, :3]
    return raw.reshape(-1, 3)


def _existing_lidar_path(path):
    if path and os.path.exists(path):
        return path
    base = os.path.basename(path or '')
    for candidate in (
            os.path.join('data/nuscenes/samples/LIDAR_TOP', base),
            os.path.join('data/nuscenes/sweeps/LIDAR_TOP', base),
            os.path.join('demo/data/nuscenes', base)):
        if os.path.exists(candidate):
            return candidate
    return path


def accumulate_sweeps(info, sweeps_num, remove_close=True):
    """Return keyframe plus past sweeps, all in the keyframe LiDAR frame.

    Uses the same ``lidar2sensor`` conversion as
    ``LoadPointsFromMultiSweeps``. ``sweeps_num`` is the number of past sweeps,
    so 9 produces the 10-frame cloud BEVFusion detects on.
    """
    key_path = _existing_lidar_path(info['lidar_points']['lidar_path'])
    clouds = [_load_bin_xyz(key_path)]
    sweeps = list(info.get('lidar_sweeps') or [])[:sweeps_num]
    used = 0
    for sweep in sweeps:
        lidar = sweep.get('lidar_points') or {}
        path = _existing_lidar_path(lidar.get('lidar_path', ''))
        transform = lidar.get('lidar2sensor')
        if not path or not os.path.exists(path) or transform is None:
            continue
        pts = _load_bin_xyz(path)
        if remove_close and len(pts):
            close = (np.abs(pts[:, 0]) < 1.0) & (np.abs(pts[:, 1]) < 1.0)
            pts = pts[~close]
        lidar2sensor = np.asarray(transform, dtype=np.float64)
        xyz = pts.astype(np.float64) @ lidar2sensor[:3, :3]
        xyz -= lidar2sensor[:3, 3]
        clouds.append(xyz.astype(np.float32))
        used += 1
    merged = np.concatenate(clouds, axis=0)
    print(f'Accumulated keyframe + {used} sweeps -> {len(merged):,} points',
          flush=True)
    return merged


def select_sweep_frame(data_list, lidar_name, sweeps_num):
    """Pick this frame, or the next frame in the log that has past sweeps."""
    log = lidar_name.split('__')[0]
    start = 0
    for i, info in enumerate(data_list):
        if os.path.basename(info['lidar_points']['lidar_path']) == lidar_name:
            start = i
            break
    chosen = None
    for i in range(start, len(data_list)):
        info = data_list[i]
        name = os.path.basename(info['lidar_points']['lidar_path'])
        if log not in name:
            break
        if len(info.get('lidar_sweeps') or []) >= sweeps_num:
            chosen = (i, info, name)
            break
    if chosen is None:
        raise FileNotFoundError(
            f'No frame in {log} has {sweeps_num} past LiDAR sweeps')
    index, info, name = chosen
    if name != lidar_name:
        print(f'{lidar_name} has no past sweeps; using the next keyframe '
              f'{name} (infos index {index})', flush=True)
    else:
        print(f'Using {name} (infos index {index})', flush=True)
    return index, info


def detect_frame(infos_path, index, config, checkpoint, score_thr):
    """Run BEVFusion and return detection dicts for ``make_scene``."""
    import mmcv
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet3d.apis import init_model
    from mmdet3d.registry import DATASETS

    cfg = Config.fromfile(config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = init_model(config, checkpoint, device='cuda:0')
    ann_file = os.path.basename(infos_path)
    ds_cfg = cfg.test_dataloader.dataset.copy()
    ds_cfg['ann_file'] = ann_file
    ds_cfg['test_mode'] = True
    ds_cfg['pipeline'] = cfg.test_dataloader.dataset.pipeline
    dataset = DATASETS.build(ds_cfg)
    data = dataset[index]
    with torch.no_grad():
        result = model.test_step(pseudo_collate([data]))[0]

    pred = result.pred_instances_3d
    keep = pred.scores_3d >= score_thr
    labels = pred.labels_3d[keep].detach().cpu().numpy().astype(int)
    bboxes = pred.bboxes_3d.tensor[keep].detach().cpu().numpy()
    names = list((model.dataset_meta or {}).get('classes', CLASS_NAMES))
    motions = classify_detections(labels, bboxes, names)
    boxes = []
    for label, bbox, motion in zip(labels, bboxes, motions):
        boxes.append({
            'label': int(label),
            'class_name': motion['class_name'],
            'motion': motion['motion'],
            'bbox_3d': bbox.tolist(),
        })
    img_paths = result.metainfo.get('img_path', [])
    if not isinstance(img_paths, list):
        img_paths = [img_paths]
    images = [
        mmcv.imconvert(mmcv.imread(path), 'bgr', 'rgb') for path in img_paths
    ]
    print(f'Detections: {len(boxes)} boxes above score {score_thr}', flush=True)
    return boxes, images, result.metainfo


def load_scene(pcd_path, detections_path, ann_path=None, img_dir=None,
               max_points=120000):
    """Return ``(xyz, rgb, boxes)`` in the LiDAR frame."""
    import json

    xyz = load_lidar_xyz(pcd_path)
    rgb = np.repeat(np.asarray(DEFAULT_LIDAR_RGB, dtype=np.uint8)[None],
                    len(xyz), axis=0)
    if ann_path:
        paths, mats = _resolve_images(ann_path, img_dir)
        images = load_images_rgb(paths)
        sampled, colored = colorize_lidar(xyz, images, mats)
        rgb = sampled
        keep = colored
        xyz = xyz[keep]
        rgb = rgb[keep]
        print(f'Colored {int(colored.sum()):,}/{len(colored):,} LiDAR points',
          flush=True)

    if len(xyz) > max_points:
        step = int(math.ceil(len(xyz) / max_points))
        xyz = xyz[::step]
        rgb = rgb[::step]
    with open(detections_path, 'r', encoding='utf-8') as f:
        boxes = json.load(f).get('boxes', [])
    print(f'Streaming {len(xyz):,} points and {len(boxes)} boxes', flush=True)
    return xyz, rgb, boxes


def load_sweep_scene(lidar_name, infos_path, sweeps_num, config, checkpoint,
                     score_thr, max_points):
    """Detect one keyframe and return its multi-sweep colored cloud."""
    with open(infos_path, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    index, info = select_sweep_frame(data_list, lidar_name, sweeps_num)
    xyz = accumulate_sweeps(info, sweeps_num)
    boxes, images, meta = detect_frame(
        infos_path, index, config, checkpoint, score_thr)
    rgb = _colorize_points(xyz, images, meta)
    if rgb is None:
        rgb = np.repeat(np.asarray(DEFAULT_LIDAR_RGB, dtype=np.uint8)[None],
                        len(xyz), axis=0)
    if len(xyz) > max_points:
        step = int(math.ceil(len(xyz) / max_points))
        xyz = xyz[::step]
        rgb = rgb[::step]
    print(f'Streaming {len(xyz):,} points and {len(boxes)} boxes', flush=True)
    return xyz, rgb, boxes


def make_pointcloud(xyz, rgb):
    """Pack XYZRGB into a Foxglove ``PointCloud``."""
    count = len(xyz)
    rec = np.zeros(
        count,
        dtype=np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
        ], align=False),
    )
    rec['x'] = xyz[:, 0]
    rec['y'] = xyz[:, 1]
    rec['z'] = xyz[:, 2]
    rec['r'] = rgb[:, 0]
    rec['g'] = rgb[:, 1]
    rec['b'] = rgb[:, 2]
    rec['a'] = 255
    numeric = PackedElementFieldNumericType
    fields = [
        PackedElementField(name='x', offset=0, type=numeric.Float32),
        PackedElementField(name='y', offset=4, type=numeric.Float32),
        PackedElementField(name='z', offset=8, type=numeric.Float32),
        PackedElementField(name='red', offset=12, type=numeric.Uint8),
        PackedElementField(name='green', offset=13, type=numeric.Uint8),
        PackedElementField(name='blue', offset=14, type=numeric.Uint8),
        PackedElementField(name='alpha', offset=15, type=numeric.Uint8),
    ]
    return PointCloud(
        frame_id=FRAME_ID,
        pose=_IDENTITY,
        point_stride=16,
        fields=fields,
        data=rec.tobytes(),
    )


def _box_color(box):
    label = int(box.get('label', 0))
    class_color = CLASS_COLORS[label % len(CLASS_COLORS)]
    return motion_color(box.get('motion', 'static'), class_color, 'motion')


def make_scene(boxes):
    """Build wireframes and translucent cubes for predicted boxes."""
    entities = []
    corners = lidar_box_corners(
        [box['bbox_3d'] for box in boxes]) if boxes else np.zeros((0, 8, 3))
    for i, box in enumerate(boxes):
        r, g, b = _box_color(box)
        bbox = np.asarray(box['bbox_3d'], dtype=np.float64)
        points = []
        for a, edge_b in BOX_EDGES:
            for idx in (a, edge_b):
                points.append(Point3(
                    x=float(corners[i, idx, 0]),
                    y=float(corners[i, idx, 1]),
                    z=float(corners[i, idx, 2]),
                ))
        name = box.get('class_name', 'object')
        motion = box.get('motion', 'static')
        entities.append(SceneEntity(
            frame_id=FRAME_ID,
            id=f'{i:03d}_{motion}_{name}',
            cubes=[CubePrimitive(
                pose=Pose(
                    position=Vector3(
                        x=float(bbox[0]),
                        y=float(bbox[1]),
                        z=float(bbox[2] + 0.5 * bbox[5]),
                    ),
                    orientation=_yaw_quaternion(bbox[6]),
                ),
                size=Vector3(
                    x=float(bbox[3]), y=float(bbox[4]), z=float(bbox[5])),
                color=Color(r=r, g=g, b=b, a=0.28),
            )],
            lines=[LinePrimitive(
                type=LinePrimitiveLineType.LineList,
                pose=_IDENTITY,
                thickness=0.06,
                scale_invariant=False,
                points=points,
                color=Color(r=r, g=g, b=b, a=1.0),
                colors=[],
                indices=[],
            )],
        ))
    return SceneUpdate(entities=entities)


def serve(pointcloud, scene, host, port, once=False):
    points = PointCloudChannel('/lidar')
    detections = SceneUpdateChannel('/detections')
    if once:
        with foxglove.open_mcap('/tmp/bevfusion_foxglove.mcap'):
            points.log(pointcloud)
            detections.log(scene)
        print('Wrote /tmp/bevfusion_foxglove.mcap')
        return

    foxglove.start_server(host=host, port=port)
    url = f'ws://{host}:{port}'
    print(f'Foxglove WebSocket listening on {url}', flush=True)
    print('Open https://app.foxglove.dev', flush=True)
    print('Choose Open connection -> Foxglove WebSocket ->', url, flush=True)
    print('Add a 3D panel, set its display frame to "lidar",', flush=True)
    print('and enable topics /lidar and /detections.', flush=True)
    print('Ctrl-C stops the stream. Coordinates stay in the LiDAR frame;',
          flush=True)
    print('z=0 is the sensor, about 1.7 m above the road.', flush=True)
    while True:
        points.log(pointcloud)
        detections.log(scene)
        time.sleep(1.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pcd',
        default=('demo/data/nuscenes/'
                 'n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951'
                 '.pcd.bin'))
    parser.add_argument(
        '--detections',
        default='demo/motion_out/colorized_pc/detections.json')
    parser.add_argument(
        '--ann',
        default=('demo/data/nuscenes/'
                 'n015-2018-07-24-11-22-45+0800.pkl'))
    parser.add_argument('--img-dir', default='demo/data/nuscenes')
    parser.add_argument(
        '--sweeps', type=int, default=9,
        help='past LiDAR sweeps to merge into the keyframe; 0 keeps one sweep')
    parser.add_argument('--infos', default=INFOS)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--score-thr', type=float, default=0.2)
    parser.add_argument('--max-points', type=int, default=350000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument(
        '--once', action='store_true',
        help='write one MCAP to /tmp and exit instead of serving')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sweeps > 0:
        xyz, rgb, boxes = load_sweep_scene(
            os.path.basename(args.pcd), args.infos, args.sweeps, args.config,
            args.checkpoint, args.score_thr, args.max_points)
    else:
        xyz, rgb, boxes = load_scene(
            args.pcd, args.detections, args.ann, args.img_dir, args.max_points)
    serve(
        make_pointcloud(xyz, rgb), make_scene(boxes), args.host, args.port,
        once=args.once)


if __name__ == '__main__':
    main()
