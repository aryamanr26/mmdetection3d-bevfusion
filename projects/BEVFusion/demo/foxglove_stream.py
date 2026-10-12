# Copyright (c) OpenMMLab. All rights reserved.
"""Stream one BEVFusion scene to Foxglove over a local WebSocket.

The Foxglove browser app connects to this process and renders the
camera-colored LiDAR cloud plus oriented detection boxes. Nothing is uploaded
to Foxglove's servers: the browser pulls the stream from this machine.

By default the cloud is the keyframe plus the nine previous LiDAR sweeps used
by BEVFusion, transformed into the keyframe LiDAR frame. Those sweeps and the
six camera images are read once per sample: the viewer uses the LiDAR-frame
xyz and the original photos, and the network gets its own range-cropped and
resized copies. The first sample of a log has no past sweeps, so the stream
steps forward to the next keyframe that does and runs detection there.

Example:
    python projects/BEVFusion/demo/foxglove_stream.py

Then open https://app.foxglove.dev, choose Open connection, Foxglove WebSocket,
and accept ws://localhost:8765. In a 3D panel, set the display frame to
``lidar`` and enable ``/lidar`` and ``/detections``. Each camera is also
published as its original JPEG on ``/CAM_FRONT``, ``/CAM_FRONT_LEFT``, and
the other four ``/CAM_*`` topics.
"""
import argparse
import math
import os
import pickle
import sys
import threading
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
from foxglove.channels import (  # noqa: E402
    CompressedImageChannel, PointCloudChannel, SceneUpdateChannel)
from foxglove.websocket import (  # noqa: E402
    Capability, PlaybackCommand, PlaybackControlRequest, PlaybackState,
    PlaybackStatus, ServerListener)
from foxglove.messages import (  # noqa: E402
    Color, CompressedImage, CubePrimitive, LinePrimitive, LinePrimitiveLineType,
    PackedElementField, PackedElementFieldNumericType, Point3, PointCloud,
    Pose, Quaternion, SceneEntity, SceneEntityDeletion,
    SceneEntityDeletionType, SceneUpdate, TextPrimitive, Vector3)


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


# Disk loaders replaced by the shared sensor read. Later transforms still run
# on copies, because the model crops and resizes and the viewer does not.
_DISK_LOADERS = {
    'BEVLoadMultiViewImageFromFiles',
    'LoadMultiViewImageFromFiles',
    'LoadPointsFromFile',
    'LoadPointsFromMultiSweeps',
}


class _SensorCache:
    """Scene-level cache for raw lidar bins and static camera intrinsics.

    A keyframe bin is often listed again as a sweep of a later sample, so the
    raw array is kept until the scene has been built. Intrinsics do not change
    across a log. Per-frame extrinsics are not cached: ``lidar2cam`` moves
    with the vehicle. Decoded images are not kept, because each sample has its
    own photos and they are only needed while that frame is packed.
    """

    def __init__(self):
        self.points = {}
        self.intrinsics = {}
        self.tail = None
        self.num_views = 6
        self.num_ref_frames = -1

    def intrinsic(self, cam_id, cam2img):
        cached = self.intrinsics.get(cam_id)
        if cached is not None:
            return cached
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.asarray(cam2img, dtype=np.float32)
        self.intrinsics[cam_id] = mat
        return mat

    def bind_pipeline(self, dataset):
        if self.tail is not None:
            return
        tail = []
        for transform in dataset.pipeline.transforms:
            name = transform.__class__.__name__
            if name in _DISK_LOADERS:
                if hasattr(transform, 'num_views'):
                    self.num_views = transform.num_views
                    self.num_ref_frames = transform.num_ref_frames
                continue
            tail.append(transform)
        self.tail = tail

    def release_points(self):
        self.points.clear()


def _cached_points(path, cache):
    """Return one lidar bin as ``(N, 5)`` float32, reading it at most once."""
    resolved = os.path.normpath(_existing_lidar_path(path))
    if cache is not None and resolved in cache.points:
        return cache.points[resolved]
    raw = np.fromfile(resolved, dtype=np.float32)
    if raw.size % 5 != 0:
        raise ValueError(f'{resolved} is not packed as 5 float32 values per point')
    points = raw.reshape(-1, 5)
    if cache is not None:
        cache.points[resolved] = points
    return points


def _load_sweep_points(info, sweeps_num, cache=None, remove_close=True):
    """Load the keyframe and the nearest past sweeps one time.

    The result is ``(N, 5)`` in the keyframe LiDAR frame: xyz, intensity, and
    the time delta the model expects (0 on the keyframe). This is the raw
    model cloud, before the detection range crop. Column ``:3`` is the cloud
    to draw. ``sweeps_num`` is the number of past sweeps, so 9 is the 10-frame
    cloud BEVFusion detects on. The nearest sweeps are kept every time.
    """
    key = _cached_points(info['lidar_points']['lidar_path'], cache).copy()
    key[:, 4] = 0
    parts = [key]
    sweeps = list(info.get('lidar_sweeps') or [])[:sweeps_num]
    timestamp = info['timestamp']
    used = 0
    for sweep in sweeps:
        lidar = sweep.get('lidar_points') or {}
        path = lidar.get('lidar_path', '')
        transform = lidar.get('lidar2sensor')
        resolved = _existing_lidar_path(path) if path else ''
        if not resolved or not os.path.exists(resolved) or transform is None:
            continue
        pts = _cached_points(path, cache).copy()
        if remove_close and len(pts):
            close = (np.abs(pts[:, 0]) < 1.0) & (np.abs(pts[:, 1]) < 1.0)
            pts = pts[~close]
        # Same conversion as LoadPointsFromMultiSweeps: row vectors, then
        # subtract the translation. Applied before any display-only step.
        lidar2sensor = np.asarray(transform)
        pts[:, :3] = pts[:, :3] @ lidar2sensor[:3, :3]
        pts[:, :3] -= lidar2sensor[:3, 3]
        pts[:, 4] = timestamp - sweep['timestamp']
        parts.append(pts)
        used += 1
    merged = np.concatenate(parts, axis=0)
    print(f'Accumulated keyframe + {used} sweeps -> {len(merged):,} points',
          flush=True)
    return merged


def accumulate_sweeps(info, sweeps_num, remove_close=True):
    """Return keyframe plus past sweeps, all in the keyframe LiDAR frame."""
    merged = _load_sweep_points(
        info, sweeps_num, cache=None, remove_close=remove_close)
    return np.ascontiguousarray(merged[:, :3])


def _resolve_image(cam_id, path):
    if path and os.path.exists(path):
        return path
    base = os.path.basename(path or '')
    candidate = os.path.join('data/nuscenes/samples', cam_id, base)
    if os.path.exists(candidate):
        return candidate
    return path


def _load_cameras(info, cache):
    """Decode each camera once and build this frame's projection matrices.

    Images stay at the original resolution. The model later resizes a copy.
    Intrinsics come from the scene cache. ``lidar2cam`` is taken from this
    sample because it is not constant over the log.
    """
    import mmcv
    from mmengine.fileio import get

    images, paths, names, jpegs = [], [], [], []
    cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], []
    for cam_id, cam_item in info['images'].items():
        path = _resolve_image(cam_id, cam_item['img_path'])
        payload = bytes(get(path))
        img = mmcv.imfrombytes(
            payload, flag='color', backend='pillow', channel_order='rgb')
        images.append(img)
        paths.append(path)
        names.append(cam_id)
        jpegs.append(payload)
        intrinsic = cache.intrinsic(cam_id, cam_item['cam2img'])
        extrinsic = np.asarray(cam_item['lidar2cam'], dtype=np.float32)
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3:4]
        camera_to_lidar = np.eye(4)
        camera_to_lidar[:3, :3] = rotation.T
        camera_to_lidar[:3, 3:4] = -rotation.T @ translation.reshape(3, 1)
        cam2img.append(intrinsic)
        lidar2cam.append(extrinsic)
        cam2lidar.append(camera_to_lidar)
        lidar2img.append(intrinsic @ extrinsic)
    shapes = {image.shape for image in images}
    if len(shapes) > 1:
        pad_h = max(image.shape[0] for image in images)
        pad_w = max(image.shape[1] for image in images)
        images = [
            mmcv.impad(image, shape=(pad_h, pad_w), pad_val=0)
            for image in images
        ]
    return {
        'images': images,
        'paths': paths,
        'names': names,
        'jpeg': jpegs,
        'cam2img': np.stack(cam2img, axis=0),
        'lidar2cam': np.stack(lidar2cam, axis=0),
        'cam2lidar': np.stack(cam2lidar, axis=0),
        'lidar2img': np.stack(lidar2img, axis=0),
    }


def _pack_model_inputs(dataset, info, points, cameras, cache):
    """Run the non-loading transforms on copies of the shared sensor data.

    ``points`` stays in the keyframe LiDAR frame with intensity and time.
    The range filter crops the model copy only. Image augmentation resizes a
    float copy and records ``img_aug_matrix``; it does not replace the
    original ``lidar2img`` used to color the displayed cloud.
    """
    from mmdet3d.structures.points import LiDARPoints

    cache.bind_pipeline(dataset)
    results = info
    results['points'] = LiDARPoints(
        points.copy(), points_dim=points.shape[1])
    results['img'] = [image.astype(np.float32) for image in cameras['images']]
    results['img_path'] = list(cameras['paths'])
    results['filename'] = results['img_path']
    results['cam2img'] = cameras['cam2img'].copy()
    results['ori_cam2img'] = cameras['cam2img'].copy()
    results['lidar2cam'] = cameras['lidar2cam'].copy()
    results['cam2lidar'] = cameras['cam2lidar'].copy()
    results['lidar2img'] = cameras['lidar2img'].copy()
    shape = cameras['images'][0].shape[:2]
    results['img_shape'] = shape
    results['ori_shape'] = shape
    results['pad_shape'] = shape
    results['scale_factor'] = 1.0
    results['img_norm_cfg'] = dict(
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
        to_rgb=False)
    results['num_views'] = cache.num_views
    results['num_ref_frames'] = cache.num_ref_frames
    results['box_type_3d'] = dataset.box_type_3d
    results['box_mode_3d'] = dataset.box_mode_3d
    for transform in cache.tail:
        results = transform(results)
    return results


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


def gt_boxes_from_info(info):
    """Return nuScenes boxes converted to the LiDAR bottom-center convention."""
    from mmdet3d.structures import LiDARInstance3DBoxes

    rows, names = [], []
    for obj in info.get('instances', []):
        if not obj.get('bbox_3d_isvalid', True):
            continue
        label = int(obj.get('bbox_label_3d', -1))
        if label < 0:
            continue
        rows.append(obj['bbox_3d'][:7])
        names.append(CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label))
    if not rows:
        return []
    converted = LiDARInstance3DBoxes(
        np.asarray(rows, dtype=np.float32),
        box_dim=7,
        origin=(0.5, 0.5, 0.5),
    )
    boxes = []
    for name, row in zip(names, converted.tensor.detach().cpu().numpy()):
        boxes.append({'class_name': name, 'bbox_3d': row[:7].tolist()})
    print(f'Ground truth: {len(boxes)} boxes', flush=True)
    return boxes


def load_detector(infos_path, config, checkpoint):
    """Load BEVFusion and the nuScenes dataset once for a sequence."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet3d.apis import init_model
    from mmdet3d.registry import DATASETS

    cfg = Config.fromfile(config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = init_model(config, checkpoint, device='cuda:0')
    ds_cfg = cfg.test_dataloader.dataset.copy()
    ds_cfg['ann_file'] = os.path.basename(infos_path)
    ds_cfg['test_mode'] = True
    ds_cfg['pipeline'] = cfg.test_dataloader.dataset.pipeline
    dataset = DATASETS.build(ds_cfg)
    names = list((model.dataset_meta or {}).get('classes', CLASS_NAMES))
    return model, dataset, names


def infer_loaded(model, dataset, index, score_thr, class_names, sweeps_num=9,
                 cache=None):
    """Run one sample and return boxes, original images, calibration, and xyz.

    Sweeps and camera images are read once. The displayed cloud is the xyz
    columns of that sweep cloud. Point colors are sampled from those decoded
    images. The network receives a separate copy that is range-filtered, plus
    resized image copies.
    """
    import torch
    from mmengine.dataset import pseudo_collate

    if cache is None:
        cache = _SensorCache()
    info = dataset.get_data_info(index)
    points = _load_sweep_points(info, sweeps_num, cache)
    cameras = _load_cameras(info, cache)
    packed = _pack_model_inputs(dataset, info, points, cameras, cache)
    # Copy boxes to CPU inside inference mode. The .cpu() call waits until
    # the GPU forward has finished.
    with torch.inference_mode():
        result = model.test_step(pseudo_collate([packed]))[0]
        pred = result.pred_instances_3d
        keep = pred.scores_3d >= score_thr
        labels = pred.labels_3d[keep].detach().cpu().numpy().astype(int)
        bboxes = pred.bboxes_3d.tensor[keep].detach().cpu().numpy()
    motions = classify_detections(labels, bboxes, class_names)
    boxes = [{
        'label': int(label),
        'class_name': motion['class_name'],
        'motion': motion['motion'],
        'bbox_3d': bbox.tolist(),
    } for label, bbox, motion in zip(labels, bboxes, motions)]
    display_xyz = np.ascontiguousarray(points[:, :3])
    meta = {
        'lidar2img': cameras['lidar2img'],
        'img_path': cameras['paths'],
        'camera_names': cameras['names'],
        'camera_jpeg': cameras['jpeg'],
    }
    return boxes, cameras['images'], meta, display_xyz


def scene_rows(data_list, lidar_name):
    """Return one log's keyframes sorted by timestamp."""
    log = os.path.basename(lidar_name).split('__')[0]
    rows = []
    for index, info in enumerate(data_list):
        name = os.path.basename(info['lidar_points']['lidar_path'])
        if name.startswith(log):
            rows.append((index, info, name))
    rows.sort(key=lambda item: float(item[1]['timestamp']))
    if not rows:
        raise FileNotFoundError(f'No keyframes found for {log}')
    return log, rows


def _subsample(xyz, rgb, max_points):
    if len(xyz) <= max_points:
        return xyz, rgb
    step = int(math.ceil(len(xyz) / max_points))
    return xyz[::step], rgb[::step]


def build_playback_frame(model, dataset, class_names, info, index, sweeps_num,
                         score_thr, max_points, cache):
    """Infer one keyframe and pack synchronized Foxglove messages."""
    boxes, images, meta, xyz = infer_loaded(
        model, dataset, index, score_thr, class_names, sweeps_num, cache)
    rgb = _colorize_points(xyz, images, meta)
    if rgb is None:
        rgb = np.repeat(np.asarray(DEFAULT_LIDAR_RGB, dtype=np.uint8)[None],
                        len(xyz), axis=0)
    xyz, rgb = _subsample(xyz, rgb, max_points)
    gt_boxes = gt_boxes_from_info(info)
    return {
        'time': int(float(info['timestamp']) * 1_000_000_000),
        'cloud': make_pointcloud(xyz, rgb),
        'detections': make_scene(boxes, replace=True),
        'ground_truth': make_gt_scene(gt_boxes, replace=True),
        'cameras': make_camera_images(
            meta.get('camera_names', []), meta.get('camera_jpeg', [])),
        'predictions': len(boxes),
        'ground_truth_count': len(gt_boxes),
        'points': len(xyz),
    }


class _ScenePlayer:
    """Clock for one cached scene. Foxglove play, pause, and seek drive it."""

    def __init__(self):
        self.frames = []
        self.index = 0
        self.status = PlaybackStatus.Paused
        self.speed = 1.0
        self.ready = False
        self._wall = time.perf_counter()
        self._anchor = 0

    def current_time(self):
        if not self.frames:
            return 0
        return self.frames[self.index]['time']

    def play(self):
        if not self.frames:
            return
        if self.status == PlaybackStatus.Ended:
            self.index = 0
        self.status = PlaybackStatus.Playing
        self._wall = time.perf_counter()
        self._anchor = self.current_time()

    def pause(self):
        if self.status != PlaybackStatus.Ended:
            self.status = PlaybackStatus.Paused

    def set_speed(self, speed):
        if speed is None:
            return
        self.speed = float(speed) or 1.0
        self._wall = time.perf_counter()
        self._anchor = self.current_time()

    def seek(self, timestamp):
        if not self.frames:
            return
        target = int(timestamp)
        self.index = min(
            range(len(self.frames)),
            key=lambda i: abs(self.frames[i]['time'] - target),
        )
        if self.status == PlaybackStatus.Ended:
            self.status = PlaybackStatus.Paused
        self._wall = time.perf_counter()
        self._anchor = self.current_time()

    def tick(self):
        """Advance to the frame due at the current wall clock."""
        if self.status != PlaybackStatus.Playing or len(self.frames) < 2:
            return
        elapsed = (time.perf_counter() - self._wall) * max(self.speed, 0.0) * 1e9
        target = self._anchor + elapsed
        while (self.index + 1 < len(self.frames)
               and self.frames[self.index + 1]['time'] <= target):
            self.index += 1
        hold = int(0.5e9 / max(self.speed, 0.05))
        if self.index == len(self.frames) - 1 and target > self.frames[-1]['time'] + hold:
            self.status = PlaybackStatus.Ended


class _PlaybackListener(ServerListener):
    """Translate Foxglove's pause, play, and seek buttons into player state."""

    def __init__(self, player, lock):
        self._player = player
        self._lock = lock

    def on_playback_control_request(
            self, playback_control_request: PlaybackControlRequest):
        with self._lock:
            player = self._player
            did_seek = playback_control_request.seek_time is not None
            if did_seek:
                player.seek(playback_control_request.seek_time)
            player.set_speed(playback_control_request.playback_speed)
            command = playback_control_request.playback_command
            if command == PlaybackCommand.Play:
                player.play()
            elif command == PlaybackCommand.Pause:
                player.pause()
            return PlaybackState(
                status=player.status,
                current_time=player.current_time(),
                playback_speed=player.speed,
                did_seek=did_seek,
                request_id=playback_control_request.request_id,
            )


def _publish_frame(points, detections, ground_truth, camera_channels, frame):
    points.log(frame['cloud'], log_time=frame['time'])
    detections.log(frame['detections'], log_time=frame['time'])
    ground_truth.log(frame['ground_truth'], log_time=frame['time'])
    for name, message in frame.get('cameras', {}).items():
        channel = camera_channels.get(name)
        if channel is not None:
            channel.log(message, log_time=frame['time'])


def _camera_channels(names):
    """One compressed-image topic per camera, named ``/CAM_FRONT`` and so on."""
    return {name: CompressedImageChannel(f'/{name}') for name in names}


def play_scene(lidar_name, infos_path, sweeps_num, config, checkpoint,
               score_thr, max_points, host, port):
    """Infer a full scene, then play it under Foxglove's timeline controls."""
    with open(infos_path, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    log, rows = scene_rows(data_list, lidar_name)
    duration = float(rows[-1][1]['timestamp']) - float(rows[0][1]['timestamp'])
    start_ns = int(float(rows[0][1]['timestamp']) * 1_000_000_000)
    end_ns = int(float(rows[-1][1]['timestamp']) * 1_000_000_000)
    print(f'Scene {log}: {len(rows)} keyframes over {duration:.1f}s',
          flush=True)
    model, dataset, class_names = load_detector(
        infos_path, config, checkpoint)
    sensors = _SensorCache()
    points = PointCloudChannel('/lidar')
    detections = SceneUpdateChannel('/detections')
    ground_truth = SceneUpdateChannel('/ground_truth')
    camera_names = list(rows[0][1].get('images', {}).keys())
    camera_channels = _camera_channels(camera_names)
    player = _ScenePlayer()
    lock = threading.Lock()
    server = foxglove.start_server(
        host=host,
        port=port,
        capabilities=[Capability.PlaybackControl, Capability.Time],
        playback_time_range=(start_ns, end_ns),
        server_listener=_PlaybackListener(player, lock),
    )
    print(f'Foxglove WebSocket listening on ws://{host}:{port}', flush=True)
    print('Reconnect, then use Foxglove pause and play. The scene stops at',
          flush=True)
    print('the last frame instead of looping. Drag the timeline to seek.',
          flush=True)
    print('Image panels: ' + ', '.join(f'/{name}' for name in camera_names),
          flush=True)

    frames = []
    for number, (index, info, name) in enumerate(rows, 1):
        started = time.perf_counter()
        frame = build_playback_frame(
            model, dataset, class_names, info, index, sweeps_num, score_thr,
            max_points, sensors)
        frames.append(frame)
        print(f'[{number}/{len(rows)}] {name} {frame["points"]:,} pts, '
              f'{frame["predictions"]} pred, {frame["ground_truth_count"]} GT, '
              f'{time.perf_counter() - started:.1f}s', flush=True)
    sensors.release_points()

    with lock:
        player.frames = frames
        player.ready = True
        player.play()
    print(f'Playing {len(frames)} frames. Pause and play from the Foxglove '
          f'timeline.', flush=True)

    published = None
    last_status = None
    while True:
        with lock:
            player.tick()
            index = player.index
            frame = player.frames[index]
            status = player.status
            current = player.current_time()
            speed = player.speed
        if index != published:
            _publish_frame(
                points, detections, ground_truth, camera_channels, frame)
            server.broadcast_time(current)
            published = player.index
        if status != last_status:
            server.broadcast_playback_state(PlaybackState(
                status=status,
                current_time=current,
                playback_speed=speed,
                did_seek=False,
                request_id=None,
            ))
            last_status = status
        time.sleep(0.02)


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
    cameras = _jpeg_messages(_camera_jpegs(ann_path, img_dir) if ann_path else [])
    return xyz, rgb, boxes, cameras


def load_sweep_scene(lidar_name, infos_path, sweeps_num, config, checkpoint,
                     score_thr, max_points):
    """Detect one keyframe and return its multi-sweep colored cloud."""
    with open(infos_path, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    index, info = select_sweep_frame(data_list, lidar_name, sweeps_num)
    model, dataset, class_names = load_detector(
        infos_path, config, checkpoint)
    boxes, images, meta, xyz = infer_loaded(
        model, dataset, index, score_thr, class_names, sweeps_num)
    print(f'Detections: {len(boxes)} boxes above score {score_thr}', flush=True)
    rgb = _colorize_points(xyz, images, meta)
    if rgb is None:
        rgb = np.repeat(np.asarray(DEFAULT_LIDAR_RGB, dtype=np.uint8)[None],
                        len(xyz), axis=0)
    if len(xyz) > max_points:
        step = int(math.ceil(len(xyz) / max_points))
        xyz = xyz[::step]
        rgb = rgb[::step]
    gt_boxes = gt_boxes_from_info(info)
    cameras = make_camera_images(
        meta.get('camera_names', []), meta.get('camera_jpeg', []))
    print(f'Streaming {len(xyz):,} points, {len(boxes)} predictions, '
          f'{len(gt_boxes)} ground-truth boxes, and {len(cameras)} cameras',
          flush=True)
    return xyz, rgb, boxes, gt_boxes, cameras


def make_camera_images(names, jpegs):
    """Pack the original JPEG bytes for each camera. No re-encode."""
    images = {}
    for name, payload in zip(names, jpegs):
        if not payload:
            continue
        images[name] = CompressedImage(
            frame_id=name,
            data=payload,
            format='jpeg',
        )
    return images


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


def make_scene(boxes, replace=False):
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
    return SceneUpdate(
        deletions=([SceneEntityDeletion(type=SceneEntityDeletionType.All)]
                   if replace else []),
        entities=entities,
    )


def make_gt_scene(boxes, replace=False):
    """White wireframes and class names for ground-truth boxes."""
    entities = []
    corners = lidar_box_corners(
        [box['bbox_3d'] for box in boxes]) if boxes else np.zeros((0, 8, 3))
    white = Color(r=1.0, g=1.0, b=1.0, a=1.0)
    for i, box in enumerate(boxes):
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
        entities.append(SceneEntity(
            frame_id=FRAME_ID,
            id=f'gt_{i:03d}_{name}',
            lines=[LinePrimitive(
                type=LinePrimitiveLineType.LineList,
                pose=_IDENTITY,
                thickness=0.08,
                scale_invariant=False,
                points=points,
                color=white,
                colors=[],
                indices=[],
            )],
            texts=[TextPrimitive(
                pose=Pose(
                    position=Vector3(
                        x=float(bbox[0]),
                        y=float(bbox[1]),
                        z=float(bbox[2] + bbox[5] + 0.4),
                    ),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                billboard=True,
                font_size=16,
                scale_invariant=True,
                color=white,
                text=f'GT {name}',
            )],
        ))
    return SceneUpdate(
        deletions=([SceneEntityDeletion(type=SceneEntityDeletionType.All)]
                   if replace else []),
        entities=entities,
    )


def _camera_jpegs(ann_path, img_dir):
    """Return ``(name, jpeg_bytes)`` for each camera in an annotation pkl."""
    with open(ann_path, 'rb') as f:
        images = pickle.load(f)['data_list'][0].get('images', {})
    pairs = []
    for name, cam_item in images.items():
        path = _resolve_image(name, cam_item.get('img_path', ''))
        if img_dir:
            candidate = os.path.join(
                img_dir, os.path.basename(cam_item.get('img_path') or ''))
            if os.path.exists(candidate):
                path = candidate
        if path and os.path.exists(path):
            with open(path, 'rb') as handle:
                pairs.append((name, handle.read()))
    return pairs


def _jpeg_messages(pairs):
    return make_camera_images(
        [name for name, _payload in pairs],
        [payload for _name, payload in pairs])


def serve(pointcloud, scene, host, port, gt_scene=None, once=False,
          cameras=None):
    points = PointCloudChannel('/lidar')
    detections = SceneUpdateChannel('/detections')
    ground_truth = SceneUpdateChannel('/ground_truth')
    cameras = cameras or {}
    camera_channels = _camera_channels(cameras)
    if gt_scene is None:
        gt_scene = SceneUpdate(entities=[])
    if once:
        with foxglove.open_mcap('/tmp/bevfusion_foxglove.mcap'):
            points.log(pointcloud)
            detections.log(scene)
            ground_truth.log(gt_scene)
            for name, message in cameras.items():
                camera_channels[name].log(message)
        print('Wrote /tmp/bevfusion_foxglove.mcap')
        return

    foxglove.start_server(host=host, port=port)
    url = f'ws://{host}:{port}'
    print(f'Foxglove WebSocket listening on {url}', flush=True)
    print('Open https://app.foxglove.dev', flush=True)
    print('Choose Open connection -> Foxglove WebSocket ->', url, flush=True)
    print('Add a 3D panel, set its display frame to "lidar",', flush=True)
    print('and enable /lidar, /detections, and /ground_truth.', flush=True)
    if camera_channels:
        print('Image panels: ' + ', '.join(
            f'/{name}' for name in camera_channels), flush=True)
    print('Predictions stay colored. Ground truth is white wireframes',
          flush=True)
    print('labeled "GT <class>".', flush=True)
    print('Ctrl-C stops the stream. Coordinates stay in the LiDAR frame;',
          flush=True)
    print('z=0 is the sensor, about 1.7 m above the road.', flush=True)
    while True:
        points.log(pointcloud)
        detections.log(scene)
        ground_truth.log(gt_scene)
        for name, message in cameras.items():
            camera_channels[name].log(message)
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
        '--play-scene', action='store_true',
        help='infer every keyframe in this log, then replay it in order')
    parser.add_argument(
        '--once', action='store_true',
        help='write one MCAP to /tmp and exit instead of serving')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.play_scene:
        play_scene(
            os.path.basename(args.pcd), args.infos, args.sweeps, args.config,
            args.checkpoint, args.score_thr, args.max_points, args.host,
            args.port)
        return
    gt_boxes = []
    cameras = {}
    if args.sweeps > 0:
        xyz, rgb, boxes, gt_boxes, cameras = load_sweep_scene(
            os.path.basename(args.pcd), args.infos, args.sweeps, args.config,
            args.checkpoint, args.score_thr, args.max_points)
    else:
        xyz, rgb, boxes, cameras = load_scene(
            args.pcd, args.detections, args.ann, args.img_dir, args.max_points)
    serve(
        make_pointcloud(xyz, rgb), make_scene(boxes), args.host, args.port,
        gt_scene=make_gt_scene(gt_boxes), once=args.once, cameras=cameras)


if __name__ == '__main__':
    main()
