"""Publish SparseOcc occupancy to Foxglove the same way BEVFusion plays a scene.

Reads ``preds/<scene>.json`` and ``lidar/<sample_token>.npz`` written by
``sparseocc_infer.py``, caches every keyframe, then serves Foxglove playback
(pause, play, seek) on port 8767. BEVFusion on 8765 is unaffected.

    .venv/bin/python projects/OccupancyFusion/occupancy_foxglove.py --scene scene-0103

Then open https://app.foxglove.dev, Open connection, Foxglove WebSocket,
ws://localhost:8767. In a 3D panel, set the display frame to ``lidar``,
color mode RGBA, and enable ``/occupancy`` and ``/lidar``.
"""
import argparse
import json
import os
import os.path as osp
import sys
import threading
import time

import numpy as np

import foxglove
from foxglove.channels import CompressedImageChannel, PointCloudChannel
from foxglove.messages import (
    CompressedImage, PackedElementField, PackedElementFieldNumericType,
    PointCloud, Pose, Quaternion, Vector3)
from foxglove.websocket import (
    Capability, PlaybackCommand, PlaybackControlRequest, PlaybackState,
    PlaybackStatus, ServerListener)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from ego_to_lidar import OCC_CLASS_NAMES  # noqa: E402

FRAME_ID = 'lidar'
_IDENTITY = Pose(position=Vector3(x=0.0, y=0.0, z=0.0),
                 orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
CLASS_RGB = np.array([
    [0, 0, 0],
    [255, 120, 50],
    [255, 192, 203],
    [255, 255, 0],
    [0, 150, 245],
    [0, 255, 255],
    [200, 180, 0],
    [255, 0, 0],
    [255, 240, 150],
    [135, 60, 0],
    [160, 32, 240],
    [255, 0, 255],
    [175, 0, 75],
    [75, 0, 75],
    [150, 240, 80],
    [230, 230, 250],
    [0, 175, 0],
    [255, 255, 255],
], dtype=np.uint8)


def make_pointcloud(xyz, rgb):
    """Same XYZRGBA packing as ``foxglove_stream.make_pointcloud``."""
    rec = np.zeros(len(xyz), dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
    ], align=False))
    rec['x'] = xyz[:, 0]
    rec['y'] = xyz[:, 1]
    rec['z'] = xyz[:, 2]
    rec['r'] = rgb[:, 0]
    rec['g'] = rgb[:, 1]
    rec['b'] = rgb[:, 2]
    rec['a'] = 255
    numeric = PackedElementFieldNumericType
    return PointCloud(
        frame_id=FRAME_ID, pose=_IDENTITY, point_stride=16,
        fields=[
            PackedElementField(name='x', offset=0, type=numeric.Float32),
            PackedElementField(name='y', offset=4, type=numeric.Float32),
            PackedElementField(name='z', offset=8, type=numeric.Float32),
            PackedElementField(name='red', offset=12, type=numeric.Uint8),
            PackedElementField(name='green', offset=13, type=numeric.Uint8),
            PackedElementField(name='blue', offset=14, type=numeric.Uint8),
            PackedElementField(name='alpha', offset=15, type=numeric.Uint8),
        ],
        data=rec.tobytes())


def occupancy_cloud(npz_path):
    payload = np.load(npz_path)
    xyz = payload['xyz']
    labels = payload['label'].astype(np.int64)
    rgb = CLASS_RGB[np.clip(labels, 0, len(CLASS_RGB) - 1)]
    return make_pointcloud(xyz, rgb), len(xyz)


def lidar_cloud(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
    pts = pts[np.linalg.norm(pts[:, :2], axis=1) < 40]
    rgb = np.full((len(pts), 3), 180, dtype=np.uint8)
    return make_pointcloud(pts, rgb), len(pts)


def load_scene_frames(pred_path, lidar_dir, dataroot, version):
    from nuscenes.nuscenes import NuScenes
    with open(pred_path) as handle:
        scene = json.load(handle)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    frames = []
    for record in scene['frames']:
        sample = nusc.get('sample', record['sample_token'])
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cam_sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        lidar_path = osp.join(dataroot, lidar_sd['filename'])
        cam_path = osp.join(dataroot, cam_sd['filename'])
        occ_path = osp.join(lidar_dir, record['sample_token'] + '.npz')
        with open(cam_path, 'rb') as handle:
            jpeg = handle.read()
        occ, n_occ = occupancy_cloud(occ_path)
        cloud, n_pts = lidar_cloud(lidar_path)
        frames.append(dict(
            time=int(record['timestamp'] * 1e9),
            occupancy=occ,
            cloud=cloud,
            camera=CompressedImage(frame_id='CAM_FRONT', data=jpeg, format='jpeg'),
            points=n_pts,
            occupied=n_occ,
        ))
    return scene, frames


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


def _publish_frame(channels, frame):
    channels['occupancy'].log(frame['occupancy'], log_time=frame['time'])
    channels['cloud'].log(frame['cloud'], log_time=frame['time'])
    channels['image'].log(frame['camera'], log_time=frame['time'])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default='scene-0103')
    parser.add_argument('--pred-dir', default='demo/eval_out/occupancy/preds')
    parser.add_argument('--lidar-dir', default='demo/eval_out/occupancy/lidar')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--mcap', help='write one pass to this MCAP file and exit')
    args = parser.parse_args()

    scene, frames = load_scene_frames(
        osp.join(args.pred_dir, args.scene + '.json'),
        args.lidar_dir, args.dataroot, args.version)
    channels = dict(
        occupancy=PointCloudChannel('/occupancy'),
        cloud=PointCloudChannel('/lidar'),
        image=CompressedImageChannel('/CAM_FRONT'))

    if args.mcap:
        with foxglove.open_mcap(args.mcap, allow_overwrite=True):
            for frame in frames:
                _publish_frame(channels, frame)
        print('Wrote %d frames of %s to %s'
              % (len(frames), args.scene, args.mcap))
        return

    start_ns = frames[0]['time']
    end_ns = frames[-1]['time']
    player = _ScenePlayer()
    lock = threading.Lock()
    url = f'ws://{args.host}:{args.port}'
    server = foxglove.start_server(
        host=args.host,
        port=args.port,
        capabilities=[Capability.PlaybackControl, Capability.Time],
        playback_time_range=(start_ns, end_ns),
        server_listener=_PlaybackListener(player, lock),
    )
    with lock:
        player.frames = frames
        player.ready = True
        player.play()
    print('Foxglove WebSocket listening on %s' % url, flush=True)
    print('Open https://app.foxglove.dev, Open connection, Foxglove WebSocket,',
          flush=True)
    print(url, flush=True)
    print('Reconnect, then use Foxglove pause and play. Display frame "lidar".',
          flush=True)
    print('Enable /occupancy (voxels) and /lidar (sweep). Image: /CAM_FRONT.',
          flush=True)
    print('%s: %d frames, ~%d occupied voxels, classes %s'
          % (scene['scene'], len(frames),
             int(np.mean([frame['occupied'] for frame in frames])),
             ','.join(OCC_CLASS_NAMES[:-1])), flush=True)

    published = None
    last_status = None
    last_pub = 0.0
    try:
        while True:
            with lock:
                player.tick()
                index = player.index
                frame = player.frames[index]
                status = player.status
                current = player.current_time()
                speed = player.speed
            now = time.perf_counter()
            if index != published or now - last_pub >= 0.5:
                _publish_frame(channels, frame)
                server.broadcast_time(current)
                published = player.index
                last_pub = now
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
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
