"""Standalone MapTRv2 inference on nuScenes-mini keyframes (no HD map input).

Runs inside the ``maptrv2:cu111`` container (see ``docker/Dockerfile``)::

    docker run --rm --gpus all --shm-size 8g \
        -v $PWD:/workspace -w /workspace maptrv2:cu111 \
        python projects/LaneMapping/maptr_infer.py

Inputs are the six cameras of each keyframe and their calibration, read with
the nuScenes devkit; the upstream MapTRv2 test pipeline (0.5x resize,
normalize, pad to 32) is reused unchanged. Map elements are written in the
LIDAR_TOP frame, the same ``lidar`` frame the Foxglove stream uses.

Outputs (``--out-dir``):
    preds/<scene>.json   per-frame elements with points, class and score,
                         plus the centerline graph
    vis/<scene>_<idx>.png  BEV over LiDAR, and predictions projected into
                         the six cameras
    summary.json         latency and GPU memory
    latency.csv          per-frame stage times with wall-clock stamps
"""
import argparse
import importlib
import json
import os
import os.path as osp
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from pyquaternion import Quaternion

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from centerline_graph import build_centerline_graph  # noqa: E402

MAPTR_ROOT = os.environ.get('MAPTR_ROOT', '/opt/MapTR')
CONFIGS = {
    'centerline': 'projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.py',
    'base': 'projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py',
}
CHECKPOINTS = {
    'centerline': 'checkpoints/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.pth',
    'base': 'checkpoints/maptrv2/maptrv2_nusc_r50_24ep.pth',
}
# Camera order of MapTR's nuScenes info converter.
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
COLORS = {'divider': '#ff9900', 'ped_crossing': '#1f77ff',
          'boundary': '#e8102e', 'centerline': '#22bb33'}
# Map elements are flat; project them at the road surface below LIDAR_TOP.
GROUND_Z = -1.8


def _pose(rec):
    m = np.eye(4)
    m[:3, :3] = Quaternion(rec['rotation']).rotation_matrix
    m[:3, 3] = rec['translation']
    return m


def build_input(nusc, sample):
    """MapTR ``get_data_info`` equivalent built from devkit records."""
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar2ego = _pose(nusc.get('calibrated_sensor',
                               lidar_sd['calibrated_sensor_token']))
    ego2global = _pose(nusc.get('ego_pose', lidar_sd['ego_pose_token']))
    lidar2global = ego2global @ lidar2ego

    filenames, lidar2img, camera2ego, intrinsics = [], [], [], []
    for cam in CAMERAS:
        sd = nusc.get('sample_data', sample['data'][cam])
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        cam2ego = _pose(cs)
        cam2global = _pose(nusc.get('ego_pose', sd['ego_pose_token'])) @ cam2ego
        lidar2cam = np.linalg.inv(cam2global) @ lidar2global
        viewpad = np.eye(4)
        viewpad[:3, :3] = cs['camera_intrinsic']
        filenames.append(osp.join(nusc.dataroot, sd['filename']))
        lidar2img.append(viewpad @ lidar2cam)
        camera2ego.append(cam2ego.astype(np.float32))
        intrinsics.append(viewpad.astype(np.float32))

    rotation = Quaternion(matrix=ego2global[:3, :3])
    yaw = np.degrees(rotation.yaw_pitch_roll[0]) % 360
    # Only the LSS encoder runs, which ignores can_bus; fill it as MapTR does.
    can_bus = np.zeros(18)
    can_bus[:3] = ego2global[:3, 3]
    can_bus[3:7] = rotation.elements
    can_bus[-2] = np.radians(yaw)
    can_bus[-1] = yaw
    return dict(
        sample_idx=sample['token'], scene_token=sample['scene_token'],
        timestamp=sample['timestamp'] / 1e6, img_filename=filenames,
        lidar2img=lidar2img, camera2ego=camera2ego,
        camera_intrinsics=intrinsics, lidar2ego=lidar2ego.astype(np.float32),
        lidar2global=lidar2global, can_bus=can_bus,
        pts_filename=osp.join(nusc.dataroot, lidar_sd['filename']))


def load_model(variant, checkpoint, fp16):
    cfg = Config.fromfile(osp.join(MAPTR_ROOT, CONFIGS[variant]))
    importlib.import_module('projects.mmdet3d_plugin')
    from mmdet3d.datasets.pipelines import Compose
    from mmdet3d.models import build_model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if fp16 and cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    ckpt = load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()
    return cfg, model, Compose(cfg.data.test.pipeline), ckpt


def decode(result, classes, score_thr):
    pts_bbox = result[0]['pts_bbox']
    scores = pts_bbox['scores_3d'].numpy()
    labels = pts_bbox['labels_3d'].numpy()
    pts = pts_bbox['pts_3d'].numpy()
    order = np.argsort(-scores)
    elements = []
    for k in order:
        if scores[k] < score_thr:
            continue
        elements.append(dict(
            id=len(elements), cls=classes[labels[k]],
            score=round(float(scores[k]), 4),
            points=np.round(pts[k], 3).tolist()))
    return elements


def graph_for(elements, score_thr):
    keep = [e for e in elements
            if e['cls'] == 'centerline' and e['score'] >= score_thr]
    graph = build_centerline_graph([e['points'] for e in keep])
    ids = [e['id'] for e in keep]
    for edge in graph['edges']:
        edge['lane'] = ids[edge['lane']]
    for key in ('successors', 'forks', 'merges'):
        graph[key] = [[ids[i], ids[j]] for i, j in graph[key]]
    graph['score_thr'] = score_thr
    return graph


class GpuMemorySampler(threading.Thread):
    """Peak device memory in use (MiB), from one ``nvidia-smi -lms`` process.

    Spawning nvidia-smi per sample costs enough CPU on this 4-core host to
    inflate the measured latency, so a single looping process is read instead.
    """
    QUERY = ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits', '-i', '0']

    def __init__(self, interval_ms=100):
        super().__init__(daemon=True)
        self.baseline = int(subprocess.run(
            self.QUERY, capture_output=True, text=True).stdout.strip() or 0)
        self.peak = self.baseline
        self._proc = subprocess.Popen(self.QUERY + ['-lms', str(interval_ms)],
                                      stdout=subprocess.PIPE, text=True)

    def run(self):
        for line in self._proc.stdout:
            if line.strip().isdigit():
                self.peak = max(self.peak, int(line))

    def stop(self):
        self._proc.terminate()
        self.join()


def _stats(values):
    v = np.asarray(values)
    return dict(mean=round(float(v.mean()), 2),
                p50=round(float(np.percentile(v, 50)), 2),
                p95=round(float(np.percentile(v, 95)), 2),
                max=round(float(v.max()), 2), n=int(len(v)))


def visualize(path, inp, elements, score_thr, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    keep = [e for e in elements if e['score'] >= score_thr]
    fig = plt.figure(figsize=(22, 9))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.9])
    order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for slot, cam in enumerate(order):
        i = CAMERAS.index(cam)
        ax = fig.add_subplot(grid[slot // 3, slot % 3])
        img = np.asarray(Image.open(inp['img_filename'][i]))
        ax.imshow(img)
        h, w = img.shape[:2]
        for e in keep:
            p = np.asarray(e['points'])
            p = np.c_[p, np.full(len(p), GROUND_Z), np.ones(len(p))]
            uvw = (inp['lidar2img'][i] @ p.T).T
            front = uvw[:, 2] > 1.0
            if front.sum() < 2:
                continue
            uv = uvw[front, :2] / uvw[front, 2:3]
            ax.plot(uv[:, 0], uv[:, 1], color=COLORS[e['cls']], lw=2)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(cam, fontsize=9)
        ax.axis('off')

    ax = fig.add_subplot(grid[:, 3])
    pts = np.fromfile(inp['pts_filename'], dtype=np.float32).reshape(-1, 5)
    near = (np.abs(pts[:, 0]) < 15) & (np.abs(pts[:, 1]) < 30) & (pts[:, 2] < 0.5)
    ax.scatter(pts[near, 0], pts[near, 1], s=0.3, c='#999999')
    for e in keep:
        p = np.asarray(e['points'])
        ax.plot(p[:, 0], p[:, 1], color=COLORS[e['cls']], lw=2,
                alpha=0.4 + 0.6 * e['score'])
        if e['cls'] == 'centerline':
            ax.annotate('', xy=p[-1], xytext=p[-2],
                        arrowprops=dict(arrowstyle='->', color=COLORS['centerline']))
    ax.add_patch(plt.Rectangle((-0.9, -2.0), 1.8, 4.5, color='k'))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.set_title('BEV (LIDAR_TOP, y forward), score >= %.2f' % score_thr, fontsize=9)
    for cls, color in COLORS.items():
        ax.plot([], [], color=color, label=cls)
    ax.legend(loc='lower right', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=sorted(CONFIGS), default='centerline')
    parser.add_argument('--checkpoint')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--scenes', nargs='*', help='scene names (default: all)')
    parser.add_argument('--out-dir', default='demo/eval_out/lane_perception')
    parser.add_argument('--export-thr', type=float, default=0.1)
    parser.add_argument('--graph-thr', type=float, default=0.3)
    parser.add_argument('--vis-thr', type=float, default=0.3)
    parser.add_argument('--vis-every', type=int, default=10,
                        help='save a figure every N frames per scene (0: none)')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--no-fp16', action='store_true')
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import mini_val

    out_dir = osp.join(args.out_dir, args.variant)
    os.makedirs(osp.join(out_dir, 'preds'), exist_ok=True)
    os.makedirs(osp.join(out_dir, 'vis'), exist_ok=True)

    sampler = GpuMemorySampler()
    sampler.start()
    checkpoint = args.checkpoint or CHECKPOINTS[args.variant]
    cfg, model, pipeline, ckpt = load_model(args.variant, checkpoint,
                                            fp16=not args.no_fp16)
    classes = list(cfg.map_classes)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scenes = sorted(nusc.scene, key=lambda s: s['name'])
    if args.scenes:
        scenes = [s for s in scenes if s['name'] in args.scenes]

    pre_ms, net_ms, post_ms = [], [], []
    frame_log = ['unix_time,sample_token,preprocess_ms,network_ms,postprocess_ms']
    frames_done = 0
    torch.cuda.reset_peak_memory_stats()
    for scene in scenes:
        split = 'val' if scene['name'] in mini_val else 'train'
        token, frames = scene['first_sample_token'], []
        while token:
            sample = nusc.get('sample', token)
            t0 = time.perf_counter()
            inp = build_input(nusc, sample)
            data = collate([pipeline(dict(inp))], samples_per_gpu=1)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            elements = decode(result, classes, args.export_thr)
            graph = (graph_for(elements, args.graph_thr)
                     if 'centerline' in classes else None)
            t3 = time.perf_counter()
            if frames_done >= args.warmup:
                pre_ms.append((t1 - t0) * 1e3)
                net_ms.append((t2 - t1) * 1e3)
                post_ms.append((t3 - t2) * 1e3)
                frame_log.append('%.3f,%s,%.2f,%.2f,%.2f' % (
                    time.time(), sample['token'], pre_ms[-1], net_ms[-1], post_ms[-1]))
            frames.append(dict(
                sample_token=sample['token'], timestamp=inp['timestamp'],
                frame='LIDAR_TOP', lidar2ego=inp['lidar2ego'].tolist(),
                elements=elements, centerline_graph=graph))
            index = len(frames) - 1
            if args.vis_every and index % args.vis_every == 0:
                name = '%s_%02d.png' % (scene['name'], index)
                visualize(osp.join(out_dir, 'vis', name), inp, elements, args.vis_thr,
                          '%s (%s) frame %d, MapTRv2 %s' % (scene['name'], split,
                                                            index, args.variant))
            frames_done += 1
            token = sample['next']
        with open(osp.join(out_dir, 'preds', scene['name'] + '.json'), 'w') as f:
            json.dump(dict(scene=scene['name'], split=split, model='MapTRv2-' + args.variant,
                           classes=classes, frames=frames), f)
        n = [sum(e['score'] >= args.vis_thr for e in fr['elements']) for fr in frames]
        print('%s (%s): %d frames, %.1f elements/frame at score >= %.2f'
              % (scene['name'], split, len(frames), np.mean(n), args.vis_thr), flush=True)
    sampler.stop()

    total = np.asarray(pre_ms) + np.asarray(net_ms) + np.asarray(post_ms)
    summary = dict(
        model='MapTRv2-' + args.variant, checkpoint=checkpoint,
        checkpoint_meta={k: str(v) for k, v in (ckpt.get('meta') or {}).items()
                         if k in ('mmdet_version', 'mmdet3d_version', 'epoch')},
        fp16=not args.no_fp16, gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__, frames=frames_done, warmup=args.warmup,
        latency_ms=dict(preprocess=_stats(pre_ms), network=_stats(net_ms),
                        postprocess=_stats(post_ms), total=_stats(total)),
        gpu_memory_mib=dict(
            torch_max_allocated=round(torch.cuda.max_memory_allocated() / 2**20),
            torch_max_reserved=round(torch.cuda.max_memory_reserved() / 2**20),
            device_baseline=sampler.baseline, device_peak=sampler.peak,
            device_peak_minus_baseline=sampler.peak - sampler.baseline))
    with open(osp.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(osp.join(out_dir, 'latency.csv'), 'w') as f:
        f.write('\n'.join(frame_log) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
