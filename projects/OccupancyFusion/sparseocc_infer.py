"""Standalone SparseOcc inference on nuScenes-mini keyframes (no Occ3D GT).

Runs inside the ``sparseocc:cu118`` container (see ``docker/Dockerfile``)::

    docker run --rm --gpus all --shm-size 8g \\
        -v $PWD:/workspace -w /workspace sparseocc:cu118 \\
        python projects/OccupancyFusion/sparseocc_infer.py

Inputs are the six cameras of each keyframe plus seven history sweeps, read
with the nuScenes devkit. The official test image pipeline is reused
(``BEVAug`` off, ``RandomTransformImage`` at the 256x704 test crop) and
``LoadOccGTFromFile`` is skipped. Predictions are written in the ego frame
and as occupied voxel centres in LIDAR_TOP, the same ``lidar`` frame the
Foxglove stream uses.

Outputs (``--out-dir``):
    ego/<sample_token>.npz     uint8 [200, 200, 16] ego-frame grid
    lidar/<sample_token>.npz   occupied centres in LIDAR_TOP
    preds/<scene>.json         per-frame tokens, ego2lidar, timings
    summary.json               latency and GPU memory
    latency.csv                per-frame stage times
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
import types

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from pyquaternion import Quaternion

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from ego_to_lidar import (  # noqa: E402
    FREE_CLASS, OCC_CLASS_NAMES, OCC_SIZE, dense_from_sparse,
    occupied_lidar_points)

SPARSEOCC_ROOT = os.environ.get('SPARSEOCC_ROOT', '/opt/SparseOcc')
CONFIG = 'configs/r50_nuimg_704x256_8f.py'
CHECKPOINT = 'checkpoints/sparseocc/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth'
DEFAULT_SCENES = ('scene-0103', 'scene-0916')
CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]


def _force_offline_sweeps():
    """Official single-GPU val uses online sweep loading (FPS path).

    That skips history JPEGs and relies on a sequential feature cache. This
    wrapper always loads the seven history sweeps so each keyframe is
    independent.
    """
    from loaders.pipelines.loading import LoadMultiViewImageFromMultiSweeps

    def _call(self, results):
        if self.sweeps_num == 0:
            return results
        return self.load_offline(results)

    LoadMultiViewImageFromMultiSweeps.__call__ = _call


def _pose(rec):
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rec['rotation']).rotation_matrix
    matrix[:3, 3] = rec['translation']
    return matrix


def _sensor2global(nusc, sample_data):
    """Same row-vector convention as SparseOcc ``gen_sweep_info.py``."""
    pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor2ego_t = np.array(calib['translation'])
    sensor2ego_r = Quaternion(calib['rotation']).rotation_matrix
    ego2global_t = np.array(pose['translation'])
    ego2global_r = Quaternion(pose['rotation']).rotation_matrix
    return {
        'data_path': osp.join(nusc.dataroot, sample_data['filename']),
        'sensor2global_rotation': sensor2ego_r.T @ ego2global_r.T,
        'sensor2global_translation': (
            sensor2ego_t @ ego2global_r.T + ego2global_t),
        'cam_intrinsic': np.array(calib['camera_intrinsic']),
        'timestamp': sample_data['timestamp'],
    }


def _sensor2lidar(nusc, sample_data, lidar_sd):
    """mmdet3d v1.0.0rc6 ``sensor2lidar_{rotation,translation}`` fields."""
    l2e = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    e2g = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    s2e = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    s2g = nusc.get('ego_pose', sample_data['ego_pose_token'])
    l2e_r = Quaternion(l2e['rotation']).rotation_matrix
    e2g_r = Quaternion(e2g['rotation']).rotation_matrix
    s2e_r = Quaternion(s2e['rotation']).rotation_matrix
    s2g_r = Quaternion(s2g['rotation']).rotation_matrix
    l2e_t = np.array(l2e['translation'])
    e2g_t = np.array(e2g['translation'])
    s2e_t = np.array(s2e['translation'])
    s2g_t = np.array(s2g['translation'])
    rotation = (s2e_r.T @ s2g_r.T) @ (
        np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T)
    translation = (s2e_t @ s2g_r.T + s2g_t) @ (
        np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T)
    translation -= e2g_t @ (
        np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T
    ) + l2e_t @ np.linalg.inv(l2e_r).T
    return rotation.T, translation


def _cam_record(nusc, sample_data, lidar_sd):
    record = _sensor2global(nusc, sample_data)
    rotation, translation = _sensor2lidar(nusc, sample_data, lidar_sd)
    record['sensor2lidar_rotation'] = rotation
    record['sensor2lidar_translation'] = translation
    return record


def build_scene_infos(nusc, scene):
    """SparseOcc-style infos for one scene, without Occ3D labels."""
    infos, token = [], scene['first_sample_token']
    while token:
        sample = nusc.get('sample', token)
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar2ego = nusc.get(
            'calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        ego2global = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        cams = {cam: _cam_record(
            nusc, nusc.get('sample_data', sample['data'][cam]), lidar_sd)
            for cam in CAMERAS}
        current = {cam: nusc.get('sample_data', sample['data'][cam])
                   for cam in CAMERAS}
        sweeps = []
        if sample['prev']:
            for _ in range(5):
                sweep = {}
                incomplete = False
                for cam in CAMERAS:
                    if current[cam]['prev'] == '':
                        incomplete = True
                        break
                    prev = nusc.get('sample_data', current[cam]['prev'])
                    sweep[cam] = _cam_record(nusc, prev, lidar_sd)
                    current[cam] = prev
                if incomplete:
                    break
                sweeps.append(sweep)
        infos.append(dict(
            token=sample['token'],
            timestamp=sample['timestamp'],
            scene_name=scene['name'],
            lidar_path=osp.join(nusc.dataroot, lidar_sd['filename']),
            lidar2ego_translation=list(lidar2ego['translation']),
            lidar2ego_rotation=list(lidar2ego['rotation']),
            ego2global_translation=list(ego2global['translation']),
            ego2global_rotation=list(ego2global['rotation']),
            cams=cams,
            sweeps=sweeps,
        ))
        token = sample['next']
    return infos


def collect_sweeps(infos, index, into_past=150):
    """Same walk as ``NuSceneOcc.collect_sweeps``, stopped at the scene edge."""
    previous, curr = [], index
    scene = infos[index]['scene_name']
    while len(previous) < into_past:
        sweeps = infos[curr]['sweeps']
        if not sweeps:
            break
        previous.extend(sweeps)
        if curr < 1 or infos[curr - 1]['scene_name'] != scene:
            break
        previous.append(infos[curr - 1]['cams'])
        curr -= 1
    return previous


def build_input(info, sweeps_prev):
    """``NuSceneOcc.get_data_info`` without ``occ_path``."""
    from nuscenes.utils.geometry_utils import transform_matrix

    ego2global_r = Quaternion(info['ego2global_rotation']).rotation_matrix
    lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    ego2lidar = transform_matrix(
        info['lidar2ego_translation'],
        Quaternion(info['lidar2ego_rotation']),
        inverse=True)
    img_paths, img_timestamps, lidar2img = [], [], []
    for cam in CAMERAS:
        cam_info = info['cams'][cam]
        img_paths.append(os.path.relpath(cam_info['data_path']))
        img_timestamps.append(cam_info['timestamp'] / 1e6)
        lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
        lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        viewpad = np.eye(4)
        intrinsic = cam_info['cam_intrinsic']
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img.append(viewpad @ lidar2cam_rt.T)
    return dict(
        sample_idx=info['token'],
        sweeps={'prev': sweeps_prev, 'next': []},
        timestamp=info['timestamp'] / 1e6,
        ego2global_translation=np.array(info['ego2global_translation']),
        ego2global_rotation=ego2global_r,
        lidar2ego_translation=np.array(info['lidar2ego_translation']),
        lidar2ego_rotation=lidar2ego_r,
        ego2lidar=[ego2lidar for _ in CAMERAS],
        img_filename=img_paths,
        img_timestamp=img_timestamps,
        lidar2img=lidar2img,
        lidar_path=info['lidar_path'],
    )


def infer_pipeline(cfg):
    """Official val pipeline minus Occ3D GT keys."""
    from mmdet3d.datasets.pipelines import Compose

    steps = []
    for step in cfg.data.val.pipeline:
        step = dict(step)
        if step['type'] == 'LoadOccGTFromFile':
            continue
        if step['type'] == 'DefaultFormatBundle3D':
            step['with_gt'] = False
            step['with_label'] = False
        if step['type'] == 'Collect3D':
            step['keys'] = ['img']
        steps.append(step)
    return Compose(steps)


def _import_sparseocc():
    """Register model + image pipelines without compiling RayIoU's DVR ops."""
    sys.path.insert(0, SPARSEOCC_ROOT)
    loaders = types.ModuleType('loaders')
    loaders.__path__ = [osp.join(SPARSEOCC_ROOT, 'loaders')]
    sys.modules['loaders'] = loaders
    importlib.import_module('models')
    importlib.import_module('loaders.pipelines')


def load_model(config_path, checkpoint):
    _import_sparseocc()
    _force_offline_sweeps()
    cfg_path = (config_path if osp.isabs(config_path)
                else osp.join(SPARSEOCC_ROOT, config_path))
    cfg = Config.fromfile(cfg_path)
    from mmdet3d.models import build_model
    model = build_model(cfg.model)
    ckpt = load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()
    return cfg, model, infer_pipeline(cfg), ckpt


class GpuMemorySampler(threading.Thread):
    """Peak device memory in use (MiB), from one ``nvidia-smi -lms`` process."""
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
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=round(float(values.mean()), 2),
                p50=round(float(np.percentile(values, 50)), 2),
                p95=round(float(np.percentile(values, 95)), 2),
                max=round(float(values.max()), 2), n=int(len(values)))


def gpu_memory_mib():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.strip().splitlines():
        pid, used = (part.strip() for part in line.split(','))
        if int(pid) == os.getpid():
            return int(used)
    return None


def decode_grid(result):
    sem_pred = np.asarray(result['sem_pred']).reshape(-1)
    occ_loc = np.asarray(result['occ_loc']).reshape(-1, 3)
    return dense_from_sparse(occ_loc, sem_pred, OCC_SIZE, FREE_CLASS)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--scenes', nargs='*', default=list(DEFAULT_SCENES))
    parser.add_argument('--out-dir', default='demo/eval_out/occupancy')
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import mini_val

    out_dir = args.out_dir
    os.makedirs(osp.join(out_dir, 'ego'), exist_ok=True)
    os.makedirs(osp.join(out_dir, 'lidar'), exist_ok=True)
    os.makedirs(osp.join(out_dir, 'preds'), exist_ok=True)

    sampler = GpuMemorySampler()
    sampler.start()
    checkpoint = osp.abspath(args.checkpoint)
    cfg, model, pipeline, ckpt = load_model(args.config, checkpoint)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scenes = [scene for scene in nusc.scene if scene['name'] in args.scenes]
    scenes.sort(key=lambda scene: scene['name'])

    pre_ms, net_ms, post_ms, full_ms = [], [], [], []
    frame_log = [
        'unix_time,sample_token,preprocess_ms,network_ms,postprocess_ms,full_ms']
    frames_done = 0
    torch.cuda.reset_peak_memory_stats()
    for scene in scenes:
        split = 'val' if scene['name'] in mini_val else 'train'
        infos = build_scene_infos(nusc, scene)
        frames = []
        for index, info in enumerate(infos):
            t0 = time.perf_counter()
            inp = build_input(info, collect_sweeps(infos, index))
            data = collate([pipeline(inp)], samples_per_gpu=1)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)[0]
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            grid = decode_grid(result)
            ego2lidar = inp['ego2lidar'][0]
            xyz, labels = occupied_lidar_points(grid, ego2lidar)
            np.savez_compressed(
                osp.join(out_dir, 'ego', info['token'] + '.npz'), pred=grid)
            np.savez_compressed(
                osp.join(out_dir, 'lidar', info['token'] + '.npz'),
                xyz=xyz, label=labels)
            t3 = time.perf_counter()
            if frames_done >= args.warmup:
                pre_ms.append((t1 - t0) * 1e3)
                net_ms.append((t2 - t1) * 1e3)
                post_ms.append((t3 - t2) * 1e3)
                full_ms.append((t3 - t0) * 1e3)
                frame_log.append('%.3f,%s,%.2f,%.2f,%.2f,%.2f' % (
                    time.time(), info['token'],
                    pre_ms[-1], net_ms[-1], post_ms[-1], full_ms[-1]))
            frames.append(dict(
                sample_token=info['token'],
                timestamp=inp['timestamp'],
                frame='LIDAR_TOP',
                ego2lidar=np.asarray(ego2lidar).tolist(),
                n_occupied=int(len(labels)),
                lidar_path=info['lidar_path'],
            ))
            frames_done += 1
        with open(osp.join(out_dir, 'preds', scene['name'] + '.json'), 'w') as handle:
            json.dump(dict(
                scene=scene['name'], split=split, model='SparseOcc-r50-8f',
                classes=list(OCC_CLASS_NAMES), frames=frames), handle)
        occupied = [frame['n_occupied'] for frame in frames]
        print('%s (%s): %d frames, %.0f occupied voxels/frame'
              % (scene['name'], split, len(frames), float(np.mean(occupied))),
              flush=True)
    sampler.stop()

    summary = dict(
        model='SparseOcc-r50-8f', checkpoint=checkpoint,
        checkpoint_meta={key: str(value)
                         for key, value in (ckpt.get('meta') or {}).items()
                         if key in ('epoch', 'mmdet_version', 'mmdet3d_version')},
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        frames=frames_done, warmup=args.warmup, scenes=args.scenes,
        latency_ms=dict(
            preprocess=_stats(pre_ms), network=_stats(net_ms),
            postprocess=_stats(post_ms), full=_stats(full_ms)),
        gpu_memory_mib=dict(
            torch_max_allocated=round(
                torch.cuda.max_memory_allocated() / 2**20, 1),
            torch_max_reserved=round(
                torch.cuda.max_memory_reserved() / 2**20, 1),
            process_nvidia_smi=gpu_memory_mib(),
            device_baseline=sampler.baseline, device_peak=sampler.peak,
            device_peak_minus_baseline=sampler.peak - sampler.baseline,
            device_total=round(
                torch.cuda.get_device_properties(0).total_memory / 2**20, 1)))
    with open(osp.join(out_dir, 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
    with open(osp.join(out_dir, 'latency.csv'), 'w') as handle:
        handle.write('\n'.join(frame_log) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
