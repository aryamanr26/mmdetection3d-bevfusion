# Copyright (c) OpenMMLab. All rights reserved.
"""Run BEVFusion on 5 distinct nuScenes logs and write colorized PLYs."""
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import mmcv
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_utils import classify_detections  # noqa: E402
from multi_modality_demo import CLASS_NAMES, save_offline_vis  # noqa: E402

SKIP_LOGS = {'n015-2018-07-24-11-22-45+0800'}  # already exported
WANTED = 5
SCORE_THR = 0.2
OUT_ROOT = 'demo/motion_out/scenes'
CONFIG = (
    'projects/BEVFusion/configs/'
    'bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')
CKPT = 'checkpoints/bevfusion_lidar-cam.pth'


def _log_name(info):
    return osp.basename(info['lidar_points']['lidar_path']).split('__')[0]


def _pick_samples(data_list):
    by = defaultdict(list)
    for i, info in enumerate(data_list):
        by[_log_name(info)].append(i)
    picked = []
    for log, idxs in sorted(by.items(), key=lambda kv: kv[0]):
        if log in SKIP_LOGS:
            continue
        picked.append((log, idxs[len(idxs) // 2]))
        if len(picked) == WANTED:
            break
    return picked


def _build_ds(cfg, ann_file):
    ds_cfg = cfg.test_dataloader.dataset.copy()
    ds_cfg['ann_file'] = ann_file
    ds_cfg['test_mode'] = True
    ds_cfg['pipeline'] = cfg.test_dataloader.dataset.pipeline
    return DATASETS.build(ds_cfg)


def main():
    cfg = Config.fromfile(CONFIG)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = init_model(CONFIG, CKPT, device='cuda:0')

    jobs = []
    for ann, tag in (('nuscenes_infos_val.pkl', 'val'),
                     ('nuscenes_infos_train.pkl', 'train')):
        with open(osp.join(cfg.data_root, ann), 'rb') as f:
            data_list = pickle.load(f)['data_list']
        picked = _pick_samples(data_list)
        if not picked:
            continue
        ds = _build_ds(cfg, ann)
        for log, idx in picked:
            if any(j['log'] == log for j in jobs):
                continue
            jobs.append(dict(log=log, idx=idx, ds=ds, split=tag,
                             token=data_list[idx]['token']))
        if len(jobs) >= WANTED:
            break
    jobs = jobs[:WANTED]
    print(f'Exporting {len(jobs)} scenes -> {OUT_ROOT}')

    for n, job in enumerate(jobs, 1):
        data = job['ds'][job['idx']]
        batch = pseudo_collate([data])
        with torch.no_grad():
            result = model.test_step(batch)[0]

        pred = result.pred_instances_3d
        keep = pred.scores_3d >= SCORE_THR
        labels = pred.labels_3d[keep].detach().cpu().numpy().astype(int)
        bboxes = pred.bboxes_3d.tensor[keep].detach().cpu().numpy()
        class_names = list((model.dataset_meta or {}).get('classes', CLASS_NAMES))
        motions = classify_detections(labels, bboxes, class_names)

        img_paths = result.metainfo.get('img_path', [])
        if not isinstance(img_paths, list):
            img_paths = [img_paths]
        imgs = []
        for p in img_paths:
            im = mmcv.imread(p)
            imgs.append(mmcv.imconvert(im, 'bgr', 'rgb'))

        stamp = osp.basename(
            job['ds'][job['idx']].get('lidar_path', '')
            or f"{job['token'][:8]}")
        out_dir = osp.join(OUT_ROOT, f"{n:02d}_{job['log']}")
        print(f'[{n}/{len(jobs)}] {job["split"]} idx={job["idx"]} {job["log"]}')
        saved = save_offline_vis(
            result, data, imgs, out_dir, SCORE_THR, class_names,
            motions=motions)
        for path in saved:
            if path.endswith('.ply'):
                print(f'  {path}')


if __name__ == '__main__':
    main()
