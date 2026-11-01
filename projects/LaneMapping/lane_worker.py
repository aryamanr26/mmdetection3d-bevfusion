"""MapTRv2 sidecar. Run this inside the maptrv2:cu111 container.

One inference at a time. A newer request replaces a queued one, and a finished
result is written with the sequence it actually processed so a late result
cannot be mistaken for a newer one.

    docker run --rm --gpus all --network host -u $(id -u):$(id -g) \\
        -e HOME=/tmp -v $PWD:/workspace -w /workspace maptrv2:cu111 \\
        python projects/LaneMapping/lane_worker.py
"""
import json
import os
import os.path as osp
import time

import maptr_infer as infer
from nuscenes.nuscenes import NuScenes


def _read(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _write(path, payload):
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-dir', default='demo/eval_out/lane_perception/live')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--variant', default='centerline')
    args = parser.parse_args()
    os.makedirs(args.live_dir, exist_ok=True)
    request_path = osp.join(args.live_dir, 'request.json')
    result_path = osp.join(args.live_dir, 'result.json')
    log_path = osp.join(args.live_dir, 'worker.jsonl')

    _cfg, model, pipeline, _ckpt = infer.load_model(
        args.variant, infer.CHECKPOINTS[args.variant], fp16=True)
    classes = list(_cfg.map_classes)
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    print('lane worker ready', flush=True)
    done = -1
    while True:
        request = _read(request_path)
        if request is None or int(request.get('seq', -1)) <= done:
            time.sleep(0.05)
            continue
        job = request
        started = time.perf_counter()
        try:
            sample = nusc.get('sample', job['sample_token'])
            prepared = infer.build_input(nusc, sample)
            from mmcv.parallel import collate
            data = collate([pipeline(dict(prepared))], samples_per_gpu=1)
            import torch
            torch.cuda.synchronize()
            net_started = time.perf_counter()
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            net_s = time.perf_counter() - net_started
            elements = infer.decode(result, classes, 0.1)
            payload = dict(
                seq=int(job['seq']), sample_token=job['sample_token'],
                timestamp=float(job['timestamp']),
                lidar2global=job['lidar2global'], elements=elements,
                network_ms=round(net_s * 1e3, 2),
                total_ms=round((time.perf_counter() - started) * 1e3, 2),
                wall_time=time.time())
        except Exception as exc:  # worker must keep running
            payload = dict(seq=int(job['seq']), error=str(exc),
                           timestamp=float(job.get('timestamp', 0)))
        _write(result_path, payload)
        with open(log_path, 'a') as handle:
            handle.write(json.dumps(payload if 'error' in payload else {
                'seq': payload['seq'], 'network_ms': payload['network_ms'],
                'total_ms': payload['total_ms'],
                'timestamp': payload['timestamp']}) + '\n')
        done = int(job['seq'])


if __name__ == '__main__':
    main()
