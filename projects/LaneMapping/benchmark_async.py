"""BEVFusion latency while MapTRv2 runs as a 2 Hz sidecar.

Starts the MapTRv2 container, asks it for a lane update every 0.5 s, and in
the same interval runs the existing BEVFusion latency benchmark. BEVFusion is
not waiting on the sidecar. Lane age is how old the last accepted estimate is.

Stop Foxglove playback first so the GPU is shared only by these two jobs.
"""
import json
import os
import os.path as osp
import pickle
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lane_motion import LaneHold  # noqa: E402
from lane_schedule import LatestFrameScheduler  # noqa: E402

LIVE = 'demo/eval_out/lane_perception/live'
OUT = 'demo/eval_out/lane_perception/async_benchmark'


def _poses(infos_path):
    with open(infos_path, 'rb') as handle:
        rows = pickle.load(handle)['data_list']
    frames = []
    for info in rows:
        pose = (np.asarray(info['ego2global'], float)
                @ np.asarray(info['lidar_points']['lidar2ego'], float))
        frames.append(dict(
            sample_token=info['token'], timestamp=float(info['timestamp']),
            lidar2global=pose.tolist()))
    return frames


def _write(path, payload):
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def _read(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _cpu_sample():
    parts = open('/proc/stat').readline().split()[1:8]
    return np.asarray(parts, dtype=float)


def main():
    os.makedirs(LIVE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(['docker', 'rm', '-f', 'maptrv2-lane-worker'],
                   capture_output=True)
    for name in ('request.json', 'result.json', 'worker.jsonl'):
        path = osp.join(LIVE, name)
        if osp.exists(path):
            os.remove(path)
    log_path = osp.join(OUT, 'worker.log')
    worker_log = open(log_path, 'w')
    worker = subprocess.Popen(
        ['docker', 'run', '--name', 'maptrv2-lane-worker', '--rm', '--gpus', 'all',
         '--cpuset-cpus', '0',
         '--network', 'host', '-u', f'{os.getuid()}:{os.getgid()}',
         '-e', 'HOME=/tmp', '-v', f'{os.getcwd()}:/workspace', '-w', '/workspace',
         'maptrv2:cu111', 'python', 'projects/LaneMapping/lane_worker.py',
         '--live-dir', LIVE],
        stdout=worker_log, stderr=subprocess.STDOUT)
    ready = time.time() + 180
    while time.time() < ready:
        worker_log.flush()
        text = open(log_path).read()
        if 'ready' in text:
            print(text[-500:], flush=True)
            break
        if worker.poll() is not None:
            raise RuntimeError(text[-1000:] or 'lane worker exited')
        time.sleep(0.5)
    else:
        worker.terminate()
        raise RuntimeError('lane worker did not start')

    benchmark = subprocess.Popen(
        ['taskset', '-c', '1-3', sys.executable,
         'projects/BEVFusion/demo/benchmark_latency.py',
         '--out', osp.join(OUT, 'bevfusion_latency.json')],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    frames = _poses('data/nuscenes/nuscenes_infos_val.pkl')
    scheduler = LatestFrameScheduler(period_s=0.5)
    hold = LaneHold(stale_s=2.0)
    ages, updates = [], 0
    cpu0 = _cpu_sample()
    gpu_used = []
    last_gpu = 0.0
    started = time.perf_counter()
    index = 0
    seq = 0
    while benchmark.poll() is None:
        now = time.perf_counter()
        seq += 1
        frame = dict(frames[index % len(frames)], seq=seq)
        job, scheduled = scheduler.submit(frame, now)
        if scheduled:
            index += 1
        if job is not None:
            _write(osp.join(LIVE, 'request.json'), job)
        result = _read(osp.join(LIVE, 'result.json'))
        if result and 'elements' in result and scheduler.accept_result(result['seq']):
            nxt = scheduler.complete(result['seq'])
            if hold.update(result['seq'], result['wall_time'],
                           result['lidar2global'], result['elements']):
                updates += 1
            if nxt is not None:
                _write(osp.join(LIVE, 'request.json'), nxt)
        view = hold.query(np.eye(4), time.time())
        if view is not None:
            ages.append(view['age_s'])
        if time.time() - last_gpu > 1.0:
            last_gpu = time.time()
            text = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True).stdout.strip()
            if text:
                gpu_used.append([float(part) for part in text.split(',')])
        time.sleep(0.05)
    cpu1 = _cpu_sample()
    bench_out = benchmark.stdout.read()
    print(bench_out[-2000:], flush=True)
    worker_rows = []
    log_jsonl = osp.join(LIVE, 'worker.jsonl')
    if osp.exists(log_jsonl):
        for line in open(log_jsonl):
            row = json.loads(line)
            if 'network_ms' in row:
                worker_rows.append(row)

    def _stat(key):
        values = [row[key] for row in worker_rows]
        if not values:
            return None
        return dict(mean=round(float(np.mean(values)), 1),
                    p50=round(float(np.median(values)), 1),
                    p95=round(float(np.percentile(values, 95)), 1),
                    max=round(float(np.max(values)), 1),
                    n=len(values))
    busy = cpu1[:3].sum() - cpu0[:3].sum()
    total = cpu1.sum() - cpu0.sum()
    summary = dict(
        bevfusion=json.load(open(osp.join(OUT, 'bevfusion_latency.json'))),
        maptr_updates=updates,
        maptr_network_ms=_stat('network_ms'),
        maptr_total_ms=_stat('total_ms'),
        wall_s=round(time.perf_counter() - started, 1),
        update_hz=round(updates / max(time.perf_counter() - started, 1e-6), 3),
        lane_age_s=dict(
            mean=round(float(np.mean(ages)), 3) if ages else None,
            p95=round(float(np.percentile(ages, 95)), 3) if ages else None),
        cpu_busy_fraction=round(float(busy / max(total, 1e-6)), 3),
        gpu=dict(
            util_mean=round(float(np.mean([row[0] for row in gpu_used])), 1) if gpu_used else None,
            memory_peak_mib=round(float(np.max([row[1] for row in gpu_used])), 0) if gpu_used else None),
    )
    with open(osp.join(OUT, 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    subprocess.run(['docker', 'rm', '-f', 'maptrv2-lane-worker'],
                   capture_output=True)
    worker_log.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        subprocess.run(['docker', 'rm', '-f', 'maptrv2-lane-worker'],
                       capture_output=True)
