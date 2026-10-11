# Copyright (c) OpenMMLab. All rights reserved.
"""Rebuild 3D PLY visualizations from a lidar file + detections.json.

Does not run the detector. Use this to turn an existing ``motion_out`` run
into colored point-cloud PLYs (gray lidar + motion-colored 3D boxes).

Example (gray cloud)::

    python projects/BEVFusion/demo/export_ply.py \\
      demo/data/nuscenes/n015-...__LIDAR_TOP__....pcd.bin \\
      demo/motion_out/detections.json \\
      --out-dir demo/motion_out

Example (camera-colorized cloud)::

    python projects/BEVFusion/demo/export_ply.py \\
      demo/data/nuscenes/n015-...__LIDAR_TOP__....pcd.bin \\
      demo/motion_out/detections.json \\
      --out-dir demo/motion_out \\
      --ann demo/data/nuscenes/n015-...pkl \\
      --img-dir demo/data/nuscenes
"""
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_modality_demo import export_ply_from_files  # noqa: E402


def parse_args():
    parser = ArgumentParser(
        description='Export BEVFusion detections onto a lidar PLY')
    parser.add_argument('pcd', help='LiDAR file (.pcd.bin / .bin)')
    parser.add_argument('detections', help='detections.json from the demo')
    parser.add_argument('--out-dir', default='demo', help='output directory')
    parser.add_argument(
        '--motion-filter',
        choices=['all', 'dynamic', 'static'],
        default='all',
        help='which motion class to include (all writes 3 PLYs)')
    parser.add_argument(
        '--color-by', choices=['motion', 'class'], default='motion')
    parser.add_argument(
        '--arrow-horizon',
        type=float,
        default=1.0,
        help='seconds of constant velocity represented by each arrow')
    parser.add_argument(
        '--no-arrows',
        action='store_true',
        help='disable velocity arrows on dynamic boxes')
    parser.add_argument(
        '--ann',
        default=None,
        help='BEVFusion ann pkl; enables camera colorization via lidar2img')
    parser.add_argument(
        '--img-dir',
        default=None,
        help='directory holding the camera images referenced by the ann pkl')
    return parser.parse_args()


def main():
    args = parse_args()
    paths = export_ply_from_files(
        args.pcd, args.detections, args.out_dir,
        motion_filter=args.motion_filter, color_by=args.color_by,
        ann_path=args.ann, img_dir=args.img_dir,
        arrow_horizon=args.arrow_horizon,
        draw_arrows=not args.no_arrows)
    for path in paths:
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
