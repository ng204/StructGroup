"""
Export boundary_scores as XYZ+scalar text files for heatmap visualization.

Usage example:
    python tools/export_boundary_scores_xyz.py \\
        --prediction_path /home/ng204/xueshuang/SoftGroup/results/Area45-50new \\
        --room_name Area_50_Panax_49 \\
        --out /home/ng204/xueshuang/SoftGroup/vis_output/boundary_heatmap/Area_50_Panax_49_boundary.xyz
"""

import argparse
import os
import os.path as osp

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Export boundary_scores as XYZ+scalar file for StructGroup')
    parser.add_argument(
        '--prediction_path',
        type=str,
        required=True,
        help='Path to prediction results (same as --out in tools/test.py)')
    parser.add_argument(
        '--room_name',
        type=str,
        required=True,
        help='Scene / room name, e.g. Area_50_Panax_49')
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output XYZ file path')
    args = parser.parse_args()

    coords_file = osp.join(args.prediction_path, 'coords', args.room_name + '.npy')
    boundary_file = osp.join(args.prediction_path, 'boundary_scores', args.room_name + '.npy')

    assert osp.isfile(coords_file), f'coords file not found: {coords_file}'
    assert osp.isfile(boundary_file), f'boundary_scores file not found: {boundary_file}'

    xyz = np.load(coords_file)  # (N, 3)
    boundary_scores = np.load(boundary_file)  # (N,)

    if boundary_scores.ndim > 1:
        boundary_scores = boundary_scores.reshape(-1)

    assert xyz.shape[0] == boundary_scores.shape[0], \
        f'coords ({xyz.shape[0]}) and boundary_scores ({boundary_scores.shape[0]}) length mismatch'

    os.makedirs(osp.dirname(args.out), exist_ok=True)

    # Stack as [x, y, z, s]
    data = np.concatenate([xyz.astype(np.float32),
                           boundary_scores.astype(np.float32).reshape(-1, 1)],
                          axis=1)

    # Save as ASCII XYZ with scalar field (CloudCompare: 4th column as scalar)
    np.savetxt(args.out, data, fmt='%.6f')
    print(f'XYZ+scalar file saved to: {args.out}')


if __name__ == '__main__':
    main()


