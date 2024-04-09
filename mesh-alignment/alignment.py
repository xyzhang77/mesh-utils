import copy
import logging

import evo.core.transformations as tr
from evo.tools import plot, file_interface, log

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--est', type=str, help='Path to estimated pose file', required=True)
    parser.add_argument('--ref', type=str, help='Path to gt pose file', required=True)
    parser.add_argument('--mesh', type=str, help='Path to mesh file', default=None)
    parser.add_argument('-o', '--output', type=str, help='Path to output path', required=True)
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize the alignment result')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    logger = logging.getLogger("evo")
    log.configure_logging(verbose=True)
    traj_ref = file_interface.read_kitti_poses_file(args.ref)
    traj_est = file_interface.read_kitti_poses_file(args.est)

    logger.info("\nUmeyama alignment with scaling")
    traj_est_aligned_scaled = copy.deepcopy(traj_est)
    r, t, s = traj_est_aligned_scaled.align(traj_ref, correct_scale=True)
    quat = tr.quaternion_from_matrix(r).tolist()
    tvec = t.tolist()
    trans_params = np.array(quat + tvec + [s])
    save_trans_path = os.path.join(args.output, 'aligment_transforms.txt')
    np.savetxt(save_trans_path, trans_params)

    if args.mesh is not None:
        os.system('python mesh_alignment.py -f {} -a {} '.format(args.mesh, save_trans_path))

    if args.visualize:
        fig = plt.figure(figsize=(8, 4))
        plot_mode = plot.PlotMode.xy

        ax = plot.prepare_axis(fig, plot_mode, subplot_arg=121)
        plot.traj(ax, plot_mode, traj_ref, '--', 'gray')
        plot.traj(ax, plot_mode, traj_est, '-', 'blue')
        fig.axes.append(ax)
        plt.title('not aligned')

        ax = plot.prepare_axis(fig, plot_mode, subplot_arg=122)
        plot.traj(ax, plot_mode, traj_ref, '--', 'gray')
        plot.traj(ax, plot_mode, traj_est_aligned_scaled, '-', 'blue')
        fig.axes.append(ax)
        plt.title('$\mathrm{Sim}(3)$ alignment')

        fig.tight_layout()
        plt.show()
        
        fig.savefig(os.path.join(args.output, 'alignment_result.png'))
