import trimesh
import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)

from utils.read_write_model import qvec2rotmat

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input_mesh', type=str, required=True, help='The input mesh file')
    parser.add_argument('-a', '--alignment_params', type=str, required=True, help='The alignment parameters')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    mesh = trimesh.load(args.input_mesh)
    params = np.loadtxt(args.alignment_params).reshape(-1)
    r = qvec2rotmat(params[:4])
    t = params[4:7]
    s = params[7]
    scale_matrix = np.diag([s, s, s, 1])
    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = r
    trans_matrix[:3, 3] = t
    
    out_dir, mesh_name = os.path.split(args.input_mesh)
    mesh_name = "aligned_" + mesh_name
    mesh.apply_transform(trans_matrix @ scale_matrix)
    mesh.export(os.path.join(out_dir, mesh_name))