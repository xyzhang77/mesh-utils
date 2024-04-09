import trimesh
import numpy as np
import open3d as o3d
import json
import os
from utils import filter_mesh_with_bbox
from metrics import MeshEvaluator

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--gt', type=str)
    parser.add_argument('-f', '--filter_with_bbox', action='store_true')
    parser.add_argument('-s', '--store_intermediate', action='store_true')

    return parser.parse_args()

args = get_parser()
mesh = trimesh.load(args.input)
gt_mesh = trimesh.load(args.gt)

bbox_max = np.array(gt_mesh.vertices).max(0)
bbox_min = np.array(gt_mesh.vertices).min(0)

if args.filter_with_bbox:
    filtered_mesh = filter_mesh_with_bbox(mesh, bbox_max, bbox_min)
    if args.store_intermediate:
        filtered_mesh.export(os.path.join(os.path.dirname(args.input), 'filtered_mesh.ply'))

mesh_eval = MeshEvaluator()

metric = mesh_eval.eval_mesh(filtered_mesh.vertices, gt_mesh.vertices)
metric['bbox_max'] = bbox_max.tolist()
metric['bbox_min'] = bbox_min.tolist()
metric['bbox_length'] = (bbox_max - bbox_min).tolist()

metricname = os.path.basename(args.input).split('.')[0] + '_metric.json'
with open(os.path.join(os.path.dirname(args.input), metricname), 'w') as f:
    json.dump(metric, f)
