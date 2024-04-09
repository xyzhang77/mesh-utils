import numpy as np
import trimesh

def filter_mesh_with_bbox(mesh, bbox_max, bbox_min):
    mask1 = (mesh.vertices <= bbox_max).all(1)
    mask2 = (mesh.vertices >= bbox_min).all(1)
    mask = np.logical_and(mask1, mask2)
    indices = np.zeros(len(mesh.vertices), dtype=int) - 1

    indices[mask] = np.arange(mask.sum())
    faces = indices[mesh.faces]
    faces_mask = (faces != -1).all(1)
    faces = faces[faces_mask]
    verts = mesh.vertices[mask]
    filtered_mesh = trimesh.Trimesh(vertices=verts, faces = faces)
    return filtered_mesh

