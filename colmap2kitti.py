import argparse
import os
import numpy as np
from read_write_model import read_images_text, qvec2rotmat


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/zhangxiyu/data/colmap-data/gt', help='Path to the COLMAP model directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()

    images = read_images_text(os.path.join(args.input, 'images.txt'))
    sorted_images = sorted(images.values(), key=lambda x: x.id)
    poses = []
    for image in sorted_images:
        pose = np.eye(4)
        pose[:3, :3] = qvec2rotmat(image.qvec)
        pose[:3, 3] = image.tvec
        pose = np.linalg.inv(pose)
        poses.append(pose[:3].reshape(-1).tolist())
    
    with open(os.path.join(args.input, 'poses.txt'), 'w') as f:
        f.write('\n'.join(list(map(lambda x: ' '.join(list(map(str, x))), poses))))