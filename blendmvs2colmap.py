## Brrowed from https://github.com/MichaelGrupp/evo/blob/master/doc/alignment_demo.py

import argparse
import numpy as np
import glob
import os
from tqdm import tqdm
from read_write_model import write_images_text, write_cameras_text, Image, Camera, rotmat2qvec


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the BlendedMVS dataset', default="/mnt/nas_7/datasets/blendedmvs_highresolution/dataset_full_res_90-112/59817e4a1bd4b175e7038d19/59817e4a1bd4b175e7038d19/59817e4a1bd4b175e7038d19/")
    parser.add_argument('--output', type=str, help='Path to output path', default='/home/zhangxiyu/data/colmap-data/gt')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    cams = glob.glob(os.path.join(args.input, 'cams', '*_cam.txt'))
    cams = sorted(cams, key=lambda x: int(os.path.basename(x).split('_')[0]))

    intrinsics = []
    extrinsics = []
    cameras = []
    images = []
    height = 1536
    width = 2048

    for idx, cam in tqdm(enumerate(cams)):
        with open(cam, 'r') as f:
            lines = f.readlines()
            extri = np.array(sum(map(lambda x: x.strip().split(' '), lines[1:4]), start=[]), dtype=np.float32).reshape(3, 4)
            intri = np.array(sum(map(lambda x: x.strip().split(' '), lines[7:10]), start=[]), dtype=np.float32).reshape(3, 3)
        camera = Camera(id=idx + 1, model='PINHOLE', width=width, height=height, params=[intri[0, 0], intri[1, 1], intri[0, 2], intri[1, 2]])
        cameras.append(camera)

        image = Image(
            id=idx, qvec=rotmat2qvec(extri[:3, :3]), tvec=extri[:3, 3], camera_id=idx + 1, name=f'{idx:08d}.jpg',
            xys=[], point3D_ids=[]
        )
        images.append(image)
    cameras = dict(zip(range(1, 1 + len(cameras)), cameras))
    images = dict(zip(range(1, 1 + len(images)), images))
    os.makedirs(args.output, exist_ok=True)
    write_cameras_text(cameras, os.path.join(args.output, 'cameras.txt'))
    write_images_text(images, os.path.join(args.output, 'images.txt'))