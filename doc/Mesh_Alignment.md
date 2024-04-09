# Mesh Alignment Based on [evo](https://github.com/MichaelGrupp/evo/)

This repo is for aligning meshes via estimated poses and gt poses.

## How To Use

### First, transform camera extrinsics into camera-to-world matrices and save as kitti format.

Here we provide some tools like `blendmvs2colmap.py`, `colmap2kitti.py`

For examples:
```

# here we read /path/to/blendedmvs/scene/cams/*_cams.txt files 
# and save to /path/to/output as colmap txt files

python blendmvs2colmap.py --input /path/to/blendedmvs/scene --output /path/to/output
```

Convert colmap txt to KITTI pose file.

```
## read images.txt and save as poses.txt which contains camera-to-world matrices in KITTI format
## each raw has 12 parameters for SE(3) matrix, like below, 
## r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2

python colmap2kitti.py --input /path/to/colmap
```

### Second, compute transform matrix from colmap to gt using evo.
After step 1, we have got two `poses.txt`, from colamp and from gt, then we use [evo](https://github.com/MichaelGrupp/evo/) to compute transform matrix.


```
python aligment.py  --est /path/to/estmated/poses.txt \
                    --ref /path/to/gt/poses.txt \
                    -o /path/to/output-folder \
                    [-v]
                    [--mesh /path/to/mesh] 
```
It will save transform matrix in output/alignment_trnasfroms.txt in the format of 8 parameters,
which means qvec(4), tvec(3), s(1)

If `--mesh` is provided, it will transform provided mesh to gt world space, and save to the same 
folder with a prefix `aligned_`.

if `-v` or `--visualize` is provided, it will show the aligned trajectories and save the fig in output folder 

## Acknowledgements

We would like to acknowledge the following code that we have borrowed and used in this project:

- [evo](https://github.com/MichaelGrupp/evo/): This codebase has been instrumental in computing the transform matrix from colmap to ground truth poses.
- [colmap](https://github.com/colmap/colmap): We have utilized the `read_write_model.py` module from this codebase.
