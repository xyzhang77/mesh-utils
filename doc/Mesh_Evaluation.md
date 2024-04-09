# Mesh Evaluation

We follow [NKSR](https://github.com/nv-tlabs/NKSR) to evaluate predicted mesh with gt mesh.

## How To Use

```
python eval.py --input /path/to/predicted_mesh.ply --gt /path/to/gt.ply [-f] [-s]
```

If `-f` is provided, it will filter predicted mesh with gt mesh bbox.
If `-s` is provided, it will save the intermediate results.

