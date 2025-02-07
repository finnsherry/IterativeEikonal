# IterativeEikonal
This repository contains code to perform crossing-preserving geodesic tracking on the space of positions and orientations on the plane $\mathbb{M}_2 \coloneqq T\mathbb{R}^2 / \sim \cong \mathbb{R}^2 \times S^1$ and on the space of positions and orientations on the sphere $\mathbb{W}_2 \coloneqq TS^2 / \sim$, where $X \sim Y$ if there is a $\lambda > 0$ such that $X = \lambda Y$.

## Installation
The core functionality of this repository requires:
* `python>=3.10`
* `taichi==1.6`
* `numpy`
* `scipy`
* `matplotlib`
* `tqdm`
* `diplib`
* `pillow`
* `h5py`

To reproduce the experiments, one additionally needs:
* `jupyter`

Alternatively, one can create the required conda environment from `eikivp.yml`:
```
conda env create -f eikivp.yml
```
This creates a conda environment called `eikivp`.

Subsequently, one must install the code of this project as a package, by running:
```
pip install -e .
```

## Cite
If you use this code in your own work, please cite our paper:

<a id="1">[1]</a> van den Berg, N.J. and Sherry, F.M. and Berendschot, T.T.J.M. and Duits, R. "Crossing-Preserving Geodesic Tracking on Spherical Images." 10th International Conference on Scale Space and Variational Methods in Computer Vision (SSVM) (2025).
```
@inproceedings{Berg2025CrossingImages,
  author =       {van den Berg, Nicky J. and Sherry, Finn M. and Berendschot, Tos T.J.M. and Duits, Remco},
  title =        {{Crossing-Preserving Geodesic Tracking on Spherical Images}},
  booktitle =    {10th International Conference on Scale Space and Variational Methods in Computer Vision},
  publisher =    {Springer},
  year =         {2025},
  address =      {Totnes, United Kingdom},
  pages =        {},
  doi =          {},
  editor =       {}
}
```
