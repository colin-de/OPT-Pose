<div align="center">

# Object Pose Transformer

### Unifying Unseen Object Pose Estimation

**CVPR 2026 · Category-agnostic Universal Absolute and Relative Pose Estimator**

Absolute and relative object pose.<br>
One transformer. Flexible views. No CAD models.

<p align="center">
  <a href="https://colin-de.github.io/" target="_blank" rel="noopener noreferrer">Weihang Li</a><sup>1,2</sup>,
  Lorenzo Garattoni<sup>3</sup>,
  Fabien Despinoy<sup>3</sup>,
  Nassir Navab<sup>1,2</sup>,
  <a href="https://www.asg.ed.tum.de/pf/team/benjamin-busam/" target="_blank" rel="noopener noreferrer">Benjamin Busam</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Technical University of Munich &nbsp;&nbsp;
  <sup>2</sup>Munich Center for Machine Learning &nbsp;&nbsp;
  <sup>3</sup>Toyota Motor Europe    &nbsp;&nbsp;
</p>

<p>
  <a href="https://arxiv.org/abs/2603.23370"><img src="https://img.shields.io/badge/arXiv-2603.23370-182832?style=flat-square" alt="arXiv paper"></a>
  <a href="https://colin-de.github.io/OPT-Pose/"><img src="https://img.shields.io/badge/Project-Page-147d81?style=flat-square" alt="Project page"></a>
  <a href="https://github.com/colin-de/OPT-Pose"><img src="https://img.shields.io/badge/GitHub-Code-182832?style=flat-square&amp;logo=github" alt="GitHub code"></a>
  <a href="https://huggingface.co/colin1842/OPT-Pose"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-e5c568?style=flat-square" alt="Hugging Face weights"></a>
</p>

[Overview](#overview) · [Setup](#setup) · [Checkpoints](#checkpoints) · [Datasets](#datasets) · [Inference](#inference) · [Training](#training) · [Citation](#citation)

</div>

---

<p align="center">
  <img src="assets/teaser.jpg" alt="OPT-Pose predicts camera geometry and canonical object coordinates for absolute and relative pose estimation" width="95%">
</p>

OPT-Pose utilizes a feed-forward transformer to predict point map, depth, NOCS, and camera parameters. Existing category-level methods predict canonical absolute 9-DoF SA(3) poses (equivalent to Depth + NOCS), but require predefined category labels and calibrated cameras. Relative pose methods align unseen objects across views in 6-DoF SE(3) (equivalent to Pointmap + Depth), but do not support single-view absolute pose prediction. OPT-Pose enables the simultaneous recovery of both unseen-object relative and category-level absolute poses (right-most column) for flexible single or multi-view RGB or RGB-D input, without the need for CAD models or semantic labels.

## Overview

| Absolute + relative | Single or multiple views | Model-free |
| :--- | :--- | :--- |
| Canonical SA(3) and cross-view SE(3) poses | RGB input with optional sensor depth | No CAD models or semantic labels; inferred camera intrinsics |

### Capabilities and benchmarks

<p align="center">
  <img src="assets/functional_capability_matrix_chart.png" alt="OPT-Pose functional capability matrix" width="85%">
</p>

<p align="center">
  <img src="assets/benchmark_radar_chart.png" alt="OPT-Pose benchmark radar chart" width="70%">
</p>

## Setup

### Recommended environment

The recommended setup is the conda environment because the code uses CUDA
extensions and GPU rendering dependencies:

```bash
conda env create -f environment.yml
conda activate opt-pose
pip install -e .
```

<details>
<summary><strong>Use an existing PyTorch environment</strong></summary>

Install the pip dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

</details>

### CUDA extensions

Install PyTorch and CUDA-compatible dependencies according to your local GPU setup if the pinned conda packages in `environment.yml` do not match your system.

The SONATA components require `spconv` and `torch-scatter`; install versions compatible with your PyTorch/CUDA build.

Build the PointNet++ CUDA extension if your checkpoint uses the geometry encoder:

```bash
cd geo_models/pointnet2
python setup.py install
cd ../..
```

## Checkpoints

Weights are distributed outside git. Download the release checkpoints from the
[OPT-Pose Hugging Face repository](https://huggingface.co/colin1842/OPT-Pose):

```bash
mkdir -p pretrained
curl -L -o pretrained/abs_pose_housecat.pt \
  https://huggingface.co/colin1842/OPT-Pose/resolve/main/abs_pose_housecat.pt
curl -L -o pretrained/abs_pose_omni6dpose.pt \
  https://huggingface.co/colin1842/OPT-Pose/resolve/main/abs_pose_omni6dpose.pt
curl -L -o pretrained/rel_pose.pt \
  https://huggingface.co/colin1842/OPT-Pose/resolve/main/rel_pose.pt
```

Pass the downloaded checkpoint path explicitly when running inference.

| Task | Dataset | Checkpoint |
| :--- | :--- | :--- |
| Absolute pose | HouseCat6D | [`abs_pose_housecat.pt`](https://huggingface.co/colin1842/OPT-Pose/blob/main/abs_pose_housecat.pt) |
| Absolute pose | Omni6DPose | [`abs_pose_omni6dpose.pt`](https://huggingface.co/colin1842/OPT-Pose/blob/main/abs_pose_omni6dpose.pt) |
| Relative pose | NOCS / TOYL | [`rel_pose.pt`](https://huggingface.co/colin1842/OPT-Pose/blob/main/rel_pose.pt) |

## Datasets

Dataset roots are passed as command-line arguments and are not hard-coded. Expected inputs:

- `<HouseCat6D>`: HouseCat6D root with the official train/test folders and object model files.
- `<Oryon_NOCS>`: prepared NOCS relative-pose runtime root containing `camera.json`, `fixed_split/cross_scene_test/instance_list.txt`, object models, RGB/depth/mask data, and ground-truth pose files.
- `<Oryon_TOYL>`: prepared TOYL relative-pose runtime root containing `camera.json`, `fixed_split/cross_scene_test/instance_list.txt`, BOP models, RGB/depth/mask data, and scene metadata.

## Inference

### HouseCat6D absolute pose

```bash
python test_abs_housecat6d.py \
  --data_root <HouseCat6D> \
  --checkpoint <housecat_ckpt> \
  --save_path <out_dir>
```

### NOCS relative pose

```bash
python test_rel_pose_nocs.py \
  --base_dir <Oryon_NOCS> \
  --checkpoint <rel_pose_ckpt> \
  --output <out_dir>
```

### TOYL relative pose

```bash
python test_rel_pose_toyl.py \
  --base_dir <Oryon_TOYL> \
  --checkpoint <rel_pose_ckpt> \
  --output <out_dir>
```

## Training

```bash
torchrun --nproc_per_node=<N> training/launch.py \
  --config housecat_default \
  data.train.dataset.dataset_configs.0.data_root=<HouseCat6D> \
  checkpoint.resume_checkpoint_path=<checkpoint>
```

## Citation

If you find OPT-Pose useful, please cite our paper:

```bibtex
@InProceedings{Li_2026_CVPR,
  author    = {Li, Weihang and Garattoni, Lorenzo and Despinoy, Fabien and Navab, Nassir and Busam, Benjamin},
  title     = {Object Pose Transformer: Unifying Unseen Object Pose Estimation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  month     = {June},
  year      = {2026},
  pages     = {436-446}
}
```

## License

Project-owned code and documentation are released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See [LICENSE](LICENSE).

Some files contain upstream third-party notices. Those notices are preserved, and additional attribution is listed in [NOTICE](NOTICE).
