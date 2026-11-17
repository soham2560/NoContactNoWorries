<p align="center">
  <h1 align="center">
    NoContactNoWorries: Estimating Contact through Vision and Proprioception for In-Hand Dexterous Manipulation
  </h1>

  <p align="center">
    Soham Patil<sup>1</sup>,
    Avirup Das<sup>2</sup>,
    Sourabh Bhosale<sup>1</sup>,
    Spandan Roy<sup>1</sup>
  </p>

  <p align="center">
    <sup>1</sup>Robotics Research Center (RRC), International Institute of Information Technology (IIIT), Hyderabad, India
    &nbsp;
    <sup>2</sup>Department of Computer Science, The University of Manchester
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2606.24450">
      <img src="https://img.shields.io/badge/arXiv-2606.24450-b31b1b" alt="arXiv">
    </a>
    <a href="https://soham2560.github.io/no-contact-no-worries/" style="padding-left: 0.5rem;">
      <img src="https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue" alt="Project Page">
    </a>
  </p>

  <p align="center">
    <strong>Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026!</strong>
  </p>
</p>

## todo

- [x] release code (model + training)
- [x] docker setup
- [ ] release the dataset (soon, folder layout is below so u can collect ur own in the meantime)
- [ ] release trained checkpoints
- [ ] release our modified leap hand sim for data collection
- [ ] real world deployment code w/ the fsr sensors

## Contents

- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
  - [Docker (dev container)](#docker-dev-container)
  - [Minimal Docker (for training)](#minimal-docker-for-training)
  - [Local Setup (without Docker)](#local-setup-without-docker)
  - [AsymFormer Checkpoint](#asymformer-checkpoint)
- [Dataset](#dataset)
  - [Dataset Collection](#dataset-collection)
  - [Dataset Folder Structure](#dataset-folder-structure)
  - [Checking Your Data](#checking-your-data)
- [Training](#training)
  - [Cross-Validation Splits](#cross-validation-splits)
- [References](#references)
- [Citation](#citation)

## Pre-requisites
Tested on:

1. Ubuntu 22.04 LTS
2. Nvidia Driver 565.77
3. CUDA 12.4

## Installation

### Docker (dev container)
- Pull the image
    ```bash
    docker pull soham2560/nocontactnoworries
    ```
- Open this repo in VSCode, open the Command Palette with `Ctrl+Shift+P` and select **Dev Containers: Rebuild and Reopen in Container**. The repo is installed in editable mode on container start.

Note: this needs the [VSCode Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

### Minimal Docker (for training)
A minimal image for training with HuggingFace 🤗 Accelerate on one or more GPUs (see [docker/](./docker/README.md) to build it yourself):

```bash
docker run --gpus all -it -v $(pwd):/workspace -w /workspace avirupdas55/ncnw:v1
```

### Local Setup (without Docker)
```bash
conda create -n ncnw python=3.10
conda activate ncnw
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/soham2560/NoContactNoWorries.git
cd NoContactNoWorries
pip install -r requirements.txt
pip install -e .
```

The AsymFormer architecture is already vendored in [models/](./models), so there is no need to clone the AsymFormer repo separately.

### AsymFormer Checkpoint
The RGB-D encoder is a frozen [AsymFormer](https://github.com/Fourier7754/AsymFormer) pretrained on NYUv2. Download it into `checkpoints/` with:

```bash
./setup.sh
```

or grab it manually from [here](https://drive.google.com/file/d/1Pg6r3eJ245GaKbHfZob0Ek0CVhiS7VaR/view) and place it at `checkpoints/AsymFormer_NYUv2.pth`.

Note: building the model also pulls ImageNet ConvNeXt-T weights through `torch.hub` the first time (they get overwritten by the AsymFormer checkpoint), so the first run needs internet access.

## Dataset

### Dataset Collection
The dataset is collected in simulation with a modified version of [LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim) (to be released). We record rollouts of in-hand reorientation on five training object geometries: 50 trajectories of 15 s each (10 per object). Each trajectory comes from an independently trained policy (different seeds, converged checkpoints and state-action occupancy), so no two trajectories share a controller.

We evaluate on 7 objects: the five training (seen) geometries, cuboid (`cube`), pentagonal prism (`pent`), extruded star (`star`), dodecahedron (`dodec`) and stairs (`stairs`), plus two novel (held-out) objects, extruded letter R (`letter_r`) and hexagonal prism (`hex_prism`), which never appear in the training data and are only used for zero-shot evaluation.

Each timestep stores:

| modality | shape | file |
|---|---|---|
| RGB | `240 x 320 x 3` | `image_history.npy` |
| depth | `240 x 320 x 1` | `depth_history.npy` |
| joint state | `16` | `hand_joint_pose_history.npy` |
| commanded joint state | `16` | `desired_hand_joint_pose_history.npy` |
| contact | `4 x 3` contact force vectors | `contact_history.npy` |

Ground-truth contact comes from the simulator's physics engine (PhysX) at four fingertip sites (K = 4), at the center of the distal phalanges of the four active fingers on the LEAP hand. These match where the real-world FSR sensors are mounted for hardware evaluation. A sensor counts as in contact when its force vector is non-zero.

Inside the model, RGB and depth are resized from 240x320 to 480x640 (the AsymFormer NYUv2 input size) and normalised with the NYUv2 statistics AsymFormer was trained with (see `ScaleNorm` / `Normalize` in [utils/utils.py](./utils/utils.py)). Keep your depth in the same units as the AsymFormer setup. AsymFormer gives a 120x160x256 feature map, which a small CNN reduces to 30x40 = 1200 visual tokens per frame.

### Dataset Folder Structure
Every trajectory is a folder of `.npy` files, all with the same number of timesteps `T`. The default [configs/train_config.yaml](./configs/train_config.yaml) expects:

```bash
data/
|--- train_data/
|    |--- <traj_id>/
|    |    |--hand_joint_pose_history.npy           # (T, 16)
|    |    |--desired_hand_joint_pose_history.npy   # (T, 16)
|    |    |--image_history.npy                     # (T, 240, 320, 3)
|    |    |--depth_history.npy                     # (T, 240, 320, 1)
|    |    |--contact_history.npy                   # (T, 4, 3)
|    :
|--- val_data/
|    |--- cube/       <traj_id>/ ...
|    |--- dodec/      <traj_id>/ ...
|    |--- pent/       <traj_id>/ ...
|    |--- star/       <traj_id>/ ...
|    |--- stairs/     <traj_id>/ ...
|    |--- letter_r/   <traj_id>/ ...   (novel)
|    |--- hex_prism/  <traj_id>/ ...   (novel)
```

Any other files in a trajectory folder are ignored. Folder names under `train_data/` and the trajectory ids don't matter, but each trajectory must have at least `horizon` timesteps.

### Checking Your Data
> **Note:** `check_data.py`, `make_splits.py` and `aggregate_splits.py` were written for this release and are still a bit rough around the edges.

Before training, you can run a quick check on your data. It only needs numpy and pyyaml, no GPU, and validates every folder in the config against the shapes above:

```bash
python scripts/check_data.py --config configs/train_config.yaml
```

## Training

Single GPU:
```bash
python3 -m utils.trainer --config configs/train_config.yaml
```

Multi GPU with 🤗 Accelerate:
```bash
accelerate config
accelerate launch -m utils.trainer --config configs/train_config.yaml
```

Validation F1 (the metric reported in the paper: computed per contact site over all timesteps, then averaged across the 4 sites), accuracy and loss are reported per object every epoch, and also appended to `<checkpoint_dir>/metrics.jsonl`. Checkpoints are saved to `checkpoint_dir` after every epoch. Logging to wandb is on by default: run `wandb login` first (optionally set `wandb_entity`), or set `use_wandb: False` in the config.

### Cross-Validation Splits
In the paper, predictive accuracy is reported over 10 object-stratified, policy-disjoint random splits. In each split, 2 trajectories per training object are held out for validation and the remaining 8 per object are used for training, so validation trajectories always come from policies not seen during training.

To reproduce this, put all trajectories in one folder per object:
```bash
data/all/
|--- cube/   <traj_id>/ ...
|--- dodec/  <traj_id>/ ...
:
|--- letter_r/   <traj_id>/ ...   (novel, eval only)
|--- hex_prism/  <traj_id>/ ...   (novel, eval only)
```
then build the splits. Only symlinks are created, nothing is copied. You get one ready-to-use config per split at `data/splits/split_k/train_config.yaml`, with its own `checkpoint_dir`:
```bash
python scripts/make_splits.py --src data/all --dst data/splits --num-splits 10 --val-per-object 2 --eval-only letter_r hex_prism
```
train on each split:
```bash
for k in $(seq 0 9); do
    accelerate launch -m utils.trainer --config data/splits/split_$k/train_config.yaml
done
```
and report F1, accuracy and loss as mean +- std per object across splits (uses the last epoch by default, `--epoch N` to pick one):
```bash
python scripts/aggregate_splits.py --splits-dir data/splits
```
Objects passed to `--eval-only` are never trained on and go fully to validation in every split. Which trajectories landed where is saved to `data/splits/splits.json`. Re-running `make_splits.py` regenerates the splits from scratch, so stale links from an earlier run can't leak into a new one.

## References

- [AsymFormer](https://github.com/Fourier7754/AsymFormer): Asymmetrical Cross-Modal Representation Learning for Mobile Platform Real-Time RGB-D Semantic Segmentation.
- [LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim): Isaac Gym repository for the LEAP Hand.

## Citation

```bibtex
@inproceedings{patil2026nocontactnoworries,
  title     = {NoContactNoWorries: Estimating Contact through Vision and Proprioception for In-Hand Dexterous Manipulation},
  author    = {Patil, Soham and Das, Avirup and Bhosale, Sourabh and Roy, Spandan},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
