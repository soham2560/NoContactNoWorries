"""
Sanity-check a dataset against what utils/sequence.py and the model expect,
without needing torch or a GPU.

    python scripts/check_data.py --config configs/train_config.yaml

Checks every directory referenced by `train_data_dir` and `<object>_val` in the config.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REQUIRED_FILES = [
    "hand_joint_pose_history.npy",
    "desired_hand_joint_pose_history.npy",
    "image_history.npy",
    "depth_history.npy",
    "contact_history.npy",
]
NUM_JOINTS = 16
NUM_SENSORS = 4


def check_trajectory(traj_dir: Path, horizon: int):
    """Returns (list of errors, number of timesteps)."""
    missing = [f for f in REQUIRED_FILES if not (traj_dir / f).exists()]
    if missing:
        return [f"{traj_dir}: missing {missing}"], 0

    arr = {f: np.load(traj_dir / f, mmap_mode="r") for f in REQUIRED_FILES}
    pose = arr["hand_joint_pose_history.npy"]
    desired = arr["desired_hand_joint_pose_history.npy"]
    image = arr["image_history.npy"]
    depth = arr["depth_history.npy"]
    contact = arr["contact_history.npy"]

    errors = []
    lengths = {f: a.shape[0] for f, a in arr.items()}
    if len(set(lengths.values())) != 1:
        errors.append(f"{traj_dir}: timestep mismatch {lengths}")
    T = pose.shape[0]

    if pose.shape[1:] != (NUM_JOINTS,):
        errors.append(f"{traj_dir}: hand_joint_pose should be (T, {NUM_JOINTS}), got {pose.shape}")
    if desired.shape[1:] != (NUM_JOINTS,):
        errors.append(f"{traj_dir}: desired_hand_joint_pose should be (T, {NUM_JOINTS}), got {desired.shape}")
    if image.ndim != 4 or image.shape[-1] != 3:
        errors.append(f"{traj_dir}: image should be (T, H, W, 3), got {image.shape}")
    elif depth.shape[1:] != image.shape[1:3] + (1,):
        h, w = image.shape[1:3]
        errors.append(f"{traj_dir}: depth should be (T, {h}, {w}, 1), got {depth.shape}")
    if contact.ndim != 3 or contact.shape[1] != NUM_SENSORS:
        # the loader marks a sensor as in contact if anything along the last axis is non-zero
        errors.append(f"{traj_dir}: contact should be (T, {NUM_SENSORS}, 3), got {contact.shape}")
    if T < horizon:
        errors.append(f"{traj_dir}: only {T} timesteps, shorter than horizon={horizon} (gives 0 windows)")
    return errors, T


def check_split_dir(name: str, data_dir: Path, horizon: int):
    if not data_dir.is_dir():
        return [f"[{name}] {data_dir} does not exist"]
    trajs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not trajs:
        return [f"[{name}] {data_dir} has no trajectory sub-folders"]

    errors, windows = [], 0
    for t in trajs:
        e, T = check_trajectory(t, horizon)
        errors += e
        windows += max(T - horizon + 1, 0)
    status = "ok" if not errors else f"{len(errors)} problem(s)"
    print(f"[{name:>10}] {data_dir}: {len(trajs)} trajectories, {windows} windows -> {status}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    horizon = int(config["horizon"])

    dirs = {"train": config["train_data_dir"]}
    dirs.update({k[: -len("_val")]: v for k, v in config.items() if k.endswith("_val")})

    errors = []
    for name, d in dirs.items():
        errors += check_split_dir(name, Path(d), horizon)

    if errors:
        print("\nfound problems:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("\nall good")


if __name__ == "__main__":
    main()
