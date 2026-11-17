"""
Build the object-stratified, policy-disjoint cross-validation splits used in the paper.

Expects every trajectory grouped by object (each trajectory comes from a different policy):

    data/all/
    |--- cube/      <traj_0>/ <traj_1>/ ...
    |--- dodec/     ...
    :
    |--- hex_prism/ ...   (eval-only objects, never trained on)

For each split, `--val-per-object` trajectories per training object are held out for validation and the
rest are used for training. Objects passed via `--eval-only` go entirely to validation in every split.
Folders are created with symlinks, so no data is copied. A ready-to-use train_config.yaml is written
per split:

    python scripts/make_splits.py --src data/all --dst data/splits --eval-only letter_r hex_prism
    accelerate launch -m utils.trainer --config data/splits/split_0/train_config.yaml
    python scripts/aggregate_splits.py --splits-dir data/splits
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml


def clear_split_dir(split_dir: Path):
    """Remove a previously generated split so stale symlinks can't leak val trajectories into train."""
    if not split_dir.exists():
        return
    for root, dirs, files in os.walk(split_dir):
        for name in dirs + files:
            p = Path(root) / name
            if not p.is_symlink() and p.is_file() and p.name != "train_config.yaml":
                raise RuntimeError(f"{p} is not a generated symlink, refusing to delete {split_dir}")
        # don't descend into symlinked trajectory folders
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
    shutil.rmtree(split_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="folder with one sub-folder per object")
    parser.add_argument("--dst", default="data/splits")
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--val-per-object", type=int, default=2)
    parser.add_argument("--eval-only", nargs="*", default=[], help="objects used only for validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-config", default="configs/train_config.yaml")
    args = parser.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst)
    objects = sorted(d.name for d in src.iterdir() if d.is_dir())
    unknown = set(args.eval_only) - set(objects)
    if unknown:
        raise ValueError(f"--eval-only objects not found in {src}: {sorted(unknown)}")

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    summary = {}
    for k in range(args.num_splits):
        rng = np.random.default_rng([args.seed, k])
        split_dir = dst / f"split_{k}"
        clear_split_dir(split_dir)
        train_dir = split_dir / "train_data"
        train_dir.mkdir(parents=True, exist_ok=True)

        config = dict(base_config)
        config["train_data_dir"] = str(train_dir)
        config["checkpoint_dir"] = os.path.join(base_config["checkpoint_dir"], f"split_{k}")
        summary[f"split_{k}"] = {}

        for obj in objects:
            trajs = sorted(d.name for d in (src / obj).iterdir() if d.is_dir())
            if obj in args.eval_only:
                val, train = trajs, []
            else:
                if len(trajs) <= args.val_per_object:
                    raise ValueError(f"{obj} has {len(trajs)} trajectories, need > {args.val_per_object}")
                perm = [trajs[i] for i in rng.permutation(len(trajs))]
                val, train = sorted(perm[: args.val_per_object]), sorted(perm[args.val_per_object:])

            val_dir = split_dir / "val_data" / obj
            val_dir.mkdir(parents=True, exist_ok=True)
            for t in val:
                (val_dir / t).symlink_to(src / obj / t, target_is_directory=True)
            for t in train:
                (train_dir / f"{obj}__{t}").symlink_to(src / obj / t, target_is_directory=True)

            config[f"{obj}_val"] = str(val_dir)
            summary[f"split_{k}"][obj] = {"train": train, "val": val}

        stale = [key for key in base_config if key.endswith("_val") and key[: -len("_val")] not in objects]
        if stale and k == 0:
            print(f"warning: {stale} not found in {src}, they still point at the base config paths")

        with open(split_dir / "train_config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    with open(dst / "splits.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.num_splits} splits to {dst}")


if __name__ == "__main__":
    main()
