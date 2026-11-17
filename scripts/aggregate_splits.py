"""
Aggregate per-object validation metrics (F1, accuracy, loss) across cross-validation splits (mean +- std).

Reads <checkpoint_dir>/metrics.jsonl for every data/splits/split_*/train_config.yaml,
written by utils/trainer.py after each epoch.

    python scripts/aggregate_splits.py --splits-dir data/splits            # last epoch of each split
    python scripts/aggregate_splits.py --splits-dir data/splits --epoch 99 # a specific epoch
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--epoch", type=int, default=None, help="epoch to report, defaults to the last one")
    args = parser.parse_args()

    split_cfgs = sorted(Path(args.splits_dir).glob("split_*/train_config.yaml"), key=lambda p: int(p.parent.name[6:]))
    if not split_cfgs:
        raise FileNotFoundError(f"no split_*/train_config.yaml in {args.splits_dir}")

    per_object = {}  # name -> {"f1": [...], "acc": [...], "loss": [...]}
    used = []
    for cfg_path in split_cfgs:
        with open(cfg_path) as f:
            ckpt_dir = Path(yaml.safe_load(f)["checkpoint_dir"])
        metrics_path = ckpt_dir / "metrics.jsonl"
        if not metrics_path.exists():
            print(f"skipping {cfg_path.parent.name}: {metrics_path} not found")
            continue
        with open(metrics_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        if args.epoch is None:
            rec = records[-1]
        else:
            matches = [r for r in records if r["epoch"] == args.epoch]
            if not matches:
                print(f"skipping {cfg_path.parent.name}: epoch {args.epoch} not found")
                continue
            rec = matches[-1]
        used.append(f"{cfg_path.parent.name}@{rec['epoch']}")
        for name, m in rec["val"].items():
            per_object.setdefault(name, {"f1": [], "acc": [], "loss": []})
            per_object[name]["f1"].append(m.get("f1", float("nan")))  # nan for runs logged before F1 was added
            per_object[name]["acc"].append(m["acc"])
            per_object[name]["loss"].append(m["loss"])

    if not used:
        raise RuntimeError("no metrics found for any split")

    print(f"aggregated {len(used)} split(s): {', '.join(used)}\n")
    cols = ["f1", "acc", "loss"]
    print(f"{'object':>10} | " + " | ".join(f"{'val ' + c + ' (mean +- std)':>23}" for c in cols))
    print("-" * (13 + 26 * len(cols)))
    for name, m in per_object.items():
        cells = []
        for c in cols:
            v = np.array(m[c], dtype=float)
            v = v[~np.isnan(v)]  # skip splits logged before F1 was added
            cells.append(f"{v.mean():>12.4f} +- {v.std():.4f}" if v.size else f"{'n/a':>23}")
        print(f"{name:>10} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
