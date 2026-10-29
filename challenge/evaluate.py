"""Score a folder of predictions against the targets of a test study.

Usage: python challenge/evaluate.py preds/my-team data/Study_2 --target Nucleus --out submissions/my-team/scores.csv

Every acquisition set of the test study that has the target must have a prediction
(set<ID>_<Target>.tif). Targets are normalised exactly as in training (percentiles 1 and 99.8).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lfp import mae, pcc, percentile_norm, ssim  # noqa: E402
from train_lmc import NAME  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("test_dir")
    ap.add_argument("--target", default="Nucleus")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    targets = {int(m["set_id"]): p for p in sorted(Path(args.test_dir).glob("*.ome.tiff"))
               if (m := NAME.fullmatch(p.name)) and m["kind"] == args.target}
    rows, missing = [], []
    for set_id, path in targets.items():
        pred_path = Path(args.pred_dir) / f"set{set_id}_{args.target}.tif"
        if not pred_path.exists():
            missing.append(set_id)
            continue
        pred = np.clip(tifffile.imread(pred_path).astype(np.float32), 0, 1)
        y = percentile_norm(np.squeeze(tifffile.imread(path)))
        if pred.ndim != 2 or pred.shape[0] > y.shape[0] or pred.shape[1] > y.shape[1] \
                or y.shape[0] - pred.shape[0] > 3 or y.shape[1] - pred.shape[1] > 3:
            raise SystemExit(f"set {set_id}: prediction shape {pred.shape} does not match target {y.shape}")
        y = y[:pred.shape[0], :pred.shape[1]]
        rows.append({"set_id": set_id, "target": args.target,
                     "MAE": mae(pred, y), "SSIM": ssim(pred, y), "PCC": pcc(pred, y)})
    if missing:
        raise SystemExit(f"missing predictions for sets {missing}")

    scores = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out, index=False)
    print(scores[["MAE", "SSIM", "PCC"]].mean().round(4).to_dict(), f"-> {args.out}")


if __name__ == "__main__":
    main()
