"""Make one PNG from a train_lmc.py run: input, prediction and target for one validation image, and the loss curves.

Usage: python make_figure.py --target Nucleus      # reads outputs/, writes outputs/figure_Nucleus.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tifffile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Nucleus")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--set-id", type=int, default=None, help="validation set to show (default: best PCC)")
    args = ap.parse_args()
    out = Path(args.out)

    scores = pd.read_csv(out / f"scores_{args.target}.csv")
    history = pd.read_csv(out / f"history_{args.target}.csv")
    row = scores.loc[scores["PCC"].idxmax()] if args.set_id is None else scores.set_index("set_id").loc[args.set_id].assign(set_id=args.set_id)
    set_id = int(row["set_id"])
    x, pred, y = tifffile.imread(out / f"set{set_id}_{args.target}_input_pred_target.tif")

    fig = plt.figure(figsize=(13, 3.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3])
    panels = [(x, f"Input ({row['modality']}, z{int(row['best_z'])})"),
              (pred, "Prediction"),
              (y, f"Target ({args.target})")]
    for k, (img, title) in enumerate(panels):
        ax = fig.add_subplot(grid[k])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    ax = fig.add_subplot(grid[3])
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_MAE"], label="validation MAE")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False)
    ax.set_title(f"SSIM {row['SSIM']:.2f} · PCC {row['PCC']:.2f} (copy input: {row['PCC_baseline']:.2f})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    path = out / f"figure_{args.target}.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path} (validation set {set_id})")


if __name__ == "__main__":
    main()
