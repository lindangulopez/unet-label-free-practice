"""Train the small U-Net on one downloaded Light My Cells study: transmitted light in, fluorescence out.

Usage: python train_lmc.py data/Study_2 --target Nucleus --epochs 30

Steps: pair files by acquisition set, pick the sharpest transmitted-light plane, normalise,
split by acquisition set (never by patch), train on random patches, then score whole
validation images with MAE, SSIM and PCC, against a "copy the input" baseline.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lfp import TinyUNet, best_focus, mae, pcc, percentile_norm, predict_full, ssim

NAME = re.compile(r"image_(?P<set_id>\d+)_(?P<kind>[A-Za-z]+)(?:_z(?P<z>\d+))?\.ome\.tiff")
INPUTS = {"BF", "PC", "DIC"}


def read2d(path):
    return np.squeeze(tifffile.imread(path)).astype(np.float32)


def load_pairs(folder, target):
    rows = [m.groupdict() | {"path": p} for p in sorted(Path(folder).glob("*.ome.tiff"))
            if (m := NAME.fullmatch(p.name))]
    files = pd.DataFrame(rows)
    files["set_id"] = files["set_id"].astype(int)
    print(files.groupby("kind").size().to_string(), "\n")

    pairs = []
    for set_id, group in files.groupby("set_id"):
        tgt = group[group["kind"] == target]
        inp = group[group["kind"].isin(INPUTS)].sort_values("z", key=lambda z: z.astype(int))
        if tgt.empty or inp.empty:
            continue
        planes = [read2d(p) for p in inp["path"]]
        z = best_focus(planes)
        x, y = percentile_norm(planes[z]), percentile_norm(read2d(tgt["path"].iloc[0]))
        if x.shape != y.shape:
            print(f"skip set {set_id}: shapes {x.shape} and {y.shape} differ")
            continue
        pairs.append({"set_id": set_id, "modality": inp["kind"].iloc[0], "best_z": z, "x": x, "y": y})
    return pairs


class Patches(Dataset):
    def __init__(self, pairs, patch, per_image, fg_frac):
        self.pairs, self.patch, self.per_image, self.fg_frac = pairs, patch, per_image, fg_frac

    def __len__(self):
        return len(self.pairs) * self.per_image

    def __getitem__(self, i):
        p = self.pairs[i % len(self.pairs)]
        h, w = p["x"].shape
        if np.random.rand() < self.fg_frac:
            # centre the patch on a bright target pixel: most of a field of view is empty background
            ys, xs = np.nonzero(p["y"] > 0.5)
            k = np.random.randint(len(ys)) if len(ys) else None
        else:
            k = None
        if k is None:
            r = np.random.randint(0, h - self.patch + 1)
            c = np.random.randint(0, w - self.patch + 1)
        else:
            r = int(np.clip(ys[k] - self.patch // 2, 0, h - self.patch))
            c = int(np.clip(xs[k] - self.patch // 2, 0, w - self.patch))
        x = p["x"][r:r + self.patch, c:c + self.patch]
        y = p["y"][r:r + self.patch, c:c + self.patch]
        if np.random.rand() < 0.5:  # flips: augmentation that does not change the biology
            x, y = x[:, ::-1], y[:, ::-1]
        if np.random.rand() < 0.5:
            x, y = x[::-1], y[::-1]
        return torch.from_numpy(x.copy())[None], torch.from_numpy(y.copy())[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--target", default="Nucleus", choices=["Nucleus", "Mitochondria", "Tubulin", "Actin"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--per-image", type=int, default=16, help="random patches per image per epoch")
    ap.add_argument("--fg-frac", type=float, default=0.7, help="share of patches centred on the target signal")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = load_pairs(args.folder, args.target)
    if len(pairs) < 2:
        raise SystemExit(f"Need at least 2 acquisition sets with {args.target}; found {len(pairs)}.")
    ids = np.random.permutation([p["set_id"] for p in pairs])
    n_val = max(1, round(len(ids) * args.val_frac))
    val_ids = set(ids[:n_val].tolist())
    train = [p for p in pairs if p["set_id"] not in val_ids]
    val = [p for p in pairs if p["set_id"] in val_ids]
    print(f"{len(pairs)} pairs -> {len(train)} train, {len(val)} validation (split by acquisition set: {sorted(val_ids)})")

    dl = DataLoader(Patches(train, args.patch, args.per_image, args.fg_frac), batch_size=8, shuffle=True)
    model = TinyUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        val_mae = np.mean([mae(*crop_pred(model, p, device)) for p in val])
        print(f"epoch {epoch:3d}  train L1 {total / len(dl.dataset):.4f}  val MAE {val_mae:.4f}")
        history.append({"epoch": epoch, "train_L1": total / len(dl.dataset), "val_MAE": val_mae})

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows = []
    for p in val:
        pred, y = crop_pred(model, p, device)
        x = p["x"][:y.shape[0], :y.shape[1]]
        rows.append({"set_id": p["set_id"], "modality": p["modality"], "best_z": p["best_z"],
                     "MAE": mae(pred, y), "SSIM": ssim(pred, y), "PCC": pcc(pred, y),
                     "PCC_baseline": pcc(x, y)})
        tifffile.imwrite(out / f"set{p['set_id']}_{args.target}_input_pred_target.tif",
                         np.stack([x, pred, y]).astype(np.float32))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / f"scores_{args.target}.csv", index=False)
    pd.DataFrame(history).to_csv(out / f"history_{args.target}.csv", index=False)
    print("\nValidation images (PCC_baseline = input used as the prediction):")
    print(scores.round(3).to_string(index=False))
    print("\nMean:", scores[["MAE", "SSIM", "PCC", "PCC_baseline"]].mean().round(3).to_dict())
    torch.save(model.state_dict(), out / f"tiny_unet_{args.target}.pt")


def crop_pred(model, p, device):
    pred, (h, w) = predict_full(model, p["x"], device)
    return np.clip(pred, 0, 1), p["y"][:h, :w]


if __name__ == "__main__":
    main()
