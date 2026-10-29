"""Write predictions for every acquisition set of a study, in the format the challenge expects.

Usage: python predict.py data/Study_2 --target Nucleus --model outputs/tiny_unet_Nucleus.pt --out preds/my-team
       python predict.py data/Study_2 --target Nucleus --baseline copy --out preds/copy-input

Each prediction is saved as set<ID>_<Target>.tif: a 2D float32 image in [0, 1].
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch

from lfp import TinyUNet, predict_full
from train_lmc import load_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--target", default="Nucleus")
    ap.add_argument("--model", help="weights saved by train_lmc.py")
    ap.add_argument("--baseline", choices=["copy"], help="no model: use the input image as the prediction")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not (args.model or args.baseline):
        raise SystemExit("Give --model or --baseline.")

    model = None
    if args.model:
        model = TinyUNet()
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in load_pairs(args.folder, args.target):
        pred = p["x"] if model is None else np.clip(predict_full(model, p["x"])[0], 0, 1)
        tifffile.imwrite(out / f"set{p['set_id']}_{args.target}.tif", pred.astype(np.float32))
    print(f"predictions written to {out}")


if __name__ == "__main__":
    main()
