# unet-label-free-practice

A small, self-contained practice project: predict a fluorescence image (for example the nucleus) from a transmitted-light microscopy image with a small U-Net in PyTorch, and score it the way image-analysis challenges do (MAE, SSIM, Pearson correlation).

It is a learning exercise, not a competitive method. Everything runs on a laptop CPU.

## Contents

| File | What it does |
|---|---|
| `refresher.py` | numpy, pandas, the three metrics and a small U-Net, on synthetic data. About 10 seconds on a CPU |
| `download_study.py` | Downloads one study of the public Light My Cells database from the BioImage Archive |
| `train_lmc.py` | Trains the U-Net on a downloaded study and scores the validation images |
| `predict.py` | Writes predictions for a study, from a trained model or the "copy the input" baseline |
| `make_figure.py` | One PNG from a training run: input, prediction, target and the loss curves |
| `lfp.py` | Shared code: the U-Net, normalisation, best-focus selection, metrics |
| [`CHALLENGE.md`](CHALLENGE.md), `challenge/`, `submissions/` | A mini challenge on this task: rules, evaluation, and a leaderboard rebuilt by GitHub Actions |

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch
```

## 1. Refresher on synthetic data

```bash
.venv/bin/python refresher.py
```

Sections, in order:

1. **numpy**: an image as an array, `uint16` to `float32`, percentile normalisation, maximum intensity projection, best-focus plane by normalised variance, adding batch and channel axes.
2. **pandas**: metadata from file names with a regex, pairing inputs and targets with `merge`, and a train/test split by study rather than by image.
3. **Metrics**: MAE, SSIM (scikit-image) and the Pearson correlation coefficient written by hand, then a leaderboard built by averaging ranks per organelle.
4. **PyTorch**: a `Dataset` of random patches, a two-level U-Net with skip connections, and the training loop (`zero_grad`, forward, loss, `backward`, `step`).

## 2. Real data: one Light My Cells study

```bash
.venv/bin/python download_study.py 2          # Study_2: 144 files, about 140 MB, DIC input
.venv/bin/python train_lmc.py data/Study_2 --target Nucleus --epochs 30
```

`train_lmc.py` pairs each transmitted-light z-stack with its fluorescence target, keeps the sharpest transmitted-light plane, splits the data by acquisition set, trains on random flipped patches, then prints MAE, SSIM and PCC for each validation image. `PCC_baseline` is the score of simply copying the input: the model should beat it. Each validation image is saved to `outputs/` as a three-plane TIFF (input, prediction, target) to open in napari or Fiji.

Small studies, with the fluorescence targets they contain (file counts from the archive listing, October 2026):

| Study | Size | Targets |
|---|---|---|
| 2 | about 140 MB | Nucleus (10), Mitochondria (9) |
| 23 | about 270 MB | Nucleus (23), Mitochondria (23) |
| 6 | about 260 MB | Nucleus (10), Actin (10) |
| 7 | about 320 MB | Nucleus (10), Tubulin (10) |
| 4 | about 400 MB | Nucleus, Mitochondria, Tubulin (31 each) |

With a single study, the split can only be by acquisition set. Across several studies, split by study: images from one study share a microscope, a cell line and a protocol, and a model scored on the same study it trained on looks better than it is.

Then make a figure of the run: `.venv/bin/python make_figure.py --target Nucleus` writes `outputs/figure_Nucleus.png`.

With only 8 training images (Study 2), expect the model to learn little: most of each field of view is empty background, and a model can get a low MAE by predicting background everywhere. `train_lmc.py` therefore centres 70% of the patches on the target signal (`--fg-frac`). More acquisition sets help more than more epochs.

Things to try, one at a time: `--target Mitochondria`, more epochs, `--patch 64`, another study, `nn.MSELoss()` instead of `nn.L1Loss()` in `train_lmc.py`.

## Data

Light My Cells database, BioImage Archive accession [S-BIAD1047](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1047), licensed CC BY 4.0. Data are downloaded to `data/` and are not part of this repository. Please cite the data descriptor:

> Kauffmann, D., Gay, G., Mateos-Langerak, J. et al. 2D Multimodal Image Collection Data Descriptor for Fluorescence Prediction from Transmitted Light Microscopy. *Scientific Data* 13, 743 (2026). https://doi.org/10.1038/s41597-026-07004-w

The dataset's own preparation code is at [codeberg.org/FranceBioImaging/LightMyCellsDatabase](https://codeberg.org/FranceBioImaging/LightMyCellsDatabase).

## References

- Ronneberger, O., Fischer, P. and Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
- Maier-Hein, L. et al. Why rankings of biomedical image analysis competitions should be interpreted with care. *Nature Communications* 9, 5217 (2018).

## Licence and AI use

Code under the MIT licence (see `LICENSE`). The code and this README were drafted with the help of an AI assistant (Claude Code) and reviewed by the author.
