# Mini challenge: light to fluorescence

A small practice challenge, run entirely on GitHub, to learn how an image-analysis challenge is designed: a question, a reference dataset, a split that cannot leak, metrics, a ranking rule, and results that others can re-run.

## The question

Can a model predict the **nucleus fluorescence image** from a **transmitted-light image** (DIC) of the same cells, in a study it has never seen?

Label-free prediction saves a fluorescent stain, and with it phototoxicity and preparation time. The hard part is generalisation: a model that works on one microscope often fails on the next.

## Data

From the public Light My Cells database (BioImage Archive [S-BIAD1047](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1047), CC BY 4.0, [data descriptor](https://doi.org/10.1038/s41597-026-07004-w)).

| Split | Study | What you may do with it |
|---|---|---|
| Training | Study 23: 23 acquisition sets, DIC z-stacks with nucleus targets | Anything: training, validation, tuning |
| Test | Study 2: 10 acquisition sets, DIC z-stacks with nucleus targets | Predictions only. No training, no tuning, no looking at the targets |

The split is **by study**, never by image: images of one study share a microscope, a cell line and a protocol, so a model tested on its own study looks better than it is.

```bash
python download_study.py 23   # training
python download_study.py 2    # test
```

**Known limitation.** The test targets are public in the archive, so this practice challenge runs on trust. A real challenge keeps them hidden, for example by running submitted containers on the organisers' side, as Grand Challenge does. Re-running submitted code (the "re-run" badge) is the check that replaces trust here.

## Submitting

1. Train on Study 23 (for example `python train_lmc.py data/Study_23 --target Nucleus`).
2. Predict every test set: `python predict.py data/Study_2 --target Nucleus --model outputs/tiny_unet_Nucleus.pt --out preds/<team>`. Each file is `set<ID>_Nucleus.tif`, a 2D float image in [0, 1].
3. Score: `python challenge/evaluate.py preds/<team> data/Study_2 --target Nucleus --out submissions/<team>/scores.csv`.
4. Copy `submissions/TEMPLATE/submission.json` to `submissions/<team>/` and fill it in.
5. Open a pull request that adds only `submissions/<team>/`. Do not commit predictions or data.

A GitHub Action checks every pull request (all test sets scored, required fields present). After merge, it rebuilds [LEADERBOARD.md](LEADERBOARD.md).

## Metrics and ranking

Each test image is scored on three metrics, against the target normalised with percentiles 1 and 99.8:

| Metric | Measures | Better |
|---|---|---|
| MAE | Mean absolute pixel error | Lower |
| SSIM | Local structure, contrast and luminance | Higher |
| PCC | Pearson correlation of pixel intensities | Higher |

Teams are ranked on their mean of each metric over the 10 test images. The final score is the **mean of the three ranks**, so metrics on different scales weigh the same. Ties are broken by PCC, then by the earliest submission.

Read the ranking with care: with 10 test images from one study, a small change in the test set can reorder close teams (Maier-Hein et al., *Nature Communications* 2018).

## Reproducibility badges

Badges sit next to the score and do not change the ranking:

| Badge | Condition (in `submission.json`) |
|---|---|
| **archived** | The code is archived in [Software Heritage](https://archive.softwareheritage.org/save/) and `swhid` gives its identifier (a `swh:1:rev:` or `swh:1:dir:` SWHID) |
| **described** | The code repository has a `codemeta.json` file, and `codemeta` gives its URL |
| **re-run** | Someone other than the team re-ran the code from the archived version and got the same scores within 0.01; `rerun` names who and when |

## Baselines

| Baseline | How |
|---|---|
| Copy the input | `python predict.py data/Study_2 --target Nucleus --baseline copy --out preds/copy-input` |
| Small U-Net | `train_lmc.py` with default settings, then `predict.py --model ...` |

A method that does not beat "copy the input" on PCC has learned nothing useful.

## Rules

- Train only on Study 23. Pre-trained weights from other public data are allowed if named in `method`.
- One submission per team per week; the leaderboard keeps the latest.
- Code must be public, with an open-source licence.
- The organisers may ask for the code to be re-run before awarding the re-run badge.
