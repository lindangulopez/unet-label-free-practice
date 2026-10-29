"""Python refresher: numpy, pandas, image metrics and a small PyTorch U-Net, on synthetic data.

Runs on a CPU in about 10 seconds. Each section matches a section of the README.
"""
# --- 1. numpy
import numpy as np

rng = np.random.default_rng(0)
stack = rng.integers(0, 4096, size=(12, 256, 256), dtype=np.uint16)  # z, y, x
print(stack.shape, stack.dtype, stack.ndim)

img = stack[5]
crop = img[64:192, 64:192]
mip = stack.max(axis=0)
img_f = img.astype(np.float32)
norm = (img_f - img_f.mean()) / (img_f.std() + 1e-8)
lo, hi = np.percentile(img_f, [1, 99.8])
scaled = np.clip((img_f - lo) / (hi - lo), 0, 1).astype(np.float32)
mask = scaled > 0.5
print(mask.mean())
flat = stack.reshape(stack.shape[0], -1)
best_z = int(np.argmax(flat.var(axis=1) / flat.mean(axis=1)))
print("best z", best_z)
batch = scaled[np.newaxis, np.newaxis]
print(batch.shape)

# --- 2. pandas
import pandas as pd

files = pd.DataFrame({
    "file": ["image_60_BF_z0.ome.tiff", "image_60_Nucleus.ome.tiff",
             "image_255_DIC_z0.ome.tiff", "image_255_Tubulin.ome.tiff",
             "image_2491_PC_z10.ome.tiff", "image_2491_Mitochondria.ome.tiff"],
    "study": [3, 3, 7, 7, 28, 28],
})
parts = files["file"].str.extract(r"image_(?P<set_id>\d+)_(?P<kind>[A-Za-z]+)(?:_z(?P<z>\d+))?")
files = files.join(parts)
files["set_id"] = files["set_id"].astype(int)
files["is_input"] = files["kind"].isin(["BF", "PC", "DIC"])
print(files)
print(files.groupby("kind").size())
inputs = files[files["is_input"]][["set_id", "study", "kind", "file"]]
targets = files[~files["is_input"]][["set_id", "kind", "file"]]
pairs = inputs.merge(targets, on="set_id", suffixes=("_in", "_out"))
print(pairs)
print(pd.crosstab(pairs["kind_in"], pairs["kind_out"]))

studies = files["study"].unique()
rng = np.random.default_rng(42)
test_studies = rng.choice(studies, size=1, replace=False)
files["split"] = np.where(files["study"].isin(test_studies), "test", "train")
print(files.groupby(["split", "study"]).size())

# --- 3. metrics
def pcc(pred, target):
    p = pred.ravel().astype(np.float64)
    t = target.ravel().astype(np.float64)
    p = p - p.mean()
    t = t - t.mean()
    return float((p * t).sum() / np.sqrt((p ** 2).sum() * (t ** 2).sum()))

from skimage.metrics import structural_similarity

target = rng.random((128, 128)).astype(np.float32)
noisy = np.clip(target + rng.normal(0, 0.1, target.shape).astype(np.float32), 0, 1)
print("PCC", round(pcc(noisy, target), 3), "numpy check", round(np.corrcoef(noisy.ravel(), target.ravel())[0, 1], 3))
print("SSIM", round(structural_similarity(noisy, target, data_range=1.0), 3))
print("MAE", round(float(np.abs(noisy - target).mean()), 3))

scores = pd.DataFrame({
    "team": ["A", "B", "C"] * 2,
    "organelle": ["Nucleus"] * 3 + ["Mitochondria"] * 3,
    "ssim": [0.80, 0.75, 0.78, 0.60, 0.66, 0.61],
    "pcc": [0.90, 0.88, 0.86, 0.70, 0.74, 0.69],
})
for m in ["ssim", "pcc"]:
    scores[f"rank_{m}"] = scores.groupby("organelle")[m].rank(ascending=False)
rank_cols = [c for c in scores if c.startswith("rank_")]
leaderboard = scores.groupby("team")[rank_cols].mean().mean(axis=1).sort_values()
print(leaderboard)

# --- 4. PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

t = torch.from_numpy(scaled)
print(t.shape, t.dtype, t.device)
device = "cuda" if torch.cuda.is_available() else "cpu"

class PairDataset(Dataset):
    """Transmitted-light input -> fluorescence target, both 2D float arrays."""
    def __init__(self, inputs, targets, patch=64):
        self.inputs, self.targets, self.patch = inputs, targets, patch
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, i):
        x, y = self.inputs[i], self.targets[i]
        r = np.random.randint(0, x.shape[0] - self.patch + 1)
        c = np.random.randint(0, x.shape[1] - self.patch + 1)
        x = x[r:r + self.patch, c:c + self.patch]
        y = y[r:r + self.patch, c:c + self.patch]
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]

def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(),
    )

class TinyUNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.enc1, self.enc2 = block(1, c), block(c, 2 * c)
        self.pool = nn.MaxPool2d(2)
        self.bottom = block(2 * c, 4 * c)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), block(4 * c, 2 * c)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottom(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

# fake pairs: the "fluorescence" is a blurred, thresholded version of the input
from scipy.ndimage import gaussian_filter
xs = [rng.random((128, 128)).astype(np.float32) for _ in range(40)]
ys = [(gaussian_filter(x, 3) > 0.5).astype(np.float32) for x in xs]
train_dl = DataLoader(PairDataset(xs[:32], ys[:32]), batch_size=8, shuffle=True)
val_dl = DataLoader(PairDataset(xs[32:], ys[32:]), batch_size=8)

torch.manual_seed(0)
model = TinyUNet().to(device)
loss_fn = nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(sum(p.numel() for p in model.parameters()), "parameters")

for epoch in range(30):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val = sum(loss_fn(model(x.to(device)), y.to(device)).item() for x, y in val_dl) / len(val_dl)
    print(f"epoch {epoch}  train {loss.item():.3f}  val {val:.3f}")

torch.save(model.state_dict(), "tiny_unet.pt")
with torch.no_grad():
    pred = model(torch.from_numpy(xs[39])[None, None]).squeeze().numpy()
print("val image PCC", round(pcc(pred, ys[39]), 3))
