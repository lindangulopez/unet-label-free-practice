"""Shared helpers: a small U-Net, image normalisation, best-focus selection and metrics."""
import numpy as np
import torch
from torch import nn
from skimage.metrics import structural_similarity


def percentile_norm(img, lo=1, hi=99.8):
    """Scale an image to about 0-1 with percentiles, so a few saturated pixels do not dominate."""
    img = img.astype(np.float32)
    p_lo, p_hi = np.percentile(img, [lo, hi])
    return np.clip((img - p_lo) / (p_hi - p_lo + 1e-8), 0, 1).astype(np.float32)


def normalised_variance(img):
    """Sharpness score of one plane: variance divided by mean (higher is sharper)."""
    img = img.astype(np.float64)
    return img.var() / (img.mean() + 1e-8)


def best_focus(planes):
    """Index of the sharpest plane in a list of 2D arrays."""
    return int(np.argmax([normalised_variance(p) for p in planes]))


def pcc(pred, target):
    """Pearson correlation coefficient between two images."""
    p = pred.ravel().astype(np.float64)
    t = target.ravel().astype(np.float64)
    p = p - p.mean()
    t = t - t.mean()
    return float((p * t).sum() / np.sqrt((p ** 2).sum() * (t ** 2).sum()))


def ssim(pred, target):
    return float(structural_similarity(pred, target, data_range=1.0))


def mae(pred, target):
    return float(np.abs(pred - target).mean())


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(),
    )


class TinyUNet(nn.Module):
    """Two-level U-Net: input height and width must be multiples of 4."""
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


def predict_full(model, img, device="cpu"):
    """Predict a whole 2D image, cropping it to a multiple of 4 first."""
    h, w = (img.shape[0] // 4) * 4, (img.shape[1] // 4) * 4
    x = torch.from_numpy(img[:h, :w])[None, None].to(device)
    model.eval()
    with torch.no_grad():
        return model(x).squeeze().cpu().numpy(), (h, w)
