"""Data loading, augmentation and metrics shared by every experiment."""

import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

ROOT = os.path.join(os.path.dirname(__file__), "..")
NPZ = os.path.join(ROOT, "data", "bloodmnist_64.npz")
COMPARTMENTS = os.path.join(ROOT, "data", "compartments_64_p8.npz")

CLASSES = ["basophil", "eosinophil", "erythroblast", "immature_granulocyte",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
MONO, IG = 5, 3


def load_split(split):
    """-> images uint8 tensor (N, 3, 64, 64), labels int64 tensor (N,)"""
    d = np.load(NPZ)
    x = torch.from_numpy(d[f"{split}_images"]).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(d[f"{split}_labels"].ravel().astype(np.int64))
    return x, y


def load_fractions(split):
    """-> per-patch [background, cytoplasm, nucleus] fractions, (N, 8, 8, 3).
    Produced once by `python src/make_compartments.py`."""
    f = np.load(COMPARTMENTS)[split]
    return torch.from_numpy(f).view(-1, 8, 8, 3)


def normalise(x_uint8):
    """uint8 (B, 3, H, W) -> float in [-1, 1]"""
    return x_uint8.float().div_(127.5).sub_(1.0)


def dihedral(x, fr=None):
    """One random flip / 90-degree rotation for the whole batch. A blood cell
    has no canonical orientation, so all 8 are label-preserving. The patch
    fraction grid `fr` (B, 8, 8, 3) is transformed the same way."""
    k = int(torch.randint(4, (1,)))
    flip = bool(torch.randint(2, (1,)))
    x = torch.rot90(x, k, dims=(2, 3))
    if flip:
        x = x.flip(3)
    if fr is None:
        return x
    fr = torch.rot90(fr, k, dims=(1, 2))
    if flip:
        fr = fr.flip(2)
    return x, fr


def stratified_subset(y, frac, seed, min_per_class=2):
    """Indices of a class-balanced random subset holding `frac` of each class
    (at least `min_per_class` images per class). Same seed -> same subset."""
    g = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = []
    for c in np.unique(y):
        members = np.where(y == c)[0]
        n = max(min_per_class, int(round(frac * len(members))))
        idx.append(g.choice(members, n, replace=False))
    return torch.from_numpy(np.sort(np.concatenate(idx)))


def class_weights(y, n_classes=8):
    """Inverse-frequency ('balanced') weights: N / (n_classes * count_c)."""
    counts = torch.bincount(y, minlength=n_classes).float()
    return len(y) / (n_classes * counts)


def metrics(y_true, y_pred, n_classes=8):
    """Everything we report, from one confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    tp = cm.diagonal().astype(float)
    recall = tp / np.maximum(cm.sum(1), 1)
    precision = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "monocyte_recall": float(recall[MONO]),
        "mono_as_ig": int(cm[MONO, IG]),
        "ig_as_mono": int(cm[IG, MONO]),
        "recall": recall.round(4).tolist(),
        "precision": precision.round(4).tolist(),
        "confusion": cm.tolist(),
    }


def strong_augment(x):
    """Extra augmentation for robustness to other labs (x: float batch in [-1, 1]).
    One random draw per batch:
      * scale: zoom out to 60-100% (cell looks smaller, border replicated) or in to 100-125%;
        external CellaVision images show cells at about half the area fraction of BloodMNIST;
      * colour: brightness +-0.15, contrast x0.8-1.2, per-channel gain x0.8-1.2 and shift +-0.1,
        saturation x0.7-1.3,
        imitating a different stain intensity or white balance."""
    B, C, H, W = x.shape
    s = float(torch.empty(1).uniform_(0.6, 1.25))
    if s < 1.0:                                                    # zoom out
        h = int(round(H * s))
        small = torch.nn.functional.interpolate(x, size=(h, h), mode="bilinear", align_corners=False)
        p0 = (H - h) // 2
        x = torch.nn.functional.pad(small, (p0, H - h - p0, p0, H - h - p0), mode="replicate")
    elif s > 1.0:                                                  # zoom in (centre crop)
        h = int(round(H / s))
        o = (H - h) // 2
        x = torch.nn.functional.interpolate(x[:, :, o:o + h, o:o + h], size=(H, W), mode="bilinear",
                                            align_corners=False)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * float(torch.empty(1).uniform_(0.8, 1.2)) + mean + float(torch.empty(1).uniform_(-0.15, 0.15))
    x = x * torch.empty(1, C, 1, 1).uniform_(0.8, 1.2) + torch.empty(1, C, 1, 1).uniform_(-0.1, 0.1)
    grey = x.mean(dim=1, keepdim=True)
    x = grey + (x - grey) * float(torch.empty(1).uniform_(0.7, 1.3))
    return x.clamp(-1, 1)
