"""
Score the stain-derived compartment maps against expert ground truth.

Uses the public WBC segmentation dataset (Zheng et al., Micron 2018):
  Dataset 1: 300 images, 120x120, rapid stain, yellow background
  Dataset 2: 100 images, 300x300, CellaVision, same look as BloodMNIST

Split, fixed before any tuning:
  dev  = Dataset 1 (all) + Dataset 2 images 1-50   -> used to choose the method
  test = Dataset 2 images 51-100                   -> reported, never tuned on

Everything is scored at 64x64, the resolution the classifier sees, both per
pixel (Dice) and per 8x8 patch (does the patch get the right majority label).

Run:  python src/eval_masks.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from compartments import segment, patch_fractions

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "segmentation_WBC")
SIZE = 64


def load(ds, i):
    img = Image.open(os.path.join(ROOT, ds, f"{i:03d}.bmp")).convert("RGB")
    gt = Image.open(os.path.join(ROOT, ds, f"{i:03d}.png")).convert("L")
    img = np.array(img.resize((SIZE, SIZE), Image.BILINEAR))
    gt = np.array(gt.resize((SIZE, SIZE), Image.NEAREST))
    lab = np.zeros_like(gt)
    lab[gt > 64] = 1          # grey  = cytoplasm
    lab[gt > 192] = 2         # white = nucleus
    return img, lab


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0


def score(items):
    rows = []
    for ds, i in items:
        img, gt = load(ds, i)
        pr = segment(img)
        pg, pp = patch_fractions(gt).argmax(1), patch_fractions(pr).argmax(1)
        rows.append([
            dice(pr == 2, gt == 2),                 # nucleus
            dice(pr == 1, gt == 1),                 # cytoplasm
            dice(pr > 0, gt > 0),                   # whole cell
            (pp == pg).mean(),                      # patch majority-label agreement
        ])
    r = np.array(rows)
    return {
        "n": len(rows),
        "dice_nucleus": float(r[:, 0].mean()),
        "dice_cytoplasm": float(r[:, 1].mean()),
        "dice_cell": float(r[:, 2].mean()),
        "patch_agreement": float(r[:, 3].mean()),
        "dice_nucleus_median": float(np.median(r[:, 0])),
        "dice_cytoplasm_median": float(np.median(r[:, 1])),
    }


if __name__ == "__main__":
    splits = {
        "dev_dataset1": [("Dataset 1", i) for i in range(1, 301)],
        "dev_dataset2_1to50": [("Dataset 2", i) for i in range(1, 51)],
    }
    if "--test" in sys.argv:
        splits["test_dataset2_51to100"] = [("Dataset 2", i) for i in range(51, 101)]
    out = {k: score(v) for k, v in splits.items()}
    for k, v in out.items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    if "--test" in sys.argv:
        os.makedirs("results", exist_ok=True)
        json.dump(out, open("results/mask_quality.json", "w"), indent=2)
