"""
Stain / scanner shift: test BloodMNIST-trained models on images from elsewhere.

External data: the WBC segmentation dataset (Zheng et al., Micron 2018), which
also carries a class label per image (5 white-cell classes; the README warns
some labels were added by the authors without expert review).

  Dataset 1: 300 images, 120x120, a different rapid stain, yellow background
  Dataset 2: 100 images, 300x300, CellaVision, close to BloodMNIST's look

Nothing here is trained on these images; they are only used for testing.
Models still predict all 8 BloodMNIST classes, so predicting e.g. "platelet"
for a neutrophil counts as an error.

Each model is also tested after two test-time fixes (no retraining):
  scale - crop around the cell so it fills the same share of the frame as in BloodMNIST;
  stain - Reinhard colour normalisation to the BloodMNIST average colour statistics.

Run:  python src/external_eval.py     -> results/external.json
"""

import csv
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, normalise, CLASSES
from probe import extract, fit_probe, load_encoder
from train_baseline import build

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "segmentation_WBC")
# their label -> BloodMNIST class index
LABEL_MAP = {1: 6, 2: 4, 3: 5, 4: 1, 5: 0}     # neutrophil, lymphocyte, monocyte, eosinophil, basophil
FIVE = [0, 1, 4, 5, 6]


TARGET_CELL_FRACTION = 0.41      # median white-cell area fraction in BloodMNIST images


def _labels(ds):
    labels = {}
    with open(os.path.join(ROOT, f"Class Labels of {ds}.csv")) as fh:
        for row in csv.reader(fh):
            if row and row[0].strip().isdigit():
                labels[int(row[0])] = int(row[1])
    return labels


def scale_match(img):
    """Crop around the cell so it fills the same share of the frame as in BloodMNIST, then
    resize. The cell is found with the stain-based segmentation on a 64x64 preview."""
    from compartments import segment
    small = np.array(Image.fromarray(img).resize((64, 64), Image.BILINEAR))
    m = segment(small) > 0
    if m.sum() < 20:
        return small
    ys, xs = np.nonzero(m)
    c = min(1.0, float(np.sqrt(m.mean() / TARGET_CELL_FRACTION)))    # crop side, as a fraction
    H, W = img.shape[:2]
    side = int(c * min(H, W))
    y0 = int(np.clip(ys.mean() / 64 * H - side / 2, 0, H - side))
    x0 = int(np.clip(xs.mean() / 64 * W - side / 2, 0, W - side))
    return np.array(Image.fromarray(img[y0:y0 + side, x0:x0 + side]).resize((64, 64), Image.BILINEAR))


_LAB_TARGET = None


def stain_normalise(img):
    """Reinhard colour normalisation: match the image's mean and spread in LAB colour space
    to the BloodMNIST average."""
    global _LAB_TARGET
    from skimage.color import rgb2lab, lab2rgb
    if _LAB_TARGET is None:
        x = load_split("train")[0][::10]
        lab = np.stack([rgb2lab(im.permute(1, 2, 0).numpy() / 255.0) for im in x]).reshape(len(x), -1, 3)
        _LAB_TARGET = (lab.mean(1).mean(0), lab.std(1).mean(0))
    tm, ts = _LAB_TARGET
    lab = rgb2lab(img / 255.0)
    mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    return (np.clip(lab2rgb((lab - mu) / sd * ts + tm), 0, 1) * 255).round().astype(np.uint8)


def load_external(ds, variant="raw"):
    """-> uint8 tensor (N, 3, 64, 64), labels. variant: raw | scale | stain | scale+stain."""
    imgs, ys = [], []
    for i, lab in sorted(_labels(ds).items()):
        full = np.array(Image.open(os.path.join(ROOT, ds, f"{i:03d}.bmp")).convert("RGB"))
        if "scale" in variant:
            img = scale_match(full)
        else:
            img = np.array(Image.fromarray(full).resize((64, 64), Image.BILINEAR))
        if "stain" in variant:
            img = stain_normalise(img)
        imgs.append(img)
        ys.append(LABEL_MAP[lab])
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous(), np.array(ys)


VARIANTS = ("raw", "scale", "stain", "scale+stain")


def summarise(y, pred):
    recall = {CLASSES[c]: float((pred[y == c] == c).mean()) for c in FIVE if (y == c).any()}
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_recall": float(np.mean(list(recall.values()))), "recall": recall,
            "counts": {CLASSES[c]: int((y == c).sum()) for c in FIVE},
            "predicted": {CLASSES[c]: int((pred == c).sum()) for c in range(8)}}


@torch.no_grad()
def predict_supervised(ckpt, x):
    blob = torch.load(ckpt, map_location="cpu")
    m = build(blob["model"])
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m(normalise(x)).argmax(1).numpy()


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    ext = {(v, ds): load_external(ds, v) for v in VARIANTS for ds in ("Dataset 1", "Dataset 2")}

    def per_variant(predict):
        return {v: {ds: summarise(y, predict(x)) for (vv, ds), (x, y) in ext.items() if vv == v}
                for v in VARIANTS}

    def show(name, res):
        print(name, {v: [round(res[v][ds]["accuracy"], 3) for ds in ("Dataset 1", "Dataset 2")]
                     for v in VARIANTS}, flush=True)

    out = {}
    for ckpt in sorted(glob.glob("checkpoints/baseline_*_s*.pt")):
        name = os.path.basename(ckpt)[:-3]
        if "_f" in name.split("baseline_")[1]:          # skip label-fraction runs
            continue
        out[name] = per_variant(lambda x: predict_supervised(ckpt, x))
        show(name, out[name])

    train = load_split("train")
    for ckpt in sorted(glob.glob("checkpoints/mae_*_s*.pt")):
        name = os.path.basename(ckpt)[:-3]
        enc = load_encoder(ckpt)
        scaler, clf = fit_probe(extract(enc, train[0]), train[1].numpy(), C=1.0, max_iter=2000)
        out[name] = per_variant(lambda x: clf.predict(scaler.transform(extract(enc, x))))
        show(name, out[name])
    json.dump(out, open("results/external.json", "w"), indent=1)


if __name__ == "__main__":
    main()
