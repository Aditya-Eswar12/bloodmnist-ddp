"""
Is the conditional GAN any good, and does its output help a classifier?

1. Class fidelity: a trained CNN classifies 500 generated images per class.
   What fraction does it assign to the class the generator was asked for?
2. Feature distance (an FID-style score): compare real and generated images in
   the CNN's 128-number feature space with the Frechet distance

       FD = |mu_r - mu_g|^2 + Tr( S_r + S_g - 2 (S_r S_g)^(1/2) )

   mu = mean feature vector, S = covariance matrix, r = real, g = generated.
   0 means identical feature statistics. As a floor, the same score between
   real training and real test images is reported. (Standard FID uses an
   Inception network trained on ImageNet; this one uses our own CNN, so the
   numbers are only comparable with each other.)
3. Writes data/synthetic_cgan.npz: the three smallest classes topped up to
   2,000 images each, for `train_baseline.py --synthetic`.

Run:  python src/cgan_eval.py      -> results/cgan_eval.json
"""

import json
import os
import sys

import numpy as np
import torch
from scipy import linalg

sys.path.insert(0, os.path.dirname(__file__))
from cgan import Generator, synthesise
from data import load_split, normalise, CLASSES
from train_baseline import build


@torch.no_grad()
def cnn_features(model, x, bs=500):
    """Output of the 128-unit dense layer (after ReLU) of the VGG-style CNN."""
    model.eval()
    feats = []
    for i in range(0, len(x), bs):
        h = model.features(normalise(x[i:i + bs]))
        feats.append(model.head[:3](h))
    return torch.cat(feats).numpy().astype(np.float64)


@torch.no_grad()
def cnn_predict(model, x, bs=500):
    model.eval()
    return torch.cat([model(normalise(x[i:i + bs])).argmax(1) for i in range(0, len(x), bs)]).numpy()


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)            # newer SciPy returns only the matrix
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    covmean = covmean.real                      # drop tiny imaginary parts from round-off
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def main(n_per_class=500, top_up_to=2000, cnn_ckpt="checkpoints/baseline_cnn_s0.pt"):
    torch.manual_seed(0)
    blob = torch.load("checkpoints/cgan.pt", map_location="cpu")
    G = Generator(ngf=blob["ngf"])
    G.load_state_dict(blob["G"])
    cnn_blob = torch.load(cnn_ckpt, map_location="cpu")
    cnn = build(cnn_blob["model"])
    cnn.load_state_dict(cnn_blob["state_dict"])

    xtr, ytr = load_split("train")
    xte, yte = load_split("test")
    out = {"cgan_epoch": blob["epoch"], "classifier": os.path.basename(cnn_ckpt), "per_class": {}}
    for c, name in enumerate(CLASSES):
        fake = synthesise(G, c, n_per_class)
        pred = cnn_predict(cnn, fake)
        f_fake = cnn_features(cnn, fake)
        f_test = cnn_features(cnn, xte[yte == c])
        f_train = cnn_features(cnn, xtr[ytr == c][:2000])
        out["per_class"][name] = {
            "fidelity": float((pred == c).mean()),
            "most_common_wrong": CLASSES[int(np.bincount(pred[pred != c], minlength=8).argmax())]
            if (pred != c).any() else None,
            "fd_fake_vs_test": frechet(f_fake, f_test),
            "fd_train_vs_test": frechet(f_train, f_test),
        }
        print(name, {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in out["per_class"][name].items()}, flush=True)
    pc = out["per_class"].values()
    out["mean_fidelity"] = float(np.mean([v["fidelity"] for v in pc]))
    out["mean_fd_fake"] = float(np.mean([v["fd_fake_vs_test"] for v in pc]))
    out["mean_fd_real"] = float(np.mean([v["fd_train_vs_test"] for v in pc]))

    # top up the three smallest classes for the downstream experiment
    counts = torch.bincount(ytr, minlength=8)
    smallest = counts.argsort()[:3].tolist()
    xs, ys = [], []
    for c in smallest:
        n = int(top_up_to - counts[c])
        xs.append(synthesise(G, c, n))
        ys.append(torch.full((n,), c, dtype=torch.long))
    np.savez_compressed("data/synthetic_cgan.npz", images=torch.cat(xs).numpy(), labels=torch.cat(ys).numpy())
    out["synthetic_added"] = {CLASSES[c]: int(top_up_to - counts[c]) for c in smallest}
    json.dump(out, open("results/cgan_eval.json", "w"), indent=1)
    print("mean fidelity", round(out["mean_fidelity"], 3), "| FD fake", round(out["mean_fd_fake"], 1),
          "vs real floor", round(out["mean_fd_real"], 1), "| added", out["synthetic_added"])


if __name__ == "__main__":
    main()
