"""
Linear probe and confusion feedback.

Linear probe: freeze the encoder, turn every image into one feature vector,
and fit a plain logistic regression on top. It measures how much class
information the pre-training put into the features, with no fine-tuning to
hide a weak encoder.

Confusion feedback (element C): during pre-training, fit a quick probe, find
the pair of classes it confuses most, then ask which compartment the probe
relies on to tell that pair apart, by blanking one compartment at a time.
Masking is then shifted toward that compartment.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import normalise, metrics, CLASSES


def load_encoder(ckpt):
    """Rebuild a pre-trained ViT encoder, reading its size from the run's result file."""
    import json
    import os
    from models import ViT
    name = os.path.basename(ckpt)[:-3]
    cfg = {"dim": 128, "depth": 4, "heads": 2}
    res = os.path.join("results", name + ".json")
    if os.path.exists(res):
        a = json.load(open(res))["args"]
        cfg = {k: a.get(k, v) for k, v in cfg.items()}
    enc = ViT(**cfg)
    enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return enc


@torch.no_grad()
def extract(encoder, x, bs=512):
    encoder.eval()
    return torch.cat([encoder.features(normalise(x[i:i + bs])) for i in range(0, len(x), bs)]).numpy()


def fit_probe(ftr, ytr, C=1.0, max_iter=300):
    scaler = StandardScaler().fit(ftr)
    clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")
    clf.fit(scaler.transform(ftr), ytr)
    return scaler, clf


def linear_probe(encoder, data, Cs=(0.1, 1.0, 10.0)):
    """Fit on train, pick C on validation macro-F1, report validation and test."""
    f = {s: extract(encoder, data[s][0]) for s in ("train", "val", "test")}
    y = {s: data[s][1].numpy() for s in ("train", "val", "test")}
    best = None
    for C in Cs:
        scaler, clf = fit_probe(f["train"], y["train"], C, max_iter=1000)
        val = metrics(y["val"], clf.predict(scaler.transform(f["val"])))
        if best is None or val["macro_f1"] > best[0]["macro_f1"]:
            best = (val, scaler, clf, C)
    val, scaler, clf, C = best
    test = metrics(y["test"], clf.predict(scaler.transform(f["test"])))
    return {"C": C, "val": val, "test": test}


def blank_compartment(x, fr, k):
    """Replace every patch that is mostly compartment k with that image's
    background colour. x uint8 (B, 3, 64, 64), fr (B, 8, 8, 3)."""
    x = x.clone()
    hide = (fr.argmax(-1) == k)                                    # (B, 8, 8)
    hide_px = hide.repeat_interleave(8, 1).repeat_interleave(8, 2)[:, None]
    bg = x.float().flatten(2).quantile(0.95, dim=2).to(x.dtype)    # (B, 3)
    return torch.where(hide_px, bg[:, :, None, None].expand_as(x), x)


def confusion_feedback(encoder, data, fr_val, n_fit=4000, temperature=0.05, floor=0.2):
    """-> dict with the worst-confused pair and new (p_hide_cytoplasm, p_hide_nucleus).

    Uses training labels (probe) and validation labels (confusion), never test.
    """
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    sub = torch.randperm(len(xtr))[:n_fit]
    scaler, clf = fit_probe(extract(encoder, xtr[sub]), ytr[sub].numpy())
    pred = clf.predict(scaler.transform(extract(encoder, xva)))
    m = metrics(yva.numpy(), pred)
    cm = np.array(m["confusion"], dtype=float)
    rate = cm / cm.sum(1, keepdims=True)
    np.fill_diagonal(rate, 0)
    pair_rate = rate + rate.T                       # confusion in either direction
    a, b = np.unravel_index(pair_rate.argmax(), pair_rate.shape)

    sel = (yva == a) | (yva == b)
    xs, ys, frs = xva[sel], yva[sel].numpy(), fr_val[sel]
    base = (clf.predict(scaler.transform(extract(encoder, xs))) == ys).mean()
    drops = []
    for k in (1, 2):                                # cytoplasm, nucleus
        acc = (clf.predict(scaler.transform(extract(encoder, blank_compartment(xs, frs, k)))) == ys).mean()
        drops.append(base - acc)
    drops = np.array(drops)
    soft = np.exp((drops - drops.max()) / temperature)
    probs = floor * 0.5 + (1 - floor) * soft / soft.sum()
    return {"pair": [CLASSES[a], CLASSES[b]], "pair_accuracy": float(base),
            "drop_cytoplasm": float(drops[0]), "drop_nucleus": float(drops[1]),
            "view_probs": [float(probs[0]), float(probs[1])],
            "val_macro_f1": m["macro_f1"]}
