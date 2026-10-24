"""
Label efficiency of the pre-trained encoders.

The question masked pre-training can actually answer on BloodMNIST: if only 1%,
5% or 10% of the training images had labels, how good is a linear classifier
on top of each frozen encoder? (The supervised CNN at the same label fractions
is trained by train_baseline.py --train_frac.)

Features are extracted once per encoder; the probe is refitted on each subset,
with its regularisation chosen on the validation set.

Run:  python src/label_efficiency.py            -> results/labeleff_<encoder>.json
"""

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, metrics, stratified_subset
from probe import extract, fit_probe, load_encoder

FRACTIONS = (0.01, 0.05, 0.10, 1.0)
SUBSET_SEEDS = (0, 1, 2)


def probe_on_subset(f, y, idx, Cs=(0.1, 1.0, 10.0)):
    best = None
    for C in Cs:
        scaler, clf = fit_probe(f["train"][idx], y["train"][idx], C, max_iter=2000)
        val = metrics(y["val"], clf.predict(scaler.transform(f["val"])))
        if best is None or val["macro_f1"] > best[0]["macro_f1"]:
            best = (val, scaler, clf, C)
    val, scaler, clf, C = best
    test = metrics(y["test"], clf.predict(scaler.transform(f["test"])))
    return {"C": C, "n_train": int(len(idx)), "val": val, "test": test}


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    data = {s: load_split(s) for s in ("train", "val", "test")}
    y = {s: data[s][1].numpy() for s in data}
    for ckpt in sorted(glob.glob("checkpoints/mae_*_s*.pt")):
        name = os.path.basename(ckpt)[:-3]
        out = f"results/labeleff_{name}.json"
        if os.path.exists(out):
            continue
        enc = load_encoder(ckpt)
        f = {s: extract(enc, data[s][0]) for s in data}
        rows = []
        for frac in FRACTIONS:
            for ss in (SUBSET_SEEDS if frac < 1.0 else (0,)):
                idx = stratified_subset(y["train"], frac, ss).numpy() if frac < 1.0 \
                    else np.arange(len(y["train"]))
                r = probe_on_subset(f, y, idx)
                rows.append({"frac": frac, "subset_seed": ss, **r})
                print(f"{name} frac {frac:.2f} subset {ss}: test acc {r['test']['accuracy']:.3f} "
                      f"macroF1 {r['test']['macro_f1']:.3f}", flush=True)
        json.dump({"encoder": name, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
