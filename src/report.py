"""Collect every result file into results/RESULTS.md (mean +/- sd over seeds).

Run:  python src/report.py
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

MAIN_ROWS = [
    ("baseline_cnn", "VGG-style CNN, supervised"),
    ("baseline_resnet18", "ResNet-18, supervised"),
    ("baseline_vit", "ViT, supervised from scratch"),
    ("baseline_cnn_19ep", "VGG-style CNN, 19 epochs (step-matched control for the cGAN row)"),
    ("baseline_cnn_cgan", "VGG-style CNN + cGAN top-up of 3 smallest classes"),
    ("baseline_cnn_strongaug", "VGG-style CNN + scale and colour augmentation"),
    ("mae_random", "MAE, random masking + linear probe"),
    ("mae_A", "MAE, A: hide cell patches 3x + linear probe"),
    ("mae_B", "MAE, B: cross-compartment views + linear probe"),
    ("mae_C", "MAE, C: B + confusion feedback + linear probe"),
]
COLS = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
        ("monocyte_recall", "Monocyte recall"), ("mono_as_ig", "Mono→IG (of 284)")]


def fmt(v, k):
    f = "{:.0f}" if k == "mono_as_ig" else "{:.3f}"
    v = np.asarray(v, dtype=float)
    return f.format(v.mean()) + (" ± " + f.format(v.std(ddof=1)) if len(v) > 1 else "")


def load_runs(folder):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        name = os.path.basename(path)[:-5]
        if "_s" not in name or name.startswith(("labeleff_", "amp_")):
            continue
        r = json.load(open(path))
        test = r["probe"]["test"] if "probe" in r else r.get("test")
        if test is not None:
            runs[name.rsplit("_s", 1)[0]].append(test)
    return runs


def main_table(runs):
    lines = ["| Model | Seeds | " + " | ".join(c for _, c in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for key, label in MAIN_ROWS:
        if key in runs:
            lines.append(f"| {label} | {len(runs[key])} | " +
                         " | ".join(fmt([t[k] for t in runs[key]], k) for k, _ in COLS) + " |")
    return "\n".join(lines)


def label_table(runs, folder):
    fr = [("01", 0.01), ("05", 0.05), ("10", 0.10)]
    rows = [("CNN, trained end to end", "baseline_cnn_f{}", "baseline_cnn"),
            ("ViT from scratch, end to end", "baseline_vit_f{}", "baseline_vit"),
            ("ViT fine-tuned from MAE (random masking)", "baseline_vit_maerandom_f{}", None),
            ("ViT fine-tuned from MAE (policy B)", "baseline_vit_maeB_f{}", None)]
    lines = ["| Method | 1% labels (119 img) | 5% (598) | 10% (1,196) | 100% |", "|---|---|---|---|---|"]
    for label, pat, full in rows:
        cells = [fmt([t["macro_f1"] for t in runs[pat.format(f)]], "macro_f1") if pat.format(f) in runs else "–"
                 for f, _ in fr]
        cells.append(fmt([t["macro_f1"] for t in runs[full]], "macro_f1") if full and full in runs else "–")
        if any(c != "–" for c in cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    # linear probes: average over encoder seeds and subset draws
    for pol, plabel in [("random", "random masking"), ("A", "policy A"), ("B", "policy B"), ("C", "policy C")]:
        vals = defaultdict(list)
        for path in glob.glob(os.path.join(folder, f"labeleff_mae_{pol}_s*.json")):
            for row in json.load(open(path))["rows"]:
                vals[row["frac"]].append(row["test"]["macro_f1"])
        if vals:
            cells = [fmt(vals[f], "macro_f1") if f in vals else "–" for f in (0.01, 0.05, 0.10, 1.0)]
            lines.append(f"| Linear probe on MAE ({plabel}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def external_table(folder):
    path = os.path.join(folder, "external.json")
    if not os.path.exists(path):
        return ""
    ext = json.load(open(path))
    variants = ["raw", "scale", "stain", "scale+stain"]
    sets = ("Dataset 1", "Dataset 2")
    groups = defaultdict(lambda: defaultdict(list))
    for name, by_v in ext.items():
        key = name.rsplit("_s", 1)[0]
        for v in variants:
            for ds in sets:
                if v in by_v:
                    groups[key][(v, ds)].append(by_v[v][ds]["accuracy"])
    head = " | ".join(f"D{ds[-1]} {v}" for ds in sets for v in variants)
    lines = [f"| Model | {head} |", "|---|" + "---|" * 8]
    for key in sorted(groups):
        cells = [fmt(groups[key][(v, ds)], "accuracy") if groups[key][(v, ds)] else "-"
                 for ds in sets for v in variants]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(folder="results"):
    runs = load_runs(folder)
    parts = ["# Results (BloodMNIST 64×64 test set: 3,421 images, unless stated)", "",
             "## Main comparison", "", main_table(runs), "",
             "## Label efficiency (test macro-F1)", "", label_table(runs, folder), ""]
    ext = external_table(folder)
    if ext:
        parts += ["## External images (accuracy; 5 white-cell classes; never used for training)", "",
                  "D1: 300 images, a different stain. D2: 100 CellaVision images from other labs. "
                  "raw = resized as is; scale = cell cropped to the BloodMNIST size in frame; "
                  "stain = Reinhard colour normalisation.", "", ext, ""]
    text = "\n".join(parts)
    with open(os.path.join(folder, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.reconfigure(encoding="utf-8")      # Windows consoles default to cp1252
    print(text)


if __name__ == "__main__":
    main()
