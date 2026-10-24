"""
End-to-end demo on SIMULATED slides.

BloodMNIST has single cells, not patients or smears. To exercise the slide-level
pipeline we build "virtual smears": for each slide we draw test-set cell images
in chosen proportions (a normal differential, raised monocytes, raised
eosinophils, immature granulocytes, ...). The true mixture is known, so we can
check whether the classifier + SQL recover it. These are not real patients.

Run:  .venv/Scripts/python serve/demo_cohort.py   -> results/cohort_demo.json, results/cohort_demo.md
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agents import SlidePipeline, ReportAgent    # noqa: E402
from data import load_split, CLASSES               # noqa: E402
from db import Store                               # noqa: E402
from model_service import Classifier               # noqa: E402

# % of white cells (erythroblasts per 100 white cells); platelets added on top as clutter
PROFILES = {
    "normal":          {"neutrophil": 60.5, "lymphocyte": 30, "monocyte": 6, "eosinophil": 3, "basophil": 0.5},
    "monocytosis":     {"neutrophil": 52, "lymphocyte": 30, "monocyte": 15, "eosinophil": 2, "basophil": 1},
    "eosinophilia":    {"neutrophil": 55, "lymphocyte": 25, "monocyte": 6, "eosinophil": 13, "basophil": 1},
    "left_shift":      {"neutrophil": 68, "lymphocyte": 18, "monocyte": 5, "eosinophil": 2,
                        "immature_granulocyte": 7},
    "lymphocytosis":   {"neutrophil": 35, "lymphocyte": 55, "monocyte": 7, "eosinophil": 2, "basophil": 1},
    "nucleated_reds":  {"neutrophil": 60.5, "lymphocyte": 30, "monocyte": 6, "eosinophil": 3, "basophil": 0.5,
                        "erythroblast": 6},
}
WBC_PER_SLIDE = 200


def build_slide(rng, x, y, profile):
    idx, truth = [], []
    for name, pct in profile.items():
        c = CLASSES.index(name)
        n = int(round(WBC_PER_SLIDE * pct / 100))
        pool = np.where(y == c)[0]
        idx += list(rng.choice(pool, n, replace=True))
        truth += [name] * n
    pool = np.where(y == CLASSES.index("platelet"))[0]       # platelet clutter, excluded by the SQL
    idx += list(rng.choice(pool, 20, replace=True))
    truth += ["platelet"] * 20
    order = rng.permutation(len(idx))
    return x[torch.as_tensor(np.array(idx)[order])], [truth[i] for i in order]


def main():
    rng = np.random.default_rng(0)
    x, y = load_split("test")
    y = y.numpy()
    db = "results/cohort_demo.sqlite"
    if os.path.exists(db):
        os.remove(db)
    store = Store(db)
    pipe = SlidePipeline(Classifier(), store, ReportAgent())
    print("model", pipe.version, "| report mode", pipe.writer.mode)

    out = {"model_version": pipe.version, "report_mode": pipe.writer.mode, "slides": []}
    for rep in range(2):                                      # two slides per profile
        for name, profile in PROFILES.items():
            sid = f"SIM-{name}-{rep + 1}"
            xs, truth = build_slide(rng, x, y, profile)
            pipe.add_cells(sid, xs, true_classes=truth, source="simulated")
            r = pipe.report(sid)
            true_pct = {k: v for k, v in profile.items()}
            got = {row["cell_type"]: row["pct"] for row in r["differential"]}
            out["slides"].append({"slide_id": sid, "profile": name, "status": r["status"],
                                  "author": r["author"], "true_pct": true_pct, "measured_pct": got,
                                  "flags": {row["cell_type"]: row["flag"] for row in r["differential"]
                                            if row["flag"] != "normal"},
                                  "verifier_notes": r["verifier_notes"], "report": r["text"]})
            print(sid, r["status"], {k: got.get(k) for k in profile})

    out["abnormal_cohort"] = store.query("abnormal_cohort")
    out["confidence_trend"] = store.query("confidence_trend")
    out["per_class_accuracy"] = store.query("per_class_accuracy")
    json.dump(out, open("results/cohort_demo.json", "w"), indent=1)

    # short markdown summary
    lines = ["# Simulated-slide demo", "", f"Model `{pipe.version}`, report writer: {pipe.writer.mode}.", "",
             "| Slide | Profile | Status | Flags raised |", "|---|---|---|---|"]
    for s in out["slides"]:
        flags = ", ".join(f"{k} {v}" for k, v in s["flags"].items()) or "none"
        lines.append(f"| {s['slide_id']} | {s['profile']} | {s['status']} | {flags} |")
    lines += ["", "## Example report", "", "```", out["slides"][1]["report"], "```"]
    open("results/cohort_demo.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("abnormal cohort rows:", len(out["abnormal_cohort"]))


if __name__ == "__main__":
    main()
