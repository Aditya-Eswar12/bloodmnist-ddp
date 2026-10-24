"""
Can the classifier's confidence be trusted?

1. Calibration on the BloodMNIST test set: when the CNN says "90% sure", is it
   right 90% of the time? Measured with the Expected Calibration Error

       ECE = sum over bins b of (n_b / N) * | accuracy(b) - mean confidence(b) |

   (15 equal-width confidence bins). Then temperature scaling: divide the
   logits by one number T, fitted on the validation set by minimising
   cross-entropy, and measure ECE again.
2. Out-of-distribution warning: is max-softmax confidence lower on external
   images (other labs / another stain) than on BloodMNIST test images? Reported
   as the AUROC of "confidence separates in-distribution from external".
3. What the API's confidence gate (0.70) would reject on each set.

Run:  python src/calibration.py      -> results/calibration.json
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, normalise
from external_eval import load_external
from train_baseline import build

GATE = 0.70


@torch.no_grad()
def logits_of(model, x, bs=500):
    model.eval()
    return torch.cat([model(normalise(x[i:i + bs])) for i in range(0, len(x), bs)])


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = (correct[m].mean() - conf[m].mean()).abs().item()
            total += m.float().mean().item() * gap
            rows.append({"bin": [round(lo.item(), 3), round(hi.item(), 3)], "n": int(m.sum()),
                         "accuracy": round(correct[m].mean().item(), 4), "confidence": round(conf[m].mean().item(), 4)})
    return total, rows


def fit_temperature(logits, y):
    """One scalar T > 0 minimising validation cross-entropy of softmax(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def main(ckpt="checkpoints/baseline_cnn_s0.pt"):
    blob = torch.load(ckpt, map_location="cpu")
    model = build(blob["model"])
    model.load_state_dict(blob["state_dict"])
    (xv, yv), (xt, yt) = load_split("val"), load_split("test")
    lv, lt = logits_of(model, xv), logits_of(model, xt)

    e_before, rows_before = ece(lt.softmax(1), yt)
    T = fit_temperature(lv, yv)
    e_after, rows_after = ece((lt / T).softmax(1), yt)
    out = {"model": os.path.basename(ckpt), "temperature": T,
           "test_ece_before": e_before, "test_ece_after": e_after,
           "test_nll_before": float(F.cross_entropy(lt, yt)), "test_nll_after": float(F.cross_entropy(lt / T, yt)),
           "reliability_before": rows_before, "reliability_after": rows_after}

    conf_in = lt.softmax(1).max(1).values.numpy()
    acc_in = (lt.argmax(1) == yt).float().numpy()
    out["gate"] = {"bloodmnist_test": {"rejected": float((conf_in < GATE).mean()),
                                       "accuracy_kept": float(acc_in[conf_in >= GATE].mean()),
                                       "accuracy_rejected": float(acc_in[conf_in < GATE].mean())
                                       if (conf_in < GATE).any() else None}}
    for ds in ("Dataset 1", "Dataset 2"):
        x, y = load_external(ds)
        le = logits_of(model, x)
        conf_out = le.softmax(1).max(1).values.numpy()
        acc_out = (le.argmax(1).numpy() == y).astype(float)
        labels = np.r_[np.zeros(len(conf_in)), np.ones(len(conf_out))]      # 1 = external
        out["gate"][ds] = {"rejected": float((conf_out < GATE).mean()),
                           "accuracy_all": float(acc_out.mean()),
                           "accuracy_kept": float(acc_out[conf_out >= GATE].mean()) if (conf_out >= GATE).any() else None,
                           "mean_confidence": float(conf_out.mean()),
                           "auroc_low_confidence_flags_external": float(roc_auc_score(labels, -np.r_[conf_in, conf_out]))}
    out["gate"]["bloodmnist_test"]["mean_confidence"] = float(conf_in.mean())
    json.dump(out, open("results/calibration.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("reliability")}, indent=1))


if __name__ == "__main__":
    main()
