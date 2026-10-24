"""
Make every figure used in the interview slides, from the saved results.

Run from the repository root:
    python figures/make_figures.py

Output: figures/*.png  (about 20 seconds; no training is run)
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # draw to files, no window
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# colours: one categorical order, used the same way in every figure
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

SHORT = ["baso", "eos", "ebl", "IG", "lymph", "mono", "neut", "plt"]


def load(name):
    return json.load(open(os.path.join("results", name + ".json")))


def seeds(prefix, key="macro_f1"):
    vals = [json.load(open(f))["test"][key] for f in sorted(glob.glob(f"results/{prefix}_s[0-9].json"))]
    return np.array(vals)


# ------------------------------------------------------------------ 1. confusion matrices
def fig_confusion():
    july = np.array([[244, 0, 0, 0, 0, 0, 0, 0], [2, 614, 0, 0, 0, 0, 8, 0], [8, 1, 276, 11, 10, 0, 4, 1],
                     [97, 0, 0, 408, 0, 1, 73, 0], [13, 0, 2, 23, 205, 0, 0, 0], [13, 0, 1, 171, 4, 66, 27, 2],
                     [13, 1, 0, 2, 0, 0, 650, 0], [0, 0, 0, 0, 0, 0, 0, 470]])
    fixed = np.array(load("baseline_cnn_s0")["test"]["confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cm, title in [(axes[0], july, "July: last-epoch weights — 85.7% accuracy"),
                          (axes[1], fixed, "Re-run: best validation checkpoint — 98.0%")]:
        rate = cm / cm.sum(1, keepdims=True)                 # each row sums to 1 = recall view
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if rate[i, j] > 0.5 else INK)
        ax.set_xticks(range(8), SHORT, rotation=45)
        ax.set_yticks(range(8), SHORT)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(title, fontsize=11, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].add_patch(plt.Rectangle((2.5, 4.5), 1, 1, fill=False, ec=ORANGE, lw=2.5))   # mono -> IG cell
    axes[0].annotate("171 of 284 monocytes called immature granulocyte", xy=(3, 5.5), xytext=(3.5, 9.9),
                     fontsize=9, color=INK, ha="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
    plt.tight_layout(w_pad=4)
    plt.savefig(os.path.join(OUT, "fig1_confusion_before_after.png"), dpi=200)
    plt.close()


def fig_confusion_final():
    """The final model's test confusion matrix on its own (used on slide 4)."""
    cm = np.array(load("baseline_cnn_s0")["test"]["confusion"])
    rate = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if rate[i, j] > 0.5 else INK)
    ax.set_xticks(range(8), SHORT, rotation=45)
    ax.set_yticks(range(8), SHORT)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Test set, 3,421 images (seed 0): 98.0% correct", fontsize=11, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig1b_confusion_final.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------------ 2. dataset examples
def fig_dataset():
    from data import load_split, CLASSES
    x, y = load_split("train")
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    for c in range(8):
        for r in range(2):
            i = rng.choice(np.where(y.numpy() == c)[0])
            axes[r, c].imshow(x[i].permute(1, 2, 0).numpy())
            axes[r, c].axis("off")
        axes[0, c].set_title(CLASSES[c].replace("_", "\n"), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_dataset_examples.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------------ 3. compartment maps
def fig_compartments():
    from data import load_split, CLASSES
    from compartments import segment
    from matplotlib.colors import ListedColormap
    x, y = load_split("train")
    cmap = ListedColormap(["#f3f1ec", "#9db7d5", "#3b3470"])
    picks = [5, 3, 6, 4, 1]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 5, figsize=(9, 3.9))
    for j, c in enumerate(picks):
        for i in rng.permutation(np.where(y.numpy() == c)[0])[:40]:
            img = x[i].permute(1, 2, 0).numpy()
            m = segment(img)
            if 0.08 < (m == 1).mean() < 0.45 and (m == 2).mean() > 0.08:
                break
        axes[0, j].imshow(img)
        axes[1, j].imshow(m, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        axes[0, j].set_title(CLASSES[c].replace("_", " "), fontsize=9)
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("image", fontsize=9)
    axes[1, 0].set_ylabel("stain-derived map", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig3_compartments.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------------ 4. label efficiency
def fig_label_efficiency():
    fr = [1, 5, 10, 100]
    series = [
        ("CNN, trained end to end", BLUE, ["baseline_cnn_f01", "baseline_cnn_f05", "baseline_cnn_f10", "baseline_cnn"]),
        ("ViT from scratch", ORANGE, ["baseline_vit_f01", "baseline_vit_f05", "baseline_vit_f10", "baseline_vit"]),
        ("ViT pre-trained (MAE), random masking", AQUA,
         ["baseline_vit_maerandom_f01", "baseline_vit_maerandom_f05", "baseline_vit_maerandom_f10", None]),
        ("ViT pre-trained (MAE), anatomy masking", YELLOW,
         ["baseline_vit_maeB_f01", "baseline_vit_maeB_f05", "baseline_vit_maeB_f10", None]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, colour, names in series:
        xs, ms, sds = [], [], []
        for f, n in zip(fr, names):
            if n is None:
                continue
            v = seeds(n)
            xs.append(f)
            ms.append(v.mean())
            sds.append(v.std(ddof=1) if len(v) > 1 else 0)
        ax.errorbar(xs, ms, yerr=sds, color=colour, lw=2, marker="o", ms=7, capsize=3,
                    mec="white", mew=1.5, label=label)
        if names[0] in ("baseline_cnn_f01", "baseline_vit_f01", "baseline_vit_maerandom_f01"):
            ax.text(0.86, ms[0], f"{ms[0]:.2f}", color=INK, fontsize=9, va="center", ha="right")
    ax.set_xscale("log")
    ax.set_xticks(fr, ["1%\n(119 images)", "5%", "10%", "100%\n(11,959)"])
    ax.set_xlim(0.6, 150)
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("test macro-F1")
    ax.set_xlabel("share of training images with labels")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("With 1% of labels, pre-training lifts the ViT from 0.71 to 0.82", fontsize=11, loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig4_label_efficiency.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------------ 5. external images
def fig_external():
    ext = json.load(open("results/external.json"))

    def acc(prefix, variant, ds):
        return np.mean([v[variant][ds]["accuracy"] for k, v in ext.items() if k.rsplit("_s", 1)[0] == prefix])

    groups = ["BloodMNIST\ntest (home)", "other labs\n(CellaVision)", "other labs,\nscale-matched",
              "other stain", "other stain,\nstain-normalised"]
    cnn = [seeds("baseline_cnn", "accuracy").mean(), acc("baseline_cnn", "raw", "Dataset 2"),
           acc("baseline_cnn", "scale", "Dataset 2"), acc("baseline_cnn", "raw", "Dataset 1"),
           acc("baseline_cnn", "stain", "Dataset 1")]
    aug = [seeds("baseline_cnn_strongaug", "accuracy").mean(), acc("baseline_cnn_strongaug", "raw", "Dataset 2"),
           acc("baseline_cnn_strongaug", "scale", "Dataset 2"), acc("baseline_cnn_strongaug", "raw", "Dataset 1"),
           acc("baseline_cnn_strongaug", "stain", "Dataset 1")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = np.arange(len(groups))
    w = 0.38
    b1 = ax.bar(xs - w / 2 - 0.01, cnn, w, color=BLUE, label="CNN")
    b2 = ax.bar(xs + w / 2 + 0.01, aug, w, color=ORANGE, label="CNN + scale/colour augmentation")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.0%}",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(xs, groups, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("98% at home, 24% and 8% on other labs' images", fontsize=11, loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig5_external.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------------ 6. GAN before / after
def fig_gan():
    a = Image.open("results/cgan_dcgan_collapsed/cgan_samples_ep005.png").convert("RGB")
    b = Image.open("results/cgan_samples_ep025.png").convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, im, t in [(axes[0], a, "First GAN: every sample of a class identical\n(mode collapse)"),
                      (axes[1], b, "After the fixes: varied, class-specific cells")]:
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(t, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig6_gan_before_after.png"), dpi=170)
    plt.close()


if __name__ == "__main__":
    for f in (fig_confusion, fig_confusion_final, fig_dataset, fig_compartments, fig_label_efficiency, fig_external, fig_gan):
        f()
        print("made", f.__name__)
    print("figures in", OUT)
