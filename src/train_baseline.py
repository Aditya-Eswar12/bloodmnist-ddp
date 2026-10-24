"""
Supervised baselines: the VGG-style CNN, ResNet-18, or a ViT trained from scratch.

Same recipe as the original Keras notebook (inverse-frequency class weights,
MixUp alpha=0.2), with two fixes: the checkpoint is chosen by validation
macro-F1, and every run is seeded and repeated.

Run:  python src/train_baseline.py --model cnn --epochs 30 --seeds 0 1 2
      python src/train_baseline.py --model cnn --train_frac 0.05 --tag _f05   (5% of the labels)
      python src/train_baseline.py --model cnn --amp                          (mixed precision)
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, normalise, dihedral, class_weights, metrics, stratified_subset, strong_augment
from models import VGGStyleCNN, resnet18, ViT
from amp import autocast, make_scaler


def build(name):
    return {"cnn": VGGStyleCNN, "resnet18": resnet18, "vit": ViT}[name]()


@torch.no_grad()
def predict(model, x, bs=512, device="cpu", amp=False):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        with autocast(device, amp):
            out.append(model(normalise(x[i:i + bs].to(device))).argmax(1).cpu())
    return torch.cat(out)


def train_supervised(model, data, epochs, lr, bs=64, mixup_alpha=0.2, wd=1e-4,
                     device="cpu", amp=False, log=print, eval_every=1, strong_aug=False):
    """Trains `model`, returns (best state dict, best validation metrics)."""
    (xtr, ytr), (xva, yva) = data["train"], data["val"]
    model.to(device)
    w = class_weights(ytr).to(device)
    bs = min(bs, len(xtr))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * (len(xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = make_scaler(device, amp)
    best, best_f1, best_val = None, -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        t0, perm, tot = time.time(), torch.randperm(len(xtr)), 0.0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x, y = dihedral(normalise(xtr[idx].to(device))), ytr[idx].to(device)
            if strong_aug:
                x = strong_augment(x)
            # MixUp: blend each image with another one, and blend the losses the same way
            lam = float(np.random.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.0
            j = torch.randperm(bs, device=device)
            with autocast(device, amp):
                logits = model(lam * x + (1 - lam) * x[j])
                loss = lam * F.cross_entropy(logits.float(), y, weight=w) \
                    + (1 - lam) * F.cross_entropy(logits.float(), y[j], weight=w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()      # scaling is a no-op unless fp16 on a GPU
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        if ep % eval_every and ep != epochs:
            continue
        val = metrics(yva.numpy(), predict(model, xva, device=device, amp=amp).numpy())
        if val["macro_f1"] > best_f1:
            best_f1, best_val = val["macro_f1"], val
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"  ep {ep:2d} loss {tot / max(1, len(perm) // bs):.3f} val acc {val['accuracy']:.3f} "
            f"macroF1 {val['macro_f1']:.3f} mono recall {val['monocyte_recall']:.3f} "
            f"{time.time() - t0:.0f}s")
    return best, best_val


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn", choices=["cnn", "resnet18", "vit"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--train_frac", type=float, default=1.0,
                   help="fraction of the training labels to use (stratified by class)")
    p.add_argument("--synthetic", default="",
                   help="npz of extra (generated) training images: keys images (N,3,64,64) uint8, labels")
    p.add_argument("--init", default="",
                   help="ViT only: start from a masked-pre-trained encoder checkpoint (fine-tuning)")
    p.add_argument("--strong_aug", action="store_true",
                   help="add scale (zoom 0.6-1.25) and colour jitter, for robustness to other labs")
    p.add_argument("--amp", action="store_true", help="mixed precision (fp16 on GPU, bf16 on CPU)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0, help="0 = PyTorch default")
    p.add_argument("--tag", default="")
    p.add_argument("--no_save", action="store_true", help="do not write the model checkpoint")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    full = {s: load_split(s) for s in ("train", "val", "test")}
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = dict(full)
        if args.train_frac < 1.0:
            idx = stratified_subset(full["train"][1], args.train_frac, seed)
            data["train"] = (full["train"][0][idx], full["train"][1][idx])
        if args.synthetic:
            syn = np.load(args.synthetic)
            data["train"] = (torch.cat([data["train"][0], torch.from_numpy(syn["images"])]),
                             torch.cat([data["train"][1], torch.from_numpy(syn["labels"]).long()]))
        # keep the number of optimiser steps roughly constant when the training set shrinks
        # (capped at 300 epochs); validation is then checked about 30 times per run
        epochs = args.epochs if args.train_frac >= 1.0 else \
            int(min(args.epochs / args.train_frac, 300))
        model = build(args.model)
        if args.init:
            from probe import load_encoder
            enc = load_encoder(args.init)                     # same ViT, without a classifier head
            missing, unexpected = model.load_state_dict(enc.state_dict(), strict=False)
            assert not unexpected and set(missing) <= {"head.weight", "head.bias"}, (missing, unexpected)
        name = f"baseline_{args.model}{args.tag}_s{seed}"
        print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
              f"{len(data['train'][1])} training images, {epochs} epochs, amp={args.amp}")
        t0 = time.time()
        state, val = train_supervised(model, data, epochs, args.lr, device=device, amp=args.amp,
                                      eval_every=max(1, epochs // 30), strong_aug=args.strong_aug)
        model.load_state_dict(state)
        test = metrics(full["test"][1].numpy(),
                       predict(model, full["test"][0], device=device, amp=args.amp).numpy())
        print(f"  TEST acc {test['accuracy']:.4f} macroF1 {test['macro_f1']:.4f} "
              f"mono recall {test['monocyte_recall']:.4f} mono->IG {test['mono_as_ig']}")
        if not args.no_save:
            torch.save({"model": args.model, "state_dict": state}, f"checkpoints/{name}.pt")
        json.dump({"model": args.model, "seed": seed, "epochs": epochs, "lr": args.lr,
                   "train_frac": args.train_frac, "synthetic": args.synthetic, "init": args.init, "strong_aug": args.strong_aug, "n_train": int(len(data["train"][1])),
                   "amp": args.amp, "device": str(device),
                   "minutes": (time.time() - t0) / 60, "val": val, "test": test},
                  open(f"{args.out}/{name}.json", "w"), indent=1)
