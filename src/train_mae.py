"""
Masked pre-training with a chosen masking policy, then a linear probe.

Run:  python src/train_mae.py --policy random --epochs 100 --seed 0
      python src/train_mae.py --policy C --epochs 100 --seed 0

Policies are defined in masking.py. Pre-training uses the training images
only. Policies random / A / B never look at a label. Policy C uses training
and validation labels for its feedback step, never test labels.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, load_fractions, normalise, dihedral
from masking import make_policy, sample
from models import ViT, MAE
from probe import linear_probe, confusion_feedback
from amp import autocast, make_scaler


def pretrain(args, data, fr, log=print):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    encoder = ViT(dim=args.dim, depth=args.depth, heads=args.heads)
    mae = MAE(encoder).to(device)
    scaler = make_scaler(device, args.amp)
    policy = make_policy(args.policy, tuple(args.pi))
    state = {"view_probs": (0.5, 0.5)}
    history = {"loss": [], "feedback": []}

    xtr = data["train"][0]
    bs = args.batch
    steps_per_epoch = len(xtr) // bs
    total, warmup = args.epochs * steps_per_epoch, args.warmup * steps_per_epoch
    opt = torch.optim.AdamW(mae.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    step = 0
    for ep in range(1, args.epochs + 1):
        mae.train()
        t0, perm, tot = time.time(), torch.randperm(len(xtr)), 0.0
        for i in range(0, steps_per_epoch * bs, bs):
            idx = perm[i:i + bs]
            x, f = dihedral(normalise(xtr[idx]), fr["train"][idx])
            keep, masked = sample(policy(f.reshape(bs, -1, 3), state), args.mask_ratio)
            x, keep, masked = x.to(device), keep.to(device), masked.to(device)
            # linear warm-up then cosine decay
            scale = step / warmup if step < warmup else \
                0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
            for g in opt.param_groups:
                g["lr"] = args.lr * scale
            with autocast(device, args.amp):
                loss = mae(x, keep, masked)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            step += 1
        history["loss"].append(tot / steps_per_epoch)
        msg = f"  ep {ep:3d} loss {history['loss'][-1]:.4f} {time.time() - t0:.0f}s"
        if args.policy == "C" and ep % args.feedback_every == 0 and ep < args.epochs:
            encoder.cpu()                       # the probe and feedback run on the CPU
            fb = confusion_feedback(encoder, data, fr["val"])
            encoder.to(device)
            state["view_probs"] = tuple(fb["view_probs"])
            history["feedback"].append({"epoch": ep, **fb})
            msg += (f" | feedback: {fb['pair'][0]} vs {fb['pair'][1]}, drop cyto "
                    f"{fb['drop_cytoplasm']:+.3f} nucleus {fb['drop_nucleus']:+.3f} -> "
                    f"p(hide cyto, hide nucleus) = ({fb['view_probs'][0]:.2f}, {fb['view_probs'][1]:.2f})")
        log(msg)
    return encoder.cpu(), history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="random", choices=["random", "A", "B", "C"])
    p.add_argument("--pi", type=float, nargs=3, default=[1.0, 3.0, 3.0],
                   help="policy A: masking propensity for background, cytoplasm, nucleus")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--threads", type=int, default=0, help="0 = PyTorch default")
    p.add_argument("--feedback_every", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="mixed precision (fp16 + GradScaler on GPU)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    data = {s: load_split(s) for s in ("train", "val", "test")}
    fr = {s: load_fractions(s) for s in ("train", "val")}
    name = f"mae_{args.policy}{args.tag}_s{args.seed}"
    print(name, vars(args))
    t0 = time.time()
    encoder, history = pretrain(args, data, fr)
    probe = linear_probe(encoder, data)
    t = probe["test"]
    print(f"  PROBE val macroF1 {probe['val']['macro_f1']:.4f} | TEST acc {t['accuracy']:.4f} "
          f"macroF1 {t['macro_f1']:.4f} mono recall {t['monocyte_recall']:.4f} mono->IG {t['mono_as_ig']}")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(encoder.state_dict(), f"checkpoints/{name}.pt")
    json.dump({"name": name, "args": vars(args), "minutes": (time.time() - t0) / 60,
               "history": history, "probe": probe},
              open(f"{args.out}/{name}.json", "w"), indent=1)
