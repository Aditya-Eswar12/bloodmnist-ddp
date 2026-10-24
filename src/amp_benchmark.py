"""
Mixed-precision benchmark: same model, same batches, fp32 vs AMP.

Measures training throughput (images per second), peak GPU memory, and whether
the loss still goes down the same way. Meant for a GPU (Colab T4 or better);
on a laptop CPU, bfloat16 autocast is usually SLOWER because most CPUs have no
fast 16-bit math units.

Run:  python src/amp_benchmark.py --model resnet18 --steps 200   -> results/amp_<model>_<device>.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from amp import autocast, make_scaler
from data import load_split, normalise
from train_baseline import build


def run(model_name, amp, steps, bs, device):
    torch.manual_seed(0)
    model = build(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = make_scaler(device, amp)
    x, y = load_split("train")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses, t0 = [], None
    for step in range(steps + 10):
        if step == 10:                                   # skip warm-up steps (kernel selection, caching)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        idx = torch.randint(0, len(x), (bs,))
        xb, yb = normalise(x[idx].to(device)), y[idx].to(device)
        with autocast(device, amp):
            loss = F.cross_entropy(model(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    return {"amp": amp, "images_per_sec": steps * bs / secs, "seconds": secs,
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
            "loss_first20": sum(losses[:20]) / 20, "loss_last20": sum(losses[-20:]) / 20,
            "grad_scale_final": scaler.get_scale() if scaler.is_enabled() else None}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18", choices=["cnn", "resnet18", "vit"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    rows = [run(args.model, amp, args.steps, args.batch, device) for amp in (False, True)]
    out = {"model": args.model, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
           "batch": args.batch, "steps": args.steps, "fp32": rows[0], "amp": rows[1],
           "speedup": rows[1]["images_per_sec"] / rows[0]["images_per_sec"]}
    if rows[0]["peak_mem_mb"]:
        out["memory_ratio"] = rows[1]["peak_mem_mb"] / rows[0]["peak_mem_mb"]
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/amp_{args.model}_{device.type}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
