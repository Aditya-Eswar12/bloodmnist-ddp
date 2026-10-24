#!/usr/bin/env bash
# Phase 3: does masked pre-training help a ViT when labels are scarce?
# Same ViT, fine-tuned end to end with 1%, 5%, 10% of the labels, starting from
#   (a) random weights, (b) the random-masking MAE encoder, (c) the policy-B MAE encoder.
# Seed 0 only (CPU budget). Safe to re-run: finished runs are skipped.
cd "$(dirname "$0")"
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=python

run() {
  local name=$1; shift
  if [ -f "results/$name.json" ]; then echo "skip $name"; return; fi
  echo "start $name  $(date +%H:%M)"
  "$PY" -u -W ignore "$@" > "logs/$name.log" 2>&1 &
}

for f in 01 05 10; do
  common="--model vit --epochs 8 --lr 5e-4 --seeds 0 --train_frac 0.$f --threads 2 --no_save"
  run baseline_vit_f${f}_s0            src/train_baseline.py $common --tag _f$f
  run baseline_vit_maerandom_f${f}_s0  src/train_baseline.py $common --tag _maerandom_f$f --init checkpoints/mae_random_s0.pt
  run baseline_vit_maeB_f${f}_s0       src/train_baseline.py $common --tag _maeB_f$f --init checkpoints/mae_B_s0.pt
  wait
done
# CNN trained with scale + colour augmentation, for robustness to other labs (3 seeds)
for s in 0 1 2; do
  run baseline_cnn_strongaug_s$s src/train_baseline.py --model cnn --epochs 15 --seeds $s --strong_aug       --tag _strongaug --threads 2
done; wait

# re-run the external test so it includes the new models and the test-time fixes
rm -f results/external.json
THREADS=4 "$PY" -u -W ignore src/external_eval.py > logs/external_eval.log 2>&1

"$PY" src/report.py
echo "phase 3 done $(date +%H:%M)"
