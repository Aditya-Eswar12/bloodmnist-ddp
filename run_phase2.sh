#!/usr/bin/env bash
# Phase 2: label efficiency, stain shift, conditional GAN. Run after run_all.sh.
# Safe to re-run: finished steps are skipped.
#
#   bash run_phase2.sh
cd "$(dirname "$0")"
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=python
mkdir -p logs results results/archive

run() {   # run <result-name> <command...>   (background, skipped if done)
  local name=$1; shift
  if [ -f "results/$name.json" ]; then echo "skip $name"; return; fi
  echo "start $name  $(date +%H:%M)"
  "$PY" -u -W ignore "$@" > "logs/$name.log" 2>&1 &
}

# 1. CNN on all labels again, this time saving the weights (needed by the
#    endpoint, the external test and the GAN evaluation). The first CNN runs did
#    not save checkpoints; their result files are kept in results/archive.
for s in 0 1 2; do
  if [ ! -f checkpoints/baseline_cnn_s$s.pt ] && [ -f results/baseline_cnn_s$s.json ]; then
    mv results/baseline_cnn_s$s.json results/archive/
  fi
  run baseline_cnn_s$s src/train_baseline.py --model cnn --epochs 15 --seeds $s --threads 2
done; wait

# 2. CNN with 1%, 5%, 10% of the labels, three seeds each
for f in 01 05 10; do
  for s in 0 1 2; do
    run baseline_cnn_f${f}_s$s src/train_baseline.py --model cnn --epochs 15 --seeds $s \
        --train_frac 0.$f --tag _f$f --threads 2
  done; wait
done

# 3. Linear probes on every pre-trained encoder at the same label fractions
THREADS=4 "$PY" -u -W ignore src/label_efficiency.py > logs/label_efficiency.log 2>&1

# 4. Stain / scanner shift
[ -f results/external.json ] || THREADS=4 "$PY" -u -W ignore src/external_eval.py > logs/external_eval.log 2>&1

# 4b. Calibration and the confidence gate on in- and out-of-distribution images
[ -f results/calibration.json ] || "$PY" -u -W ignore src/calibration.py > logs/calibration.log 2>&1

# 4c. End-to-end demo of the serving pipeline on simulated slides
[ -f results/cohort_demo.json ] || "$PY" -u -W ignore serve/demo_cohort.py > logs/cohort_demo.log 2>&1

# 5. Conditional GAN, its evaluation, and a CNN trained with its images
[ -f checkpoints/cgan.pt ] || "$PY" -u -W ignore src/cgan.py --epochs 25 --threads 4 > logs/cgan.log 2>&1
[ -f results/cgan_eval.json ] || "$PY" -u -W ignore src/cgan_eval.py > logs/cgan_eval.log 2>&1
for s in 0 1 2; do
  run baseline_cnn_cgan_s$s src/train_baseline.py --model cnn --epochs 15 --seeds $s \
      --synthetic data/synthetic_cgan.npz --tag _cgan --no_save --threads 2
done; wait

"$PY" src/report.py
echo "phase 2 done $(date +%H:%M)"
