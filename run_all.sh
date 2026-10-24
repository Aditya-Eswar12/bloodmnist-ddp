#!/usr/bin/env bash
# Runs every experiment for the pilot, in order of importance. Safe to re-run:
# a run whose result file already exists is skipped.
#
#   bash run_all.sh
#
# CPU budget (4-core laptop): about 4-5 hours in total.
cd "$(dirname "$0")"
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=python
mkdir -p logs results
MAE_EPOCHS=${MAE_EPOCHS:-40}

run() {   # run <result-name> <command...>   (background, skipped if done)
  local name=$1; shift
  if [ -f "results/$name.json" ]; then echo "skip $name"; return; fi
  echo "start $name  $(date +%H:%M)"
  "$PY" -u -W ignore "$@" > "logs/$name.log" 2>&1 &
}

# 1. CNN baseline, three seeds side by side
for s in 0 1 2; do
  run baseline_cnn_s$s src/train_baseline.py --model cnn --epochs 15 --seeds $s --threads 2
done; wait

# 2. Masked pre-training: the four policies side by side, one seed at a time
for s in 0 1 2; do
  for pol in random A B C; do
    run mae_${pol}_s$s src/train_mae.py --policy $pol --epochs $MAE_EPOCHS --seed $s --threads 2
  done; wait
done

# 3. Reference models
run baseline_resnet18_s0 src/train_baseline.py --model resnet18 --epochs 12 --seeds 0 --threads 4
run baseline_vit_s0 src/train_baseline.py --model vit --epochs 15 --seeds 0 --threads 4
wait
"$PY" src/report.py
echo "all done $(date +%H:%M)"
