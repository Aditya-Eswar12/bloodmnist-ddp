# Blood-cell classification, stain-physics priors and external validation (BloodMNIST)

Dual Degree Project (B.Tech + M.Tech, Biological Engineering), IIT Madras, 2026 — Kancharla Aditya Eswar.
Guide: Prof. R. Jayaganthan. The first Keras model and the first conditional GAN were built with my teammates in July 2026;
the PyTorch pipeline, evaluation protocol, segmentation, masked pre-training, external validation, calibration and serving code are mine.

What the project asks:

1. How accurate is a classifier of 8 peripheral blood cell types when it is evaluated correctly?
2. Can a biological prior (stain optical density) make self-supervised pre-training more useful when labels are scarce?
3. Does the model still work on other laboratories' images, and if not, why?

Data are not included: BloodMNIST (MedMNIST v2, 64×64) and the Zheng et al. (2018) WBC segmentation dataset are public.
Trained checkpoints are not included; every number below can be regenerated with the scripts.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-serve.txt pytest
```

Put `bloodmnist_64.npz` in `data/`. For the mask-quality check and the stain-shift test, clone
`https://github.com/zxaoyou/segmentation_WBC` into `data/segmentation_WBC`.

## Layout

| Path | What it does |
|---|---|
| `src/data.py` | loading, normalising, flips/rotations, stratified label subsets, metrics |
| `src/models.py` | VGG-style CNN, ResNet-18 (64×64 stem), small ViT, masked autoencoder |
| `src/train_baseline.py` | supervised training; `--train_frac`, `--amp`, `--synthetic`, checkpoint selection on validation macro-F1 |
| `src/compartments.py`, `src/eval_masks.py`, `src/make_compartments.py` | label-free nucleus / cytoplasm maps from stain optical density, scored on expert masks |
| `src/masking.py`, `src/train_mae.py`, `src/probe.py` | masking policies (random, A, B, C), MAE pre-training, linear probe, confusion feedback |
| `src/label_efficiency.py` | linear probes on every encoder with 1 / 5 / 10 / 100% of labels |
| `src/external_eval.py` | test on external images (other labs, another stain) |
| `src/calibration.py` | expected calibration error, temperature scaling, confidence gate in and out of distribution |
| `src/cgan.py`, `src/cgan_eval.py` | conditional DCGAN; class fidelity, Fréchet distance, synthetic top-up set |
| `src/amp.py`, `src/amp_benchmark.py` | mixed precision helpers; fp32 vs AMP benchmark (run on a GPU) |
| `src/report.py` | collects all results into `results/RESULTS.md` |
| `serve/app.py` | FastAPI service: `/predict`, slide endpoints, cohort and monitoring queries |
| `serve/sql/` | SQLite schema and analytics (CTEs, window functions) |
| `serve/agents.py` | classifier → QC gate → SQL differential → LLM or template report → rule-based verifier |
| `serve/demo_cohort.py` | end-to-end run on simulated slides |
| `serve/tests/` | API and pipeline tests (`python -m pytest serve/tests -q`) |
| `Dockerfile`, `.github/workflows/docker.yml` | container for the service; CI builds and smoke-tests it |
| `colab/ddp_gpu.ipynb`, `make_colab_bundle.py` | GPU runs: AMP benchmark, 200-epoch pre-training |
| `run_all.sh`, `run_phase2.sh` | all CPU experiments in order; re-runnable, finished runs are skipped |
| `figures/make_figures.py` | the result figures in `figures/` (confusion matrix, segmentation examples, label efficiency, external tests) |

## Results in one table (details: `results/RESULTS.md`)

| Question | Answer (BloodMNIST test set unless stated) |
|---|---|
| How good is a small CNN with correct model selection? | 97.9 ± 0.1% accuracy, macro-F1 0.979 (3 seeds); ResNet-18 the same at 16× the parameters |
| Can stain physics find nucleus and cytoplasm without labels? | Dice 0.95 nucleus, 0.65 cytoplasm on held-out expert masks |
| Does anatomy-aware masking beat random masking? | Not at full labels; +2.6 macro-F1 for frozen features at 1% labels (9/9 paired), gone after fine-tuning |
| Does masked pre-training help with 1% of labels? | Fine-tuned ViT 0.707 → 0.82 macro-F1; the CNN trained end to end still reaches 0.863 |
| Does the model work on other labs' images? | No: 24% (CellaVision, other labs) and 8% (other stain); scale matching 48%, augmentation roughly doubles raw accuracy |
| Is confidence calibrated? | ECE 2.2% → 0.6% after temperature scaling (T = 0.64, under-confident); unreliable off-distribution |
| Do GAN images help? | No gain beyond a step-matched control (0.982 vs 0.980) |
| Does the service work end to end? | Yes on simulated smears; 7 API tests pass |

## Reproduce

```bash
python src/eval_masks.py --test
python src/make_compartments.py
bash run_all.sh
bash run_phase2.sh
.venv/Scripts/python -m pytest serve/tests -q
.venv/Scripts/python -m uvicorn serve.app:app --port 8000     # then open http://127.0.0.1:8000/docs
```

Every run writes `results/<name>.json` (arguments, seed, validation and test metrics, confusion matrix) and
`logs/<name>.log`. Model selection uses the validation split only; the test split is scored once per run.
