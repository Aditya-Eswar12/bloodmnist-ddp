"""API and pipeline tests. Run:  .venv/Scripts/python -m pytest serve/tests -q"""

import io
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

os.environ["SERVE_NO_AUTOAPP"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from fastapi.testclient import TestClient   # noqa: E402

from agents import VerifierAgent, template_report, DISCLAIMER, MIN_WBC   # noqa: E402
from app import create_app                   # noqa: E402
from data import load_split, CLASSES          # noqa: E402

CKPT = os.environ.get("MODEL_CKPT", "checkpoints/baseline_cnn_s0.pt")


def png_bytes(arr_chw_uint8):
    buf = io.BytesIO()
    Image.fromarray(arr_chw_uint8.permute(1, 2, 0).numpy()).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "test.sqlite"
    return TestClient(create_app(db_path=str(db), ckpt=CKPT))


@pytest.fixture(scope="module")
def test_images():
    return load_split("test")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["report_mode"] in ("llm", "template")


def test_predict_returns_valid_probabilities(client, test_images):
    x, y = test_images
    r = client.post("/predict", files={"file": ("cell.png", png_bytes(x[0]), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["class"] in CLASSES
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert 0.0 <= body["confidence"] <= 1.0


@pytest.mark.skipif(not os.path.exists(CKPT), reason="needs a trained checkpoint")
def test_trained_model_is_accurate_through_the_api(client, test_images):
    x, y = test_images
    idx = torch.arange(0, len(y), 50)[:40]
    correct = 0
    for i in idx:
        r = client.post("/predict", files={"file": ("c.png", png_bytes(x[i]), "image/png")})
        correct += r.json()["class"] == CLASSES[int(y[i])]
    assert correct / len(idx) > 0.9      # same preprocessing as training, so API accuracy matches


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_slide_report_pipeline(client, test_images):
    x, y = test_images
    files = [("files", (f"c{i}.png", png_bytes(x[i]), "image/png")) for i in range(150)]
    r = client.post("/slides/S1/cells", files=files)
    assert r.status_code == 200 and r.json()["n"] == 150
    d = client.get("/slides/S1/differential").json()
    assert {row["cell_type"] for row in d["differential"]} >= {"neutrophil", "lymphocyte", "monocyte"}
    rep = client.post("/slides/S1/report").json()
    assert rep["status"] in ("draft_for_review", "blocked_llm_draft_replaced", "insufficient_cells")
    assert DISCLAIMER in rep["text"]


def test_small_slide_is_refused(client, test_images):
    x, _ = test_images
    files = [("files", (f"c{i}.png", png_bytes(x[i]), "image/png")) for i in range(5)]
    client.post("/slides/TINY/cells", files=files)
    rep = client.post("/slides/TINY/report").json()
    assert rep["status"] == "insufficient_cells"


def test_verifier_blocks_invented_numbers_and_diagnoses():
    table = [{"cell_type": "monocyte", "n": 12, "pct": 12.0, "low_pct": 2.0, "high_pct": 8.0,
              "flag": "high", "wbc_counted": 100},
             {"cell_type": "neutrophil", "n": 88, "pct": 88.0, "low_pct": 40.0, "high_pct": 70.0,
              "flag": "high", "wbc_counted": 100}]
    good = template_report("S", table, [])
    assert VerifierAgent().run(good, table)[0]
    bad = "Monocytes 15% and neutrophils 88%, consistent with a diagnosis of leukemia. " + DISCLAIMER
    ok, notes = VerifierAgent().run(bad, table)
    assert not ok
    assert any("15" in n for n in notes) and any("leukemia" in n for n in notes)
    no_disclaimer = "Monocytes 12.0% above range; neutrophils 88.0% above range."
    assert not VerifierAgent().run(no_disclaimer, table)[0]
