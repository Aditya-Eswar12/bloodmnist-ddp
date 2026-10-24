"""
FastAPI service for the blood-cell classifier.

Run locally:   .venv/Scripts/python -m uvicorn serve.app:app --port 8000
Then open:     http://127.0.0.1:8000/docs   (interactive API page)

Endpoints
  GET  /health                         model version, whether it is trained, report mode
  POST /predict                        one image -> class, confidence, all 8 probabilities
  POST /slides/{slide_id}/cells        many images from one smear -> logged predictions
  GET  /slides/{slide_id}/differential % differential with reference-range flags (SQL)
  POST /slides/{slide_id}/report       run the QC -> differential -> writer -> verifier pipeline
  GET  /cohort/abnormal                slides outside reference ranges, ranked (SQL window functions)
  GET  /monitoring/confidence          per-slide confidence trend (SQL window functions)
"""

import os
import sys
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agents import SlidePipeline, ReportAgent          # noqa: E402
from db import Store                                     # noqa: E402
from data import CLASSES                                 # noqa: E402
from model_service import Classifier, DEFAULT_CKPT       # noqa: E402

MAX_BYTES = 5 * 1024 * 1024


def create_app(db_path=None, ckpt=None, report_agent=None):
    db_path = db_path or os.environ.get("DB_PATH", "predictions.sqlite")
    clf = Classifier(ckpt or DEFAULT_CKPT)
    store = Store(db_path)
    pipeline = SlidePipeline(clf, store, report_agent or ReportAgent())
    app = FastAPI(title="Blood-cell classifier", version="1.0")

    async def read_image(f: UploadFile):
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename}: image larger than 5 MB")
        try:
            return clf.preprocess(data)
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, f"{f.filename}: not a readable image")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": clf.version, "trained": clf.trained,
                "report_mode": pipeline.writer.mode}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), slide_id: Optional[str] = Form(None)):
        x = await read_image(file)
        cls, conf, probs = pipeline.add_cells(slide_id or "adhoc", x)[0]
        return {"class": cls, "confidence": round(conf, 4),
                "accepted": pipeline.qc.accept_cell(conf),
                "probabilities": {c: round(p, 4) for c, p in zip(CLASSES, probs)},
                "model_version": clf.version}

    @app.post("/slides/{slide_id}/cells")
    async def add_cells(slide_id: str, files: List[UploadFile] = File(...)):
        x = torch.cat([await read_image(f) for f in files])
        res = pipeline.add_cells(slide_id, x)
        return {"slide_id": slide_id, "n": len(res),
                "accepted": sum(pipeline.qc.accept_cell(c) for _, c, _ in res),
                "predictions": [{"class": c, "confidence": round(p, 4)} for c, p, _ in res]}

    @app.get("/slides/{slide_id}/differential")
    def differential(slide_id: str):
        return {"slide_id": slide_id, "differential": store.query("differential", slide_id=slide_id),
                "qc": (store.query("slide_qc", slide_id=slide_id) or [None])[0]}

    @app.post("/slides/{slide_id}/report")
    def report(slide_id: str):
        return pipeline.report(slide_id)

    @app.get("/cohort/abnormal")
    def cohort():
        return store.query("abnormal_cohort")

    @app.get("/monitoring/confidence")
    def monitoring():
        return store.query("confidence_trend")

    app.state.pipeline, app.state.store = pipeline, store
    return app


app = create_app() if os.environ.get("SERVE_NO_AUTOAPP") != "1" else None
