"""Loads a trained classifier checkpoint and turns an image into class probabilities."""

import hashlib
import io
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import CLASSES, normalise          # noqa: E402
from train_baseline import build             # noqa: E402

DEFAULT_CKPT = os.environ.get("MODEL_CKPT", "checkpoints/baseline_cnn_s0.pt")


class Classifier:
    def __init__(self, ckpt=DEFAULT_CKPT):
        if os.path.exists(ckpt):
            blob = torch.load(ckpt, map_location="cpu")
            self.model = build(blob["model"])
            self.model.load_state_dict(blob["state_dict"])
            digest = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()[:10]
            self.version = f"{blob['model']}-{digest}"
            self.trained = True
        else:
            # lets the API and tests run before any model is trained; /health reports it
            torch.manual_seed(0)
            self.model = build("cnn")
            self.version = "cnn-UNTRAINED"
            self.trained = False
        self.model.eval()

    @staticmethod
    def preprocess(image_bytes):
        """Any image file -> uint8 tensor (1, 3, 64, 64), the format used in training."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64), Image.BILINEAR)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].contiguous()

    @torch.no_grad()
    def predict_tensor(self, x_uint8):
        """(B, 3, 64, 64) uint8 -> (B, 8) probabilities."""
        return torch.softmax(self.model(normalise(x_uint8)), dim=1).numpy()

    def predict(self, image_bytes):
        p = self.predict_tensor(self.preprocess(image_bytes))[0]
        k = int(p.argmax())
        return {"class": CLASSES[k], "confidence": float(p[k]),
                "probabilities": {c: float(v) for c, v in zip(CLASSES, p)}}
