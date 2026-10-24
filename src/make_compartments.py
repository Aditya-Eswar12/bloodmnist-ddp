"""Compute the stain-derived compartment fractions for every BloodMNIST image
once and cache them (takes about two minutes).

Run:  python src/make_compartments.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compartments import segment, patch_fractions
from data import NPZ, COMPARTMENTS

if __name__ == "__main__":
    d = np.load(NPZ)
    out = {}
    for split in ("train", "val", "test"):
        out[split] = np.stack([patch_fractions(segment(im)) for im in d[f"{split}_images"]])
        print(split, out[split].shape)
    np.savez_compressed(COMPARTMENTS, **out)
