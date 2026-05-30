"""Verify that saved topic classifier models match their label encoders.

This script checks the label count for L1 and L2 models against the
corresponding LabelEncoder class counts before you start the app.
"""

import json
import os
import pickle
import sys
from typing import Dict, Tuple

from transformers import DistilBertForSequenceClassification


MODELS_BASE = os.path.join(os.path.dirname(__file__), "Models")
MODEL_L1_DIR = os.path.join(MODELS_BASE, "model_level1")
MODEL_L2_DIR = os.path.join(MODELS_BASE, "model_level2")
ENCODER_DIR = os.path.join(MODELS_BASE, "label_encoders")


def load_count(path: str) -> int:
    with open(path, "rb") as f:
        encoder = pickle.load(f)
    return len(encoder.classes_)


def model_num_labels(model_dir: str) -> int:
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    return int(model.config.num_labels or 0)


def check_pair(model_dir: str, encoder_path: str) -> Tuple[bool, str]:
    model_labels = model_num_labels(model_dir)
    encoder_labels = load_count(encoder_path)
    if model_labels != encoder_labels:
        return False, (
            f"Mismatch for {os.path.basename(model_dir)}: model has {model_labels} labels, "
            f"encoder has {encoder_labels}."
        )
    return True, (
        f"OK for {os.path.basename(model_dir)}: model labels={model_labels}, encoder labels={encoder_labels}."
    )


def main() -> int:
    checks = [
        (MODEL_L1_DIR, os.path.join(ENCODER_DIR, "le_level1.pkl")),
        (MODEL_L2_DIR, os.path.join(ENCODER_DIR, "le_level2.pkl")),
    ]

    all_ok = True
    print("Model compatibility check\n")

    for model_dir, encoder_path in checks:
        ok, message = check_pair(model_dir, encoder_path)
        print(message)
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll model and encoder pairs are compatible.")
        return 0

    print("\nAt least one model/encoder pair is mismatched.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
