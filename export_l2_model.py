"""Copy a trained 41-class L2 classifier into Models/model_level2.

Usage:
    python export_l2_model.py --source <trained_l2_model_dir>

The source directory must contain a Hugging Face sequence-classification model
with the real 41-class L2 head.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from transformers import DistilBertForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_L2_DIR = PROJECT_ROOT / "Models" / "model_level2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained L2 model into Models/model_level2")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the directory containing the trained 41-class L2 model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing Models/model_level2 directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()

    if not source.exists():
        print(f"Source directory does not exist: {source}")
        return 1

    model = DistilBertForSequenceClassification.from_pretrained(str(source))
    num_labels = int(model.config.num_labels or 0)

    if num_labels != 41:
        print(f"Refusing to export: source model has {num_labels} labels, expected 41.")
        return 1

    if MODEL_L2_DIR.exists():
        if not args.force:
            print(
                f"Target already exists: {MODEL_L2_DIR}\n"
                "Use --force to replace it, or delete the folder first."
            )
            return 1
        shutil.rmtree(MODEL_L2_DIR)

    shutil.copytree(source, MODEL_L2_DIR)
    print(f"Copied 41-class L2 model from {source} to {MODEL_L2_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
