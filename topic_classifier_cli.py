"""Command-line topic classification utility.

Loads the level 1 and level 2 DistilBERT models plus label encoders,
then lets you type text in the terminal and prints the predicted topics.
"""

import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertForSequenceClassification


MODELS_BASE = os.path.join(os.path.dirname(__file__), "Models")
MODEL_L1_DIR = os.path.join(MODELS_BASE, "model_level1")
MODEL_L2_DIR = os.path.join(MODELS_BASE, "model_level2")
ENCODER_DIR = os.path.join(MODELS_BASE, "label_encoders")
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THR = 0.75


def load_models() -> Dict:
    """Load tokenizer, models, encoders, and taxonomy from disk."""
    le_l1_path = os.path.join(ENCODER_DIR, "le_level1.pkl")
    le_l2_path = os.path.join(ENCODER_DIR, "le_level2.pkl")
    taxonomy_path = os.path.join(ENCODER_DIR, "taxonomy.json")

    if not all(os.path.exists(path) for path in [le_l1_path, le_l2_path, taxonomy_path]):
        missing = [p for p in [le_l1_path, le_l2_path, taxonomy_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing model metadata files: {missing}")

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_L1_DIR)
    except Exception:
        checkpoint_dir = os.path.join(MODELS_BASE, "checkpoint-438")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    load_kwargs = {"low_cpu_mem_usage": True}
    model_l1 = DistilBertForSequenceClassification.from_pretrained(MODEL_L1_DIR, **load_kwargs).to(DEVICE).eval()
    model_l2 = DistilBertForSequenceClassification.from_pretrained(MODEL_L2_DIR, **load_kwargs).to(DEVICE).eval()

    with open(le_l1_path, "rb") as f:
        le_l1 = pickle.load(f)
    with open(le_l2_path, "rb") as f:
        le_l2 = pickle.load(f)
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    print(f"L1 model labels: {model_l1.config.num_labels} | L1 encoder labels: {len(le_l1.classes_)}")
    print(f"L2 model labels: {model_l2.config.num_labels} | L2 encoder labels: {len(le_l2.classes_)}")

    return {
        "tokenizer": tokenizer,
        "model_l1": model_l1,
        "model_l2": model_l2,
        "le_l1": le_l1,
        "le_l2": le_l2,
        "taxonomy": taxonomy,
    }


def safe_decode_label(encoder, idx: int, model_name: str) -> str:
    """Decode a class index safely and raise a clear error if the mapping is invalid."""
    if idx < 0 or idx >= len(encoder.classes_):
        raise ValueError(
            f"{model_name} predicted class index {idx}, but the saved encoder only has "
            f"{len(encoder.classes_)} classes. The model and encoder files are mismatched."
        )
    return encoder.inverse_transform([idx])[0]


def align_probs_to_encoder(probs: np.ndarray, encoder, model_name: str) -> np.ndarray:
    """Align a probability vector to the encoder class count."""
    class_count = len(encoder.classes_)
    if probs.shape[0] == class_count:
        return probs

    if probs.shape[0] > class_count:
        print(
            f"⚠️ {model_name} produced {probs.shape[0]} logits, but the encoder has {class_count} classes. "
            f"Using the first {class_count} logits to keep the CLI usable."
        )
        return probs[:class_count]

    raise ValueError(
        f"{model_name} produced only {probs.shape[0]} logits, but the encoder expects {class_count} classes."
    )


def infer(model, tokenizer, text: str) -> Tuple[int, np.ndarray]:
    """Run a single forward pass and return the predicted index and probabilities."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


def predict_topic(text: str, models: Dict) -> Dict:
    """Predict L1 and L2 topics for the provided text."""
    tokenizer = models["tokenizer"]
    model_l1 = models["model_l1"]
    model_l2 = models["model_l2"]
    le_l1 = models["le_l1"]
    le_l2 = models["le_l2"]
    taxonomy = models["taxonomy"]

    l1_idx, l1_probs = infer(model_l1, tokenizer, text)
    l1_probs = align_probs_to_encoder(l1_probs, le_l1, "L1 model")
    l1_idx = int(np.argmax(l1_probs))
    l1_label = safe_decode_label(le_l1, l1_idx, "L1 model")
    l1_conf = float(l1_probs[l1_idx])

    if l1_conf < CONFIDENCE_THR:
        return {
            "level1": "General",
            "level2": "Miscellaneous/General Knowledge",
            "level1_confidence": l1_conf,
            "level2_confidence": 0.0,
            "confidence": l1_conf,
            "used_fallback": True,
        }

    l2_idx, l2_probs = infer(model_l2, tokenizer, text)
    if l2_probs.shape[0] != len(le_l2.classes_):
        return {
            "level1": l1_label,
            "level2": "Unavailable",
            "level1_confidence": l1_conf,
            "level2_confidence": 0.0,
            "confidence": l1_conf,
            "used_fallback": True,
            "note": (
                f"L2 model/encoder mismatch: model emits {l2_probs.shape[0]} logits, "
                f"but the saved L2 encoder expects {len(le_l2.classes_)} classes."
            ),
        }

    l2_probs = align_probs_to_encoder(l2_probs, le_l2, "L2 model")
    l2_idx = int(np.argmax(l2_probs))
    l2_label = safe_decode_label(le_l2, l2_idx, "L2 model")
    l2_conf = float(l2_probs[l2_idx])

    valid_subdomains = taxonomy.get(l1_label, [])
    if l2_label not in valid_subdomains and valid_subdomains:
        valid_indices = [i for i, cls in enumerate(le_l2.classes_) if cls in valid_subdomains]
        if valid_indices:
            best_idx = max(valid_indices, key=lambda i: l2_probs[i])
            l2_label = le_l2.inverse_transform([best_idx])[0]
            l2_conf = float(l2_probs[best_idx])

    return {
        "level1": l1_label,
        "level2": l2_label,
        "level1_confidence": l1_conf,
        "level2_confidence": l2_conf,
        "confidence": l1_conf,
        "used_fallback": False,
    }


def main() -> None:
    print("Loading topic classification models...")
    print(f"Device: {DEVICE}")
    models = load_models()
    print("Models loaded successfully.")
    print("Type text and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Enter text: ").strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break

        try:
            result = predict_topic(text, models)
            print(f"\n✅ [{result['level1']} > {result['level2']}]  ({result['confidence'] * 100:.1f}%)")
            print(f"   L1 confidence: {result['level1_confidence'] * 100:.1f}%")
            print(f"   L2 confidence: {result['level2_confidence'] * 100:.1f}%")
            if result.get("used_fallback"):
                print("   Fallback: low-confidence General classification")
            if result.get("note"):
                print(f"   Note: {result['note']}")
            print()
        except ValueError as e:
            print(f"\n❌ {e}\n")
        except Exception as e:
            print(f"\n❌ Classification failed: {e}\n")


if __name__ == "__main__":
    main()
