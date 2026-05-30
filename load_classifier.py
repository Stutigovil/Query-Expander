"""Quick loader to verify models, tokenizers, and encoders load correctly."""
import os
import json
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, DistilBertForSequenceClassification

_MODELS_BASE = os.path.join(os.path.dirname(__file__), "Models")
MODEL_L1_DIR = os.path.join(_MODELS_BASE, "model_level1")
MODEL_L2_DIR = os.path.join(_MODELS_BASE, "model_level2")
ENCODER_DIR = os.path.join(_MODELS_BASE, "label_encoders")
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# Check files
le1_p = os.path.join(ENCODER_DIR, "le_level1.pkl")
le2_p = os.path.join(ENCODER_DIR, "le_level2.pkl")
tax_p = os.path.join(ENCODER_DIR, "taxonomy.json")
print("Checking encoder files:", os.path.exists(le1_p), os.path.exists(le2_p), os.path.exists(tax_p))

# Load encoders
with open(le1_p, 'rb') as f:
    le1 = pickle.load(f)
with open(le2_p, 'rb') as f:
    le2 = pickle.load(f)
with open(tax_p) as f:
    tax = json.load(f)

print(f"Loaded encoders: L1={len(le1.classes_)} classes, L2={len(le2.classes_)} classes, taxonomy entries={len(tax)}")

# Load tokenizer (try local model dir, then checkpoint, then Hugging Face default)
tok = None
try:
    tok = AutoTokenizer.from_pretrained(MODEL_L1_DIR)
    print('Tokenizer loaded from model_level1')
except Exception as e1:
    print('Tokenizer load from model_level1 failed:', e1)
    try:
        checkpoint_dir = os.path.join(_MODELS_BASE, 'checkpoint-438')
        tok = AutoTokenizer.from_pretrained(checkpoint_dir)
        print('Tokenizer loaded from checkpoint-438')
    except Exception as e2:
        print('Tokenizer load from checkpoint failed:', e2)
        try:
            tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            print('Tokenizer loaded from distilbert-base-uncased (Hugging Face)')
        except Exception as e3:
            print('Tokenizer load failed entirely:', e3)
            raise

# Load models
m1 = DistilBertForSequenceClassification.from_pretrained(MODEL_L1_DIR, low_cpu_mem_usage=True).to(DEVICE).eval()
m2 = DistilBertForSequenceClassification.from_pretrained(MODEL_L2_DIR, low_cpu_mem_usage=True).to(DEVICE).eval()
print('Models loaded')

# Quick inference
def infer(text):
    enc = tok(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        p1 = F.softmax(m1(**enc).logits, dim=-1)[0].cpu().numpy()
        p2 = F.softmax(m2(**enc).logits, dim=-1)[0].cpu().numpy()
    l1_idx = int(np.argmax(p1))
    l2_idx = int(np.argmax(p2))
    l1 = le1.inverse_transform([l1_idx])[0]
    l2 = le2.inverse_transform([l2_idx])[0]
    print('Text:', text)
    print('L1:', l1, f'(conf={p1[l1_idx]:.3f})')
    print('L2:', l2)

if __name__ == '__main__':
    infer('What is machine learning?')
