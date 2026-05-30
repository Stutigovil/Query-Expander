import os, pickle, json
_MODELS_BASE = os.path.join(os.path.dirname(__file__), "Models")
ENCODER_DIR = os.path.join(_MODELS_BASE, "label_encoders")
le1_p = os.path.join(ENCODER_DIR, "le_level1.pkl")
le2_p = os.path.join(ENCODER_DIR, "le_level2.pkl")
tax_p = os.path.join(ENCODER_DIR, "taxonomy.json")
print('Paths:', le1_p, le2_p, tax_p)
print('Exists:', os.path.exists(le1_p), os.path.exists(le2_p), os.path.exists(tax_p))
if os.path.exists(le1_p):
    with open(le1_p,'rb') as f:
        le1 = pickle.load(f)
    print('le1 classes:', len(le1.classes_))
if os.path.exists(le2_p):
    with open(le2_p,'rb') as f:
        le2 = pickle.load(f)
    print('le2 classes:', len(le2.classes_))
if os.path.exists(tax_p):
    with open(tax_p) as f:
        tax = json.load(f)
    print('taxonomy entries:', len(tax))
