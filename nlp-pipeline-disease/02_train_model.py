"""
02_train_model.py
==================
NLP Pipeline - Plant Disease Text Classification
Multimodal Ensemble Framework Project

What this script does:
1. Loads TF-IDF matrices from preprocessing
2. Trains Logistic Regression model
3. Saves model as nlp_model.pkl
4. Verifies predict_proba for ARI fusion
"""

import os
import numpy as np
import joblib
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = r"data"
MODELS_DIR = r"models"

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  NLP PIPELINE — 02_train_model.py")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading preprocessed TF-IDF data...")

X_train = sparse.load_npz(os.path.join(DATA_DIR, "X_train_tfidf.npz"))
X_test  = sparse.load_npz(os.path.join(DATA_DIR, "X_test_tfidf.npz"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"  → Train shape : {X_train.shape}")
print(f"  → Test shape  : {X_test.shape}")


# ─────────────────────────────────────────────
# 2. TRAIN LOGISTIC REGRESSION
# ─────────────────────────────────────────────
print("\n[STEP 2] Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"  → Accuracy  : {acc*100:.2f}%")
print(f"  → Precision : {prec:.4f}")
print(f"  → Recall    : {rec:.4f}")
print(f"  → F1-Score  : {f1:.4f}")


# ─────────────────────────────────────────────
# 3. VERIFY predict_proba FOR ARI FUSION
#    D = disease probability score
# ─────────────────────────────────────────────
print("\n[STEP 3] Verifying predict_proba for ARI fusion...")

tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

sample_texts = [
    "leaves have yellow patches and brown spots on edges",
    "plant looks green and growing well"
]

for text in sample_texts:
    vec   = tfidf.transform([text])
    proba = model.predict_proba(vec)[0]
    print(f"\n  Text              : '{text}'")
    print(f"  Healthy prob  (0) : {proba[0]:.4f}")
    print(f"  Diseased prob (1) : {proba[1]:.4f}  ← D value in ARI")


# ─────────────────────────────────────────────
# 4. SAVE MODEL
# ─────────────────────────────────────────────
print("\n[STEP 4] Saving model...")

model_path = os.path.join(MODELS_DIR, "nlp_model.pkl")
joblib.dump(model, model_path)
print(f"  ✅ nlp_model.pkl saved → {model_path}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING COMPLETE ✅")
print("=" * 60)
print(f"  Model      : Logistic Regression")
print(f"  Accuracy   : {acc*100:.2f}%")
print(f"  Precision  : {prec:.4f}")
print(f"  Recall     : {rec:.4f}")
print(f"  F1-Score   : {f1:.4f}")
print(f"\n  Files saved to: {MODELS_DIR}/")
print(f"    ├── nlp_model.pkl")
print(f"    └── tfidf_vectorizer.pkl")
print(f"\n  Next step → run 03_evaluate_model.py")
print("=" * 60)