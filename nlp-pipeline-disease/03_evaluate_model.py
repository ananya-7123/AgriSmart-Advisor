"""
03_evaluate_model.py
=====================
NLP Pipeline - Plant Disease Text Classification
Multimodal Ensemble Framework Project

What this script does:
1. Loads trained Logistic Regression model
2. Runs predictions on test set
3. Generates classification report
4. Plots confusion matrix
5. Saves results for ARI fusion
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = r"data"
MODELS_DIR  = r"models"
RESULTS_DIR = r"results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  NLP PIPELINE — 03_evaluate_model.py")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. LOAD MODEL + DATA
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading model and test data...")

model = joblib.load(os.path.join(MODELS_DIR, "nlp_model.pkl"))
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

X_test = sparse.load_npz(os.path.join(DATA_DIR, "X_test_tfidf.npz"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"  ✅ Model loaded  : Logistic Regression")
print(f"  ✅ Test samples  : {X_test.shape[0]}")


# ─────────────────────────────────────────────
# 2. PREDICTIONS
# ─────────────────────────────────────────────
print("\n[STEP 2] Running predictions...")

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"  → Accuracy  : {acc*100:.2f}%")
print(f"  → Precision : {prec:.4f}")
print(f"  → Recall    : {rec:.4f}")
print(f"  → F1-Score  : {f1:.4f}")


# ─────────────────────────────────────────────
# 3. CLASSIFICATION REPORT
# ─────────────────────────────────────────────
print("\n[STEP 3] Classification Report...")

report = classification_report(
    y_test, y_pred,
    target_names=["Healthy (0)", "Diseased (1)"],
    digits=4
)
print(report)

report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Logistic Regression — NLP Disease Detection\n")
    f.write(f"Accuracy  : {acc*100:.2f}%\n")
    f.write(f"Precision : {prec:.4f}\n")
    f.write(f"Recall    : {rec:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n\n")
    f.write(report)

print(f"  ✅ Report saved → {report_path}")


# ─────────────────────────────────────────────
# 4. CONFUSION MATRIX
# ─────────────────────────────────────────────
print("\n[STEP 4] Plotting confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Healthy", "Diseased"],
    yticklabels=["Healthy", "Diseased"],
    ax=ax
)
ax.set_title("Confusion Matrix — NLP (Logistic Regression)", fontsize=12)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"  ✅ Confusion matrix saved → {cm_path}")


# ─────────────────────────────────────────────
# 5. SAVE PREDICTION CONFIDENCE FOR ARI FUSION
#    D = disease probability for each sample
# ─────────────────────────────────────────────
print("\n[STEP 5] Saving prediction confidence for ARI fusion...")

test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))

confidence_df = pd.DataFrame({
    "text"             : test_df["cleaned_text"].values,
    "true_label"       : y_test,
    "pred_label"       : y_pred,
    "healthy_prob"     : y_pred_proba[:, 0],
    "disease_prob"     : y_pred_proba[:, 1],   # ← D in ARI formula
    "correct"          : (y_pred == y_test).astype(int)
})

conf_path = os.path.join(RESULTS_DIR, "prediction_confidence.csv")
confidence_df.to_csv(conf_path, index=False)
print(f"  ✅ Confidence scores saved → {conf_path}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("  EVALUATION COMPLETE ✅")
print("=" * 60)
print(f"  Accuracy   : {acc*100:.2f}%")
print(f"  Precision  : {prec:.4f}")
print(f"  Recall     : {rec:.4f}")
print(f"  F1-Score   : {f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    True Positives  (Diseased → Diseased) : {tp}")
print(f"    True Negatives  (Healthy  → Healthy)  : {tn}")
print(f"    False Positives (Healthy  → Diseased) : {fp}")
print(f"    False Negatives (Diseased → Healthy)  : {fn}")
print(f"\n  Results saved to: {RESULTS_DIR}/")
print(f"    ├── classification_report.txt")
print(f"    ├── confusion_matrix.png")
print(f"    └── prediction_confidence.csv")
print(f"\n  NLP Pipeline complete! Ready for ARI fusion.")
print("=" * 60)