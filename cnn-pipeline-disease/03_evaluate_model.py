"""
03_evaluate_model.py
=====================
CNN Pipeline - Plant Disease Classification
Multimodal Ensemble Framework Project

What this script does:
1. Loads the best trained model
2. Runs predictions on the test set
3. Generates classification report (precision, recall, F1)
4. Plots confusion matrix
5. Shows per-class accuracy
6. Saves confidence scores for ARI fusion later
7. Plots F1 score per class
8. Plots Precision/Recall/F1 comparison
9. Plots overall metrics summary
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\cnn-pipeline-disease\data"
MODEL_DIR   = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\cnn-pipeline-disease\models"
RESULTS_DIR = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\cnn-pipeline-disease\results"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  CNN PIPELINE — 03_evaluate_model.py")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. LOAD MODEL + MAPPINGS
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading model and class mappings...")

model_path = os.path.join(MODEL_DIR, "best_model_phase2.keras")
model      = load_model(model_path)
print(f"  ✅ Model loaded from → {model_path}")

with open(os.path.join(DATA_DIR, "class_mapping.json"), "r") as f:
    mapping = json.load(f)

with open(os.path.join(DATA_DIR, "keras_class_indices.json"), "r") as f:
    keras_class_indices = json.load(f)

NUM_CLASSES  = mapping["num_classes"]
class_to_idx = mapping["class_to_idx"]

idx_to_class_keras = {v: k for k, v in keras_class_indices.items()}
class_names        = [idx_to_class_keras[i] for i in range(NUM_CLASSES)]

print(f"  → Classes : {NUM_CLASSES}")


# ─────────────────────────────────────────────
# 2. TEST GENERATOR
# ─────────────────────────────────────────────
print("\n[STEP 2] Loading test set...")

df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"  → Test samples : {len(df_test)}")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# ─────────────────────────────────────────────
# 3. PREDICTIONS
# ─────────────────────────────────────────────
print("\n[STEP 3] Running predictions on test set...")
print("  (This may take a few minutes...)\n")

y_pred_probs = model.predict(test_generator, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_generator.classes


# ─────────────────────────────────────────────
# 4. OVERALL METRICS
# ─────────────────────────────────────────────
print("\n[STEP 4] Computing overall metrics...")

top1_acc = accuracy_score(y_true, y_pred)
top5_acc = top_k_accuracy_score(y_true, y_pred_probs, k=5)

print(f"  → Top-1 Accuracy : {top1_acc:.4f} ({top1_acc*100:.2f}%)")
print(f"  → Top-5 Accuracy : {top5_acc:.4f} ({top5_acc*100:.2f}%)")


# ─────────────────────────────────────────────
# 5. CLASSIFICATION REPORT
# ─────────────────────────────────────────────
print("\n[STEP 5] Classification Report...")

report = classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=4
)
print(report)

report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)\n")
    f.write(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)\n\n")
    f.write(report)
print(f"  ✅ Report saved → {report_path}")

report_dict = classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=4,
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = os.path.join(RESULTS_DIR, "classification_report.csv")
report_df.to_csv(report_csv_path)
print(f"  ✅ Report CSV saved → {report_csv_path}")


# ─────────────────────────────────────────────
# 6. PER CLASS ACCURACY
# ─────────────────────────────────────────────
print("\n[STEP 6] Per-class accuracy...")

cm = confusion_matrix(y_true, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

print(f"\n  {'Class':<50} {'Accuracy':>10}")
print(f"  {'-'*62}")
for i, (cls, acc) in enumerate(zip(class_names, per_class_acc)):
    flag = " ⚠️" if acc < 0.80 else ""
    print(f"  {cls:<50} {acc*100:>9.2f}%{flag}")

per_class_df = pd.DataFrame({
    "class": class_names,
    "accuracy": per_class_acc
}).sort_values("accuracy", ascending=False)

per_class_path = os.path.join(RESULTS_DIR, "per_class_accuracy.csv")
per_class_df.to_csv(per_class_path, index=False)
print(f"\n  ✅ Per-class accuracy saved → {per_class_path}")


# ─────────────────────────────────────────────
# 7. CONFUSION MATRIX (NORMALIZED + CLEAR VISUALIZATION)
# ─────────────────────────────────────────────
print("\n[STEP 7] Plotting confusion matrix...")

# Normalize confusion matrix row-wise
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(20, 18))

sns.heatmap(
    cm_normalized,
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    annot=False,
    linewidths=0.2,
    linecolor="gray",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={"label": "Prediction Probability"},
    ax=ax
)

ax.set_title("Normalized Confusion Matrix — MobileNetV2", fontsize=16)
ax.set_xlabel("Predicted Class", fontsize=12)
ax.set_ylabel("True Class", fontsize=12)

plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)

plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()

print(f"  ✅ Confusion matrix saved → {cm_path}")



# ─────────────────────────────────────────────
# 8. PER CLASS ACCURACY BAR CHART
# ─────────────────────────────────────────────
print("\n[STEP 8] Plotting per-class accuracy bar chart...")

sorted_df  = per_class_df.sort_values("accuracy", ascending=True)
colors_acc = ["#FF5722" if a < 0.80 else "#4CAF50" for a in sorted_df["accuracy"]]

fig, ax = plt.subplots(figsize=(10, 14))
ax.barh(sorted_df["class"], sorted_df["accuracy"] * 100, color=colors_acc)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Per-Class Accuracy — MobileNetV2\n(Red = below 80%)", fontsize=12)
ax.axvline(x=80, color="red",  linestyle="--", alpha=0.5, label="80% threshold")
ax.axvline(x=90, color="blue", linestyle="--", alpha=0.5, label="90% threshold")
ax.legend()
plt.tight_layout()

bar_path = os.path.join(RESULTS_DIR, "per_class_accuracy_bar.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"  ✅ Bar chart saved → {bar_path}")


# ─────────────────────────────────────────────
# 9. SAVE PREDICTION CONFIDENCE SCORES
# ─────────────────────────────────────────────
print("\n[STEP 9] Saving prediction confidence scores for ARI fusion...")

max_confidence   = np.max(y_pred_probs, axis=1)
pred_class_names = [class_names[i] for i in y_pred]
true_class_names = [class_names[i] for i in y_true]

healthy_keywords = ["healthy", "Healthy", "fresh_leaf", "fresh_plant"]

def is_diseased(class_name):
    return not any(kw in class_name for kw in healthy_keywords)

pred_is_diseased = [1 if is_diseased(c) else 0 for c in pred_class_names]
true_is_diseased = [1 if is_diseased(c) else 0 for c in true_class_names]

confidence_df = pd.DataFrame({
    "true_class"       : true_class_names,
    "pred_class"       : pred_class_names,
    "confidence"       : max_confidence,
    "pred_is_diseased" : pred_is_diseased,
    "true_is_diseased" : true_is_diseased,
    "correct"          : (np.array(y_pred) == np.array(y_true)).astype(int)
})

confidence_path = os.path.join(RESULTS_DIR, "prediction_confidence.csv")
confidence_df.to_csv(confidence_path, index=False)
print(f"  ✅ Confidence scores saved → {confidence_path}")


# ─────────────────────────────────────────────
# 10. F1 SCORE PER CLASS BAR CHART
# ─────────────────────────────────────────────
print("\n[STEP 10] Plotting F1 score per class...")

f1_scores  = [report_dict[c]["f1-score"] for c in class_names]
sorted_idx = np.argsort(f1_scores)
sorted_cls = [class_names[i] for i in sorted_idx]
sorted_f1  = [f1_scores[i] for i in sorted_idx]
colors_f1  = ["#FF5722" if v < 0.80 else "#FFC107" if v < 0.90 else "#4CAF50" for v in sorted_f1]

fig, ax = plt.subplots(figsize=(10, 16))
bars = ax.barh(sorted_cls, sorted_f1, color=colors_f1, edgecolor="white", height=0.7)

for bar, val in zip(bars, sorted_f1):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val*100:.1f}%", va="center", fontsize=7.5)

ax.axvline(x=0.80, color="red",    linestyle="--", alpha=0.6, linewidth=1)
ax.axvline(x=0.90, color="orange", linestyle="--", alpha=0.6, linewidth=1)

red_patch    = mpatches.Patch(color="#FF5722", label="F1 < 80%")
yellow_patch = mpatches.Patch(color="#FFC107", label="F1 80–90%")
green_patch  = mpatches.Patch(color="#4CAF50", label="F1 > 90%")
ax.legend(handles=[red_patch, yellow_patch, green_patch], fontsize=9, loc="lower right")

ax.set_xlabel("F1 Score", fontsize=11)
ax.set_title("Per-Class F1 Score — MobileNetV2\n(42 Plant Disease Classes)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlim(0, 1.08)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()

f1_path = os.path.join(RESULTS_DIR, "f1_score_per_class.png")
plt.savefig(f1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ F1 chart saved → {f1_path}")


# ─────────────────────────────────────────────
# 11. PRECISION / RECALL / F1 GROUPED BAR
# ─────────────────────────────────────────────
print("\n[STEP 11] Plotting Precision/Recall/F1 comparison...")

precision_scores = [report_dict[c]["precision"] for c in class_names]
recall_scores    = [report_dict[c]["recall"]    for c in class_names]

x     = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(20, 8))
ax.bar(x - width, precision_scores, width, label="Precision", color="#2196F3", alpha=0.85)
ax.bar(x,         recall_scores,    width, label="Recall",    color="#4CAF50", alpha=0.85)
ax.bar(x + width, f1_scores,        width, label="F1 Score",  color="#FF9800", alpha=0.85)

ax.set_xlabel("Disease Class", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Precision / Recall / F1-Score per Class — MobileNetV2",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=90, fontsize=7)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4, linewidth=1)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

prf_path = os.path.join(RESULTS_DIR, "precision_recall_f1.png")
plt.savefig(prf_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Precision/Recall/F1 chart saved → {prf_path}")


# ─────────────────────────────────────────────
# 12. OVERALL METRICS SUMMARY BAR
# ─────────────────────────────────────────────
print("\n[STEP 12] Plotting overall metrics summary...")

macro    = report_dict["macro avg"]
metrics  = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1-Score\n(Macro)"]
values   = [top1_acc, macro["precision"], macro["recall"], macro["f1-score"]]
colors_s = ["#9C27B0", "#2196F3", "#4CAF50", "#FF9800"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics, values, color=colors_s, width=0.5, edgecolor="white")

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val*100:.2f}%", ha="center", fontsize=11, fontweight="bold")

ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Overall Model Performance — MobileNetV2\n"
             f"(Test Set: {len(y_true)} images, {NUM_CLASSES} classes)",
             fontsize=12, fontweight="bold", pad=15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

summary_path = os.path.join(RESULTS_DIR, "overall_metrics_summary.png")
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Overall metrics summary saved → {summary_path}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
worst_classes = per_class_df.nsmallest(3, "accuracy")
best_classes  = per_class_df.nlargest(3, "accuracy")

print("\n" + "=" * 60)
print("  EVALUATION COMPLETE ✅")
print("=" * 60)
print(f"  Top-1 Test Accuracy  : {top1_acc*100:.2f}%")
print(f"  Top-5 Test Accuracy  : {top5_acc*100:.2f}%")

print(f"\n  Best predicted classes:")
for _, row in best_classes.iterrows():
    print(f"    {row['class']:<45} {row['accuracy']*100:.2f}%")

print(f"\n  Hardest classes (lowest accuracy):")
for _, row in worst_classes.iterrows():
    print(f"    {row['class']:<45} {row['accuracy']*100:.2f}%")

print(f"\n  Results saved to: {RESULTS_DIR}")
print(f"    ├── classification_report.txt")
print(f"    ├── classification_report.csv")
print(f"    ├── per_class_accuracy.csv")
print(f"    ├── confusion_matrix.png")
print(f"    ├── per_class_accuracy_bar.png")
print(f"    ├── prediction_confidence.csv")
print(f"    ├── f1_score_per_class.png")
print(f"    ├── precision_recall_f1.png")
print(f"    └── overall_metrics_summary.png")
print("\n  CNN Pipeline complete! Ready for ARI fusion.")
print("=" * 60)