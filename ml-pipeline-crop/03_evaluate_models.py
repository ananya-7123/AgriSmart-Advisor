"""
03_evaluate_models.py
Evaluates all trained models and creates comparison tables
Generates visualizations and detailed performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import time
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# STEP 1: LOAD TEST DATA AND MODELS
# ============================================================================
print("=" * 70)
print("STEP 1: LOADING TEST DATA AND TRAINED MODELS")
print("=" * 70)

# Load test data
X_test = np.load('/data/processed/X_test.npy')
y_test = np.load('/data/processed/y_test.npy')

# Load label encoder
label_encoder = joblib.load('/models/label_encoder.pkl')

# Load all trained models
rf_model = joblib.load('models/rf_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
nb_model = joblib.load('models/nb_model.pkl')

print(f"✓ Test data loaded: {X_test.shape}")
print(f"✓ Random Forest model loaded")
print(f"✓ SVM model loaded")
print(f"✓ Naive Bayes model loaded")
print(f"✓ Number of classes: {len(label_encoder.classes_)}")

# ============================================================================
# STEP 2: MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: MAKING PREDICTIONS")
print("=" * 70)

# Dictionary to store models and their predictions
models = {
    'Random Forest': rf_model,
    'SVM': svm_model,
    'Naive Bayes': nb_model
}

predictions = {}
prediction_times = {}

for name, model in models.items():
    print(f"\nPredicting with {name}...")
    start_time = time.time()
    predictions[name] = model.predict(X_test)
    prediction_times[name] = time.time() - start_time
    print(f"✓ {name} predictions complete ({prediction_times[name]:.4f}s)")

# ============================================================================
# STEP 3: CALCULATE METRICS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: CALCULATING PERFORMANCE METRICS")
print("=" * 70)

results = []

for name, y_pred in predictions.items():
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Prediction Time (s)': prediction_times[name]
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# ============================================================================
# STEP 4: CREATE COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: MODEL COMPARISON TABLE")
print("=" * 70)

# Create dataframe
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.iloc[0]['Model']
best_accuracy = comparison_df.iloc[0]['Accuracy']

print(f"\n BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Save comparison table
import os
os.makedirs('results', exist_ok=True)
comparison_df.to_csv('results/model_comparison.csv', index=False)
print(f"\n✓ Saved: results/model_comparison.csv")

# ============================================================================
# STEP 5: DETAILED CLASSIFICATION REPORT (BEST MODEL)
# ============================================================================
print("\n" + "=" * 70)
print(f"STEP 5: DETAILED CLASSIFICATION REPORT - {best_model_name.upper()}")
print("=" * 70)

best_predictions = predictions[best_model_name]

print("\nClassification Report:")
print(classification_report(
    y_test, 
    best_predictions,
    target_names=label_encoder.classes_,
    digits=4
))

# ============================================================================
# STEP 6: CONFUSION MATRIX (BEST MODEL)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: GENERATING CONFUSION MATRIX")
print("=" * 70)

os.makedirs('results/plots', exist_ok=True)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar_kws={'label': 'Count'}
)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/confusion_matrix.png")
plt.close()

# ============================================================================
# STEP 7: MODEL COMPARISON VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: CREATING COMPARISON VISUALIZATIONS")
print("=" * 70)

# Plot 1: Accuracy Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

colors = ['#2ecc71', '#3498db', '#e74c3c']

# Accuracy
axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'], color=colors)
axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim([0.85, 1.0])
axes[0, 0].grid(axis='y', alpha=0.3)
for i, (model, acc) in enumerate(zip(comparison_df['Model'], comparison_df['Accuracy'])):
    axes[0, 0].text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontweight='bold')

# Precision
axes[0, 1].bar(comparison_df['Model'], comparison_df['Precision'], color=colors)
axes[0, 1].set_title('Precision Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_ylim([0.85, 1.0])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, (model, prec) in enumerate(zip(comparison_df['Model'], comparison_df['Precision'])):
    axes[0, 1].text(i, prec + 0.01, f'{prec:.4f}', ha='center', fontweight='bold')

# Recall
axes[1, 0].bar(comparison_df['Model'], comparison_df['Recall'], color=colors)
axes[1, 0].set_title('Recall Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_ylim([0.85, 1.0])
axes[1, 0].grid(axis='y', alpha=0.3)
for i, (model, rec) in enumerate(zip(comparison_df['Model'], comparison_df['Recall'])):
    axes[1, 0].text(i, rec + 0.01, f'{rec:.4f}', ha='center', fontweight='bold')

# F1-Score
axes[1, 1].bar(comparison_df['Model'], comparison_df['F1-Score'], color=colors)
axes[1, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_ylim([0.85, 1.0])
axes[1, 1].grid(axis='y', alpha=0.3)
for i, (model, f1) in enumerate(zip(comparison_df['Model'], comparison_df['F1-Score'])):
    axes[1, 1].text(i, f1 + 0.01, f'{f1:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/metrics_comparison.png")
plt.close()

# Plot 2: All Metrics Combined
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(comparison_df))
width = 0.2

ax.bar(x - 1.5*width, comparison_df['Accuracy'], width, label='Accuracy', color='#2ecc71')
ax.bar(x - 0.5*width, comparison_df['Precision'], width, label='Precision', color='#3498db')
ax.bar(x + 0.5*width, comparison_df['Recall'], width, label='Recall', color='#e74c3c')
ax.bar(x + 1.5*width, comparison_df['F1-Score'], width, label='F1-Score', color='#f39c12')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'])
ax.legend()
ax.set_ylim([0.85, 1.0])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/all_metrics_comparison.png")
plt.close()

# ============================================================================
# STEP 8: FEATURE IMPORTANCE (FOR RANDOM FOREST)
# ============================================================================
if best_model_name == 'Random Forest':
    print("\n" + "=" * 70)
    print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance Ranking:")
    print(importance_df.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/plots/feature_importance.png")
    plt.close()

# ============================================================================
# STEP 9: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("🎯 EVALUATION COMPLETE - FINAL SUMMARY")
print("=" * 70)

print("\nModel Performance:")
for _, row in comparison_df.iterrows():
    print(f"\n{row['Model']}:")
    print(f"  Accuracy:  {row['Accuracy']:.4f}")
    print(f"  Precision: {row['Precision']:.4f}")
    print(f"  Recall:    {row['Recall']:.4f}")
    print(f"  F1-Score:  {row['F1-Score']:.4f}")

print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

print("\n✅ Generated Files:")
print("  ✓ results/model_comparison.csv")
print("  ✓ results/plots/confusion_matrix.png")
print("  ✓ results/plots/metrics_comparison.png")
print("  ✓ results/plots/all_metrics_comparison.png")
if best_model_name == 'Random Forest':
    print("  ✓ results/plots/feature_importance.png")

print("\n" + "=" * 70)
print("✅ ALL EVALUATION COMPLETE!")    
