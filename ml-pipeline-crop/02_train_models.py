
"""
02_train_models.py
Trains Random Forest, SVM, and Naive Bayes models
Saves trained models to models/ folder
"""

import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 1: LOADING PREPROCESSED DATA")
print("=" * 70)

# Load preprocessed data from 01_preprocessing.py
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ X_test shape:  {X_test.shape}")
print(f"✓ y_train shape: {y_train.shape}")
print(f"✓ y_test shape:  {y_test.shape}")

# Load label encoder to show class names (saved by 01_preprocessing.py to repo root /models/)
import os as _os
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
label_encoder = joblib.load(_os.path.join(_repo_root, 'models', 'label_encoder.pkl'))
print(f"✓ Number of classes: {len(label_encoder.classes_)}")

# ============================================================================
# STEP 2: TRAIN MODEL 1 - RANDOM FOREST
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: TRAINING RANDOM FOREST CLASSIFIER")
print("=" * 70)

# Initialize Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_depth=15,               # Maximum depth of trees
    min_samples_split=5,        # Minimum samples to split a node
    random_state=42,            # For reproducibility
    n_jobs=-1,                  # Use all CPU cores
    verbose=0
)

# Train the model
print("Training Random Forest...")
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

print(f"✓ Random Forest trained successfully!")
print(f"  Training time: {rf_train_time:.2f} seconds")
print(f"  Number of trees: {rf_model.n_estimators}")
print(f"  Max depth: {rf_model.max_depth}")
print(f"  Training accuracy: {rf_model.score(X_train, y_train):.4f}")
print(f"  Testing accuracy: {rf_model.score(X_test, y_test):.4f}")

# Save the model
joblib.dump(rf_model, 'models/rf_model.pkl')
print(f"✓ Saved: models/rf_model.pkl")

# ============================================================================
# STEP 3: TRAIN MODEL 2 - SUPPORT VECTOR MACHINE (SVM)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: TRAINING SVM CLASSIFIER")
print("=" * 70)

# Initialize SVM
svm_model = SVC(
    kernel='rbf',               # Radial basis function kernel
    C=10,                       # Regularization parameter
    gamma='scale',              # Kernel coefficient
    random_state=42
)

# Train the model
print("Training SVM...")
start_time = time.time()
svm_model.fit(X_train, y_train)
svm_train_time = time.time() - start_time

print(f"✓ SVM trained successfully!")
print(f"  Training time: {svm_train_time:.2f} seconds")
print(f"  Kernel: {svm_model.kernel}")
print(f"  C parameter: {svm_model.C}")

# Save the model
joblib.dump(svm_model, 'models/svm_model.pkl')
print(f" Saved: models/svm_model.pkl")

# ============================================================================
# STEP 4: TRAIN MODEL 3 - NAIVE BAYES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: TRAINING NAIVE BAYES CLASSIFIER")
print("=" * 70)

# Initialize Naive Bayes
nb_model = GaussianNB()

# Train the model
print("Training Naive Bayes...")
start_time = time.time()
nb_model.fit(X_train, y_train)
nb_train_time = time.time() - start_time

print(f"✓ Naive Bayes trained successfully!")
print(f"  Training time: {nb_train_time:.2f} seconds")
print(f"  Algorithm: Gaussian Naive Bayes")

# Save the model
joblib.dump(nb_model, 'models/nb_model.pkl')
print(f"✓ Saved: models/nb_model.pkl")

# ============================================================================
# STEP 5: TRAINING SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print("\nTraining Times:")
print(f"  Random Forest: {rf_train_time:.2f}s")
print(f"  SVM:           {svm_train_time:.2f}s")
print(f"  Naive Bayes:   {nb_train_time:.2f}s")

print("\nSaved Models:")
print("  ✓ models/rf_model.pkl")
print("  ✓ models/svm_model.pkl")
print("  ✓ models/nb_model.pkl")

print("\n" + "=" * 70)
print(" ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
print("=" * 70)
print("\nNext step: Run 03_evaluate_models.py to compare performance")