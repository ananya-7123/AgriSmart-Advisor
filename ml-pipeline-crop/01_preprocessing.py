# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

df = pd.read_csv("../datasets/structured/Crop_recommendation.csv")

print(" Dataset loaded successfully")
print(f"  Location: ../datasets/structured/Crop_recommendation.csv")

# ============================================================================
# STEP 2: BASIC PREVIEW
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: BASIC PREVIEW")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "-" * 60)
print("SHAPE (ROWS, COLUMNS)")
print("-" * 60)
print(df.shape)

print("\n" + "-" * 60)
print("COLUMN NAMES")
print("-" * 60)
print(df.columns.tolist())

print("\n" + "-" * 60)
print("DATA TYPES")
print("-" * 60)
print(df.dtypes)

# ============================================================================
# STEP 3: DATA QUALITY CHECKS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: DATA QUALITY CHECKS")
print("=" * 60)

# Check 1: Missing Values
print("\n" + "-" * 60)
print("CHECK 1: MISSING VALUES")
print("-" * 60)
missing_values = df.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print(" No missing values found")
else:
    print(" Missing values detected")

# Check 2: Duplicate Rows
print("\n" + "-" * 60)
print("CHECK 2: DUPLICATE ROWS")
print("-" * 60)
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")
if duplicate_count == 0:
    print(" No duplicate rows found")
else:
    print(" Duplicate rows detected")

# Check 3: Negative Values
print("\n" + "-" * 60)
print("CHECK 3: NEGATIVE VALUE CHECK")
print("-" * 60)
numeric_cols = df.drop(columns=['label'])
negative_values = (numeric_cols < 0).sum()
print(negative_values)
if negative_values.sum() == 0:
    print(" No negative values found")
else:
    print(" Negative values detected")

# ============================================================================
# STEP 4: STATISTICAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

# ============================================================================
# STEP 5: TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: TARGET VARIABLE ANALYSIS")
print("=" * 60)

print("Unique crop labels:")
print(df['label'].unique())

print(f"\nNumber of unique crops: {df['label'].nunique()}")

print("\nCrop distribution:")
print(df['label'].value_counts().sort_index())

# ============================================================================
# STEP 6: OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: OUTLIER DETECTION (IQR METHOD)")
print("=" * 60)

for col in numeric_cols.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
    print(f"{col:15s}: {outliers:3d} potential outliers")

# ============================================================================
# STEP 7: DATA CLEANLINESS VERDICT
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: FINAL CLEANLINESS VERDICT")
print("=" * 60)

issues = []

if df.isnull().sum().sum() > 0:
    issues.append("Missing values found")

if df.duplicated().sum() > 0:
    issues.append("Duplicate rows found")

if (numeric_cols < 0).sum().sum() > 0:
    issues.append("Negative values found")

if len(issues) == 0:
    print(" DATASET IS CLEAN AND READY FOR ML PIPELINE")
else:
    print("  DATASET HAS ISSUES:")
    for issue in issues:
        print(f"   - {issue}")

# ============================================================================
# STEP 8: FEATURE-TARGET SEPARATION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: FEATURE-TARGET SEPARATION")
print("=" * 60)

X = df.drop(columns=['label'])
y = df['label']

print(f"X shape (features): {X.shape}")
print(f"y shape (target):   {y.shape}")

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget sample values:")
print(y.head())

# ============================================================================
# STEP 9: LABEL ENCODING
# ============================================================================
print("\n" + "=" * 60)
print("STEP 9: LABEL ENCODING")
print("=" * 60)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
import joblib
joblib.dump(label_encoder, '../models/label_encoder.pkl')
print(" Label encoder saved successfully")

print(f"Encoded target shape: {y_encoded.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")

print("\nLabel mapping (all crops):")
for i, crop in enumerate(label_encoder.classes_):
    print(f"  {crop:15s} -> {i}")

print("\nFirst 10 encoded labels:")
print(y_encoded[:10])

# ============================================================================
# STEP 10: FEATURE SCALING
# ============================================================================
print("\n" + "=" * 60)
print("STEP 10: FEATURE SCALING (STANDARDIZATION)")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Scaled feature shape: {X_scaled.shape}")

print("\nSample BEFORE scaling (first row):")
print(X.iloc[0].values)

print("\nSample AFTER scaling (first row):")
print(X_scaled[0])

print("\n Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 11: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 60)
print("STEP 11: TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

print("\n" + "-" * 60)
print("CLASS DISTRIBUTION CHECK")
print("-" * 60)
train_dist = np.bincount(y_train)
test_dist = np.bincount(y_test)

print("\nTrain set distribution:")
for i, count in enumerate(train_dist):
    crop_name = label_encoder.classes_[i]
    print(f"  {crop_name:15s}: {count:4d} samples")

print("\nTest set distribution:")
for i, count in enumerate(test_dist):
    crop_name = label_encoder.classes_[i]
    print(f"  {crop_name:15s}: {count:4d} samples")

# ============================================================================
# STEP 12: SAVE PREPROCESSED DATA
# ============================================================================
print("\n" + "=" * 60)
print("STEP 12: SAVING PREPROCESSED DATA")
print("=" * 60)

import joblib
import os

# Create output directory
os.makedirs('/models', exist_ok=True)
os.makedirs('/data/processed', exist_ok=True)

# Save preprocessors
joblib.dump(scaler, '/models/scaler.pkl')
joblib.dump(label_encoder, '/models/label_encoder.pkl')

# Save train-test splits
np.save('/data/processed/X_train.npy', X_train)
np.save('/data/processed/X_test.npy', X_test)
np.save('/data/processed/y_train.npy', y_train)
np.save('/data/processed/y_test.npy', y_test)

print(" Scaler saved:         /models/scaler.pkl")
print(" Label encoder saved:  /models/label_encoder.pkl")
print(" Train-test splits saved in /data/processed/")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print(" PREPROCESSING COMPLETE - SUMMARY")
print("=" * 60)
print(f"✓ Total samples:        {len(df)}")
print(f"✓ Features:             {X.shape[1]}")
print(f"✓ Classes:              {len(label_encoder.classes_)}")
print(f"✓ Training samples:     {len(X_train)}")
print(f"✓ Testing samples:      {len(X_test)}")
print(f"✓ Train-Test ratio:     80-20")
print(f"✓ Data quality:         {'CLEAN' if len(issues) == 0 else 'NEEDS ATTENTION'}")
print("\n Ready for model training (Step 02)")
print("=" * 60)