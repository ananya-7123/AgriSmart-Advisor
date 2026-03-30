"""
01_preprocessing.py
====================
CNN Pipeline - Plant Disease Classification
Multimodal Ensemble Framework Project

What this script does:
1. Scans all class folders and collects image paths + labels
2. Checks and filters corrupt/unreadable images
3. Stratified split → Train (70%) / Val (15%) / Test (15%)
4. Saves split CSVs + class label mapping (JSON)
5. Defines and tests ImageDataGenerators for Train/Val/Test
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ─────────────────────────────────────────────
# 0. CONFIG — change DATASET_DIR to your path
# ─────────────────────────────────────────────
DATASET_DIR = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\datasets\images\plant_disease"
OUTPUT_DIR  = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\cnn-pipeline-disease\data"

IMG_SIZE    = (224, 224)   # MobileNetV2 default input size
BATCH_SIZE  = 32
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15        # whatever remains after train+val

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  CNN PIPELINE — 01_preprocessing.py")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. SCAN DATASET — collect all image paths + labels
# ─────────────────────────────────────────────
print("\n[STEP 1] Scanning dataset folders...")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = []
labels      = []

class_folders = sorted([
    f for f in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, f))
])

print(f"  → Found {len(class_folders)} class folders\n")

for class_name in class_folders:
    class_path = os.path.join(DATASET_DIR, class_name)
    files = os.listdir(class_path)
    valid_files = [
        f for f in files
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]
    for fname in valid_files:
        image_paths.append(os.path.join(class_path, fname))
        labels.append(class_name)
    print(f"  [{class_name}]  →  {len(valid_files)} images")

print(f"\n  ✅ Total images collected: {len(image_paths)}")


# ─────────────────────────────────────────────
# 2. CORRUPT IMAGE CHECK
# ─────────────────────────────────────────────
print("\n[STEP 2] Checking for corrupt images...")

clean_paths  = []
clean_labels = []
corrupt_count = 0

for path, label in zip(image_paths, labels):
    try:
        img = Image.open(path)
        img.verify()          # checks file integrity without fully loading
        clean_paths.append(path)
        clean_labels.append(label)
    except Exception:
        corrupt_count += 1

print(f"  → Corrupt images removed : {corrupt_count}")
print(f"  → Clean images remaining : {len(clean_paths)}")


# ─────────────────────────────────────────────
# 3. CLASS LABEL MAPPING — encode class names to integers
# ─────────────────────────────────────────────
print("\n[STEP 3] Building class label mapping...")

unique_classes = sorted(set(clean_labels))
class_to_idx   = {cls: idx for idx, cls in enumerate(unique_classes)}
idx_to_class   = {idx: cls for cls, idx in class_to_idx.items()}
NUM_CLASSES    = len(unique_classes)

print(f"  → Total classes : {NUM_CLASSES}")

# Save mapping as JSON (needed later in train + evaluate scripts)
mapping_path = os.path.join(OUTPUT_DIR, "class_mapping.json")
with open(mapping_path, "w") as f:
    json.dump({
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes" : NUM_CLASSES
    }, f, indent=4)

print(f"  ✅ Class mapping saved → {mapping_path}")


# ─────────────────────────────────────────────
# 4. STRATIFIED SPLIT — Train / Val / Test
#    stratify=y ensures EVERY class is equally
#    represented in all three splits (no bias!)
# ─────────────────────────────────────────────
print("\n[STEP 4] Stratified splitting (70 / 15 / 15)...")

X = np.array(clean_paths)
y = np.array(clean_labels)

# Step A: Split off TEST set (15%) from full data
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED,
    stratify=y               # ← key: equal class distribution
)

# Step B: From remaining 85%, split VAL (15% of original = ~17.6% of 85%)
val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_ratio_adjusted,
    random_state=RANDOM_SEED,
    stratify=y_trainval      # ← stratify again here too!
)

print(f"  → Train : {len(X_train)} images")
print(f"  → Val   : {len(X_val)}   images")
print(f"  → Test  : {len(X_test)}  images")

# Suggestion 3 — class distribution % (great for methodology section of paper)
print("\n  📊 Train class distribution (%):")
train_dist = pd.Series(y_train).value_counts(normalize=True).mul(100).round(2)
for cls, pct in train_dist.items():
    print(f"     {cls:<50} {pct}%")

print("\n  📊 Val class distribution (%):")
val_dist = pd.Series(y_val).value_counts(normalize=True).mul(100).round(2)
for cls, pct in val_dist.items():
    print(f"     {cls:<50} {pct}%")

print("\n  ✅ If all percentages are roughly equal → stratification worked perfectly!")


# ─────────────────────────────────────────────
# 5. SAVE SPLITS AS CSV
#    Each CSV has two columns: filepath, label
#    These CSVs will be used by 02_train_model.py
# ─────────────────────────────────────────────
print("\n[STEP 5] Saving split CSVs...")

def save_split_csv(paths, labels, filename):
    df = pd.DataFrame({"filepath": paths, "label": labels})
    save_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"  ✅ Saved: {save_path}  ({len(df)} rows)")
    return df

df_train = save_split_csv(X_train, y_train, "train.csv")
df_val   = save_split_csv(X_val,   y_val,   "val.csv")
df_test  = save_split_csv(X_test,  y_test,  "test.csv")


# ─────────────────────────────────────────────
# 6. VERIFY STRATIFICATION — plot class distribution
#    All 3 bars per class should be roughly equal proportions
# ─────────────────────────────────────────────
print("\n[STEP 6] Verifying stratification (saving plot)...")

train_counts = Counter(y_train)
val_counts   = Counter(y_val)
test_counts  = Counter(y_test)

classes_sorted = sorted(class_to_idx.keys())

train_vals = [train_counts[c] for c in classes_sorted]
val_vals   = [val_counts[c]   for c in classes_sorted]
test_vals  = [test_counts[c]  for c in classes_sorted]

x_pos = np.arange(len(classes_sorted))
width = 0.3

fig, ax = plt.subplots(figsize=(20, 6))
ax.bar(x_pos - width, train_vals, width, label="Train", color="#4CAF50")
ax.bar(x_pos,         val_vals,   width, label="Val",   color="#2196F3")
ax.bar(x_pos + width, test_vals,  width, label="Test",  color="#FF5722")

ax.set_xticks(x_pos)
ax.set_xticklabels(classes_sorted, rotation=90, fontsize=7)
ax.set_title("Class Distribution Across Train / Val / Test Splits")
ax.set_ylabel("Image Count")
ax.legend()
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  ✅ Distribution plot saved → {plot_path}")


# ─────────────────────────────────────────────
# 7. IMAGE DATA GENERATORS
#    Train  → augmentation ON  (prevents overfitting)
#    Val    → NO augmentation  (we want real performance)
#    Test   → NO augmentation  (same reason)
# ─────────────────────────────────────────────
print("\n[STEP 7] Setting up ImageDataGenerators...")

# MobileNetV2 expects pixel values in [-1, 1]
# preprocess_input handles this, but rescale=1./255 + manual norm also works
# We use rescale here for simplicity and compatibility

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # maps pixels to [-1, 1] for MobileNetV2
    rotation_range=20,          # randomly rotate images ±20°
    width_shift_range=0.1,      # randomly shift left/right by 10%
    height_shift_range=0.1,     # randomly shift up/down by 10%
    shear_range=0.1,            # shear transformation
    zoom_range=0.2,             # random zoom in/out
    horizontal_flip=True,       # flip left-right (valid for leaves)
    fill_mode="nearest"         # fill empty pixels after transform
)

# Val and Test — only preprocess_input, NO augmentation
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

print("  ✅ Train generator  : augmentation ON  | preprocess_input [-1, 1]")
print("  ✅ Val/Test generator: augmentation OFF | preprocess_input [-1, 1]")


# ─────────────────────────────────────────────
# 8. CREATE FLOW GENERATORS from CSV
#    These will be used directly in 02_train_model.py
# ─────────────────────────────────────────────
print("\n[STEP 8] Creating flow generators from CSVs...")

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",   # one-hot encoded for multi-class
    shuffle=True,
    seed=RANDOM_SEED
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False               # keep order for evaluation
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"\n  → Train batches : {len(train_generator)}")
print(f"  → Val batches   : {len(val_generator)}")
print(f"  → Test batches  : {len(test_generator)}")


# ─────────────────────────────────────────────
# 9. SAVE GENERATOR CLASS INDEX MAPPING
#    Keras assigns its own internal class indices
#    when using flow_from_dataframe — save these
#    so 02_train and 03_evaluate stay consistent
# ─────────────────────────────────────────────
print("\n[STEP 9] Saving Keras generator class indices...")

keras_class_indices = train_generator.class_indices  # {"Apple___healthy": 0, ...}
keras_mapping_path  = os.path.join(OUTPUT_DIR, "keras_class_indices.json")

with open(keras_mapping_path, "w") as f:
    json.dump(keras_class_indices, f, indent=4)

print(f"  ✅ Keras class indices saved → {keras_mapping_path}")


# ─────────────────────────────────────────────
# 10. QUICK SANITY CHECK — preview one batch
# ─────────────────────────────────────────────
print("\n[STEP 10] Sanity check — previewing one batch...")

sample_images, sample_labels = next(train_generator)
print(f"  → Batch image shape : {sample_images.shape}")   # (32, 224, 224, 3)
print(f"  → Batch label shape : {sample_labels.shape}")   # (32, 42)
print(f"  → Pixel value range : [{sample_images.min():.2f}, {sample_images.max():.2f}]")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE ✅")
print("=" * 60)
print(f"  Classes       : {NUM_CLASSES}")
print(f"  Train images  : {len(X_train)}")
print(f"  Val images    : {len(X_val)}")
print(f"  Test images   : {len(X_test)}")
print(f"  Image size    : {IMG_SIZE}")
print(f"  Batch size    : {BATCH_SIZE}")
print(f"\n  Files saved to: {OUTPUT_DIR}")
print(f"    ├── train.csv")
print(f"    ├── val.csv")
print(f"    ├── test.csv")
print(f"    ├── class_mapping.json")
print(f"    ├── keras_class_indices.json")
print(f"    └── class_distribution.png")
print("\n  Next step → run 02_train_model.py")
print("=" * 60)