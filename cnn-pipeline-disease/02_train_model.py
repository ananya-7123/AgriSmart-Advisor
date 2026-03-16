"""
02_train_model.py
==================
CNN Pipeline - Plant Disease Classification
Multimodal Ensemble Framework Project

What this script does:
1. Loads train/val generators from CSVs saved in preprocessing
2. Computes class weights to handle class imbalance (Orange dominance etc.)
3. Builds MobileNetV2 transfer learning model
4. Phase 1: Train only top layers (feature extraction)
5. Phase 2: Fine-tune last few layers of MobileNetV2 (fine-tuning)
6. Saves best model + training history plot
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\cnn-pipeline-disease\data"
MODEL_DIR  = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\cnn-pipeline-disease\models"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
RANDOM_SEED = 42

# Phase 1 — train only top layers
PHASE1_EPOCHS = 15
PHASE1_LR     = 1e-3

# Phase 2 — fine-tune last N layers of MobileNetV2
PHASE2_EPOCHS      = 15
PHASE2_LR          = 1e-4        # much lower LR for fine-tuning
FINE_TUNE_AT_LAYER = 100         # unfreeze from this layer onwards

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  CNN PIPELINE — 02_train_model.py")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. LOAD CLASS MAPPING + SPLIT CSVs
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading data splits and class mapping...")

with open(os.path.join(DATA_DIR, "class_mapping.json"), "r") as f:
    mapping = json.load(f)

NUM_CLASSES  = mapping["num_classes"]
class_to_idx = mapping["class_to_idx"]

df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

print(f"  → Train samples : {len(df_train)}")
print(f"  → Val samples   : {len(df_val)}")
print(f"  → Classes       : {NUM_CLASSES}")


# ─────────────────────────────────────────────
# 2. CLASS WEIGHTS — fix imbalance
#    compute_class_weight gives higher weight
#    to rare classes so model doesn't ignore them
# ─────────────────────────────────────────────
print("\n[STEP 2] Computing class weights for imbalance correction...")

# Map string labels to integer indices
y_train_indices = df_train["label"].map(class_to_idx).values
unique_classes  = np.array(sorted(class_to_idx.values()))

weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train_indices
)

class_weight_dict = dict(zip(unique_classes, weights))

print("  → Sample class weights (higher = rarer class):")
# Print top 5 highest and lowest weighted classes
idx_to_class = {v: k for k, v in class_to_idx.items()}
sorted_weights = sorted(class_weight_dict.items(), key=lambda x: x[1], reverse=True)

print("    Most underrepresented (highest weight):")
for idx, w in sorted_weights[:5]:
    print(f"      {idx_to_class[idx]:<50} weight={w:.4f}")

print("    Most overrepresented (lowest weight):")
for idx, w in sorted_weights[-5:]:
    print(f"      {idx_to_class[idx]:<50} weight={w:.4f}")

print(f"\n  ✅ Class weights computed for all {NUM_CLASSES} classes")


# ─────────────────────────────────────────────
# 3. DATA GENERATORS
# ─────────────────────────────────────────────
print("\n[STEP 3] Setting up data generators...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=RANDOM_SEED
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col="filepath",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("  ✅ Generators ready")


# ─────────────────────────────────────────────
# 4. BUILD MODEL — MobileNetV2 + custom head
#
#    Architecture:
#    MobileNetV2 (frozen) → GlobalAveragePooling →
#    BatchNorm → Dense(256) → Dropout(0.4) →
#    Dense(128) → Dropout(0.3) → Dense(42, softmax)
# ─────────────────────────────────────────────
print("\n[STEP 4] Building MobileNetV2 model...")

# Load MobileNetV2 without top classification layers
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,           # exclude original classifier
    weights="imagenet"           # use pretrained ImageNet weights
)

# Freeze all base model layers for Phase 1
base_model.trainable = False

print(f"  → Base model layers : {len(base_model.layers)}")
print(f"  → Base model frozen : True (Phase 1)")

# Build custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)           # flatten spatial dimensions
x = BatchNormalization()(x)               # normalize activations
x = Dense(256, activation="relu")(x)      # first dense layer
x = Dropout(0.4)(x)                       # dropout to prevent overfitting
x = Dense(128, activation="relu")(x)      # second dense layer
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)  # final output

model = Model(inputs=base_model.input, outputs=output)

print(f"  → Total parameters       : {model.count_params():,}")
print(f"  → Trainable parameters   : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")


# ─────────────────────────────────────────────
# 5. CALLBACKS
# ─────────────────────────────────────────────
print("\n[STEP 5] Setting up callbacks...")

best_model_path = os.path.join(MODEL_DIR, "best_model_phase1.h5")
csv_log_path    = os.path.join(MODEL_DIR, "training_log.csv")

callbacks_phase1 = [
    # Save best model based on val accuracy
    ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    # Stop early if val_loss doesn't improve for 5 epochs
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce LR if val_loss plateaus
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    # Log all metrics to CSV
    CSVLogger(csv_log_path, append=False)
]

print("  ✅ Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger")


# ─────────────────────────────────────────────
# 6. PHASE 1 — Feature Extraction
#    Train only the custom head we added
#    MobileNetV2 layers are frozen
# ─────────────────────────────────────────────
print("\n[STEP 6] PHASE 1 — Feature Extraction Training...")
print(f"  → Epochs : {PHASE1_EPOCHS}")
print(f"  → LR     : {PHASE1_LR}")
print(f"  → Frozen : MobileNetV2 base (only head is trained)\n")

model.compile(
    optimizer=Adam(learning_rate=PHASE1_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase1 = model.fit(
    train_generator,
    epochs=PHASE1_EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,   # ← class imbalance fix applied here
    callbacks=callbacks_phase1,
    verbose=1
)


# ─────────────────────────────────────────────
# 7. PHASE 2 — Fine-Tuning
#    Unfreeze last N layers of MobileNetV2
#    Train with very low LR to gently update
#    pretrained weights
# ─────────────────────────────────────────────
print(f"\n[STEP 7] PHASE 2 — Fine-Tuning from layer {FINE_TUNE_AT_LAYER}...")
print(f"  → Epochs : {PHASE2_EPOCHS}")
print(f"  → LR     : {PHASE2_LR} (lower to protect pretrained weights)")

# Unfreeze base model
base_model.trainable = True

# Freeze all layers BEFORE fine_tune_at, unfreeze rest
for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
    layer.trainable = False
for layer in base_model.layers[FINE_TUNE_AT_LAYER:]:
    layer.trainable = True

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"  → Trainable parameters after unfreeze: {trainable_count:,}")

best_model_phase2 = os.path.join(MODEL_DIR, "best_model_phase2.h5")
csv_log_phase2    = os.path.join(MODEL_DIR, "training_log_phase2.csv")

callbacks_phase2 = [
    ModelCheckpoint(
        filepath=best_model_phase2,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    CSVLogger(csv_log_phase2, append=False)
]

# Recompile with lower LR for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=PHASE2_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase2 = model.fit(
    train_generator,
    epochs=PHASE2_EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,   # ← class weights applied here too
    callbacks=callbacks_phase2,
    verbose=1
)


# ─────────────────────────────────────────────
# 8. SAVE FINAL MODEL
# ─────────────────────────────────────────────
print("\n[STEP 8] Saving final model...")

final_model_path = os.path.join(MODEL_DIR, "final_model.h5")
model.save(final_model_path)
print(f"  ✅ Final model saved → {final_model_path}")


# ─────────────────────────────────────────────
# 9. PLOT TRAINING HISTORY
#    Combined plot for both phases
# ─────────────────────────────────────────────
print("\n[STEP 9] Plotting training history...")

def combine_history(h1, h2, key):
    return h1.history[key] + h2.history[key]

epochs_phase1 = len(history_phase1.history["accuracy"])
epochs_phase2 = len(history_phase2.history["accuracy"])
total_epochs  = epochs_phase1 + epochs_phase2

train_acc  = combine_history(history_phase1, history_phase2, "accuracy")
val_acc    = combine_history(history_phase1, history_phase2, "val_accuracy")
train_loss = combine_history(history_phase1, history_phase2, "loss")
val_loss   = combine_history(history_phase1, history_phase2, "val_loss")

epoch_range = range(1, total_epochs + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(epoch_range, train_acc, label="Train Accuracy", color="#4CAF50")
ax1.plot(epoch_range, val_acc,   label="Val Accuracy",   color="#2196F3")
ax1.axvline(x=epochs_phase1, color="gray", linestyle="--", label="Fine-tune starts")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(epoch_range, train_loss, label="Train Loss", color="#FF5722")
ax2.plot(epoch_range, val_loss,   label="Val Loss",   color="#9C27B0")
ax2.axvline(x=epochs_phase1, color="gray", linestyle="--", label="Fine-tune starts")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("MobileNetV2 Training History (Phase 1 + Phase 2)", fontsize=13)
plt.tight_layout()

plot_path = os.path.join(MODEL_DIR, "training_history.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  ✅ Training plot saved → {plot_path}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
final_val_acc = max(val_acc)

print("\n" + "=" * 60)
print("  TRAINING COMPLETE ✅")
print("=" * 60)
print(f"  Phase 1 epochs ran   : {epochs_phase1}")
print(f"  Phase 2 epochs ran   : {epochs_phase2}")
print(f"  Best val accuracy    : {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"\n  Models saved to: {MODEL_DIR}")
print(f"    ├── best_model_phase1.h5")
print(f"    ├── best_model_phase2.h5")
print(f"    ├── final_model.h5")
print(f"    ├── training_log.csv")
print(f"    ├── training_log_phase2.csv")
print(f"    └── training_history.png")
print("\n  Next step → run 03_evaluate_model.py")
print("=" * 60)