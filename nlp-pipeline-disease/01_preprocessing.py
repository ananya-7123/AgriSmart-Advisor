"""
01_preprocessing.py
====================
NLP Pipeline - Plant Disease Text Classification
Multimodal Ensemble Framework Project

What this script does:
1. Loads the text_reports.csv dataset
2. Cleans and preprocesses text
3. Applies TF-IDF vectorization
4. Stratified split → Train (80%) / Test (20%)
5. Saves processed splits and TF-IDF vectorizer
"""

import os
import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = r"data"
MODELS_DIR  = r"models"
RANDOM_SEED = 42

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  NLP PIPELINE — 01_preprocessing.py")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading dataset...")

df = pd.read_csv(os.path.join(DATA_DIR, "text_reports.csv"))

print(f"  → Total samples  : {len(df)}")
print(f"  → Diseased (1)   : {len(df[df['label'] == 1])}")
print(f"  → Healthy  (0)   : {len(df[df['label'] == 0])}")

# Drop any empty rows
df = df.dropna(subset=["text", "label"])
df = df[df["text"].str.strip() != ""]
print(f"  → After cleaning : {len(df)} samples")


# ─────────────────────────────────────────────
# 2. TEXT CLEANING
# ─────────────────────────────────────────────
print("\n[STEP 2] Cleaning text...")

def clean_text(text):
    text = str(text).lower()                          # lowercase
    text = text.translate(                             # remove punctuation
        str.maketrans("", "", string.punctuation)
    )
    text = " ".join(text.split())                      # remove extra spaces
    return text

df["cleaned_text"] = df["text"].apply(clean_text)

print("  → Sample cleaned texts:")
for i in range(3):
    print(f"    [{df['label'].iloc[i]}] {df['cleaned_text'].iloc[i][:80]}...")

print("  ✅ Text cleaning complete")


# ─────────────────────────────────────────────
# 3. STRATIFIED SPLIT — Train (80%) / Test (20%)
#    stratify=label ensures equal diseased/healthy
#    ratio in both train and test sets
# ─────────────────────────────────────────────
print("\n[STEP 3] Stratified train/test split (80/20)...")

X = df["cleaned_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y               # ← equal class distribution in both splits
)

print(f"  → Train samples : {len(X_train)}")
print(f"  → Test samples  : {len(X_test)}")

# Verify stratification
train_counts = Counter(y_train)
test_counts  = Counter(y_test)
print(f"\n  Train — Diseased: {train_counts[1]}, Healthy: {train_counts[0]}")
print(f"  Test  — Diseased: {test_counts[1]},  Healthy: {test_counts[0]}")
print(f"  ✅ Stratification verified!")


# ─────────────────────────────────────────────
# 4. TF-IDF VECTORIZATION
#    Converts text to numerical features
#    max_features=5000 keeps top 5000 words
#    ngram_range=(1,2) captures single words
#    AND two-word phrases (bigrams)
#    e.g. "yellow leaves" as one feature
# ─────────────────────────────────────────────
print("\n[STEP 4] Applying TF-IDF Vectorization...")

tfidf = TfidfVectorizer(
    max_features=5000,        # top 5000 most important words
    ngram_range=(1, 2),       # unigrams + bigrams
    min_df=2,                 # word must appear in at least 2 docs
    max_df=0.95,              # ignore words in more than 95% of docs
    sublinear_tf=True         # apply log normalization to term frequency
)

# Fit on TRAIN only — never fit on test data!
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)     # only transform, not fit

print(f"  → Vocabulary size      : {len(tfidf.vocabulary_)}")
print(f"  → Train matrix shape   : {X_train_tfidf.shape}")
print(f"  → Test matrix shape    : {X_test_tfidf.shape}")
print(f"  → Top 20 features      : {list(tfidf.get_feature_names_out()[:20])}")


# ─────────────────────────────────────────────
# 5. SAVE PROCESSED DATA + VECTORIZER
# ─────────────────────────────────────────────
print("\n[STEP 5] Saving processed data...")

# Save train/test splits as CSV (for reference)
train_df = pd.DataFrame({
    "cleaned_text" : X_train.values,
    "label"        : y_train.values
})
test_df = pd.DataFrame({
    "cleaned_text" : X_test.values,
    "label"        : y_test.values
})

train_df.to_csv(os.path.join(DATA_DIR, "train_data.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR,  "test_data.csv"),  index=False)

print(f"  ✅ train_data.csv saved ({len(train_df)} rows)")
print(f"  ✅ test_data.csv saved  ({len(test_df)} rows)")

# Save TF-IDF vectorizer — CRITICAL for fusion later
tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
joblib.dump(tfidf, tfidf_path)
print(f"  ✅ tfidf_vectorizer.pkl saved → {tfidf_path}")

# Save numpy arrays for training script
import numpy as np
from scipy import sparse

sparse.save_npz(os.path.join(DATA_DIR, "X_train_tfidf.npz"), X_train_tfidf)
sparse.save_npz(os.path.join(DATA_DIR, "X_test_tfidf.npz"),  X_test_tfidf)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train.values)
np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test.values)

print(f"  ✅ TF-IDF matrices saved as .npz files")
print(f"  ✅ Labels saved as .npy files")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE ✅")
print("=" * 60)
print(f"  Total samples    : {len(df)}")
print(f"  Train samples    : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  Vocabulary size  : {len(tfidf.vocabulary_)}")
print(f"\n  Files saved to: {DATA_DIR}/")
print(f"    ├── train_data.csv")
print(f"    ├── test_data.csv")
print(f"    ├── X_train_tfidf.npz")
print(f"    ├── X_test_tfidf.npz")
print(f"    ├── y_train.npy")
print(f"    └── y_test.npy")
print(f"\n  Files saved to: {MODELS_DIR}/")
print(f"    └── tfidf_vectorizer.pkl")
print(f"\n  Next step → run 02_train_model.py")
print("=" * 60)