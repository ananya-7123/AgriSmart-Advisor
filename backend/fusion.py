""" 
fusion.py
==========
Multimodal Ensemble Framework
Agricultural Risk Index (ARI) Fusion

Combines:
- ML Pipeline  → Crop suitability confidence (C)
- NLP Pipeline → Text disease probability (D_text)
- CNN Pipeline → Image disease probability (D_image)

ARI = α(1 - C) + β × D_ensemble
where D_ensemble = weighted average of D_text and D_image

Risk Levels:
- Low Risk      : ARI < 0.35
- Moderate Risk : 0.35 ≤ ARI < 0.65
- High Risk     : ARI ≥ 0.65
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import string
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─────────────────────────────────────────────
# 0. CONFIG — paths to all saved models
# ─────────────────────────────────────────────

BASE = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction"

# ML Pipeline
ML_RF_MODEL       = os.path.join(BASE, "ml-pipeline-crop", "models", "rf_model.pkl")
ML_LABEL_ENCODER  = os.path.join(BASE, "models", "label_encoder.pkl")

# NLP Pipeline
NLP_MODEL         = os.path.join(BASE, "nlp-pipeline-disease", "models", "nlp_model.pkl")
NLP_TFIDF         = os.path.join(BASE, "nlp-pipeline-disease", "models", "tfidf_vectorizer.pkl")

# CNN Pipeline
CNN_MODEL         = os.path.join(BASE, "cnn-pipeline-disease", "models", "final_model.h5")
CNN_CLASS_INDICES = os.path.join(BASE, "cnn-pipeline-disease", "data", "keras_class_indices.json")

# ARI weights
ALPHA = 0.5   # weight for crop unsuitability (1 - C)
BETA  = 0.5   # weight for disease probability D

# Cross-modal ensemble weights
W_TEXT  = 0.4   # weight for NLP text model
W_IMAGE = 0.6   # weight for CNN image model

IMG_SIZE = (224, 224)

# CNN classes considered healthy
HEALTHY_KEYWORDS = ["healthy", "Healthy", "fresh_leaf", "fresh_plant"]

print("=" * 65)
print("  MULTIMODAL ENSEMBLE — ARI FUSION SYSTEM")
print("=" * 65)


# ─────────────────────────────────────────────
# 1. LOAD ALL MODELS
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading all models...")

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ML
rf_model      = joblib.load(ML_RF_MODEL)
label_encoder = joblib.load(ML_LABEL_ENCODER)
print("  ✅ ML  model loaded  (Random Forest)")

# NLP
nlp_model = joblib.load(NLP_MODEL)
tfidf     = joblib.load(NLP_TFIDF)
print("  ✅ NLP model loaded  (Logistic Regression)")

# CNN
print("CNN_MODEL path =", CNN_MODEL)
cnn_model = load_model(CNN_MODEL, compile=False)
with open(CNN_CLASS_INDICES, "r") as f:
    keras_class_indices = json.load(f)
idx_to_class = {v: k for k, v in keras_class_indices.items()}
print("  ✅ CNN model loaded  (MobileNetV2)")


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def clean_text(text):
    """Clean farmer text report"""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def get_crop_confidence(soil_features):
    """
    ML Pipeline — Crop suitability
    Input : [N, P, K, temperature, humidity, ph, rainfall]
    Output: recommended crop name, confidence score C
    """
    features    = np.array(soil_features).reshape(1, -1)
    proba       = rf_model.predict_proba(features)[0]
    pred_idx    = np.argmax(proba)
    confidence  = proba[pred_idx]
    crop_name   = label_encoder.classes_[pred_idx]
    return crop_name, confidence

def get_nlp_disease_prob(farmer_text):
    """
    NLP Pipeline — Text disease probability
    Input : raw farmer text report
    Output: disease probability D_text (0 to 1)
    """
    cleaned  = clean_text(farmer_text)
    vec      = tfidf.transform([cleaned])
    proba    = nlp_model.predict_proba(vec)[0]
    return proba[1]   # index 1 = diseased probability

def get_cnn_disease_prob(image_path):
    """
    CNN Pipeline — Image disease probability
    Input : path to crop leaf image
    Output: predicted class, disease probability D_image (0 to 1)
    """
    img      = Image.open(image_path).convert("RGB")
    img      = img.resize(IMG_SIZE)
    arr      = np.array(img, dtype=np.float32)
    arr      = preprocess_input(arr)
    arr      = np.expand_dims(arr, axis=0)

    proba    = cnn_model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(proba)
    pred_cls = idx_to_class[pred_idx]

    # Disease probability = 1 if diseased class, else confidence of healthy
    is_healthy   = any(kw in pred_cls for kw in HEALTHY_KEYWORDS)
    disease_prob = 1 - proba[pred_idx] if is_healthy else float(proba[pred_idx])

    return pred_cls, disease_prob

def compute_ari(C, D_text, D_image):
    """
    ARI Fusion
    D_ensemble = weighted average of text and image disease probs
    ARI = α(1 - C) + β × D_ensemble
    """
    D_ensemble = W_TEXT * D_text + W_IMAGE * D_image
    ARI        = ALPHA * (1 - C) + BETA * D_ensemble
    return ARI, D_ensemble

def get_risk_level(ARI):
    """Categorize ARI into risk levels"""
    if ARI < 0.35:
        return "🟢 LOW RISK"
    elif ARI < 0.65:
        return "🟡 MODERATE RISK"
    else:
        return "🔴 HIGH RISK"

def get_advisory(risk_level, crop, disease_class):
    """Generate simple advisory based on risk level"""
    if "LOW" in risk_level:
        return f"Crop '{crop}' is suitable. No significant disease detected. Continue regular monitoring."
    elif "MODERATE" in risk_level:
        return f"Crop '{crop}' is suitable but moderate disease risk detected ({disease_class}). Apply preventive fungicide/pesticide and monitor closely."
    else:
        return f"HIGH RISK! Crop '{crop}' faces serious disease threat ({disease_class}). Immediate treatment recommended. Consult agricultural expert."


# ─────────────────────────────────────────────
# 2. MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict(soil_features, farmer_text, image_path):
    """
    Full multimodal prediction pipeline

    Inputs:
        soil_features : list [N, P, K, temp, humidity, ph, rainfall]
        farmer_text   : string — farmer's description of crop condition
        image_path    : string — path to crop leaf image

    Output: full advisory report
    """
    print("\n" + "─" * 65)
    print("  RUNNING MULTIMODAL PREDICTION")
    print("─" * 65)

    # ML Pipeline
    crop_name, C = get_crop_confidence(soil_features)
    print(f"\n  [ML]  Recommended Crop  : {crop_name}")
    print(f"  [ML]  Suitability Score : {C:.4f} ({C*100:.1f}%)")

    # NLP Pipeline
    D_text = get_nlp_disease_prob(farmer_text)
    print(f"\n  [NLP] Farmer Report     : '{farmer_text[:60]}...'")
    print(f"  [NLP] Disease Prob      : {D_text:.4f} ({D_text*100:.1f}%)")

    # CNN Pipeline
    disease_class, D_image = get_cnn_disease_prob(image_path)
    print(f"\n  [CNN] Predicted Class   : {disease_class}")
    print(f"  [CNN] Disease Prob      : {D_image:.4f} ({D_image*100:.1f}%)")
    cnn_crop = disease_class.split("___")[0]

    if cnn_crop.lower() != crop_name.lower():
        print("\n  ⚠ WARNING: Crop mismatch detected!")
        print(f"     ML recommends: {crop_name}")
        print(f"     CNN image crop: {cnn_crop}")

    # ARI Fusion
    ARI, D_ensemble = compute_ari(C, D_text, D_image)
    risk_level      = get_risk_level(ARI)
    advisory        = get_advisory(risk_level, crop_name, disease_class)

    # Final Report
    print("\n" + "=" * 65)
    print("  AGRICULTURAL RISK INDEX (ARI) REPORT")
    print("=" * 65)
    print(f"  Crop Suitability (C)     : {C:.4f}")
    print(f"  Text Disease Prob        : {D_text:.4f}")
    print(f"  Image Disease Prob       : {D_image:.4f}")
    print(f"  D_ensemble               : {D_ensemble:.4f}")
    print(f"  ARI Score                : {ARI:.4f}")
    print(f"  Risk Level               : {risk_level}")
    print(f"\n  Recommended Crop         : {crop_name}")
    print(f"  Detected Disease Class   : {disease_class}")
    print(f"\n  Advisory:")
    print(f"  {advisory}")
    print("=" * 65)

    return {
        "crop"          : crop_name,
        "confidence"    : C,
        "disease_class" : disease_class,
        "D_text"        : D_text,
        "D_image"       : D_image,
        "D_ensemble"    : D_ensemble,
        "ARI"           : ARI,
        "risk_level"    : risk_level,
        "advisory"      : advisory
    }


# ─────────────────────────────────────────────
# 3. TEST WITH SAMPLE INPUT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n[STEP 2] Running sample prediction...")
    print("  (Replace the inputs below with real values)\n")

    # ── CHANGE THESE TO TEST WITH REAL DATA ──
    sample_soil = [90, 42, 43, 20.8, 82, 6.5, 202]   # N,P,K,temp,humidity,ph,rainfall
    sample_text = "the leaves are turning yellow with brown spots and the plant looks weak"
    sample_image = r"C:\Users\KIIT0001\college\minor project\github_setup\crop-analysis-disease-prediction\datasets\images\plant_disease\Tomato___Early_blight\0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG"

    result = predict(sample_soil, sample_text, sample_image)