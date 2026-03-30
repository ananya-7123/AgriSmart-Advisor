"""
app.py
=======
Flask Backend — Multimodal Agricultural Decision Support System
Exposes REST API endpoints for the ARI fusion pipeline

Endpoints:
    GET  /health     → Check if server is running
    POST /predict    → Full multimodal prediction (soil + text + image)
    POST /predict/ml → ML only (soil features)
    POST /predict/nlp → NLP only (text)
    POST /predict/cnn → CNN only (image)

Run:
    python app.py
"""

import os
import json
import string
import logging
import numpy as np
from PIL import Image
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Suppress TF verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

# Model paths
ML_RF_MODEL       = os.path.join(BASE, "ml-pipeline-crop",    "models", "rf_model.pkl")
ML_LABEL_ENCODER  = os.path.join(BASE, "models",              "label_encoder.pkl")
ML_SCALER         = os.path.join(BASE, "models",              "scaler.pkl")
NLP_MODEL         = os.path.join(BASE, "nlp-pipeline-disease","models", "nlp_model.pkl")
NLP_TFIDF         = os.path.join(BASE, "nlp-pipeline-disease","models", "tfidf_vectorizer.pkl")
CNN_MODEL         = os.path.join(BASE, "cnn-pipeline-disease", "models", "best_model_phase2.keras")
CNN_CLASS_INDICES = os.path.join(BASE, "cnn-pipeline-disease", "data",   "keras_class_indices.json")

# Upload folder for temporary image storage
UPLOAD_FOLDER    = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "JPG", "JPEG", "PNG"}

# ARI weights
ALPHA    = 0.5
BETA     = 0.5
W_TEXT   = 0.4
W_IMAGE  = 0.6

IMG_SIZE = (224, 224)
HEALTHY_KEYWORDS = ["healthy", "Healthy", "fresh_leaf", "fresh_plant"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_allowed_origins():
    """Resolve frontend origins from env for CORS configuration."""
    configured = os.environ.get("FRONTEND_ORIGINS", "").strip()
    if configured:
        return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]

    # Safe local defaults when FRONTEND_ORIGINS is not configured.
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


ALLOWED_ORIGINS = get_allowed_origins()

# ─────────────────────────────────────────────
# 1. FLASK APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16MB max upload


# ─────────────────────────────────────────────
# 2. LOAD ALL MODELS AT STARTUP
# ─────────────────────────────────────────────
print("Loading models...")

rf_model      = joblib.load(ML_RF_MODEL)
label_encoder = joblib.load(ML_LABEL_ENCODER)
scaler        = joblib.load(ML_SCALER)
print("  ✅ ML model loaded")

nlp_model = joblib.load(NLP_MODEL)
tfidf     = joblib.load(NLP_TFIDF)
print("  ✅ NLP model loaded")

cnn_model = load_model(CNN_MODEL, compile=False)
with open(CNN_CLASS_INDICES, "r") as f:
    keras_class_indices = json.load(f)
idx_to_class = {v: k for k, v in keras_class_indices.items()}
NUM_CLASSES  = len(idx_to_class)
print("  ✅ CNN model loaded")

print("All models loaded! Starting server...\n")


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg", "jpeg", "png"}

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def get_crop_prediction(soil_features):
    """ML pipeline — crop suitability"""
    features   = np.array(soil_features).reshape(1, -1)
    features   = scaler.transform(features)
    proba      = rf_model.predict_proba(features)[0]
    pred_idx   = np.argmax(proba)
    confidence = float(proba[pred_idx])
    crop_name  = label_encoder.classes_[pred_idx]

    # Top 3 crops
    top3_idx   = np.argsort(proba)[::-1][:3]
    top3       = [
        {"crop": label_encoder.classes_[i], "confidence": round(float(proba[i]), 4)}
        for i in top3_idx
    ]
    return crop_name, confidence, top3

def get_nlp_prediction(farmer_text):
    """NLP pipeline — text disease probability"""
    cleaned = clean_text(farmer_text)
    vec     = tfidf.transform([cleaned])
    proba   = nlp_model.predict_proba(vec)[0]
    return float(proba[1])   # disease probability

def get_cnn_prediction(image_path): 
    """CNN pipeline — image disease probability"""
    img      = Image.open(image_path).convert("RGB")
    img      = img.resize(IMG_SIZE)
    arr      = np.array(img, dtype=np.float32)
    arr      = preprocess_input(arr)
    arr      = np.expand_dims(arr, axis=0)

    proba    = cnn_model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    pred_cls = idx_to_class[pred_idx]
    conf     = float(proba[pred_idx])

    is_healthy   = any(kw in pred_cls for kw in HEALTHY_KEYWORDS)
    disease_prob = 1 - conf if is_healthy else conf

    return pred_cls, float(disease_prob)

def compute_ari(C, D_text, D_image):
    """ARI fusion formula"""
    D_ensemble = W_TEXT * D_text + W_IMAGE * D_image
    ARI        = ALPHA * (1 - C) + BETA * D_ensemble
    return float(ARI), float(D_ensemble)

def get_risk_level(ARI):
    if ARI < 0.35:
        return "LOW"
    elif ARI < 0.65:
        return "MODERATE"
    else:
        return "HIGH"

def get_advisory(risk_level, crop, disease_class):
    if risk_level == "LOW":
        return f"Crop '{crop}' is suitable. No significant disease detected. Continue regular monitoring."
    elif risk_level == "MODERATE":
        return f"Crop '{crop}' shows moderate disease risk ({disease_class}). Apply preventive treatment and monitor closely."
    else:
        return f"HIGH RISK! Crop '{crop}' faces serious disease threat ({disease_class}). Immediate treatment recommended. Consult an agricultural expert."


# ─────────────────────────────────────────────
# 3. API ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status"  : "running",
        "models"  : {
            "ml"  : "Random Forest",
            "nlp" : "Logistic Regression",
            "cnn" : "MobileNetV2"
        },
        "message" : "ARI Fusion API is up and running!"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Full multimodal prediction

    Form data:
        n           : float  — Nitrogen
        p           : float  — Phosphorus
        k           : float  — Potassium
        temperature : float  — Temperature (°C)
        humidity    : float  — Humidity (%)
        ph          : float  — Soil pH
        rainfall    : float  — Rainfall (mm)
        text        : string — Farmer's description
        image       : file   — Crop leaf image (jpg/png)
    """
    try:
        # ── Validate inputs ──
        required_soil = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
        for field in required_soil:
            if field not in request.form:
                return jsonify({"error": f"Missing field: {field}"}), 400

        if "text" not in request.form:
            return jsonify({"error": "Missing field: text"}), 400

        if "image" not in request.files:
            return jsonify({"error": "Missing field: image"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        if not allowed_file(image_file.filename):
            return jsonify({"error": "Invalid image format. Use JPG or PNG"}), 400

        # ── Extract soil features ──
        soil = [float(request.form[f]) for f in required_soil]
        text = request.form["text"]

        # ── Save image temporarily ──
        filename   = secure_filename(image_file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(image_path)

        # ── Run all pipelines ──
        crop_name, C, top3_crops    = get_crop_prediction(soil)
        D_text                      = get_nlp_prediction(text)
        disease_class, D_image      = get_cnn_prediction(image_path)
        ARI, D_ensemble             = compute_ari(C, D_text, D_image)
        risk_level                  = get_risk_level(ARI)
        advisory                    = get_advisory(risk_level, crop_name, disease_class)

        # ── Clean up uploaded image ──
        os.remove(image_path)

        # ── Return result ──
        return jsonify({
            "success"   : True,
            "ml": {
                "recommended_crop" : crop_name,
                "confidence"       : round(C, 4),
                "top3_crops"       : top3_crops
            },
            "nlp": {
                "disease_probability" : round(D_text, 4),
                "prediction"          : "Diseased" if D_text > 0.5 else "Healthy"
            },
            "cnn": {
                "predicted_class"     : disease_class,
                "disease_probability" : round(D_image, 4)
            },
            "fusion": {
                "C"           : round(C, 4),
                "D_text"      : round(D_text, 4),
                "D_image"     : round(D_image, 4),
                "D_ensemble"  : round(D_ensemble, 4),
                "ARI"         : round(ARI, 4),
                "risk_level"  : risk_level,
                "advisory"    : advisory
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/ml", methods=["POST"])
def predict_ml():
    """ML only — crop recommendation from soil data"""
    try:
        data = request.get_json()
        soil = [
            float(data["n"]),
            float(data["p"]),
            float(data["k"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]
        crop_name, C, top3 = get_crop_prediction(soil)
        return jsonify({
            "success"          : True,
            "recommended_crop" : crop_name,
            "confidence"       : round(C, 4),
            "top3_crops"       : top3
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/nlp", methods=["POST"])
def predict_nlp():
    """NLP only — disease detection from text"""
    try:
        data       = request.get_json()
        text       = data.get("text", "")
        D_text     = get_nlp_prediction(text)
        prediction = "Diseased" if D_text > 0.5 else "Healthy"
        return jsonify({
            "success"             : True,
            "disease_probability" : round(D_text, 4),
            "prediction"          : prediction
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/cnn", methods=["POST"])
def predict_cnn():
    """CNN only — disease detection from image"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        if not allowed_file(image_file.filename):
            return jsonify({"error": "Invalid image format"}), 400

        filename   = secure_filename(image_file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(image_path)

        disease_class, D_image = get_cnn_prediction(image_path)
        os.remove(image_path)

        return jsonify({
            "success"             : True,
            "predicted_class"     : disease_class,
            "disease_probability" : round(D_image, 4)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# 4. RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    print("=" * 50)
    print("  ARI Fusion API Server")
    print(f"  Running at http://0.0.0.0:{port}")
    print(f"  Allowed CORS origins: {', '.join(ALLOWED_ORIGINS)}")
    print("=" * 50)
    app.run(debug=debug, host="0.0.0.0", port=port)