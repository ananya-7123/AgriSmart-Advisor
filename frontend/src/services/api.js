import axios from "axios";

// In dev: Vite proxy rewrites /api → localhost:5000, so baseURL = '/api'
// In prod: no proxy exists, so hit the Render URL directly from VITE_API_BASE_URL
const BASE_URL = import.meta.env.PROD
  ? import.meta.env.VITE_API_BASE_URL // e.g. https://agrismart-api.onrender.com
  : "/api"; // Vite proxy handles this locally

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
});

/**
 * ML Model — Crop recommendation from soil parameters
 * POST /predict/ml
 * Payload: { n, p, k, temperature, humidity, ph, rainfall }
 * Response: { success, recommended_crop, confidence, top3_crops }
 */
export async function predictCrop(soilData) {
  const res = await api.post("/predict/ml", {
    n: Number(soilData.n),
    p: Number(soilData.p),
    k: Number(soilData.k),
    temperature: Number(soilData.temperature),
    humidity: Number(soilData.humidity),
    ph: Number(soilData.ph),
    rainfall: Number(soilData.rainfall),
  });
  return res.data;
}

/**
 * NLP Model — Disease detection from farmer text
 * POST /predict/nlp
 * Payload: { text }
 * Response: { success, prediction, disease_probability }
 */
export async function analyzeText(text) {
  const res = await api.post("/predict/nlp", { text });
  return res.data;
}

/**
 * CNN Model — Disease detection from leaf image
 * POST /predict/cnn
 * Payload: FormData with key 'image'
 * Response: { success, predicted_class, disease_probability }
 */
export async function classifyImage(imageFile) {
  const form = new FormData();
  form.append("image", imageFile);
  const res = await api.post("/predict/cnn", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

/**
 * Full Fusion — All three inputs combined
 * POST /predict
 * Payload: FormData with soil fields as individual strings + text + image
 * Response: { success, ml:{...}, nlp:{...}, cnn:{...}, fusion:{ARI, risk_level, advisory} }
 *
 * CRITICAL: soil fields are appended individually as strings (not JSON blob)
 * because Flask reads request.form, not request.json
 */
export async function fullAssessment(soilData, text, imageFile) {
  const form = new FormData();
  // Soil fields as individual string values
  form.append("n", String(soilData.n));
  form.append("p", String(soilData.p));
  form.append("k", String(soilData.k));
  form.append("temperature", String(soilData.temperature));
  form.append("humidity", String(soilData.humidity));
  form.append("ph", String(soilData.ph));
  form.append("rainfall", String(soilData.rainfall));
  form.append("text", text);
  form.append("image", imageFile);
  const res = await api.post("/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}
