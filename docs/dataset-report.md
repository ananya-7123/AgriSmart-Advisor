# Phase 2: Dataset Collection and Preparation

This document describes the datasets used in the project **“Multimodal Crop Health and Suitability Prediction using Machine Learning and NLP”**. The project integrates structured agricultural data with unstructured textual data to support intelligent crop recommendation and disease risk assessment.

---

## 1. Structured Dataset – Crop Recommendation

### 1.1 Dataset Overview

The primary structured dataset used for crop suitability prediction is the **Crop Recommendation Dataset**, obtained from Kaggle. This dataset is designed to support precision agriculture by recommending the most suitable crop based on soil nutrients and climatic conditions.

### 1.2 Dataset Source

- **Source:** Kaggle
- **Dataset Name:** Crop Recommendation Dataset
- **URL:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

### 1.3 Data Format and Features

- **Format:** CSV (Structured data)
- **Input Features:**
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature (°C)
  - Humidity (%)
  - Soil pH
  - Rainfall (mm)
- **Target Variable:**
  - Crop label (e.g., rice, maize, chickpea, cotton, fruits, pulses, etc.)

### 1.4 Purpose in the Project

This dataset serves as the **primary benchmark dataset** for:

- Predicting the most suitable crop based on environmental conditions
- Comparing the performance of multiple machine learning algorithms such as Naive Bayes, Support Vector Machine (SVM), and Random Forest
- Supporting data-driven agricultural decision-making

The dataset is clean, well-structured, and directly usable for machine learning model training and evaluation.

---

## 2. Unstructured Dataset – Disease Prediction (NLP)

### 2.1 Dataset Overview

The unstructured dataset is a **synthetic text-based dataset** manually curated to simulate realistic farmer field reports and crop health observations. This dataset is used for Natural Language Processing (NLP)-based disease risk prediction.

### 2.2 Dataset Type and Format

- **Type:** Unstructured textual data
- **Format:** CSV
- **Columns:**
  - `text` – Farmer-style crop health report
  - `label` – Disease status
    - `1` → Disease present
    - `0` → No disease detected

### 2.3 Dataset Creation Methodology

Due to the limited availability of labeled farmer disease reports in textual form, a **manual and synthetic data generation approach** was adopted. The dataset was curated using professionally written sentence templates inspired by real agricultural observations.

The reports include:

- Visual symptoms (leaf yellowing, spots, wilting, fungal growth)
- Crop growth conditions
- Environmental context (humidity, rainfall, crop vigor)
- Healthy growth descriptions

The dataset covers **all crop types included in the structured dataset**, ensuring consistency across both pipelines.

### 2.4 Purpose in the Project

This dataset is used for:

- Binary disease risk classification using NLP techniques
- Training and evaluating text-based classifiers (TF-IDF + ML models)
- Detecting whether a crop is likely affected by disease based on farmer-reported observations

The NLP model predicts **disease presence or absence**, independent of crop type, enabling generalization across multiple crops.

---

## 3. Dataset Integration Strategy

The project follows a **dual-pipeline approach**:

1. **Crop Recommendation Pipeline (Structured Data):**

   - Uses soil and climatic parameters to predict the most suitable crop using machine learning models.

2. **Disease Detection Pipeline (Unstructured Data):**
   - Uses NLP techniques to analyze farmer text reports and detect potential disease risk.

### 3.1 Final Integrated Output

The outputs from both pipelines are combined to generate a comprehensive advisory:

- Recommended crop based on environmental conditions
- Disease risk status based on textual analysis
- Actionable insight for crop planning and monitoring

This integrated approach enables holistic agricultural decision support by considering both **crop suitability** and **crop health risk**.

---

## 4. Summary

- The structured dataset enables accurate crop suitability prediction and machine learning model comparison.
- The unstructured dataset supports NLP-based disease risk detection using realistic farmer-style reports.
- Both datasets complement each other and form the foundation of the multimodal system.
- The dataset design and methodology are suitable for academic evaluation and future research publication.
