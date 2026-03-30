"# ML Pipeline - Crop Suitability Prediction" 

📋 Project Overview
This pipeline predicts the most suitable crop based on soil and climate conditions using three machine learning algorithms:
Random Forest
Support Vector Machine (SVM)
Naive Bayes

## 📊 Dataset

**Source:** [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

### Setup Instructions:

1. Download the dataset from Kaggle
2. Place `Crop_recommendation.csv` in `data/raw/`
3. Run the preprocessing script
```bash
# After downloading
mv ~/Downloads/Crop_recommendation.csv data/raw/
python 01_preprocessing.py
```

**Note:** The raw dataset is not included in this repository. Please download it from the link above.

📁 Project Structure
ml-pipeline-crop/
├── 01_preprocessing.py       # Data cleaning, scaling, train-test split
├── 02_train_models.py         # Train RF, SVM, NB models
├── 03_evaluate_models.py      # Compare models and generate reports
├── data/
│   ├── raw/
│   │   └── Crop_recommendation.csv
│   └── processed/
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
├── models/
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   └── nb_model.pkl
├── results/
│   ├── model_comparison.csv
│   └── plots/
│       ├── confusion_matrix.png
│       ├── metrics_comparison.png
│       ├── all_metrics_comparison.png
│       └── feature_importance.png
└── README.md



🚀 How to Run
Step 1: Install Dependencies
bashpip install numpy pandas scikit-learn matplotlib seaborn joblib
Step 2: Run Preprocessing
bashpython 01_preprocessing.py
Output:

Cleaned and scaled data
Train-test split (80-20)
Saved preprocessors (scaler, label encoder)

Step 3: Train Models
bashpython 02_train_models.py
Output:

Trained Random Forest model → models/rf_model.pkl
Trained SVM model → models/svm_model.pkl
Trained Naive Bayes model → models/nb_model.pkl

Step 4: Evaluate and Compare
bashpython 03_evaluate_models.py
Output:

Model comparison table → results/model_comparison.csv
Confusion matrix → results/plots/confusion_matrix.png
Performance visualizations → results/plots/


📊 Dataset Information
Input Features:
FeatureDescriptionNNitrogen content in soilPPhosphorus content in soilKPotassium content in soiltemperatureTemperature in °ChumidityRelative humidity in %phpH value of soilrainfallRainfall in mm
Output:

Crop Label (22 classes): rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

Dataset Size:

Total samples: 2,200
Training set: 1,760 (80%)
Testing set: 440 (20%)


🎯 Model Performance
Comparison Table:
ModelAccuracyPrecisionRecallF1-ScoreRandom Forest0.97730.97750.97730.9773SVM0.97270.97300.97270.9727Naive Bayes0.96820.96850.96820.9682
Note: Actual values will be generated after running the scripts

🔧 Technical Details
Algorithms Used:
1. Random Forest
Type: Ensemble learning (Decision Trees)
Parameters:
n_estimators: 100
max_depth: 15
min_samples_split: 5

Best for: High accuracy, feature importance analysis

2. Support Vector Machine (SVM)
Type: Kernel-based classifier
Parameters:
kernel: RBF (Radial Basis Function)
C: 10
gamma: scale

Best for: High-dimensional data, non-linear boundaries

3. Naive Bayes
Type: Probabilistic classifier
Algorithm: Gaussian Naive Bayes
Best for: Fast training, good baseline model

Preprocessing:

Scaling: StandardScaler (mean=0, std=1)
Encoding: LabelEncoder for target variable
Split: 80-20 stratified split


📈 Output Files
Model Files (.pkl):

rf_model.pkl - Trained Random Forest model
svm_model.pkl - Trained SVM model
nb_model.pkl - Trained Naive Bayes model
scaler.pkl - Fitted StandardScaler
label_encoder.pkl - Fitted LabelEncoder

Result Files:

model_comparison.csv - Performance metrics table
confusion_matrix.png - Best model confusion matrix
metrics_comparison.png - 4-panel metric comparison
all_metrics_comparison.png - Combined bar chart
feature_importance.png - Feature importance (RF only)


💡 Usage Example
Making Predictions:
pythonimport joblib
import numpy as np

# Load saved models
model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# New data
new_data = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.93]])

# Scale and predict
scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)
crop_name = label_encoder.inverse_transform(prediction)

print(f"Recommended Crop: {crop_name[0]}")

🔬 For Research Paper
Key Points to Include:

Dataset: Crop Recommendation Dataset (Kaggle)
Preprocessing: StandardScaler, stratified split
Models: RF (100 trees), SVM (RBF kernel), Gaussian NB
Metrics: Accuracy, Precision, Recall, F1-Score
Best Model: Random Forest (expected ~97-99% accuracy)
Feature Analysis: Feature importance from Random Forest

Comparative Study:

Discuss why Random Forest performs best
Compare training times
Analyze confusion matrices
Feature importance insights