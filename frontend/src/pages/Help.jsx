import {
  HelpCircle, Sprout, MessageSquare, Camera, Layers,
  AlertTriangle, BookOpen, ChevronRight
} from 'lucide-react';

export default function Help() {
  return (
    <div className="page-wrapper">
      <div className="help-page">
        <div className="container">
          <h1>Help & Documentation</h1>
          <p className="help-subtitle">Learn how to use AgriSmart Advisor effectively</p>

          {/* Getting Started */}
          <div className="help-section">
            <h2><BookOpen size={22} /> Getting Started</h2>
            <ol>
              <li><strong>Create an account</strong> — Click Login → Sign Up with your email and password.</li>
              <li><strong>Choose a tool</strong> — On the Home page, select one of the four assessment tools.</li>
              <li><strong>Enter your data</strong> — Fill in the required form fields.</li>
              <li><strong>Run analysis</strong> — Click "Run Analysis" and view your results.</li>
              <li><strong>View history</strong> — All assessments are automatically saved to your History page.</li>
            </ol>
          </div>

          {/* ML Tool */}
          <div className="help-section">
            <h2><Sprout size={22} /> ML Crop Recommendation</h2>
            <p>
              Uses a Random Forest model trained on crop-soil datasets to recommend the most suitable crop
              based on your soil and environmental conditions.
            </p>
            <table className="help-table">
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Unit</th>
                  <th>Range</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>Nitrogen (N)</td><td>kg/ha</td><td>0 – 140</td><td>Nitrogen content in soil</td></tr>
                <tr><td>Phosphorus (P)</td><td>kg/ha</td><td>0 – 145</td><td>Phosphorus content in soil</td></tr>
                <tr><td>Potassium (K)</td><td>kg/ha</td><td>0 – 205</td><td>Potassium content in soil</td></tr>
                <tr><td>Temperature</td><td>°C</td><td>0 – 50</td><td>Average temperature</td></tr>
                <tr><td>Humidity</td><td>%</td><td>0 – 100</td><td>Relative humidity percentage</td></tr>
                <tr><td>Soil pH</td><td>—</td><td>0 – 14</td><td>Acidity/alkalinity of soil</td></tr>
                <tr><td>Rainfall</td><td>mm</td><td>0 – 300</td><td>Average rainfall</td></tr>
              </tbody>
            </table>
          </div>

          {/* NLP Tool */}
          <div className="help-section">
            <h2><MessageSquare size={22} /> NLP Text Analysis</h2>
            <p>
              Uses a logistic regression model with TF-IDF vectorization to analyze textual descriptions
              of crop symptoms and determine disease probability.
            </p>
            <p><strong>Tips for effective descriptions:</strong></p>
            <ul>
              <li>Describe visible symptoms: color changes, spots, wilting, texture</li>
              <li>Mention affected plant parts: leaves, stem, roots, fruit</li>
              <li>Include timing: when symptoms first appeared, how they've progressed</li>
              <li>Example: <em>"The tomato leaves have yellow patches with brown edges, and the stems show dark lesions near the base"</em></li>
            </ul>
          </div>

          {/* CNN Tool */}
          <div className="help-section">
            <h2><Camera size={22} /> CNN Image Analysis</h2>
            <p>
              Uses a MobileNetV2 deep learning model to classify crop leaf images and detect disease from visual features.
            </p>
            <p><strong>Tips for good photos:</strong></p>
            <ul>
              <li>Use good lighting — natural daylight works best</li>
              <li>Focus on a single leaf showing symptoms clearly</li>
              <li>Avoid blurry images — hold your phone steady</li>
              <li>Supported formats: JPG, JPEG, PNG (max 10MB)</li>
              <li>The model recognizes classes in format <code>Crop___Disease</code> (e.g., <code>Tomato___Early_blight</code>)</li>
            </ul>
          </div>

          {/* Full Assessment */}
          <div className="help-section">
            <h2><Layers size={22} /> Full Smart Assessment</h2>
            <p>
              Combines all three models through a multimodal fusion engine. The fusion generates
              an Agricultural Risk Index (ARI) using this formula:
            </p>
            <div className="help-formula">
              ARI = α × (1 − C) + β × D<sub>ensemble</sub>
            </div>
            <p>Where:</p>
            <ul>
              <li><strong>C</strong> = Crop suitability confidence from ML model (0 to 1)</li>
              <li><strong>D<sub>ensemble</sub></strong> = Weighted average of NLP and CNN disease probabilities</li>
              <li><strong>D<sub>ensemble</sub></strong> = 0.4 × D<sub>text</sub> + 0.6 × D<sub>image</sub></li>
              <li><strong>α = 0.5</strong>, <strong>β = 0.5</strong> — weights for crop unsuitability vs disease risk</li>
            </ul>
          </div>

          {/* Risk Levels */}
          <div className="help-section">
            <h2><AlertTriangle size={22} /> Risk Levels</h2>
            <table className="help-table">
              <thead>
                <tr>
                  <th>ARI Range</th>
                  <th>Risk Level</th>
                  <th>Recommended Action</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>0.00 – 0.35</td>
                  <td style={{ color: 'var(--risk-low)', fontWeight: 700 }}>🟢 Low Risk</td>
                  <td>Continue regular monitoring. Crop is suitable and no significant disease detected.</td>
                </tr>
                <tr>
                  <td>0.35 – 0.65</td>
                  <td style={{ color: 'var(--risk-moderate)', fontWeight: 700 }}>🟡 Moderate Risk</td>
                  <td>Apply preventive treatment. Monitor crop closely for worsening symptoms.</td>
                </tr>
                <tr>
                  <td>0.65 – 1.00</td>
                  <td style={{ color: 'var(--risk-high)', fontWeight: 700 }}>🔴 High Risk</td>
                  <td>Immediate action needed. Apply treatment and consult an agricultural expert.</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* FAQ */}
          <div className="help-section">
            <h2><HelpCircle size={22} /> Frequently Asked Questions</h2>

            <div className="help-faq-item">
              <h3><ChevronRight size={16} style={{ display: 'inline' }} /> Analysis gives an error "Connection refused"</h3>
              <p>
                The Flask ML server must be running at <code>http://localhost:5000</code>.
                Start it with: <code>python backend/app.py</code>
              </p>
            </div>

            <div className="help-faq-item">
              <h3><ChevronRight size={16} style={{ display: 'inline' }} /> Image upload fails or times out</h3>
              <p>
                Check the image size (max 10MB) and format (JPG/PNG only). Large images may take longer to process.
                The CNN model needs the image to be an actual plant leaf photo for best results.
              </p>
            </div>

            <div className="help-faq-item">
              <h3><ChevronRight size={16} style={{ display: 'inline' }} /> My assessment history is empty</h3>
              <p>
                You must be logged in for assessments to be saved. Anonymous assessments are not stored.
                If you're logged in and still see no history, ensure the Flask server was running when you
                submitted the assessment.
              </p>
            </div>

            <div className="help-faq-item">
              <h3><ChevronRight size={16} style={{ display: 'inline' }} /> The recommended crop doesn't match the leaf image</h3>
              <p>
                This is normal — the ML crop recommendation is based on soil data, while the CNN analyzes
                an uploaded leaf photo. The Full Assessment will show both and compute ARI to give a
                combined risk level.
              </p>
            </div>

            <div className="help-faq-item">
              <h3><ChevronRight size={16} style={{ display: 'inline' }} /> How accurate are the predictions?</h3>
              <p>
                Model accuracy depends on training data quality. The ML Random Forest model has ~99% accuracy
                on the training dataset. The CNN MobileNetV2 model achieves ~93% accuracy on plant disease
                classification. Results should be used as guidance alongside expert consultation.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
