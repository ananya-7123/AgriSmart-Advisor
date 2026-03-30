import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, TrendingUp } from 'lucide-react';
import ARIBadge from './ARIBadge';

export default function ResultCard({ result, mode }) {
  if (!result) return null;

  // Determine risk level class for advisory styling
  const getRiskClass = (level) => {
    if (!level) return '';
    const l = level.toLowerCase();
    if (l.includes('high')) return 'high';
    if (l.includes('moderate')) return 'moderate';
    return '';
  };

  return (
    <motion.div
      className="result-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="result-card-inner">
        <div className="result-header">
          <CheckCircle size={22} />
          <h3>
            {mode === 'ml' && 'Crop Recommendation'}
            {mode === 'nlp' && 'Text Analysis Result'}
            {mode === 'cnn' && 'Image Analysis Result'}
            {mode === 'full' && 'Full Assessment Report'}
          </h3>
        </div>
        <div className="result-body">

          {/* ── ML Result ── */}
          {(mode === 'ml' || mode === 'full') && (
            <div>
              <div className="result-grid">
                <div className="result-item">
                  <div className="result-item-label">Recommended Crop</div>
                  <div className="result-item-value crop">
                    {mode === 'full' ? result.ml?.recommended_crop : result.recommended_crop}
                  </div>
                </div>
                <div className="result-item">
                  <div className="result-item-label">Confidence</div>
                  <div className="result-item-value">
                    {(((mode === 'full' ? result.ml?.confidence : result.confidence) || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-bar-fill"
                      style={{ width: `${((mode === 'full' ? result.ml?.confidence : result.confidence) || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              {/* Top 3 crops */}
              {((mode === 'full' ? result.ml?.top3_crops : result.top3_crops) || []).length > 0 && (
                <div className="result-section">
                  <div className="result-section-title">Top 3 Crop Suggestions</div>
                  <div className="top3-list">
                    {(mode === 'full' ? result.ml?.top3_crops : result.top3_crops).map((item, i) => (
                      <div key={i} className="top3-item">
                        <span className="crop-name">{item.crop}</span>
                        <span className="crop-conf">{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── NLP Result ── */}
          {(mode === 'nlp' || mode === 'full') && (
            <div className={mode === 'full' ? 'result-section' : ''}>
              {mode === 'full' && <div className="result-section-title">NLP Text Analysis</div>}
              <div className="result-grid">
                <div className="result-item">
                  <div className="result-item-label">Prediction</div>
                  <div className="result-item-value" style={{
                    color: (mode === 'full' ? result.nlp?.prediction : result.prediction) === 'Healthy'
                      ? 'var(--risk-low)' : 'var(--risk-high)'
                  }}>
                    {mode === 'full' ? result.nlp?.prediction : result.prediction}
                  </div>
                </div>
                <div className="result-item">
                  <div className="result-item-label">Disease Probability</div>
                  <div className="result-item-value">
                    {(((mode === 'full' ? result.nlp?.disease_probability : result.disease_probability) || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-bar-fill"
                      style={{
                        width: `${((mode === 'full' ? result.nlp?.disease_probability : result.disease_probability) || 0) * 100}%`,
                        background: 'linear-gradient(90deg, var(--risk-moderate), var(--risk-high))',
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── CNN Result ── */}
          {(mode === 'cnn' || mode === 'full') && (
            <div className={mode === 'full' ? 'result-section' : ''}>
              {mode === 'full' && <div className="result-section-title">CNN Image Analysis</div>}
              <div className="result-grid">
                <div className="result-item">
                  <div className="result-item-label">Predicted Class</div>
                  <div className="result-item-value">
                    {(() => {
                      const cls = mode === 'full' ? result.cnn?.predicted_class : result.predicted_class;
                      if (!cls) return 'N/A';
                      // Split 'Tomato___Early_blight' into crop and disease
                      const parts = cls.split('___');
                      if (parts.length === 2) {
                        return `${parts[0]} — ${parts[1].replace(/_/g, ' ')}`;
                      }
                      return cls.replace(/_/g, ' ');
                    })()}
                  </div>
                </div>
                <div className="result-item">
                  <div className="result-item-label">Disease Probability</div>
                  <div className="result-item-value">
                    {(((mode === 'full' ? result.cnn?.disease_probability : result.disease_probability) || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-bar-fill"
                      style={{
                        width: `${((mode === 'full' ? result.cnn?.disease_probability : result.disease_probability) || 0) * 100}%`,
                        background: 'linear-gradient(90deg, var(--risk-moderate), var(--risk-high))',
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── Fusion / ARI Result (full mode only) ── */}
          {mode === 'full' && result.fusion && (
            <div className="result-section">
              <div className="result-section-title">
                <TrendingUp size={16} style={{ display: 'inline', marginRight: '6px' }} />
                Agricultural Risk Index (ARI)
              </div>
              <div style={{ margin: '12px 0' }}>
                <ARIBadge score={result.fusion.ARI} riskLevel={result.fusion.risk_level} />
              </div>
              {result.fusion.advisory && (
                <div className={`advisory-box ${getRiskClass(result.fusion.risk_level)}`}>
                  <AlertTriangle size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'text-bottom' }} />
                  {result.fusion.advisory}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
