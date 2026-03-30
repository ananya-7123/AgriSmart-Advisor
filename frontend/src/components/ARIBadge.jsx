export default function ARIBadge({ score, riskLevel }) {
  const level = (riskLevel || '').toLowerCase();
  let className = 'ari-badge';
  let label = riskLevel;

  if (level.includes('low')) {
    className += ' low';
    label = 'Low Risk';
  } else if (level.includes('moderate')) {
    className += ' moderate';
    label = 'Moderate Risk';
  } else if (level.includes('high')) {
    className += ' high';
    label = 'High Risk';
  }

  return (
    <div>
      <span className={className}>{label}</span>
      {score != null && (
        <div className="ari-score-display">
          <div className="ari-meter">
            <div
              className={`ari-meter-fill ${level.includes('low') ? 'low' : level.includes('moderate') ? 'moderate' : 'high'}`}
              style={{ width: `${Math.min(score * 100, 100)}%` }}
            />
          </div>
          <span className="ari-value" style={{
            color: level.includes('low') ? 'var(--risk-low)' :
                   level.includes('moderate') ? 'var(--risk-moderate)' : 'var(--risk-high)'
          }}>
            {(score * 100).toFixed(1)}%
          </span>
        </div>
      )}
    </div>
  );
}
