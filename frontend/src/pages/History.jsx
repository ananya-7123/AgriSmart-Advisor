import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Clock, FileText, LogIn } from 'lucide-react';
import { supabase } from '../services/supabase';
import { useAuth } from '../context/AuthContext';
import ARIBadge from '../components/ARIBadge';
import LoadingSpinner from '../components/LoadingSpinner';

export default function History() {
  const { user, loading: authLoading } = useAuth();
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    if (!user) {
      setLoading(false);
      return;
    }

    async function fetchHistory() {
      try {
        // Use assessment_summaries view (all JSONB already extracted)
        const { data, error } = await supabase
          .from('assessment_summaries')
          .select('*')
          .order('created_at', { ascending: false });

        if (error) {
          // Fallback to raw assessments table if view doesn't exist
          console.warn('assessment_summaries view error, falling back to assessments:', error);
          const { data: fallback } = await supabase
            .from('assessments')
            .select('*')
            .eq('user_id', user.id)
            .order('created_at', { ascending: false });
          setRecords(fallback || []);
        } else {
          setRecords(data || []);
        }
      } catch (err) {
        console.error('Failed to fetch history:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchHistory();
  }, [user]);

  if (authLoading) return <LoadingSpinner text="Loading..." />;

  if (!user) {
    return (
      <div className="page-wrapper">
        <div className="history-login-prompt">
          <div className="history-empty-icon">
            <LogIn size={48} />
          </div>
          <h2>Sign In Required</h2>
          <p>Please log in to view your assessment history.</p>
          <Link to="/auth" className="history-login-btn" style={{ textDecoration: 'none', display: 'inline-block' }}>
            Go to Login
          </Link>
        </div>
      </div>
    );
  }

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleString('en-IN', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getTypeBadgeClass = (type) => {
    return type === 'full' ? 'history-type-badge full' : 'history-type-badge';
  };

  const getSummary = (record) => {
    const type = record.type?.toUpperCase() || 'N/A';
    const result = record.result || {};

    if (record.type === 'ml') {
      const crop = result.recommended_crop || 'N/A';
      return `Crop: ${crop}`;
    }
    if (record.type === 'nlp') {
      const pred = result.prediction || 'N/A';
      return `Prediction: ${pred}`;
    }
    if (record.type === 'cnn') {
      const cls = result.predicted_class || 'N/A';
      const parts = cls.split('___');
      return parts.length === 2 ? `${parts[0]} — ${parts[1].replace(/_/g, ' ')}` : cls;
    }
    if (record.type === 'full') {
      const crop = result.ml?.recommended_crop || 'N/A';
      return `Crop: ${crop}`;
    }
    return type + ' Assessment';
  };

  return (
    <div className="page-wrapper">
      <div className="history-page">
        <div className="container">
          <h1>Assessment History</h1>
          <p className="history-subtitle">View your past crop and disease assessments</p>
        </div>

        {loading ? (
          <LoadingSpinner text="Loading history..." />
        ) : records.length === 0 ? (
          <div className="history-empty">
            <div className="history-empty-icon">
              <FileText size={48} />
            </div>
            <h3>No assessments yet</h3>
            <p>Run your first analysis from the Home page!</p>
          </div>
        ) : (
          <div className="history-list">
            {records.map((record) => (
              <motion.div
                key={record.id}
                className="history-card"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div
                  className="history-card-header"
                  onClick={() => setExpandedId(expandedId === record.id ? null : record.id)}
                >
                  <span className={getTypeBadgeClass(record.type)}>
                    {(record.type || 'N/A').toUpperCase()}
                  </span>
                  <div className="history-card-meta">
                    <div className="history-card-summary">{getSummary(record)}</div>
                    <div className="history-card-date">
                      <Clock size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'text-bottom' }} />
                      {formatDate(record.created_at)}
                    </div>
                  </div>
                  {record.risk_level && (
                    <ARIBadge riskLevel={record.risk_level} />
                  )}
                  <ChevronDown
                    size={18}
                    className={`history-toggle-icon ${expandedId === record.id ? 'open' : ''}`}
                  />
                </div>

                <AnimatePresence>
                  {expandedId === record.id && (
                    <motion.div
                      className="history-card-body"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="history-detail-grid">
                        {record.type === 'ml' || record.type === 'full' ? (
                          <>
                            <div className="history-detail-item">
                              <div className="history-detail-label">Recommended Crop</div>
                              <div className="history-detail-value" style={{ textTransform: 'capitalize' }}>
                                {record.type === 'full'
                                  ? record.result?.ml?.recommended_crop
                                  : record.result?.recommended_crop
                                }
                              </div>
                            </div>
                            <div className="history-detail-item">
                              <div className="history-detail-label">Confidence</div>
                              <div className="history-detail-value">
                                {(((record.type === 'full'
                                  ? record.result?.ml?.confidence
                                  : record.result?.confidence) || 0) * 100).toFixed(1)}%
                              </div>
                            </div>
                          </>
                        ) : null}

                        {record.type === 'nlp' || record.type === 'full' ? (
                          <>
                            <div className="history-detail-item">
                              <div className="history-detail-label">NLP Prediction</div>
                              <div className="history-detail-value">
                                {record.type === 'full' ? record.result?.nlp?.prediction : record.result?.prediction}
                              </div>
                            </div>
                            <div className="history-detail-item">
                              <div className="history-detail-label">Disease Prob.</div>
                              <div className="history-detail-value">
                                {(((record.type === 'full'
                                  ? record.result?.nlp?.disease_probability
                                  : record.result?.disease_probability) || 0) * 100).toFixed(1)}%
                              </div>
                            </div>
                          </>
                        ) : null}

                        {record.type === 'cnn' || record.type === 'full' ? (
                          <>
                            <div className="history-detail-item">
                              <div className="history-detail-label">CNN Class</div>
                              <div className="history-detail-value">
                                {(() => {
                                  const cls = record.type === 'full'
                                    ? record.result?.cnn?.predicted_class
                                    : record.result?.predicted_class;
                                  if (!cls) return 'N/A';
                                  const parts = cls.split('___');
                                  return parts.length === 2 ? `${parts[0]} — ${parts[1].replace(/_/g, ' ')}` : cls;
                                })()}
                              </div>
                            </div>
                          </>
                        ) : null}

                        {record.type === 'full' && record.result?.fusion ? (
                          <>
                            <div className="history-detail-item">
                              <div className="history-detail-label">ARI Score</div>
                              <div className="history-detail-value">
                                {((record.result.fusion.ARI || 0) * 100).toFixed(1)}%
                              </div>
                            </div>
                            <div className="history-detail-item">
                              <div className="history-detail-label">Risk Level</div>
                              <div className="history-detail-value">
                                <ARIBadge riskLevel={record.result.fusion.risk_level} />
                              </div>
                            </div>
                            <div className="history-detail-item" style={{ gridColumn: '1 / -1' }}>
                              <div className="history-detail-label">Advisory</div>
                              <div className="history-detail-value" style={{ fontSize: '0.9rem', fontWeight: 500, lineHeight: 1.5 }}>
                                {record.result.fusion.advisory}
                              </div>
                            </div>
                          </>
                        ) : null}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
