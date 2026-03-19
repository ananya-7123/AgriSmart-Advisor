import { MessageSquare } from 'lucide-react';

const MAX_CHARS = 1000;

export default function NLPForm({ text, setText }) {
  return (
    <div className="form-section">
      <h4 className="form-section-title">
        <MessageSquare size={18} />
        Symptoms Description
      </h4>
      <div className="form-group">
        <label htmlFor="nlp-text">Describe the symptoms you observe</label>
        <textarea
          id="nlp-text"
          value={text}
          onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
          placeholder="Describe the symptoms you are seeing on your crops, e.g. yellow spots on leaves, wilting, discoloration..."
          rows={5}
        />
        <div className="char-count" style={{ color: text.length < 20 && text.length > 0 ? 'var(--risk-high)' : undefined }}>
          {text.length} / {MAX_CHARS}
          {text.length > 0 && text.length < 20 && ' (minimum 20 characters)'}
        </div>
      </div>
    </div>
  );
}
