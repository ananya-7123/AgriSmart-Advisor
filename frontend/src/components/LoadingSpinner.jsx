export default function LoadingSpinner({ text = 'Analyzing...' }) {
  return (
    <div className="spinner-overlay">
      <div style={{ textAlign: 'center' }}>
        <div className="spinner" />
        <p className="spinner-text">{text}</p>
      </div>
    </div>
  );
}
