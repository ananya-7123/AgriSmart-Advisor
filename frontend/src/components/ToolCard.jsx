import { motion } from 'framer-motion';

export default function ToolCard({ icon, title, subtitle, buttonText, isActive, isFull, onClick }) {
  return (
    <motion.div
      className={`tool-card ${isFull ? 'full' : ''} ${isActive ? 'active' : ''}`}
      onClick={onClick}
      whileHover={{ y: -3 }}
      whileTap={{ scale: 0.98 }}
      layout
    >
      <div className="tool-card-icon">{icon}</div>
      {isFull ? (
        <>
          <div className="tool-card-content">
            <h3>{title}</h3>
            <p className="tool-card-subtitle">{subtitle}</p>
          </div>
          <button className="tool-card-btn" onClick={(e) => { e.stopPropagation(); onClick(); }}>
            {buttonText}
          </button>
        </>
      ) : (
        <>
          <h3>{title}</h3>
          <p className="tool-card-subtitle">{subtitle}</p>
          <button className="tool-card-btn" onClick={(e) => { e.stopPropagation(); onClick(); }}>
            {buttonText}
          </button>
        </>
      )}
    </motion.div>
  );
}
