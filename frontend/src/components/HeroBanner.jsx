import { motion } from 'framer-motion';
import { Sparkles, ArrowRight } from 'lucide-react';

export default function HeroBanner({ onStartAssessment, onExploreTools }) {
  return (
    <section className="hero">
      <div className="hero-inner">
        <motion.div
          className="hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="hero-badge">
            <Sparkles size={14} />
            AI-Powered Agriculture
          </div>
          <h1>
            Smart Crop & Disease Support{' '}
            <span>for Farmers</span>
          </h1>
          <p className="hero-subtitle">
            Optimize your farming with AI-driven insights. Get crop recommendations,
            disease detection, and comprehensive risk assessment — all in one platform.
          </p>
          <div className="hero-actions">
            <button className="hero-btn primary" onClick={onStartAssessment}>
              Start Full Assessment
              <ArrowRight size={18} />
            </button>
            <button className="hero-btn secondary" onClick={onExploreTools}>
              Explore Tools
            </button>
          </div>
        </motion.div>

        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="hero-illustration">
            🌾
          </div>
        </motion.div>
      </div>
    </section>
  );
}
