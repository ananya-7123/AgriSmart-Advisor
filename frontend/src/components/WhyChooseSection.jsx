import { motion } from 'framer-motion';
import { CheckCircle, Bot, BarChart3, Zap } from 'lucide-react';

const pills = [
  { icon: <CheckCircle size={18} />, text: 'Easy to Use' },
  { icon: <Bot size={18} />,         text: 'AI-Powered' },
  { icon: <BarChart3 size={18} />,   text: 'Actionable Insights' },
  { icon: <Zap size={18} />,         text: 'Instant Results' },
];

export default function WhyChooseSection() {
  return (
    <section className="why-section">
      <h2>Why Choose AgriSmart Advisor?</h2>
      <div className="why-pills">
        {pills.map((pill, i) => (
          <motion.div
            key={i}
            className="why-pill"
            initial={{ opacity: 0, y: 15 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1, duration: 0.4 }}
            viewport={{ once: true }}
          >
            <span className="why-pill-icon">{pill.icon}</span>
            {pill.text}
          </motion.div>
        ))}
      </div>
    </section>
  );
}
