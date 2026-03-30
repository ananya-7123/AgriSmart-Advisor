import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sprout, MessageSquare, Camera, Zap, Send, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { supabase } from '../services/supabase';
import { predictCrop, analyzeText, classifyImage, fullAssessment } from '../services/api';
import HeroBanner from '../components/HeroBanner';
import ToolCard from '../components/ToolCard';
import WhyChooseSection from '../components/WhyChooseSection';
import MLForm from '../components/MLForm';
import NLPForm from '../components/NLPForm';
import CNNForm from '../components/CNNForm';
import ResultCard from '../components/ResultCard';
import LoadingSpinner from '../components/LoadingSpinner';

const DEFAULT_SOIL = { n: '', p: '', k: '', temperature: '', humidity: '', ph: '', rainfall: '' };

export default function Home() {
  const { user } = useAuth();
  const [activeMode, setActiveMode] = useState(null);
  const [soilData, setSoilData] = useState(DEFAULT_SOIL);
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const inputRef = useRef(null);
  const toolsRef = useRef(null);

  const handleMode = (mode) => {
    setActiveMode(prev => prev === mode ? null : mode);
    setResult(null);
    setError(null);
    // Scroll to input panel after a short delay
    setTimeout(() => {
      inputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      let data;

      if (activeMode === 'ml') {
        // Validate all soil fields filled
        const allFilled = Object.values(soilData).every(v => v !== '' && v != null);
        if (!allFilled) {
          setError('Please fill in all soil parameters.');
          setLoading(false);
          return;
        }
        data = await predictCrop(soilData);
      }

      if (activeMode === 'nlp') {
        if (text.length < 20) {
          setError('Please enter at least 20 characters of symptoms description.');
          setLoading(false);
          return;
        }
        data = await analyzeText(text);
      }

      if (activeMode === 'cnn') {
        if (!image) {
          setError('Please upload a leaf image.');
          setLoading(false);
          return;
        }
        data = await classifyImage(image);
      }

      if (activeMode === 'full') {
        const allFilled = Object.values(soilData).every(v => v !== '' && v != null);
        if (!allFilled || text.length < 20 || !image) {
          setError('Full assessment requires all soil parameters, symptoms text (20+ chars), and a leaf image.');
          setLoading(false);
          return;
        }
        data = await fullAssessment(soilData, text, image);
      }

      setResult(data);

      // Save to Supabase if logged in
      if (user && data) {
        try {
          const ari = activeMode === 'full' ? data.fusion?.ARI : null;
          const risk = activeMode === 'full' ? data.fusion?.risk_level : null;

          await supabase.from('assessments').insert({
            user_id: user.id,
            type: activeMode,
            inputs: {
              ...(activeMode === 'ml' || activeMode === 'full' ? { soilData } : {}),
              ...(activeMode === 'nlp' || activeMode === 'full' ? { text } : {}),
              ...(activeMode === 'cnn' || activeMode === 'full' ? { hasImage: true } : {}),
            },
            result: data,
            ari_score: ari ?? null,
            risk_level: risk ?? null,
          });
        } catch (saveErr) {
          console.warn('Failed to save to Supabase:', saveErr);
        }
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(
        err.response?.data?.error ||
        'Analysis failed. Make sure the Flask server is running on localhost:5000.'
      );
    } finally {
      setLoading(false);
    }
  };

  const tools = [
    {
      key: 'ml',
      icon: <Sprout size={24} />,
      title: 'Check Best Crop to Grow',
      subtitle: 'ML Model',
      buttonText: 'Get Recommendations',
    },
    {
      key: 'nlp',
      icon: <MessageSquare size={24} />,
      title: 'Describe Plant Problem',
      subtitle: 'NLP Model',
      buttonText: 'Analyze Issue',
    },
    {
      key: 'cnn',
      icon: <Camera size={24} />,
      title: 'Upload Leaf Photo',
      subtitle: 'CNN Model',
      buttonText: 'Identify Disease',
    },
  ];

  return (
    <div className="page-wrapper">
      {/* Hero */}
      <HeroBanner
        onStartAssessment={() => handleMode('full')}
        onExploreTools={() => toolsRef.current?.scrollIntoView({ behavior: 'smooth' })}
      />

      {/* Smart Tools Grid */}
      <section className="tools-section" ref={toolsRef}>
        <h2>Your Smart Tools</h2>
        <div className="tools-grid">
          {tools.map((tool) => (
            <ToolCard
              key={tool.key}
              icon={tool.icon}
              title={tool.title}
              subtitle={tool.subtitle}
              buttonText={tool.buttonText}
              isActive={activeMode === tool.key}
              onClick={() => handleMode(tool.key)}
            />
          ))}
          <ToolCard
            icon={<Zap size={24} />}
            title="Full Smart Assessment"
            subtitle="Multimodal + ARI Fusion"
            buttonText="Start Smart Assessment"
            isActive={activeMode === 'full'}
            isFull
            onClick={() => handleMode('full')}
          />
        </div>
      </section>

      {/* Input Panel (conditional) */}
      <AnimatePresence>
        {activeMode && (
          <motion.div
            ref={inputRef}
            className="input-panel"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="input-panel-inner">
              {(activeMode === 'ml' || activeMode === 'full') && (
                <MLForm soilData={soilData} setSoilData={setSoilData} />
              )}
              {(activeMode === 'nlp' || activeMode === 'full') && (
                <NLPForm text={text} setText={setText} />
              )}
              {(activeMode === 'cnn' || activeMode === 'full') && (
                <CNNForm image={image} setImage={setImage} />
              )}
              <button
                className="submit-btn"
                onClick={handleSubmit}
                disabled={loading}
              >
                {loading ? (
                  <>Analyzing...</>
                ) : (
                  <>
                    <Send size={18} />
                    Run Analysis
                  </>
                )}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading */}
      {loading && <LoadingSpinner text="Running AI analysis..." />}

      {/* Error */}
      {error && (
        <div className="error-message">
          <div className="error-message-inner">
            <AlertCircle size={18} />
            {error}
          </div>
        </div>
      )}

      {/* Result */}
      {result && !loading && (
        <ResultCard result={result} mode={activeMode} />
      )}

      {/* Why Choose */}
      <WhyChooseSection />
    </div>
  );
}
