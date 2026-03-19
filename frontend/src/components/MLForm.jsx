import { Leaf } from 'lucide-react';

const fields = [
  { key: 'n',           label: 'Nitrogen (N)',    unit: 'kg/ha',  min: 0, max: 140, step: 1 },
  { key: 'p',           label: 'Phosphorus (P)',  unit: 'kg/ha',  min: 0, max: 145, step: 1 },
  { key: 'k',           label: 'Potassium (K)',   unit: 'kg/ha',  min: 0, max: 205, step: 1 },
  { key: 'temperature', label: 'Temperature',     unit: '°C',     min: 0, max: 50,  step: 0.1 },
  { key: 'humidity',    label: 'Humidity',         unit: '%',      min: 0, max: 100, step: 0.1 },
  { key: 'ph',          label: 'Soil pH',          unit: '0–14',   min: 0, max: 14,  step: 0.01 },
  { key: 'rainfall',    label: 'Rainfall',         unit: 'mm',     min: 0, max: 300, step: 0.1 },
];

export default function MLForm({ soilData, setSoilData }) {
  const handleChange = (key, value) => {
    setSoilData(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="form-section">
      <h4 className="form-section-title">
        <Leaf size={18} />
        Soil Parameters
      </h4>
      <div className="form-grid">
        {fields.map((f) => (
          <div className="form-group" key={f.key}>
            <label htmlFor={`soil-${f.key}`}>
              {f.label} <span className="unit">({f.unit})</span>
            </label>
            <input
              id={`soil-${f.key}`}
              type="number"
              min={f.min}
              max={f.max}
              step={f.step}
              value={soilData[f.key] || ''}
              onChange={(e) => handleChange(f.key, e.target.value)}
              placeholder={`${f.min}–${f.max}`}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
