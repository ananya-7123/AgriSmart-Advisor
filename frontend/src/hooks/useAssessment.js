import { useState } from "react";

const DEFAULT_SOIL = {
  n: "",
  p: "",
  k: "",
  temperature: "",
  humidity: "",
  ph: "",
  rainfall: "",
};

/**
 * useAssessment — shared state hook for active assessment inputs and result.
 * Centralises all form state so it can be lifted if needed.
 */
export function useAssessment() {
  const [activeMode, setActiveMode] = useState(null);
  const [soilData, setSoilData] = useState(DEFAULT_SOIL);
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const reset = () => {
    setSoilData(DEFAULT_SOIL);
    setText("");
    setImage(null);
    setResult(null);
    setError(null);
  };

  return {
    activeMode,
    setActiveMode,
    soilData,
    setSoilData,
    text,
    setText,
    image,
    setImage,
    result,
    setResult,
    loading,
    setLoading,
    error,
    setError,
    reset,
  };
}
