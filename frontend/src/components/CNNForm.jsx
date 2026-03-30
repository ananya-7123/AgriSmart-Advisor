import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Camera, Upload, X } from 'lucide-react';

export default function CNNForm({ image, setImage }) {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
    }
  }, [setImage]);

  const removeImage = (e) => {
    e.stopPropagation();
    setImage(null);
    setPreview(null);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png'] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  return (
    <div className="form-section">
      <h4 className="form-section-title">
        <Camera size={18} />
        Leaf Image
      </h4>

      {preview ? (
        <div className="image-preview" {...getRootProps()}>
          <input {...getInputProps()} />
          <img src={preview} alt="Leaf preview" />
          <div className="image-preview-overlay">
            <span>Click to change image</span>
          </div>
          <button
            onClick={removeImage}
            style={{
              position: 'absolute',
              top: '8px',
              right: '8px',
              width: '28px',
              height: '28px',
              borderRadius: '50%',
              background: 'rgba(0,0,0,0.6)',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: 'none',
              zIndex: 2,
            }}
          >
            <X size={14} />
          </button>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="dropzone-icon">
            <Upload size={32} />
          </div>
          <p>
            <strong>Drag & drop</strong> a leaf image here, or <strong>click to browse</strong>
          </p>
          <p style={{ fontSize: '0.8rem', marginTop: '8px' }}>
            Supports JPG, PNG — Max 10MB
          </p>
        </div>
      )}
    </div>
  );
}
