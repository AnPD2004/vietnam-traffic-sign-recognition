import { useState } from "react";

function UploadIcon() {
  return (
    <svg
      className="upload-svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden="true"
    >
      <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" />
    </svg>
  );
}

export default function MediaUpload({
  stepTitle,
  stepDescription,
  file,
  previewUrl,
  loading,
  accept,
  mediaType,
  placeholderText,
  hintText,
  loadingText,
  onFileSelect,
  onRun,
}) {
  const [dragover, setDragover] = useState(false);

  const handleInputChange = (event) => {
    const nextFile = event.target.files?.[0];
    if (nextFile) onFileSelect(nextFile);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragover(true);
  };

  const handleDragLeave = () => {
    setDragover(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragover(false);
    const nextFile = event.dataTransfer.files?.[0];
    if (nextFile) onFileSelect(nextFile);
  };

  return (
    <section className="card upload-section">
      <header className="section-header">
        <h2 className="section-title">{stepTitle}</h2>
        <p className="section-desc">{stepDescription}</p>
      </header>

      <label
        className={`upload-zone${dragover ? " upload-zone--dragover" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input type="file" accept={accept} hidden onChange={handleInputChange} />
        {!previewUrl ? (
          <div className="upload-placeholder">
            <UploadIcon />
            <span className="upload-text">{placeholderText}</span>
            {hintText && <span className="upload-hint">{hintText}</span>}
          </div>
        ) : mediaType === "video" ? (
          <video className="preview-media" src={previewUrl} controls muted />
        ) : (
          <img className="preview-media" src={previewUrl} alt="Dữ liệu đầu vào" />
        )}
      </label>

      <div className="upload-actions">
        <button
          type="button"
          className="btn btn--primary"
          disabled={loading || !file}
          onClick={onRun}
        >
          {loading ? "Đang xử lý…" : "Chạy nhận dạng"}
        </button>
        {file && !loading && (
          <span className="upload-filename">{file.name}</span>
        )}
      </div>

      {loading && (
        <div className="status-bar" role="status">
          <span className="status-bar__spinner" aria-hidden="true" />
          <span>{loadingText}</span>
        </div>
      )}
    </section>
  );
}
