import { useState } from "react";

export default function MediaUpload({
  file,
  previewUrl,
  loading,
  accept,
  mediaType,
  placeholderIcon,
  placeholderText,
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
    <section className="upload-section">
      <label
        className={`upload-zone${dragover ? " dragover" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input type="file" accept={accept} hidden onChange={handleInputChange} />
        {!previewUrl ? (
          <div className="upload-placeholder">
            <span className="upload-icon">{placeholderIcon}</span>
            <span>{placeholderText}</span>
          </div>
        ) : mediaType === "video" ? (
          <video className="preview-video" src={previewUrl} controls muted />
        ) : (
          <img className="preview-img" src={previewUrl} alt="Ảnh đầu vào" />
        )}
      </label>

      <button
        type="button"
        className="btn btn--run"
        disabled={loading || !file}
        onClick={onRun}
      >
        Chạy nhận dạng
      </button>

      {loading && <p className="loading upload-loading">{loadingText}</p>}
    </section>
  );
}
