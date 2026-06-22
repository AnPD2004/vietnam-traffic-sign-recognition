import { useCallback, useState } from "react";
import { runAllPredictions } from "../api.js";
import { PIPELINES } from "../config/pipelines.js";
import MediaUpload from "./MediaUpload.jsx";
import PipelinePanel from "./PipelinePanel.jsx";

export default function ImageTab() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({ 1: null, 2: null });
  const [errors, setErrors] = useState({ 1: null, 2: null });

  const handleFileSelect = useCallback((nextFile) => {
    if (!nextFile || !nextFile.type.startsWith("image/")) return;

    setFile(nextFile);
    setResults({ 1: null, 2: null });
    setErrors({ 1: null, 2: null });

    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(nextFile);
    });
  }, []);

  const handleRun = async () => {
    if (!file) return;

    setLoading(true);
    setResults({ 1: null, 2: null });
    setErrors({ 1: null, 2: null });

    const { results: nextResults, errors: nextErrors } =
      await runAllPredictions(file);

    setResults(nextResults);
    setErrors(nextErrors);
    setLoading(false);
  };

  return (
    <>
      <MediaUpload
        stepTitle="Tải ảnh đầu vào"
        stepDescription="Hỗ trợ JPG, PNG, WEBP, BMP. Hai pipeline được gọi song song trên cùng một ảnh."
        file={file}
        previewUrl={previewUrl}
        loading={loading}
        accept="image/*"
        mediaType="image"
        placeholderText="Chọn file ảnh hoặc kéo thả vào vùng này"
        hintText="Định dạng: JPG, PNG, WEBP, BMP"
        loadingText="Đang chạy song song Pipeline 1 và Pipeline 2…"
        onFileSelect={handleFileSelect}
        onRun={handleRun}
      />
      <section className="results-section">
        <header className="section-header section-header--results">
          <h2 className="section-title">Kết quả so sánh pipeline</h2>
          <p className="section-desc">
            Metrics và ảnh đầu ra có gán nhãn. Nhấn vào ô &quot;Số biển báo&quot; để xem chi tiết.
          </p>
        </header>
        <main className="main">
          {PIPELINES.map((pipeline) => (
            <PipelinePanel
              key={pipeline.id}
              id={pipeline.id}
              title={pipeline.title}
              badge={pipeline.badge}
              description={pipeline.description}
              variant={pipeline.variant}
              mode="image"
              loading={loading}
              error={errors[pipeline.id]}
              result={results[pipeline.id]}
            />
          ))}
        </main>
      </section>
    </>
  );
}
