import { useCallback, useState } from "react";
import { runAllVideoPredictions } from "../api.js";
import { PIPELINES } from "../config/pipelines.js";
import MediaUpload from "./MediaUpload.jsx";
import PipelinePanel from "./PipelinePanel.jsx";

export default function VideoTab() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({ 1: null, 2: null });
  const [errors, setErrors] = useState({ 1: null, 2: null });

  const handleFileSelect = useCallback((nextFile) => {
    if (!nextFile || !nextFile.type.startsWith("video/")) return;

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
      await runAllVideoPredictions(file);

    setResults(nextResults);
    setErrors(nextErrors);
    setLoading(false);
  };

  return (
    <>
      <MediaUpload
        stepTitle="Tải video đầu vào"
        stepDescription="Hỗ trợ MP4, WEBM, AVI, MOV, MKV. Xử lý toàn bộ video."
        file={file}
        previewUrl={previewUrl}
        loading={loading}
        accept="video/*"
        mediaType="video"
        placeholderText="Chọn file video hoặc kéo thả vào vùng này"
        hintText="Định dạng: MP4, WEBM, AVI, MOV, MKV"
        loadingText="Đang xử lý video với cả hai pipeline song song…"
        onFileSelect={handleFileSelect}
        onRun={handleRun}
      />
      <section className="results-section">
        <header className="section-header section-header--results">
          <h2 className="section-title">Kết quả so sánh pipeline</h2>
          <p className="section-desc">
            Video đầu ra có tracking và ổn định bbox. Danh sách biển báo kèm thời điểm xuất hiện.
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
              mode="video"
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
