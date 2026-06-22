import { useCallback, useState } from "react";
import { runAllVideoPredictions } from "../api.js";
import MediaUpload from "./MediaUpload.jsx";
import PipelinePanel from "./PipelinePanel.jsx";

const PIPELINES = [
  { id: 1, title: "Pipeline 1", badge: "YOLO End-to-End", variant: "flow1" },
  { id: 2, title: "Pipeline 2", badge: "YOLO + CNN", variant: "flow2" },
];

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
        file={file}
        previewUrl={previewUrl}
        loading={loading}
        accept="video/*"
        mediaType="video"
        placeholderIcon="🎬"
        placeholderText="Chọn video hoặc kéo thả vào đây"
        loadingText="Đang xử lý video với cả 2 pipeline song song..."
        onFileSelect={handleFileSelect}
        onRun={handleRun}
      />
      <main className="main">
        {PIPELINES.map((pipeline) => (
          <PipelinePanel
            key={pipeline.id}
            id={pipeline.id}
            title={pipeline.title}
            badge={pipeline.badge}
            variant={pipeline.variant}
            mode="video"
            loading={loading}
            error={errors[pipeline.id]}
            result={results[pipeline.id]}
          />
        ))}
      </main>
    </>
  );
}
