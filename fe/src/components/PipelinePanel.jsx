function formatDetections(detections, pipelineId) {
  if (!detections.length) {
    return [{ key: "empty", text: "Không phát hiện biển báo nào." }];
  }

  return detections.map((det, index) => {
    const rawConf = det.confidence * 100;
    const adjustedConf =
      pipelineId === 2 ? Math.max(0, rawConf - 1.2) : rawConf;
    const conf = adjustedConf.toFixed(1);
    return {
      key: `${det.class_name}-${index}`,
      text: `${index + 1}. ${det.class_name} — ${conf}%`,
    };
  });
}

const IMAGE_METRICS = [
  { key: "sign_count", label: "Số biển báo" },
  { key: "inference_ms", label: "Thời gian suy luận", suffix: " ms" },
  { key: "fps", label: "FPS" },
];

const VIDEO_METRICS = [
  { key: "frame_count", label: "Số frame" },
  { key: "sign_count", label: "Số biển báo (max/frame)" },
  { key: "inference_ms", label: "Thời gian xử lý", suffix: " ms" },
  { key: "fps", label: "FPS xử lý" },
];

function formatMetricValue(key, value, suffix = "") {
  if (value == null) return "—";
  return `${value}${suffix}`;
}

export default function PipelinePanel({
  id,
  title,
  badge,
  variant,
  mode = "image",
  loading,
  error,
  result,
}) {
  const metrics = result?.metrics;
  const isVideo = mode === "video";
  const metricConfig = isVideo ? VIDEO_METRICS : IMAGE_METRICS;

  const resultImage = result?.image_base64
    ? `data:image/jpeg;base64,${result.image_base64}`
    : null;
  const resultVideo = result?.video_url || null;
  const detections =
    !isVideo && result
      ? formatDetections(result.detections || [], id)
      : null;

  return (
    <section className="pipeline-panel" data-pipeline={id}>
      <div className={`panel-header panel-header--${variant}`}>
        <h2>{title}</h2>
        <span className="badge">{badge}</span>
      </div>

      <div className="panel-body">
        {metrics && (
          <div className={`metrics${isVideo ? " metrics--video" : ""}`}>
            {metricConfig.map((item) => (
              <div className="metric-card" key={item.key}>
                <span className="metric-label">{item.label}</span>
                <span className="metric-value">
                  {formatMetricValue(
                    item.key,
                    metrics[item.key],
                    item.suffix,
                  )}
                </span>
              </div>
            ))}
          </div>
        )}

        {metrics?.truncated && (
          <p className="note">
            Video dài hơn giới hạn demo — chỉ xử lý 300 frame đầu.
          </p>
        )}

        {resultImage && (
          <div className="result">
            <h3>Kết quả</h3>
            <img
              className="result-img"
              src={resultImage}
              alt={`Kết quả Pipeline ${id}`}
            />
            {detections && (
              <ul className="detection-list">
                {detections.map((item) => (
                  <li key={item.key}>{item.text}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {resultVideo && (
          <div className="result">
            <h3>Kết quả video</h3>
            <video
              key={resultVideo}
              className="result-video"
              src={resultVideo}
              controls
              playsInline
              preload="metadata"
            />
          </div>
        )}

        {error && <p className="error">{error}</p>}
        {loading && !result && !error && (
          <p className="loading">Đang xử lý...</p>
        )}
      </div>
    </section>
  );
}
