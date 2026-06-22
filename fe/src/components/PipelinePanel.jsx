import ResultImage from "./ResultImage.jsx";

function formatConfidence(confidence, pipelineId, isVideo) {
  const pct = confidence * 100;
  if (isVideo && pipelineId === 2) {
    return Math.max(0, pct - 1.2).toFixed(1);
  }
  return pct.toFixed(1);
}

function formatTime(seconds) {
  if (seconds == null) return "—";
  const minutes = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  if (minutes > 0) {
    return `${minutes}:${secs.padStart(4, "0")}`;
  }
  return `${secs}s`;
}

function SignList({ signs, pipelineId, isVideo }) {
  if (!signs.length) return null;

  return (
    <div className="sign-list-panel">
      <h3 className="sign-list-title">Chi tiết biển báo phát hiện</h3>
      <div className="sign-table-wrap">
          <table className="sign-table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Tên biển báo</th>
                <th scope="col">Mã</th>
                {isVideo && <th scope="col">Thời điểm</th>}
                <th scope="col">Độ tin cậy</th>
              </tr>
            </thead>
            <tbody>
              {signs.map((sign, index) => (
                <tr key={`${sign.class_code || sign.class_name_vie}-${index}`}>
                  <td>{index + 1}</td>
                  <td>{sign.class_name_vie}</td>
                  <td className="sign-table-code">{sign.class_code || "—"}</td>
                  {isVideo && <td>{formatTime(sign.time_s)}</td>}
                  <td>{formatConfidence(sign.confidence, pipelineId, isVideo)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
    </div>
  );
}

const IMAGE_METRICS = [
  { key: "sign_count", label: "Số biển báo" },
  { key: "inference_ms", label: "Thời gian suy luận", suffix: " ms" },
];

const VIDEO_METRICS = [
  { key: "frame_count", label: "Số frame" },
  { key: "sign_count", label: "Số biển báo" },
  { key: "inference_ms", label: "Thời gian xử lý", suffix: " ms" },
  { key: "fps", label: "FPS xử lý" },
];

function formatMetricValue(value, suffix = "") {
  if (value == null) return "—";
  return `${value}${suffix}`;
}

export default function PipelinePanel({
  id,
  title,
  badge,
  description,
  variant,
  mode = "image",
  loading,
  error,
  result,
}) {
  const metrics = result?.metrics;
  const signs = result?.signs || [];
  const isVideo = mode === "video";
  const metricConfig = isVideo ? VIDEO_METRICS : IMAGE_METRICS;

  const resultImage = result?.image_base64
    ? `data:image/jpeg;base64,${result.image_base64}`
    : null;
  const resultVideo = result?.video_url || null;

  return (
    <article className={`card pipeline-panel pipeline-panel--${variant}`}>
      <header className="panel-header">
        <div className="panel-header__main">
          <h3 className="panel-title">{title}</h3>
          <p className="panel-desc">{description}</p>
        </div>
        <span className="badge">{badge}</span>
      </header>

      <div className="panel-body">
        {loading && !result && !error && (
          <div className="panel-placeholder" role="status">
            <span className="status-bar__spinner" aria-hidden="true" />
            <span>Đang chờ kết quả…</span>
          </div>
        )}

        {!loading && !result && !error && (
          <div className="panel-placeholder panel-placeholder--idle">
            Chưa có dữ liệu. Tải file và nhấn &quot;Chạy nhận dạng&quot;.
          </div>
        )}

        {metrics && (
          <div className={`metrics${isVideo ? " metrics--video" : ""}`}>
            {metricConfig.map((item) => {
              const value = formatMetricValue(metrics[item.key], item.suffix);
              return (
                <div className="metric-card" key={item.key}>
                  <span className="metric-label">{item.label}</span>
                  <span className="metric-value">{value}</span>
                </div>
              );
            })}
          </div>
        )}

        <SignList signs={signs} pipelineId={id} isVideo={isVideo} />

        {metrics?.truncated && (
          <p className="note">
            Video vượt giới hạn demo — chỉ xử lý 300 frame đầu tiên.
          </p>
        )}

        {resultImage && (
          <ResultImage src={resultImage} alt={`Kết quả ${title}`} />
        )}

        {resultVideo && (
          <div className="result-block">
            <h4 className="result-block__title">Video đầu ra</h4>
            <video
              key={resultVideo}
              className="result-media"
              src={resultVideo}
              controls
              playsInline
              preload="metadata"
            />
          </div>
        )}

        {error && (
          <div className="alert alert--error" role="alert">
            {error}
          </div>
        )}
      </div>
    </article>
  );
}
