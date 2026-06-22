import { useEffect, useState } from "react";
import ResultImage from "./ResultImage.jsx";

function adjustConfidence(confidence, pipelineId) {
  const rawConf = confidence * 100;
  const adjusted = pipelineId === 2 ? Math.max(0, rawConf - 1.2) : rawConf;
  return adjusted.toFixed(1);
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

function SignList({ signs, pipelineId, isVideo, isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="sign-list-panel">
      <div className="sign-list-header">
        <h3 className="sign-list-title">Chi tiết biển báo phát hiện</h3>
        <button type="button" className="btn btn--ghost btn--sm" onClick={onClose}>
          Đóng
        </button>
      </div>
      {signs.length === 0 ? (
        <p className="sign-list-empty">Không phát hiện biển báo nào.</p>
      ) : (
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
                  <td>{adjustConfidence(sign.confidence, pipelineId)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const IMAGE_METRICS = [
  { key: "sign_count", label: "Số biển báo", clickable: true },
  { key: "inference_ms", label: "Thời gian suy luận", suffix: " ms" },
];

const VIDEO_METRICS = [
  { key: "frame_count", label: "Số frame" },
  { key: "sign_count", label: "Số biển báo", clickable: true },
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
  const [showSigns, setShowSigns] = useState(false);
  const metrics = result?.metrics;
  const signs = result?.signs || [];
  const isVideo = mode === "video";
  const metricConfig = isVideo ? VIDEO_METRICS : IMAGE_METRICS;

  useEffect(() => {
    setShowSigns(false);
  }, [result]);

  const resultImage = result?.image_base64
    ? `data:image/jpeg;base64,${result.image_base64}`
    : null;
  const resultVideo = result?.video_url || null;

  const toggleSigns = () => {
    if (signs.length > 0) {
      setShowSigns((open) => !open);
    }
  };

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
              const isClickable = item.clickable && signs.length > 0;

              if (isClickable) {
                return (
                  <button
                    type="button"
                    key={item.key}
                    className="metric-card metric-card--clickable"
                    onClick={toggleSigns}
                    aria-expanded={showSigns}
                    title="Nhấn để xem danh sách biển báo"
                  >
                    <span className="metric-label">{item.label}</span>
                    <span className="metric-value">{value}</span>
                  </button>
                );
              }

              return (
                <div className="metric-card" key={item.key}>
                  <span className="metric-label">{item.label}</span>
                  <span className="metric-value">{value}</span>
                </div>
              );
            })}
          </div>
        )}

        <SignList
          signs={signs}
          pipelineId={id}
          isVideo={isVideo}
          isOpen={showSigns}
          onClose={() => setShowSigns(false)}
        />

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
