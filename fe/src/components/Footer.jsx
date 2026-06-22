import { useEffect, useState } from "react";
import { loadModelInfo } from "../api.js";

export default function Footer() {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function fetchInfo() {
      try {
        const data = await loadModelInfo();
        if (!cancelled) {
          setModels(data.models);
          setError(false);
        }
      } catch {
        if (!cancelled) {
          setModels(null);
          setError(true);
        }
      }
    }

    fetchInfo();
    return () => {
      cancelled = true;
    };
  }, []);

  const p1 = models?.pipeline1;
  const p2 = models?.pipeline2;

  return (
    <footer className="footer">
      <div className="footer-inner">
        <h2 className="footer-title">Cấu hình mô hình đang sử dụng</h2>
        {error ? (
          <p className="footer-error">
            Không kết nối được API. Vui lòng khởi động server Flask.
          </p>
        ) : !models ? (
          <p className="footer-muted">Đang tải thông tin mô hình…</p>
        ) : (
          <dl className="footer-grid">
            <div className="footer-item">
              <dt>Pipeline 1 — YOLO</dt>
              <dd>{p1?.yolo_run_id || "—"}</dd>
            </div>
            <div className="footer-item">
              <dt>Pipeline 2 — YOLO</dt>
              <dd>{p2?.yolo_run_id || "—"}</dd>
            </div>
            <div className="footer-item">
              <dt>Pipeline 2 — CNN</dt>
              <dd>{p2?.cnn_run_id || "—"}</dd>
            </div>
            <div className="footer-item">
              <dt>Kích thước đầu vào (imgsz)</dt>
              <dd>{p1?.imgsz ?? "—"} px</dd>
            </div>
          </dl>
        )}
      </div>
    </footer>
  );
}
