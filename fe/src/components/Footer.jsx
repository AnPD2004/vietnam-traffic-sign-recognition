import { useEffect, useState } from "react";
import { loadModelInfo } from "../api.js";

export default function Footer() {
  const [modelInfo, setModelInfo] = useState("Đang tải thông tin model...");

  useEffect(() => {
    let cancelled = false;

    async function fetchInfo() {
      try {
        const data = await loadModelInfo();
        if (cancelled) return;

        const p1 = data.models?.pipeline1;
        const p2 = data.models?.pipeline2;

        setModelInfo(
          `P1 YOLO: ${p1?.yolo_run_id || "—"} | ` +
            `P2 YOLO: ${p2?.yolo_run_id || "—"} | ` +
            `P2 CNN: ${p2?.cnn_run_id || "—"}`,
        );
      } catch {
        if (!cancelled) {
          setModelInfo(
            "Không thể tải thông tin model. Hãy chạy server Flask.",
          );
        }
      }
    }

    fetchInfo();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <footer className="footer">
      <p>{modelInfo}</p>
    </footer>
  );
}
