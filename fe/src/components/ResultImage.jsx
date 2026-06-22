import { useEffect, useRef, useState } from "react";

export default function ResultImage({ src, alt }) {
  const containerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === containerRef.current);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    const container = containerRef.current;
    if (!container) return;

    try {
      if (document.fullscreenElement === container) {
        await document.exitFullscreen();
      } else {
        await container.requestFullscreen();
      }
    } catch {
      // Browser may block fullscreen without user gesture.
    }
  };

  return (
    <div className="result-block result-media" ref={containerRef}>
      <div className="result-block__toolbar">
        <h4 className="result-block__title">Ảnh đầu ra</h4>
        <button
          type="button"
          className="btn btn--ghost btn--sm"
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? "Thoát toàn màn hình" : "Xem toàn màn hình"}
        >
          {isFullscreen ? "Thoát toàn màn hình" : "Toàn màn hình"}
        </button>
      </div>
      <img
        className="result-media result-media--image"
        src={src}
        alt={alt}
        onDoubleClick={toggleFullscreen}
      />
    </div>
  );
}
