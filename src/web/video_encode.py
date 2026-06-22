from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        raise RuntimeError(
            "ffmpeg is required for browser-compatible video output. "
            "Install imageio-ffmpeg or add ffmpeg to PATH."
        )


def transcode_to_browser_mp4(source: Path, dest: Path) -> None:
    ffmpeg = get_ffmpeg_exe()
    dest.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(stderr or "ffmpeg transcode failed")
