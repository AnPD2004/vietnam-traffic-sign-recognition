from __future__ import annotations

from pathlib import Path

import torch
from flask import Flask, send_from_directory

from src.web.models import InferenceService
from src.web.routes import api_bp


def create_app(project_root: Path | None = None) -> Flask:
    root = project_root or Path(__file__).resolve().parents[2]
    fe_dir = root / "fe"
    upload_dir = root / "tmp" / "uploads"

    app = Flask(
        __name__,
        static_folder=str(fe_dir),
        static_url_path="",
    )
    app.config["PROJECT_ROOT"] = root
    app.config["UPLOAD_DIR"] = upload_dir

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    app.extensions["inference_service"] = InferenceService(root, device=device)

    app.register_blueprint(api_bp)

    @app.get("/")
    def index():
        return send_from_directory(fe_dir, "index.html")

    return app
