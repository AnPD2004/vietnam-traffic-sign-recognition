from __future__ import annotations

from pathlib import Path

import torch
from flask import Flask, abort, send_from_directory

from src.web.models import InferenceService
from src.web.routes import api_bp


def create_app(project_root: Path | None = None) -> Flask:
    root = project_root or Path(__file__).resolve().parents[2]
    fe_dist = root / "fe" / "dist"
    upload_dir = root / "tmp" / "uploads"
    output_dir = root / "tmp" / "outputs"

    app = Flask(__name__)
    app.config["PROJECT_ROOT"] = root
    app.config["UPLOAD_DIR"] = upload_dir
    app.config["OUTPUT_DIR"] = output_dir

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    app.extensions["inference_service"] = InferenceService(root, device=device)

    app.register_blueprint(api_bp)

    @app.get("/")
    def index():
        return send_from_directory(fe_dist, "index.html")

    @app.get("/assets/<path:filename>")
    def frontend_assets(filename: str):
        return send_from_directory(fe_dist / "assets", filename)

    @app.get("/<path:path>")
    def spa_fallback(path: str):
        if path.startswith(("predict", "health", "outputs", "api", "assets")):
            abort(404)
        index = fe_dist / "index.html"
        if not index.is_file():
            abort(404)
        return send_from_directory(fe_dist, "index.html")

    return app
