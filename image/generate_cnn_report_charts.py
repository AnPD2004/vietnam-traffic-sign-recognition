"""Generate CNN benchmark figures for report sections 4.3.1–4.3.3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "artifacts/pipeline2/registry.json"
BEST_PATH = ROOT / "artifacts/pipeline2/best/best.json"

COLORS = {
    "resnet50": "#6C757D",
    "efficientnet_b0": "#2E86AB",
    "accent": "#F18F01",
    "best": "#27AE60",
    "baseline": "#7F8C8D",
    "fps": "#1B9E77",
    "bg": "#FAFAFA",
}

# Flow 2 e2e latency estimate: T_yolo + avg_dets * (T_cnn + T_crop)
# YOLO from Pipeline 1 best; avg detections from test split (1645 / 639 images).
YOLO_E2E_MS = 7.9242
AVG_DETS_PER_IMAGE = 1645 / 639
CROP_OVERHEAD_MS = 0.25


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": COLORS["bg"],
            "axes.edgecolor": "#CCCCCC",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_best(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(run: dict, key: str) -> float:
    return float(run["metrics"][key])


def _get_run(registry: dict, run_id: str) -> dict:
    return registry["runs"][run_id]


def estimate_flow2_e2e_ms(cnn_latency_ms: float) -> float:
    """End-to-end latency per image: YOLO detect + crop + CNN per bbox."""
    return YOLO_E2E_MS + AVG_DETS_PER_IMAGE * (cnn_latency_ms + CROP_OVERHEAD_MS)


def estimate_flow2_e2e_fps(cnn_latency_ms: float) -> float:
    return 1000.0 / estimate_flow2_e2e_ms(cnn_latency_ms)


def fig_431_model_compare(registry: dict, output: Path) -> None:
    """4.3.1 — So sánh baseline ResNet50 và EfficientNet-B0."""
    ids = [
        "resnet50_e50_isz224_lr0.001_b32_ft",
        "efficientnet_b0_e50_isz224_lr0.001_b32_ft",
    ]
    labels = ["ResNet50", "EfficientNet-B0"]
    colors = [COLORS["resnet50"], COLORS["efficientnet_b0"]]

    metrics_keys = ["accuracy", "precision", "recall", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-macro"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Left: classification metrics
    x = np.arange(len(metric_labels))
    w = 0.32
    for i, (rid, label, color) in enumerate(zip(ids, labels, colors, strict=True)):
        run = _get_run(registry, rid)
        vals = [_metric(run, k) for k in metrics_keys]
        bars = axes[0].bar(x + (i - 0.5) * w, vals, w, label=label, color=color, edgecolor="white")
        for bar, v in zip(bars, vals, strict=True):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0.97, 1.001)
    axes[0].set_ylabel("Giá trị")
    axes[0].set_title("Chỉ số phân loại (baseline)", fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    # Right: Flow 2 e2e latency (YOLO + CNN), FPS, CNN model size
    cnn_lat = [_metric(_get_run(registry, rid), "latency_ms_per_image") for rid in ids]
    lat_e2e = [estimate_flow2_e2e_ms(v) for v in cnn_lat]
    fps_e2e = [estimate_flow2_e2e_fps(v) for v in cnn_lat]
    size = [_metric(_get_run(registry, rid), "model_size_mb") for rid in ids]

    ax_l = axes[1]
    ax_fps = ax_l.twinx()
    ax_mb = ax_l.twinx()
    ax_mb.spines["right"].set_position(("axes", 1.14))
    ax_mb.spines["right"].set_visible(True)

    pos = np.arange(len(labels))
    w = 0.24
    bars_lat = ax_l.bar(pos - w, lat_e2e, w, color=COLORS["accent"], label="Latency e2e (ms/ảnh)")
    bars_fps = ax_fps.bar(pos, fps_e2e, w, color=COLORS["fps"], label="FPS e2e")
    bars_mb = ax_mb.bar(pos + w, size, w, color=COLORS["baseline"], label="Kích thước CNN (MB)")

    ax_l.set_xticks(pos)
    ax_l.set_xticklabels(labels)
    ax_l.set_ylabel("Latency e2e (ms/ảnh)")
    ax_fps.set_ylabel("FPS e2e", color=COLORS["fps"])
    ax_fps.tick_params(axis="y", labelcolor=COLORS["fps"])
    ax_fps.set_ylim(0, max(fps_e2e) * 1.35)
    ax_mb.set_ylabel("Kích thước CNN (MB)", color=COLORS["baseline"])
    ax_mb.tick_params(axis="y", labelcolor=COLORS["baseline"])
    ax_mb.set_ylim(0, max(size) * 1.25)
    ax_l.set_ylim(0, max(lat_e2e) * 1.28)
    ax_l.set_title("Tốc độ suy luận e2e (YOLO+CNN) & kích thước", fontweight="bold")
    ax_l.grid(axis="y", alpha=0.25)

    for bar, v in zip(bars_lat, lat_e2e, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.2f}", ha="center", fontsize=7)
    for bar, v in zip(bars_fps, fps_e2e, strict=True):
        ax_fps.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=7, color=COLORS["fps"])
    for bar, v in zip(bars_mb, size, strict=True):
        ax_mb.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.2f}", ha="center", fontsize=7)

    handles = [bars_lat, bars_fps, bars_mb]
    ax_l.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=7)

    fig.suptitle("4.3.1 — Đánh giá baseline ResNet50 và EfficientNet-B0", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_432_lr(registry: dict, output: Path) -> None:
    """4.3.2 — Ảnh hưởng learning rate (EfficientNet-B0)."""
    configs = [
        ("0.0001", "efficientnet_b0_e50_isz224_lr0.0001_b32_ft"),
        ("0.001", "efficientnet_b0_e50_isz224_lr0.001_b32_ft"),
        ("0.01", "efficientnet_b0_e50_isz224_lr0.01_b32_ft"),
    ]
    labels = [c[0] for c in configs]
    f1 = [_metric(_get_run(registry, rid), "f1_macro") for _, rid in configs]
    acc = [_metric(_get_run(registry, rid), "accuracy") for _, rid in configs]
    val_loss = [_metric(_get_run(registry, rid), "val_loss") for _, rid in configs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(labels))

    # F1 & Accuracy
    w = 0.35
    axes[0].bar(x - w / 2, f1, w, label="F1-macro", color=COLORS["efficientnet_b0"])
    axes[0].bar(x + w / 2, acc, w, label="Accuracy", color=COLORS["accent"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylim(0.86, 1.0)
    axes[0].set_title("Độ chính xác theo learning rate", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    for i, (a, b) in enumerate(zip(f1, acc, strict=True)):
        axes[0].text(i - w / 2, a + 0.006, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, b + 0.006, f"{b:.3f}", ha="center", fontsize=8)

    # Val loss
    bars = axes[1].bar(labels, val_loss, color=COLORS["accent"], width=0.5)
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Validation Loss theo learning rate", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, val_loss, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("4.3.2 — Ảnh hưởng Learning Rate (EfficientNet-B0)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_432_transfer(registry: dict, output: Path) -> None:
    """4.3.2 — Ảnh hưởng chiến lược Transfer Learning."""
    configs = [
        ("Frozen Backbone", "efficientnet_b0_e50_isz224_lr0.001_b32_frz"),
        ("Fine-Tuning", "efficientnet_b0_e50_isz224_lr0.001_b32_ft"),
    ]
    labels = [c[0] for c in configs]
    f1 = [_metric(_get_run(registry, rid), "f1_macro") for _, rid in configs]
    acc = [_metric(_get_run(registry, rid), "accuracy") for _, rid in configs]
    val_loss = [_metric(_get_run(registry, rid), "val_loss") for _, rid in configs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.35

    axes[0].bar(x - w / 2, f1, w, label="F1-macro", color=COLORS["efficientnet_b0"])
    axes[0].bar(x + w / 2, acc, w, label="Accuracy", color=COLORS["accent"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylim(0.98, 0.995)
    axes[0].set_ylabel("Giá trị")
    axes[0].set_title("So sánh chiến lược Transfer Learning", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    for i, (a, b) in enumerate(zip(f1, acc, strict=True)):
        axes[0].text(i - w / 2, a + 0.002, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, b + 0.002, f"{b:.3f}", ha="center", fontsize=8)

    bars = axes[1].bar(labels, val_loss, color=[COLORS["baseline"], COLORS["best"]], width=0.5)
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Validation Loss theo chiến lược huấn luyện", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, val_loss, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("4.3.2 — Ảnh hưởng chiến lược Transfer Learning (EfficientNet-B0)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_432_progress(registry: dict, best_overall_id: str, output: Path) -> None:
    """4.3.2 — Tiến trình F1-macro qua các giai đoạn exp1–exp3."""
    runs = registry["runs"]
    best_per_exp = registry.get("best_per_experiment", {})
    # đảm bảo đúng thứ tự exp1, exp2, exp3 nếu có
    order = ["exp1_model_compare", "exp2_lr", "exp3_transfer"]
    stage_labels = ["exp1\n(model)", "exp2\n(lr)", "exp3\n(transfer)"]
    f1_vals = []
    for k in order:
        rid = best_per_exp.get(k)
        if rid is None:
            f1_vals.append(np.nan)
        else:
            f1_vals.append(_metric(runs[rid], "f1_macro"))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(stage_labels))
    bars = ax.bar(x, f1_vals, color=[COLORS["baseline"], COLORS["efficientnet_b0"], COLORS["best"]], width=0.55)
    ax.plot(x, f1_vals, "o--", color=COLORS["accent"], linewidth=1.5, markersize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0.978, 0.992)
    ax.set_title("4.3.2 — Tiến trình cải thiện F1-macro qua các giai đoạn", fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, f1_vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0008, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    fig.savefig(output)
    plt.close(fig)


def fig_433_best_model_card(best: dict, output: Path) -> None:
    """4.3.3 — Thẻ tổng hợp mô hình CNN tốt nhất."""
    cnn = best["cnn"]
    hp = cnn["hyperparams"]
    m = cnn["metrics"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_title("4.3.3 — Mô hình CNN tốt nhất được lựa chọn", fontsize=14, fontweight="bold", pad=16)

    config_text = (
        f"{cnn['model']}  |  input_size={hp['input_size']}  |  lr={hp['lr']}  |  batch={hp['batch']}  |  strategy={hp['strategy']}\n"
        f"epochs={hp['epochs']}  |  patience={hp['patience']}  |  seed={hp['seed']}  |  weight_decay={hp['weight_decay']}"
    )
    ax.text(
        0.5,
        0.88,
        config_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4F8", edgecolor=COLORS["efficientnet_b0"]),
        transform=ax.transAxes,
    )

    cards = [
        ("Accuracy", f"{m['accuracy']:.3f}", COLORS["efficientnet_b0"]),
        ("Precision", f"{m['precision']:.3f}", COLORS["accent"]),
        ("Recall", f"{m['recall']:.3f}", COLORS["resnet50"]),
        ("F1-macro", f"{m['f1_macro']:.3f}", COLORS["best"]),
        ("Latency", f"{m['latency_ms_per_image']:.2f} ms/ảnh", COLORS["efficientnet_b0"]),
        ("Kích thước", f"{m['model_size_mb']:.2f} MB", COLORS["baseline"]),
    ]

    for idx, (label, value, color) in enumerate(cards):
        row, col = divmod(idx, 3)
        x = 0.06 + col * 0.31
        y = 0.58 - row * 0.32
        rect = mpatches.FancyBboxPatch(
            (x, y),
            0.27,
            0.22,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.135,
            y + 0.14,
            value,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=color,
            transform=ax.transAxes,
        )
        ax.text(x + 0.135, y + 0.04, label, ha="center", va="center", fontsize=9, transform=ax.transAxes)

    ax.text(
        0.5,
        0.08,
        "Tiêu chí: F1-macro → Accuracy → latency  |  Dùng làm bộ phân loại chính trong Pipeline 2",
        ha="center",
        fontsize=8,
        color="#6C757D",
        transform=ax.transAxes,
    )

    fig.savefig(output)
    plt.close(fig)


def generate_all(output_dir: Path, registry_path: Path, best_path: Path) -> list[Path]:
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    best = load_best(best_path)

    outputs = [
        output_dir / "fig_cnn_431_model_compare.png",
        output_dir / "fig_cnn_432_lr.png",
        output_dir / "fig_cnn_432_transfer.png",
        output_dir / "fig_cnn_432_progress.png",
        output_dir / "fig_cnn_433_best_model.png",
    ]

    fig_431_model_compare(registry, outputs[0])
    fig_432_lr(registry, outputs[1])
    fig_432_transfer(registry, outputs[2])
    fig_432_progress(registry, registry.get("best_overall", ""), outputs[3])
    fig_433_best_model_card(best, outputs[4])

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--best", type=Path, default=BEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_all(args.output, args.registry, args.best)
    print(f"Generated {len(outputs)} CNN figures in {args.output.resolve()}:")
    for path in outputs:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()

