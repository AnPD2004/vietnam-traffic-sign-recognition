"""Generate YOLO benchmark figures for report sections 4.2.1–4.2.3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = Path(__file__).resolve().parent.parent / "artifacts/pipeline1/report/summary.json"
BEST_PATH = Path(__file__).resolve().parent.parent / "artifacts/pipeline1/best/best.json"

COLORS = {
    "yolov5n": "#6C757D",
    "yolov8n": "#2E86AB",
    "yolov11n": "#A23B72",
    "mos0": "#95A5A6",
    "mos1": "#2E86AB",
    "accent": "#F18F01",
    "best": "#27AE60",
    "baseline": "#7F8C8D",
    "muted": "#6C757D",
    "bg": "#FAFAFA",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": COLORS["bg"],
            "axes.edgecolor": "#CCCCCC",
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


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(run: dict, key: str) -> float:
    return float(run["metrics"][key])


def _get_run(runs: dict, run_id: str) -> dict:
    return runs[run_id]


YOLO421_MODEL_LABELS = ["YOLOv5n", "YOLOv8n", "YOLOv11n"]
YOLO421_MOSAIC0_IDS = [
    "yolov5n_e100_sz640_lr0.005_b16_mos0",
    "yolov8n_e100_sz640_lr0.005_b16_mos0",
    "yolov11n_e100_sz640_lr0.005_b16_mos0",
]
YOLO421_MOSAIC1_IDS = [
    "yolov5n_e100_sz640_lr0.005_b16_mos1",
    "yolov8n_e100_sz640_lr0.005_b16_mos1",
    "yolov11n_e100_sz640_lr0.005_b16_mos1",
]
YOLO421_METRICS = [
    ("precision", "Precision", None, "{:.3f}"),
    ("recall", "Recall", None, "{:.3f}"),
    ("map50", "mAP@0.5", None, "{:.3f}"),
    ("map50_95", "mAP@0.5:0.95", None, "{:.3f}"),
    ("fps", "FPS", 180.0, "{:.1f}"),
    ("model_size_mb", "Kích thước (MB)", 10.0, "{:.2f}"),
]
YOLO421_METRIC_LABELS = [label for _, label, _, _ in YOLO421_METRICS]
YOLO421_MODEL_COLORS = [COLORS["yolov5n"], COLORS["yolov8n"], COLORS["yolov11n"]]
YOLO421_YLABEL = "Giá trị"
YOLO421_NOTE = (
    "Ghi chú: FPS (÷180) và kích thước MB (÷10) được scale để hiển thị cùng biểu đồ; "
    "nhãn trên cột là giá trị thực."
)


def _yolo421_display_value(raw: float, scale: float | None) -> float:
    return raw / scale if scale else raw


def _draw_421_model_compare_panel(
    runs: dict,
    ax,
    run_ids: list[str],
    *,
    subtitle: str,
    show_legend: bool,
) -> None:
    """Grouped bar chart: metrics on x-axis, one bar group per YOLO model."""
    x = np.arange(len(YOLO421_METRIC_LABELS))
    width = 0.25

    for i, (rid, label, color) in enumerate(zip(run_ids, YOLO421_MODEL_LABELS, YOLO421_MODEL_COLORS, strict=True)):
        run = _get_run(runs, rid)
        vals = [
            _yolo421_display_value(_metric(run, key), scale)
            for key, _, scale, _ in YOLO421_METRICS
        ]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, edgecolor="white")
        for bar, (key, _, scale, fmt) in zip(bars, YOLO421_METRICS, strict=True):
            raw = _metric(run, key)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                fmt.format(raw),
                ha="center",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(YOLO421_METRIC_LABELS)
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(YOLO421_YLABEL)
    ax.set_title(subtitle, fontweight="bold", pad=8)
    if show_legend:
        ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)


def fig_421_model_compare_combined(runs: dict, output: Path) -> None:
    """4.2.1 — So sánh 3 model: mosaic=0 và mosaic=1, cùng một template."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.subplots_adjust(hspace=0.38, top=0.90, bottom=0.10)

    _draw_421_model_compare_panel(
        runs, axes[0], YOLO421_MOSAIC0_IDS, subtitle="mosaic = 0", show_legend=True
    )
    _draw_421_model_compare_panel(
        runs, axes[1], YOLO421_MOSAIC1_IDS, subtitle="mosaic = 1", show_legend=False
    )

    fig.suptitle(
        "4.2.1 — So sánh các mô hình YOLO (mosaic = 0 và mosaic = 1)",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.03,
        YOLO421_NOTE,
        ha="center",
        fontsize=7,
        color=COLORS["yolov5n"],
    )
    fig.savefig(output)
    plt.close(fig)


def fig_421_model_compare_mosaic0(runs: dict, output: Path) -> None:
    """4.2.1 — So sánh 3 model khi mosaic=0."""
    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_421_model_compare_panel(
        runs, ax, YOLO421_MOSAIC0_IDS, subtitle="mosaic = 0", show_legend=True
    )
    ax.set_title("4.2.1 — So sánh các mô hình YOLO (mosaic = 0)", fontweight="bold", pad=12)
    fig.text(
        0.5,
        0.02,
        YOLO421_NOTE,
        ha="center",
        fontsize=7,
        color=COLORS["yolov5n"],
        transform=fig.transFigure,
    )
    fig.savefig(output)
    plt.close(fig)


def fig_421_model_compare_mosaic1(runs: dict, output: Path) -> None:
    """4.2.1 — So sánh 3 model khi mosaic=1."""
    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_421_model_compare_panel(
        runs, ax, YOLO421_MOSAIC1_IDS, subtitle="mosaic = 1", show_legend=True
    )
    ax.set_title("4.2.1 — So sánh các mô hình YOLO (mosaic = 1)", fontweight="bold", pad=12)
    fig.text(
        0.5,
        0.02,
        YOLO421_NOTE,
        ha="center",
        fontsize=7,
        color=COLORS["yolov5n"],
        transform=fig.transFigure,
    )
    fig.savefig(output)
    plt.close(fig)


def fig_421_yolov8n_mosaic_effect(runs: dict, output: Path) -> None:
    """4.2.1 — Ảnh hưởng Mosaic trên YOLOv8n."""
    mos0 = _get_run(runs, "yolov8n_e100_sz640_lr0.005_b16_mos0")
    mos1 = _get_run(runs, "yolov8n_e100_sz640_lr0.005_b16_mos1")
    keys = ["precision", "recall", "map50", "map50_95"]
    labels = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    v0 = [_metric(mos0, k) for k in keys]
    v1 = [_metric(mos1, k) for k in keys]
    axes[0].bar(x - w / 2, v0, w, label="Mosaic = 0", color=COLORS["mos0"])
    axes[0].bar(x + w / 2, v1, w, label="Mosaic = 1", color=COLORS["mos1"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.9, 1.0)
    axes[0].set_title("Chỉ số phát hiện", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    for i, (a, b) in enumerate(zip(v0, v1, strict=True)):
        axes[0].text(i - w / 2, a + 0.002, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, b + 0.002, f"{b:.3f}", ha="center", fontsize=8)

    fps0, fps1 = _metric(mos0, "fps"), _metric(mos1, "fps")
    ms0, ms1 = _metric(mos0, "inference_ms"), _metric(mos1, "inference_ms")
    axes[1].bar(["Mosaic = 0", "Mosaic = 1"], [fps0, fps1], color=[COLORS["mos0"], COLORS["mos1"]], width=0.5)
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Tốc độ suy luận", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    for i, (fps, ms) in enumerate([(fps0, ms0), (fps1, ms1)]):
        axes[1].text(i, fps + 2, f"{fps:.1f} FPS\n{ms:.2f} ms", ha="center", fontsize=9)

    fig.suptitle("4.2.1 — Ảnh hưởng Mosaic Augmentation (YOLOv8n)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_422_imgsz(runs: dict, output: Path) -> None:
    """4.2.2 — Ảnh hưởng kích thước ảnh."""
    configs = [
        ("416", "yolov8n_e100_sz416_lr0.005_b16_mos1"),
        ("640", "yolov8n_e100_sz640_lr0.005_b16_mos1"),
        ("800", "yolov8n_e100_sz800_lr0.005_b16_mos1"),
    ]
    labels = [c[0] for c in configs]
    map95 = [_metric(_get_run(runs, c[1]), "map50_95") for c in configs]
    fps = [_metric(_get_run(runs, c[1]), "fps") for c in configs]
    recall = [_metric(_get_run(runs, c[1]), "recall") for c in configs]
    precision = [_metric(_get_run(runs, c[1]), "precision") for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))
    axes[0].plot(x, map95, "o-", color=COLORS["yolov8n"], linewidth=2, markersize=8, label="mAP@0.5:0.95")
    axes[0].plot(x, recall, "s--", color=COLORS["accent"], linewidth=1.5, markersize=7, label="Recall")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}px" for s in labels])
    axes[0].set_xlabel("Kích thước ảnh đầu vào")
    axes[0].set_ylabel("Giá trị")
    axes[0].set_title("Độ chính xác theo imgsz", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    for i, v in enumerate(map95):
        axes[0].annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    bars = axes[1].bar(x, fps, color=[COLORS["accent"], COLORS["yolov8n"], COLORS["yolov11n"]], width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{s}px" for s in labels])
    axes[1].set_xlabel("Kích thước ảnh đầu vào")
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Tốc độ suy luận theo imgsz", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, fps, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("4.2.2 — Ảnh hưởng kích thước ảnh đầu vào (YOLOv8n)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_422_lr(runs: dict, output: Path) -> None:
    """4.2.2 — Ảnh hưởng learning rate."""
    configs = [
        ("0.001", "yolov8n_e100_sz800_lr0.001_b16_mos1"),
        ("0.005", "yolov8n_e100_sz800_lr0.005_b16_mos1"),
        ("0.01", "yolov8n_e100_sz800_lr0.01_b16_mos1"),
    ]
    labels = [c[0] for c in configs]
    metrics = ["precision", "recall", "map50", "map50_95"]
    metric_labels = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    fps = [_metric(_get_run(runs, c[1]), "fps") for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.18
    for i, (m, ml) in enumerate(zip(metrics, metric_labels, strict=True)):
        vals = [_metric(_get_run(runs, c[1]), m) for c in configs]
        axes[0].bar(x + (i - 1.5) * w, vals, w, label=ml)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylim(0.93, 1.0)
    axes[0].set_title("Chỉ số phát hiện theo learning rate", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    bars = axes[1].bar(labels, fps, color=COLORS["yolov8n"], width=0.5)
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Tốc độ suy luận theo learning rate", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, fps, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("4.2.2 — Ảnh hưởng Learning Rate (YOLOv8n, imgsz=800)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_422_batch(runs: dict, output: Path) -> None:
    """4.2.2 — Ảnh hưởng batch size."""
    configs = [
        ("8", "yolov8n_e100_sz800_lr0.001_b8_mos1"),
        ("16", "yolov8n_e100_sz800_lr0.001_b16_mos1"),
        ("32", "yolov8n_e100_sz800_lr0.001_b32_mos1"),
    ]
    labels = [c[0] for c in configs]
    map95 = [_metric(_get_run(runs, c[1]), "map50_95") for c in configs]
    recall = [_metric(_get_run(runs, c[1]), "recall") for c in configs]
    fps = [_metric(_get_run(runs, c[1]), "fps") for c in configs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, map95, w, label="mAP@0.5:0.95", color=COLORS["yolov8n"])
    axes[0].bar(x + w / 2, recall, w, label="Recall", color=COLORS["accent"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"batch={b}" for b in labels])
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_title("Độ chính xác theo batch size", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    for i, (a, b) in enumerate(zip(map95, recall, strict=True)):
        axes[0].text(i - w / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)

    bars = axes[1].bar(x, fps, color=[COLORS["accent"], COLORS["yolov8n"], COLORS["best"]], width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"batch={b}" for b in labels])
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Tốc độ suy luận theo batch size", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, fps, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("4.2.2 — Ảnh hưởng Batch Size (YOLOv8n, imgsz=800, lr=0.001)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_422_hyperparam_summary(summary: dict, output: Path) -> None:
    """4.2.2 — Tiến trình mAP@0.5:0.95 qua các giai đoạn."""
    best_per_exp = summary["best_per_experiment"]
    runs = summary["runs"]
    stages = ["exp1\n(model)", "exp2\n(imgsz)", "exp3\n(lr)", "exp4\n(batch)"]
    map95 = [_metric(runs[best_per_exp[k]], "map50_95") for k in best_per_exp]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(stages))
    bars = ax.bar(x, map95, color=[COLORS["baseline"], COLORS["yolov8n"], COLORS["yolov8n"], COLORS["best"]], width=0.55, edgecolor="white")
    ax.plot(x, map95, "o--", color=COLORS["accent"], linewidth=1.5, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_ylim(0.74, 0.79)
    ax.set_title("4.2.2 — Tiến trình cải thiện mAP@0.5:0.95 qua các giai đoạn", fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.25)
    for bar, v, rid in zip(bars, map95, best_per_exp.values(), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.008, rid.replace("yolov8n_e100_", ""), ha="center", fontsize=6, rotation=15)
    fig.savefig(output)
    plt.close(fig)


def fig_423_baseline_vs_best(runs: dict, best: dict, output: Path) -> None:
    """4.2.3 — So sánh baseline vs mô hình tốt nhất."""
    baseline_id = "yolov8n_e100_sz640_lr0.005_b16_mos1"
    baseline = _get_run(runs, baseline_id)
    best_run = best

    keys = ["precision", "recall", "map50", "map50_95"]
    labels = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    b_vals = [_metric(baseline, k) for k in keys]
    o_vals = [float(best_run["metrics"][k]) for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, b_vals, w, label="Baseline (sz640, lr0.005, b16)", color=COLORS["baseline"])
    axes[0].bar(x + w / 2, o_vals, w, label="Best (sz800, lr0.001, b32)", color=COLORS["best"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0.72, 1.03)
    axes[0].set_ylabel("Giá trị")
    axes[0].set_title("So sánh chỉ số phát hiện", fontweight="bold")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].axhline(0.9, color="#CCCCCC", linestyle=":", linewidth=0.8, alpha=0.7)
    for i, (a, b) in enumerate(zip(b_vals, o_vals, strict=True)):
        axes[0].text(i - w / 2, a + 0.008, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, b + 0.008, f"{b:.3f}", ha="center", fontsize=8)
        delta = b - a
        y_ann = max(a, b) + 0.025 if i < 3 else max(a, b) + 0.018
        axes[0].annotate(
            f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
            xy=(i, y_ann),
            ha="center",
            fontsize=7,
            color=COLORS["best"],
        )

    fps_b = _metric(baseline, "fps")
    fps_o = float(best_run["metrics"]["fps"])
    ms_b = _metric(baseline, "inference_ms")
    ms_o = float(best_run["metrics"]["inference_ms"])
    size_b = _metric(baseline, "model_size_mb")
    size_o = float(best_run["metrics"]["model_size_mb"])

    cat = ["FPS", "Inference\n(ms)", "Model\n(MB)"]
    b_perf = [fps_b, ms_b, size_b]
    o_perf = [fps_o, ms_o, size_o]
    x2 = np.arange(3)
    axes[1].bar(x2 - w / 2, b_perf, w, label="Baseline", color=COLORS["baseline"])
    axes[1].bar(x2 + w / 2, o_perf, w, label="Best", color=COLORS["best"])
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cat)
    axes[1].set_title("Hiệu năng & kích thước", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    for i, (a, b) in enumerate(zip(b_perf, o_perf, strict=True)):
        fmt = "{:.1f}" if i == 0 else "{:.2f}"
        axes[1].text(i - w / 2, a + 1, fmt.format(a), ha="center", fontsize=8)
        axes[1].text(i + w / 2, b + 1, fmt.format(b), ha="center", fontsize=8)

    fig.suptitle("4.2.3 — So sánh Baseline và mô hình YOLO tốt nhất", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_423_best_model_card(best: dict, output: Path) -> None:
    """4.2.3 — Thẻ tổng hợp mô hình tốt nhất."""
    m = best["metrics"]
    hp = best["hyperparams"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_title("4.2.3 — Mô hình YOLO tốt nhất được lựa chọn", fontsize=14, fontweight="bold", pad=16)

    config_text = (
        f"YOLOv8n  |  imgsz={hp['imgsz']}  |  lr={hp['lr']}  |  batch={hp['batch']}  |  mosaic={hp['mosaic']}\n"
        f"epochs={hp['epochs']}  |  patience={hp['patience']}  |  seed={hp['seed']}"
    )
    ax.text(0.5, 0.88, config_text, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4F8", edgecolor=COLORS["yolov8n"]), transform=ax.transAxes)

    cards = [
        ("Precision", f"{m['precision']:.3f}", COLORS["yolov8n"]),
        ("Recall", f"{m['recall']:.3f}", COLORS["accent"]),
        ("mAP@0.5", f"{m['map50']:.3f}", COLORS["yolov11n"]),
        ("mAP@0.5:0.95", f"{m['map50_95']:.3f}", COLORS["best"]),
        ("FPS", f"{m['fps']:.1f}", COLORS["yolov8n"]),
        ("Inference", f"{m['inference_ms']:.2f} ms", COLORS["accent"]),
        ("Kích thước", f"{m['model_size_mb']:.2f} MB", COLORS["yolov5n"]),
        ("Run ID", best["run_id"][:28] + "…", COLORS["muted"]),
    ]

    for idx, (label, value, color) in enumerate(cards[:6]):
        row, col = divmod(idx, 3)
        x = 0.06 + col * 0.31
        y = 0.58 - row * 0.32
        rect = mpatches.FancyBboxPatch(
            (x, y), 0.27, 0.22,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2, edgecolor=color, facecolor="white", transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + 0.135, y + 0.14, value, ha="center", va="center", fontsize=14, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x + 0.135, y + 0.04, label, ha="center", va="center", fontsize=9, transform=ax.transAxes)

    ax.text(0.5, 0.08,
            "Tiêu chí: mAP@0.5:0.95 → mAP@0.5 → FPS  |  "
            "Dùng cho Pipeline 1 và detector Pipeline 2",
            ha="center", fontsize=8, color="#6C757D", transform=ax.transAxes)

    fig.savefig(output)
    plt.close(fig)


def generate_all(output_dir: Path, summary_path: Path, best_path: Path) -> list[Path]:
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(summary_path)
    best = json.loads(best_path.read_text(encoding="utf-8"))
    runs = summary["runs"]

    outputs = [
        output_dir / "fig_yolo_421_mosaic_compare.png",
        output_dir / "fig_yolo_421_mosaic_effect.png",
        output_dir / "fig_yolo_422_imgsz.png",
        output_dir / "fig_yolo_422_lr.png",
        output_dir / "fig_yolo_422_batch.png",
        output_dir / "fig_yolo_422_progress.png",
        output_dir / "fig_yolo_423_baseline_vs_best.png",
        output_dir / "fig_yolo_423_best_model.png",
    ]

    fig_421_model_compare_combined(runs, outputs[0])
    fig_421_yolov8n_mosaic_effect(runs, outputs[1])
    fig_422_imgsz(runs, outputs[2])
    fig_422_lr(runs, outputs[3])
    fig_422_batch(runs, outputs[4])
    fig_422_hyperparam_summary(summary, outputs[5])
    fig_423_baseline_vs_best(runs, best, outputs[6])
    fig_423_best_model_card(best, outputs[7])

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--best", type=Path, default=BEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_all(args.output, args.summary, args.best)
    print(f"Generated {len(outputs)} YOLO figures in {args.output.resolve()}:")
    for path in outputs:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
