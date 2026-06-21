"""Generate figures for report section 4.4.2 — Kết quả đánh giá."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

P1_REGISTRY = ROOT / "artifacts/pipeline1/registry.json"
P1_BEST = ROOT / "artifacts/pipeline1/best/best.json"
P2_REGISTRY = ROOT / "artifacts/pipeline2/registry.json"
P2_BEST = ROOT / "artifacts/pipeline2/best/best.json"
BENCH_RESULTS = ROOT / "artifacts/benchmark/results.json"
BENCH_BEST = ROOT / "artifacts/benchmark/best_pipeline.json"

YOLO_E2E_MS = 7.9242
AVG_DETS_PER_IMAGE = 1645 / 639
CROP_OVERHEAD_MS = 0.25

COLORS = {
    "yolov5n": "#6C757D",
    "yolov8n": "#2E86AB",
    "yolov11n": "#A23B72",
    "resnet50": "#6C757D",
    "efficientnet_b0": "#2E86AB",
    "flow1": "#2E86AB",
    "flow2": "#F18F01",
    "accent": "#F18F01",
    "best": "#27AE60",
    "baseline": "#7F8C8D",
    "fps": "#1B9E77",
    "detection": "#5C4D7D",
    "classification": "#E07A5F",
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
            "axes.titlesize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _m(run: dict, key: str) -> float:
    return float(run["metrics"][key])


def estimate_flow2_e2e_ms(cnn_ms: float) -> float:
    return YOLO_E2E_MS + AVG_DETS_PER_IMAGE * (cnn_ms + CROP_OVERHEAD_MS)


def fig_442_stage1_yolo(p1_reg: dict, p1_best: dict, output: Path) -> None:
    """4.4.2.A — Kết quả Stage 1: so sánh YOLO & mô hình tốt nhất."""
    runs = p1_reg["runs"]
    exp1_ids = [
        "yolov5n_e100_sz640_lr0.005_b16_mos1",
        "yolov8n_e100_sz640_lr0.005_b16_mos1",
        "yolov11n_e100_sz640_lr0.005_b16_mos1",
    ]
    labels = ["YOLOv5n", "YOLOv8n", "YOLOv11n"]
    bar_colors = [COLORS["yolov5n"], COLORS["yolov8n"], COLORS["yolov11n"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: Exp1 model compare (mosaic=1)
    map_vals = [_m(runs[rid], "map50_95") for rid in exp1_ids]
    x = np.arange(len(labels))
    bars = axes[0].bar(x, map_vals, 0.55, color=bar_colors, edgecolor="white")
    best_idx = int(np.argmax(map_vals))
    bars[best_idx].set_edgecolor(COLORS["best"])
    bars[best_idx].set_linewidth(2.5)
    for bar, v in zip(bars, map_vals, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.72, 0.79)
    axes[0].set_ylabel("mAP@0.5:0.95")
    axes[0].set_title("Exp1 — So sánh kiến trúc YOLO (mosaic=1)", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)

    # Right: Best overall metrics
    bm = p1_best["metrics"]
    hp = p1_best["hyperparams"]
    cards = [
        ("mAP@0.5:0.95", f"{bm['map50_95']:.3f}", COLORS["best"]),
        ("mAP@0.5", f"{bm['map50']:.3f}", COLORS["yolov8n"]),
        ("Precision", f"{bm['precision']:.3f}", COLORS["detection"]),
        ("Recall", f"{bm['recall']:.3f}", COLORS["classification"]),
        ("FPS", f"{bm['fps']:.1f}", COLORS["fps"]),
        ("Latency", f"{bm['inference_ms']:.2f} ms", COLORS["accent"]),
    ]
    axes[1].axis("off")
    axes[1].set_title(
        f"Mô hình YOLO tốt nhất\n{hp['model']} · sz={hp['imgsz']} · lr={hp['lr']} · batch={hp['batch']}",
        fontweight="bold",
        pad=12,
    )
    for idx, (label, value, color) in enumerate(cards):
        row, col = divmod(idx, 3)
        cx = 0.08 + col * 0.31
        cy = 0.62 - row * 0.38
        rect = mpatches.FancyBboxPatch(
            (cx, cy),
            0.27,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
            transform=axes[1].transAxes,
        )
        axes[1].add_patch(rect)
        axes[1].text(cx + 0.135, cy + 0.17, value, ha="center", va="center", fontsize=12, fontweight="bold", color=color, transform=axes[1].transAxes)
        axes[1].text(cx + 0.135, cy + 0.05, label, ha="center", va="center", fontsize=8, transform=axes[1].transAxes)

    fig.suptitle("4.4.2.A — Kết quả đánh giá Stage 1 (YOLO End-to-End)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_442_stage2_cnn(p2_reg: dict, output: Path) -> None:
    """4.4.2.B — Kết quả Stage 2: benchmark CNN."""
    runs = p2_reg["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: Model compare
    ids = [
        "resnet50_e50_isz224_lr0.001_b32_ft",
        "efficientnet_b0_e50_isz224_lr0.001_b32_ft",
    ]
    labels = ["ResNet50", "EfficientNet-B0"]
    colors = [COLORS["resnet50"], COLORS["efficientnet_b0"]]
    metric_keys = ["accuracy", "precision", "recall", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-macro"]
    x = np.arange(len(metric_labels))
    w = 0.32
    for i, (rid, label, color) in enumerate(zip(ids, labels, colors, strict=True)):
        vals = [_m(runs[rid], k) for k in metric_keys]
        bars = axes[0].bar(x + (i - 0.5) * w, vals, w, label=label, color=color, edgecolor="white")
        for bar, v in zip(bars, vals, strict=True):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0.97, 1.001)
    axes[0].set_ylabel("Giá trị")
    axes[0].set_title("Exp1 — So sánh ResNet50 và EfficientNet-B0", fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    # Right: LR + transfer (F1-macro)
    lr_ids = [
        ("0.0001", "efficientnet_b0_e50_isz224_lr0.0001_b32_ft"),
        ("0.001", "efficientnet_b0_e50_isz224_lr0.001_b32_ft"),
        ("0.01", "efficientnet_b0_e50_isz224_lr0.01_b32_ft"),
    ]
    tr_ids = [
        ("Frozen", "efficientnet_b0_e50_isz224_lr0.001_b32_frz"),
        ("Fine-tune", "efficientnet_b0_e50_isz224_lr0.001_b32_ft"),
    ]
    lr_labels = [c[0] for c in lr_ids]
    lr_f1 = [_m(runs[rid], "f1_macro") for _, rid in lr_ids]
    tr_labels = [c[0] for c in tr_ids]
    tr_f1 = [_m(runs[rid], "f1_macro") for _, rid in tr_ids]

    ax_lr = axes[1]
    ax_tr = ax_lr.twinx()
    x_lr = np.arange(len(lr_labels))
    x_tr = np.arange(len(tr_labels)) + len(lr_labels) + 0.6
    bars_lr = ax_lr.bar(x_lr, lr_f1, 0.5, color=COLORS["efficientnet_b0"], label="LR sweep", alpha=0.9)
    bars_tr = ax_tr.bar(x_tr, tr_f1, 0.45, color=[COLORS["baseline"], COLORS["best"]], label="Transfer", alpha=0.9)
    ax_lr.set_xticks(list(x_lr) + list(x_tr))
    ax_lr.set_xticklabels(lr_labels + tr_labels, rotation=12, ha="right")
    ax_lr.set_ylim(0.88, 1.0)
    ax_tr.set_ylim(0.88, 1.0)
    ax_lr.set_ylabel("F1-macro (LR)")
    ax_tr.set_ylabel("F1-macro (Transfer)", color=COLORS["best"])
    ax_lr.set_title("Exp2–Exp3 — Learning rate & Transfer learning", fontweight="bold")
    ax_lr.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars_lr, lr_f1, strict=True):
        ax_lr.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    for bar, v in zip(bars_tr, tr_f1, strict=True):
        ax_tr.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7, color=COLORS["best"])

    fig.suptitle("4.4.2.B — Kết quả đánh giá Stage 2 (YOLO + CNN — phần CNN)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


def build_stage3_metrics_from_best(p1_best: dict, p2_best: dict) -> tuple[dict, dict]:
    """Stage 3 metrics sourced from pipeline best artifacts (not benchmark e2e re-eval)."""
    p1m = p1_best["metrics"]
    c2m = p2_best["cnn"]["metrics"]
    p1_f1 = f1_from_precision_recall(float(p1m["precision"]), float(p1m["recall"]))

    flow1 = {
        "map50_95": p1m["map50_95"],
        "map50": p1m["map50"],
        "f1_macro": p1_f1,
        "precision": p1m["precision"],
        "recall": p1m["recall"],
        "fps": p1m["fps"],
        "inference_ms": p1m["inference_ms"],
        "total_model_size_mb": p1m["model_size_mb"],
    }
    cnn_ms = float(c2m["latency_ms_per_image"])
    flow2_e2e_ms = estimate_flow2_e2e_ms(cnn_ms)
    flow2 = {
        "map50_95": p1m["map50_95"],
        "map50": p1m["map50"],
        "f1_macro": c2m["f1_macro"],
        "precision": c2m["precision"],
        "recall": c2m["recall"],
        "fps": 1000.0 / flow2_e2e_ms,
        "inference_ms": flow2_e2e_ms,
        "total_model_size_mb": round(p1m["model_size_mb"] + c2m["model_size_mb"], 4),
    }
    return flow1, flow2


def fig_442_stage3_pipeline(p1_best: dict, p2_best: dict, best_pipeline: dict, output: Path) -> None:
    """4.4.2.C — Stage 3: so sánh hai pipeline end-to-end từ best.json."""
    flow1, flow2 = build_stage3_metrics_from_best(p1_best, p2_best)
    flows = [flow1, flow2]
    labels = ["Flow 1\n(YOLO E2E)", "Flow 2\n(YOLO+CNN)"]
    colors = [COLORS["flow1"], COLORS["flow2"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    x = np.arange(2)

    metric_keys = ["map50_95", "map50", "f1_macro", "precision", "recall"]
    metric_labels = ["mAP@0.5:0.95", "mAP@0.5", "F1-macro", "Precision", "Recall"]
    xe = np.arange(len(metric_labels))
    bar_w = 0.32
    for i, flow, color in zip(range(2), flows, colors, strict=True):
        for j, key in enumerate(metric_keys):
            val = flow[key]
            bar = axes[0].bar(
                xe[j] + (i - 0.5) * bar_w,
                val,
                bar_w,
                label=labels[i].replace("\n", " ") if j == 0 else None,
                color=color,
                edgecolor="white",
            )
            axes[0].text(
                bar[0].get_x() + bar[0].get_width() / 2,
                val + 0.006,
                f"{val:.3f}",
                ha="center",
                fontsize=7,
            )
    axes[0].set_xticks(xe)
    axes[0].set_xticklabels(metric_labels, rotation=18, ha="right")
    axes[0].set_ylim(0.72, 1.04)
    axes[0].set_ylabel("Giá trị")
    axes[0].set_title("Chỉ số đánh giá end-to-end", fontweight="bold", y=1.08)
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].axvline(1.5, color="#CCCCCC", linestyle="--", linewidth=1, alpha=0.8)
    axes[0].text(0.75, 1.01, "Phát hiện (YOLO)", ha="center", transform=axes[0].get_xaxis_transform(), fontsize=8, color="#555555")
    axes[0].text(3.0, 1.01, "Phân loại", ha="center", transform=axes[0].get_xaxis_transform(), fontsize=8, color="#555555")
    axes[0].text(
        0.5,
        -0.28,
        "Nguồn: pipeline1/best/best.json · pipeline2/best/best.json\n"
        f"Flow 1 F1 = 2×P×R/(P+R) = {flow1['f1_macro']:.3f} · Flow 2 F1 từ CNN val",
        ha="center",
        transform=axes[0].transAxes,
        fontsize=7,
        color="#555555",
    )

    # Panel 2: runtime & size từ best.json
    lat = [flow1["inference_ms"], flow2["inference_ms"]]
    fps = [flow1["fps"], flow2["fps"]]
    size = [flow1["total_model_size_mb"], flow2["total_model_size_mb"]]

    ax_lat = axes[1]
    ax_fps = ax_lat.twinx()
    ax_mb = ax_lat.twinx()
    ax_mb.spines["right"].set_position(("axes", 1.14))
    ax_mb.spines["right"].set_visible(True)

    bw = 0.22
    bars_lat = ax_lat.bar(x - bw, lat, bw, color=COLORS["accent"], label="Latency e2e (ms)")
    bars_fps = ax_fps.bar(x, fps, bw, color=COLORS["fps"], label="FPS e2e")
    bars_mb = ax_mb.bar(x + bw, size, bw, color=COLORS["baseline"], label="Model size (MB)")

    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(labels)
    ax_lat.set_ylabel("Latency (ms/ảnh)")
    ax_fps.set_ylabel("FPS", color=COLORS["fps"])
    ax_fps.tick_params(axis="y", labelcolor=COLORS["fps"])
    ax_mb.set_ylabel("Size (MB)", color=COLORS["baseline"])
    ax_mb.tick_params(axis="y", labelcolor=COLORS["baseline"])
    ax_lat.set_ylim(0, max(lat) * 1.3)
    ax_fps.set_ylim(0, max(fps) * 1.25)
    ax_mb.set_ylim(0, max(size) * 1.3)
    ax_lat.set_title("Tốc độ & kích thước triển khai", fontweight="bold", y=1.08)
    ax_lat.grid(axis="y", alpha=0.25)

    for bar, v in zip(bars_lat, lat, strict=True):
        ax_lat.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.2f}", ha="center", fontsize=8)
    for bar, v in zip(bars_fps, fps, strict=True):
        ax_fps.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=8, color=COLORS["fps"])
    for bar, v in zip(bars_mb, size, strict=True):
        ax_mb.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)

    handles = [bars_lat, bars_fps, bars_mb]
    ax_lat.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8)

    winner_id = best_pipeline.get("winner", "")
    winner = "Flow 1 (YOLO E2E)" if winner_id == "flow1_yolo_yolo" else "Flow 2 (YOLO+CNN)"
    fig.suptitle(
        f"4.4.2.C — So sánh end-to-end hai pipeline  |  Winner: {winner}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(output)
    plt.close(fig)


def fig_442_summary_overview(p1_best: dict, p2_best: dict, output: Path) -> None:
    """4.4.2 — Tổng hợp ba giai đoạn (overview)."""
    c2m = p2_best["cnn"]["metrics"]
    flow1, flow2 = build_stage3_metrics_from_best(p1_best, p2_best)

    stages = ["Stage 1\n(YOLO)", "Stage 2\n(CNN crop)", "Stage 3\n(Flow 1)", "Stage 3\n(Flow 2)"]
    primary_metric = [
        p1_best["metrics"]["map50_95"],
        c2m["f1_macro"],
        p1_best["metrics"]["map50_95"],
        flow2["f1_macro"],
    ]
    fps_vals = [
        p1_best["metrics"]["fps"],
        1000.0 / float(c2m["latency_ms_per_image"]),
        flow1["fps"],
        flow2["fps"],
    ]
    metric_names = ["mAP@0.5:0.95", "F1-macro", "mAP@0.5:0.95", "F1-macro"]

    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(stages))
    bars = ax1.bar(x, primary_metric, 0.5, color=[COLORS["yolov8n"], COLORS["efficientnet_b0"], COLORS["flow1"], COLORS["flow2"]], edgecolor="white")
    for bar, v, name in zip(bars, primary_metric, metric_names, strict=True):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}\n({name})", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylim(0.72, 1.05)
    ax1.set_ylabel("Metric chính")
    ax1.set_title("Metric chính theo giai đoạn đánh giá", fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, fps_vals, "o--", color=COLORS["fps"], linewidth=2, markersize=8, label="FPS")
    for xi, fp in zip(x, fps_vals, strict=True):
        ax2.text(xi, fp + 4, f"{fp:.1f} FPS", ha="center", fontsize=8, color=COLORS["fps"])
    ax2.set_ylabel("FPS", color=COLORS["fps"])
    ax2.tick_params(axis="y", labelcolor=COLORS["fps"])
    ax2.set_ylim(0, max(fps_vals) * 1.2)
    ax2.legend(loc="upper right")

    fig.suptitle("4.4.2 — Tổng quan kết quả đánh giá (Stage 1 → 2 → 3)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def generate_all(output_dir: Path) -> list[Path]:
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    p1_reg = _load(P1_REGISTRY)
    p1_best = _load(P1_BEST)
    p2_reg = _load(P2_REGISTRY)
    p2_best = _load(P2_BEST)
    bench_best = _load(BENCH_BEST)

    outputs = [
        output_dir / "fig_442_stage1_yolo.png",
        output_dir / "fig_442_stage2_cnn.png",
        output_dir / "fig_442_stage3_pipeline.png",
        output_dir / "fig_442_summary_overview.png",
    ]
    fig_442_stage1_yolo(p1_reg, p1_best, outputs[0])
    fig_442_stage2_cnn(p2_reg, outputs[1])
    fig_442_stage3_pipeline(p1_best, p2_best, bench_best, outputs[2])
    fig_442_summary_overview(p1_best, p2_best, outputs[3])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_all(args.output)
    print(f"Generated {len(outputs)} figures in {args.output.resolve()}:")
    for path in outputs:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
