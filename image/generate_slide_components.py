"""Generate slide figures: model components & evaluation metrics for Pipeline 1 & 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

P1_SUMMARY = ROOT / "artifacts/pipeline1/report/summary.json"
P2_REGISTRY = ROOT / "artifacts/pipeline2/registry.json"

YOLO_E2E_MS = 7.9242
AVG_DETS_PER_IMAGE = 1645 / 639
CROP_OVERHEAD_MS = 0.25

# Standard 16:9 slide canvas — avoids stretch when inserted into PowerPoint
FIG_W, FIG_H = 12.8, 7.2

THEME = {
    "header_bg": "#1B3A5C",
    "header_text": "#FFFFFF",
    "card_bg": "#FFFFFF",
    "card_border": "#DDE3EA",
    "best": "#27AE60",
    "best_bg": "#E8F8EF",
    "yolov5n": "#6C757D",
    "yolov8n": "#2E86AB",
    "yolov11n": "#A23B72",
    "resnet50": "#6C757D",
    "efficientnet_b0": "#2E86AB",
    "metric_label": "#4A5568",
    "metric_value": "#1A202C",
    "footer": "#8896A5",
    "badge_p1": "#2E86AB",
    "badge_p2": "#F18F01",
    "row_alt": "#F4F7FA",
    "row_alt_best": "#D8F0E3",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "savefig.dpi": 150,
            "savefig.facecolor": "white",
        }
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _m(run: dict, key: str) -> float:
    return float(run["metrics"][key])


def _estimate_flow2_e2e_ms(cnn_ms: float) -> float:
    return YOLO_E2E_MS + AVG_DETS_PER_IMAGE * (cnn_ms + CROP_OVERHEAD_MS)


def _draw_metric_table(ax, metrics: list[tuple[str, str]], *, is_best: bool) -> None:
    ax.axis("off")
    rows = [[label, value] for label, value in metrics]
    table = ax.table(
        cellText=rows,
        colLabels=None,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E2E8F0")
        cell.set_linewidth(0.5)
        if col == 0:
            cell.set_text_props(ha="left", color=THEME["metric_label"], fontsize=8.5)
            cell.set_width(0.58)
        else:
            cell.set_text_props(ha="right", fontweight="bold", color=THEME["metric_value"], fontsize=9)
            cell.set_width(0.42)
        if row % 2 == 1:
            cell.set_facecolor(THEME["row_alt_best"] if is_best else THEME["row_alt"])
        else:
            cell.set_facecolor(THEME["best_bg"] if is_best else "white")


def _draw_model_card(
    fig,
    gs_slot,
    *,
    model_name: str,
    color: str,
    metrics: list[tuple[str, str]],
    is_best: bool,
) -> None:
    border_color = THEME["best"] if is_best else THEME["card_border"]
    face = THEME["best_bg"] if is_best else THEME["card_bg"]
    lw = 2.0 if is_best else 1.0

    outer = fig.add_subplot(gs_slot)
    outer.set_xlim(0, 1)
    outer.set_ylim(0, 1)
    outer.axis("off")

    outer.add_patch(
        mpatches.FancyBboxPatch(
            (0.04, 0.04),
            0.92,
            0.92,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=face,
            edgecolor=border_color,
            linewidth=lw,
            transform=outer.transAxes,
        )
    )

    outer.text(0.5, 0.90, model_name, ha="center", va="center", fontsize=11, fontweight="bold", color=color, transform=outer.transAxes)
    if is_best:
        outer.add_patch(
            mpatches.FancyBboxPatch(
                (0.32, 0.82),
                0.36,
                0.06,
                boxstyle="round,pad=0.005,rounding_size=0.02",
                facecolor=THEME["best"],
                edgecolor="none",
                transform=outer.transAxes,
            )
        )
        outer.text(0.5, 0.85, "TỐT NHẤT", ha="center", va="center", fontsize=7, fontweight="bold", color="white", transform=outer.transAxes)
        table_top, table_h = 0.08, 0.72
    else:
        table_top, table_h = 0.10, 0.76

    table_ax = outer.inset_axes([0.06, table_top, 0.88, table_h])
    _draw_metric_table(table_ax, metrics, is_best=is_best)


def _build_figure(title: str, subtitle: str, badge: str, badge_color: str, footer: str) -> tuple[plt.Figure, GridSpec]:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08)

    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.11, 0.89], hspace=0.06)

    header_ax = fig.add_subplot(gs[0])
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    header_ax.axis("off")
    header_ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="square,pad=0", facecolor=THEME["header_bg"], edgecolor="none"
        )
    )
    header_ax.text(0.02, 0.62, title, fontsize=13, fontweight="bold", color=THEME["header_text"], va="center")
    header_ax.text(0.02, 0.22, subtitle, fontsize=8, color="#B8C9D9", va="center")
    header_ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.86, 0.22), 0.12, 0.56, boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=badge_color, edgecolor="none",
        )
    )
    header_ax.text(0.92, 0.5, badge, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    fig.text(0.5, 0.02, footer, ha="center", fontsize=7.5, color=THEME["footer"])
    return fig, gs


def fig_slide_yolo_components(p1_summary: dict, output: Path) -> None:
    runs = p1_summary["runs"]
    configs = [
        ("YOLOv5n", THEME["yolov5n"], "yolov5n_e100_sz640_lr0.005_b16_mos1"),
        ("YOLOv8n", THEME["yolov8n"], "yolov8n_e100_sz640_lr0.005_b16_mos1"),
        ("YOLOv11n", THEME["yolov11n"], "yolov11n_e100_sz640_lr0.005_b16_mos1"),
    ]
    best_idx = 1

    fig, gs = _build_figure(
        "Thành phần mô hình — Pipeline 1 (YOLO End-to-End)",
        "3 kiến trúc YOLO nano · epochs=100, imgsz=640, lr=0.005, batch=16, mosaic=1",
        "PIPELINE 1",
        THEME["badge_p1"],
        "Tiêu chí: mAP@0.5:0.95 → mAP@0.5 → FPS  |  YOLOv8n → Pipeline 1 & detector Pipeline 2",
    )

    cards_gs = gs[1].subgridspec(1, 3, wspace=0.12)
    for i, (name, color, run_id) in enumerate(configs):
        run = runs[run_id]
        metrics = [
            ("Precision", f"{_m(run, 'precision'):.3f}"),
            ("Recall", f"{_m(run, 'recall'):.3f}"),
            ("mAP@0.5", f"{_m(run, 'map50'):.3f}"),
            ("mAP@0.5:0.95", f"{_m(run, 'map50_95'):.3f}"),
            ("FPS", f"{_m(run, 'fps'):.1f}"),
            ("Inference", f"{_m(run, 'inference_ms'):.2f} ms"),
            ("Kích thước", f"{_m(run, 'model_size_mb'):.2f} MB"),
        ]
        _draw_model_card(fig, cards_gs[0, i], model_name=name, color=color, metrics=metrics, is_best=(i == best_idx))

    fig.savefig(output, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def fig_slide_cnn_components(p2_registry: dict, output: Path) -> None:
    runs = p2_registry["runs"]
    configs = [
        ("ResNet50", THEME["resnet50"], "resnet50_e50_isz224_lr0.001_b32_ft"),
        ("EfficientNet-B0", THEME["efficientnet_b0"], "efficientnet_b0_e50_isz224_lr0.001_b32_ft"),
    ]
    best_idx = 1

    fig, gs = _build_figure(
        "Thành phần mô hình — Pipeline 2 (YOLO + CNN Hybrid)",
        "2 kiến trúc CNN · epochs=50, input=224, lr=0.001, batch=32, fine-tuning",
        "PIPELINE 2",
        THEME["badge_p2"],
        "Tiêu chí: F1-macro → Accuracy → Latency  |  EfficientNet-B0 + YOLO detector",
    )

    cards_gs = gs[1].subgridspec(1, 2, wspace=0.14, width_ratios=[1, 1])
    for i, (name, color, run_id) in enumerate(configs):
        run = runs[run_id]
        cnn_ms = _m(run, "latency_ms_per_image")
        e2e_ms = _estimate_flow2_e2e_ms(cnn_ms)
        metrics = [
            ("Accuracy", f"{_m(run, 'accuracy'):.3f}"),
            ("Precision", f"{_m(run, 'precision'):.3f}"),
            ("Recall", f"{_m(run, 'recall'):.3f}"),
            ("F1-macro", f"{_m(run, 'f1_macro'):.3f}"),
            ("E2E Latency", f"{e2e_ms:.2f} ms"),
            ("E2E FPS", f"{1000.0 / e2e_ms:.1f}"),
            ("Kích thước", f"{_m(run, 'model_size_mb'):.2f} MB"),
        ]
        _draw_model_card(fig, cards_gs[0, i], model_name=name, color=color, metrics=metrics, is_best=(i == best_idx))

    fig.savefig(output, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def generate_all(output_dir: Path) -> list[Path]:
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    p1 = _load(P1_SUMMARY)
    p2 = _load(P2_REGISTRY)

    outputs = [
        output_dir / "fig_slide_pipeline1_yolo_models.png",
        output_dir / "fig_slide_pipeline2_cnn_models.png",
    ]
    fig_slide_yolo_components(p1, outputs[0])
    fig_slide_cnn_components(p2, outputs[1])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_all(args.output)
    print(f"Generated {len(outputs)} slide figures ({FIG_W}x{FIG_H} in, 16:9):")
    for path in outputs:
        print(f"  - {path.resolve()}")


if __name__ == "__main__":
    main()
