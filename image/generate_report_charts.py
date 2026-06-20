"""Generate dataset statistics figures for the research report."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_crops.build_crops_for_cnn import label_path_for_image, parse_yolo_labels
from src.common.labels import get_class_names
from src.common.splits import load_splits_from_data_yaml, resolve_dataset_root

DEFAULT_DATA_YAML = ROOT / "data/vn-traffic-signs/data.yaml"
DEFAULT_CLASSES = ROOT / "data/vn-traffic-signs/classes.txt"
DEFAULT_CLASSES_VIE = ROOT / "data/vn-traffic-signs/classes_vie.txt"
OUTPUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "train": "#2E86AB",
    "val": "#A23B72",
    "accent": "#F18F01",
    "muted": "#6C757D",
    "bg": "#FAFAFA",
}


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
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def collect_stats(data_yaml: Path, classes_path: Path, classes_vie_path: Path) -> dict:
    dataset_root = resolve_dataset_root(data_yaml)
    splits = load_splits_from_data_yaml(data_yaml)
    class_codes = get_class_names(classes_path)
    class_labels_vie = get_class_names(classes_vie_path)
    if len(class_codes) != len(class_labels_vie):
        raise ValueError(
            f"class count mismatch: {classes_path}={len(class_codes)}, "
            f"{classes_vie_path}={len(class_labels_vie)}"
        )

    all_images_dir = dataset_root / "images"
    all_images = sorted(all_images_dir.glob("*.jpg"))
    split_set = {p.resolve() for paths in splits.values() for p in paths}
    unused = [p for p in all_images if p.resolve() not in split_set]

    class_counts: Counter[int] = Counter()
    boxes_per_image: list[int] = []
    split_stats: dict[str, dict] = {}

    for split_name, paths in splits.items():
        split_boxes = 0
        split_boxes_per_image: list[int] = []
        for image_path in paths:
            objects = parse_yolo_labels(label_path_for_image(image_path, dataset_root))
            if not objects:
                continue
            split_boxes_per_image.append(len(objects))
            boxes_per_image.extend([len(objects)] * 1)
            for class_id, *_ in objects:
                class_counts[class_id] += 1
                split_boxes += 1

        split_stats[split_name] = {
            "images": len(paths),
            "boxes": split_boxes,
            "boxes_per_image": split_boxes_per_image,
        }

    return {
        "total_images_disk": len(all_images),
        "used_images": sum(s["images"] for s in split_stats.values()),
        "unused_images": len(unused),
        "total_boxes": sum(class_counts.values()),
        "num_classes": len(class_codes),
        "class_codes": class_codes,
        "class_labels_vie": class_labels_vie,
        "class_counts": class_counts,
        "boxes_per_image": boxes_per_image,
        "splits": split_stats,
    }


def fig_dataset_overview(stats: dict, output: Path) -> None:
    used = stats["used_images"]
    unused = stats["unused_images"]
    total = stats["total_images_disk"]
    total_boxes = stats["total_boxes"]
    avg_boxes = total_boxes / used if used else 0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    title = "Tổng quan bộ dữ liệu nhận dạng biển báo giao thông Việt Nam"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)

    cards = [
        ("Ảnh gốc", f"{total:,}".replace(",", "."), COLORS["muted"]),
        ("Ảnh sử dụng", f"{used:,}".replace(",", "."), COLORS["train"]),
        ("Ảnh loại (nhãn rỗng)", f"{unused:,}".replace(",", "."), COLORS["accent"]),
        ("Số lớp", str(stats["num_classes"]), COLORS["val"]),
        ("Tổng bounding box", f"{total_boxes:,}".replace(",", "."), COLORS["train"]),
        ("TB bbox / ảnh", f"{avg_boxes:.2f}", COLORS["val"]),
    ]

    for idx, (label, value, color) in enumerate(cards):
        row, col = divmod(idx, 3)
        x = 0.06 + col * 0.31
        y = 0.72 - row * 0.42
        rect = mpatches.FancyBboxPatch(
            (x, y),
            0.27,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + 0.135, y + 0.19, value, ha="center", va="center", fontsize=20, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x + 0.135, y + 0.07, label, ha="center", va="center", fontsize=10, transform=ax.transAxes)

    ax.text(
        0.5,
        0.06,
        "Nguồn: thống kê từ mã nguồn dự án (data.yaml, split files, parse_yolo_labels)",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    fig.savefig(output)
    plt.close(fig)


def fig_train_val_split(stats: dict, output: Path) -> None:
    splits = stats["splits"]
    labels = {"train": "Huấn luyện", "val": "Xác thực"}
    names = [labels.get(k, k) for k in splits]
    images = [splits[k]["images"] for k in splits]
    boxes = [splits[k]["boxes"] for k in splits]
    colors = [COLORS["train"], COLORS["val"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, values, title, unit in zip(
        axes,
        [images, boxes],
        ["Phân chia số lượng ảnh", "Phân chia số lượng bounding box"],
        ["ảnh", "bbox"],
        strict=True,
    ):
        bars = ax.bar(names, values, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
        total = sum(values)
        for bar, value in zip(bars, values, strict=True):
            pct = value / total * 100 if total else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{value:,}".replace(",", ".") + f"\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_ylabel(f"Số {unit}")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Phân chia tập huấn luyện và tập xác thực (~80/20)", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(output)
    plt.close(fig)


def fig_bbox_per_image_hist(stats: dict, output: Path) -> None:
    data = stats["boxes_per_image"]
    fig, ax = plt.subplots(figsize=(9, 5))
    max_boxes = max(data)
    bins = np.arange(0.5, max_boxes + 1.5, 1)

    ax.hist(data, bins=bins, color=COLORS["train"], edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.axvline(np.mean(data), color=COLORS["accent"], linestyle="--", linewidth=2, label=f"Trung bình = {np.mean(data):.2f}")
    ax.set_xlabel("Số biển báo trong một ảnh")
    ax.set_ylabel("Số lượng ảnh")
    ax.set_title("Phân bố số lượng biển báo trên mỗi ảnh", fontweight="bold")
    ax.set_xticks(range(1, max_boxes + 1))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.savefig(output)
    plt.close(fig)


def _class_chart_label(stats: dict, class_id: int) -> str:
    code = stats["class_codes"][class_id]
    name = stats["class_labels_vie"][class_id]
    return f"{code} — {name}"


def _plot_class_bars(
    class_labels: list[str],
    counts: list[int],
    title: str,
    output: Path,
    color: str,
) -> None:
    max_label_len = max(len(label) for label in class_labels)
    fig_w = 12 if max_label_len > 30 else 10.5
    fig_h = max(5, 0.32 * len(class_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    y_pos = np.arange(len(class_labels))

    bars = ax.barh(y_pos, counts, color=color, edgecolor="white", height=0.72)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_labels, fontsize=8 if len(class_labels) > 20 else 9)
    ax.invert_yaxis()
    ax.set_xlabel("Số bounding box")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    for bar, count in zip(bars, counts, strict=True):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=8)

    fig.savefig(output)
    plt.close(fig)


def fig_class_distribution(stats: dict, output_dir: Path) -> None:
    counts = stats["class_counts"]
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    top = ranked[:15]
    bottom = sorted(counts.items(), key=lambda item: item[1])[:15]

    _plot_class_bars(
        [_class_chart_label(stats, cid) for cid, _ in top],
        [cnt for _, cnt in top],
        "15 lớp biển báo có số lượng mẫu nhiều nhất",
        output_dir / "fig04_class_top15.png",
        COLORS["train"],
    )
    _plot_class_bars(
        [_class_chart_label(stats, cid) for cid, _ in bottom],
        [cnt for _, cnt in bottom],
        "15 lớp biển báo có số lượng mẫu ít nhất",
        output_dir / "fig05_class_bottom15.png",
        COLORS["val"],
    )

    all_ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    _plot_class_bars(
        [_class_chart_label(stats, cid) for cid, _ in all_ranked],
        [cnt for _, cnt in all_ranked],
        "Phân bố số lượng mẫu theo 52 lớp biển báo",
        output_dir / "fig06_class_all52.png",
        COLORS["accent"],
    )


def fig_data_pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    ax.set_title("Quy trình sử dụng dữ liệu trong hệ thống", fontweight="bold", fontsize=14, pad=12)

    steps = [
        ("3.216 ảnh gốc\n(JPG)", COLORS["muted"]),
        ("Danh sách split\n(train / val)", COLORS["train"]),
        ("3.191 ảnh\n có nhãn", COLORS["train"]),
        ("8.334 bbox\nYOLO", COLORS["accent"]),
        ("Crop theo bbox\n(Pipeline 2)", COLORS["val"]),
        ("Huấn luyện\nYOLO / CNN", COLORS["train"]),
    ]

    n = len(steps)
    for i, (text, color) in enumerate(steps):
        x = 0.04 + i * 0.19
        rect = mpatches.FancyBboxPatch(
            (x, 0.35),
            0.15,
            0.38,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=1.5,
            edgecolor=color,
            facecolor="white",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + 0.075, 0.54, text, ha="center", va="center", fontsize=9, transform=ax.transAxes)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + 0.165, 0.54),
                xytext=(x + 0.175, 0.54),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
            )

    ax.text(0.22, 0.18, "25 ảnh loại\n(nhãn rỗng)", ha="center", fontsize=8, color=COLORS["accent"], transform=ax.transAxes)
    ax.annotate(
        "",
        xy=(0.14, 0.35),
        xytext=(0.22, 0.26),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.2, linestyle="dashed"),
    )

    fig.savefig(output)
    plt.close(fig)


def generate_all(
    output_dir: Path,
    data_yaml: Path,
    classes_path: Path,
    classes_vie_path: Path,
) -> list[Path]:
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = collect_stats(data_yaml, classes_path, classes_vie_path)

    outputs = [
        output_dir / "fig01_dataset_overview.png",
        output_dir / "fig02_train_val_split.png",
        output_dir / "fig03_bbox_per_image_hist.png",
        output_dir / "fig04_class_top15.png",
        output_dir / "fig05_class_bottom15.png",
        output_dir / "fig06_class_all52.png",
        output_dir / "fig07_data_pipeline.png",
    ]

    fig_dataset_overview(stats, outputs[0])
    fig_train_val_split(stats, outputs[1])
    fig_bbox_per_image_hist(stats, outputs[2])
    fig_class_distribution(stats, output_dir)
    fig_data_pipeline(outputs[6])

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--classes-vie", type=Path, default=DEFAULT_CLASSES_VIE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_all(args.output, args.data_yaml, args.classes, args.classes_vie)
    print(f"Generated {len(outputs)} figures in {args.output.resolve()}:")
    for path in outputs:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
