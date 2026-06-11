from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from src.common.image_ops import crop_bbox, yolo_norm_to_xyxy
from src.common.labels import class_dir_name, get_class_names
from src.common.splits import load_splits_from_data_yaml, resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cropped classification dataset for Flow 2 CNN training."
    )
    parser.add_argument(
        "--data-yaml",
        default="data/vn-traffic-signs/data.yaml",
        help="YOLO data.yaml used to resolve dataset splits.",
    )
    parser.add_argument(
        "--output",
        default="data/vn-traffic-signs/crops",
        help="Output root for crops/{split}/{class_name}/*.jpg",
    )
    parser.add_argument(
        "--classes",
        default="data/vn-traffic-signs/classes.txt",
        help="Shared class map file.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Relative bbox padding when cropping.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directory before building.",
    )
    return parser.parse_args()


def label_path_for_image(image_path: Path, dataset_root: Path) -> Path:
    try:
        relative = image_path.resolve().relative_to((dataset_root / "images").resolve())
    except ValueError:
        relative = Path(image_path.name)
    return (dataset_root / "labels" / relative).with_suffix(".txt")


def parse_yolo_labels(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_file.exists():
        return []

    objects: list[tuple[int, float, float, float, float]] = []
    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        cx, cy, width, height = (float(v) for v in parts[1:])
        objects.append((class_id, cx, cy, width, height))
    return objects


def build_split_crops(
    split_name: str,
    image_paths: list[Path],
    dataset_root: Path,
    output_root: Path,
    num_classes: int,
    padding: float,
) -> tuple[int, Counter[int]]:
    split_dir = output_root / split_name
    saved = 0
    class_counts: Counter[int] = Counter()

    for image_path in image_paths:
        if not image_path.exists():
            continue

        label_file = label_path_for_image(image_path, dataset_root)
        objects = parse_yolo_labels(label_file)
        if not objects:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for obj_idx, (class_id, cx, cy, width, height) in enumerate(objects):
                if class_id < 0 or class_id >= num_classes:
                    continue

                x1, y1, x2, y2 = yolo_norm_to_xyxy(
                    cx, cy, width, height, image.width, image.height
                )
                crop = crop_bbox(image, (x1, y1, x2, y2), padding=padding)

                class_folder = split_dir / class_dir_name(class_id)
                class_folder.mkdir(parents=True, exist_ok=True)
                out_name = f"{image_path.stem}_{obj_idx}.jpg"
                crop.save(class_folder / out_name, quality=95)
                saved += 1
                class_counts[class_id] += 1

    return saved, class_counts


def build_crops(
    data_yaml: Path,
    output_root: Path,
    classes_path: Path,
    padding: float = 0.05,
    overwrite: bool = False,
) -> dict[str, object]:
    if overwrite and output_root.exists():
        import shutil

        shutil.rmtree(output_root)

    class_names = get_class_names(classes_path)
    dataset_root = resolve_dataset_root(data_yaml)
    splits = load_splits_from_data_yaml(data_yaml)
    if not splits:
        raise ValueError(f"no splits found in {data_yaml}")

    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "data_yaml": str(data_yaml.resolve()),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root.resolve()),
        "num_classes": len(class_names),
        "padding": padding,
        "splits": {},
    }

    total_saved = 0
    for split_name, image_paths in splits.items():
        saved, class_counts = build_split_crops(
            split_name=split_name,
            image_paths=image_paths,
            dataset_root=dataset_root,
            output_root=output_root,
            num_classes=len(class_names),
            padding=padding,
        )
        total_saved += saved
        summary["splits"][split_name] = {
            "images": len(image_paths),
            "crops_saved": saved,
            "class_counts": {
                get_class_names(classes_path)[class_id]: count
                for class_id, count in sorted(class_counts.items())
            },
        }

    summary["total_crops_saved"] = total_saved
    stats_path = output_root / "dataset_stats.json"
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_crops(
        data_yaml=Path(args.data_yaml),
        output_root=Path(args.output),
        classes_path=Path(args.classes),
        padding=args.padding,
        overwrite=args.overwrite,
    )
    print(f"Saved {summary['total_crops_saved']} crops to {Path(args.output).resolve()}")
    print(f"Stats: {(Path(args.output) / 'dataset_stats.json').resolve()}")


if __name__ == "__main__":
    main()
