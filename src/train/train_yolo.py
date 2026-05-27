from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO for Flow 1 (YOLO detect + YOLO predict)."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to YOLO data.yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model checkpoint (default: yolov8n.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        default=None,
        help='Device, e.g. "0", "0,1", "cpu" (default: auto).',
    )
    parser.add_argument(
        "--project",
        default="artifacts/yolo_detect_cls",
        help="Output project directory.",
    )
    parser.add_argument(
        "--name",
        default="flow1_train",
        help="Run name under project directory.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Run validation after training.",
    )
    return parser.parse_args()


def _resolve_split_list(
    split_file: Path,
    dataset_root: Path,
    workspace_root: Path,
    temp_dir: Path,
) -> Path:
    lines = split_file.read_text(encoding="utf-8").splitlines()
    normalized: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            if "/" in line or "\\" in line:
                data_root = split_file.parent.parent.resolve()
                path_candidates = [
                    (workspace_root / candidate).resolve(),
                    candidate.resolve(),
                    (data_root / candidate).resolve(),
                    (dataset_root / candidate).resolve(),
                    (split_file.parent / candidate).resolve(),
                ]
                existing = [c for c in path_candidates if c.exists()]
                if existing:
                    candidate = existing[0]
                else:
                    candidate = (dataset_root / candidate).resolve()
            else:
                candidate = (dataset_root / "images" / candidate).resolve()
        normalized.append(candidate.as_posix())

    resolved_file = temp_dir / f"{split_file.stem}.resolved.txt"
    resolved_file.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    return resolved_file


def _prepare_data_yaml(data_path: Path) -> Path:
    import yaml

    data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(data_cfg, dict):
        return data_path

    dataset_root = Path(data_cfg.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()
    workspace_root = data_path.parent.parent.resolve()

    split_keys = ("train", "val", "test")
    needs_resolve = False
    for key in split_keys:
        value = data_cfg.get(key)
        if isinstance(value, str) and value.endswith(".txt"):
            needs_resolve = True
            break

    if not needs_resolve:
        return data_path

    temp_dir = Path(tempfile.mkdtemp(prefix="flow1_data_"))
    patched_cfg = dict(data_cfg)
    patched_cfg["path"] = str(dataset_root)

    for key in split_keys:
        value = data_cfg.get(key)
        if not isinstance(value, str) or not value.endswith(".txt"):
            continue

        split_file = Path(value)
        if not split_file.is_absolute():
            candidates = [
                (data_path.parent / split_file).resolve(),
                (dataset_root / split_file).resolve(),
            ]
            split_file = next((c for c in candidates if c.exists()), candidates[0])
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found for '{key}': {split_file}")

        resolved_file = _resolve_split_list(
            split_file=split_file,
            dataset_root=dataset_root,
            workspace_root=workspace_root,
            temp_dir=temp_dir,
        )
        patched_cfg[key] = str(resolved_file)

    resolved_yaml = temp_dir / "data.resolved.yaml"
    resolved_yaml.write_text(
        yaml.safe_dump(patched_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return resolved_yaml


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    prepared_data_path = _prepare_data_yaml(data_path.resolve())

    model = YOLO(args.model)
    model.train(
        data=str(prepared_data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
    )

    if args.val:
        model.val(data=str(prepared_data_path), imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()

