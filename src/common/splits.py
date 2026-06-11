from __future__ import annotations

from pathlib import Path


def resolve_dataset_root(data_yaml_path: Path) -> Path:
    import yaml

    data_cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data_cfg, dict):
        raise ValueError(f"invalid data.yaml: {data_yaml_path}")

    dataset_root = Path(data_cfg.get("path", data_yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()
    return dataset_root


def resolve_split_file(data_yaml_path: Path, split_key: str) -> Path | None:
    import yaml

    data_cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data_cfg, dict):
        raise ValueError(f"invalid data.yaml: {data_yaml_path}")

    value = data_cfg.get(split_key)
    if not isinstance(value, str):
        return None

    dataset_root = resolve_dataset_root(data_yaml_path)
    if value.endswith(".txt"):
        split_file = Path(value)
        if not split_file.is_absolute():
            candidates = [
                (data_yaml_path.parent / split_file).resolve(),
                (dataset_root / split_file).resolve(),
            ]
            split_file = next((c for c in candidates if c.exists()), candidates[0])
        return split_file

    if Path(value).is_absolute():
        return Path(value)
    return (dataset_root / value).resolve()


def resolve_image_path(line: str, dataset_root: Path, split_file: Path) -> Path:
    candidate = Path(line.strip())
    if not line.strip():
        raise ValueError("empty split line")

    if candidate.is_absolute():
        return candidate.resolve()

    if "/" in line or "\\" in line:
        workspace_root = dataset_root.parent.parent.resolve()
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
            return existing[0]
        return (dataset_root / candidate).resolve()

    return (dataset_root / "images" / candidate).resolve()


def load_split_image_paths(split_file: Path, dataset_root: Path) -> list[Path]:
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")

    paths: list[Path] = []
    for raw_line in split_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        paths.append(resolve_image_path(line, dataset_root, split_file))
    return paths


def load_splits_from_data_yaml(data_yaml_path: Path) -> dict[str, list[Path]]:
    dataset_root = resolve_dataset_root(data_yaml_path)
    splits: dict[str, list[Path]] = {}

    for key in ("train", "val", "test"):
        split_file = resolve_split_file(data_yaml_path, key)
        if split_file is None or not split_file.exists():
            continue
        if split_file.suffix == ".txt":
            splits[key] = load_split_image_paths(split_file, dataset_root)
        else:
            splits[key] = sorted(split_file.glob("*.*"))

    return splits
