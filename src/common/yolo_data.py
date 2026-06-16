from __future__ import annotations

import tempfile
from pathlib import Path


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


def prepare_data_yaml(data_path: Path) -> Path:
    """Resolve split .txt paths in a YOLO data.yaml for Ultralytics training/val."""
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
