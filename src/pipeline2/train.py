from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import yaml

from src.common.cnn_model import build_classifier, save_checkpoint
from src.common.image_ops import build_cnn_transform
from src.common.labels import class_dir_name, get_class_names
from src.common.memory import release_runtime_memory


def apply_transfer_strategy(model: nn.Module, strategy: str) -> None:
    if strategy == "frozen":
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
    elif strategy == "finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"unknown transfer strategy: {strategy}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")
    return torch.device(device)


def build_dataloaders(
    crops_root: Path,
    input_size: int,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = crops_root / "train"
    val_dir = crops_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"expected {train_dir} and {val_dir}; run crop preparation first"
        )

    train_ds = ImageFolder(train_dir, transform=build_cnn_transform(input_size, train=True))
    val_ds = ImageFolder(val_dir, transform=build_cnn_transform(input_size, train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds.classes


def validate_folder_class_order(folder_class_names: list[str], expected_class_names: list[str]) -> None:
    expected_folders = sorted(class_dir_name(class_id) for class_id in range(len(expected_class_names)))
    actual_folders = sorted(folder_class_names)
    if actual_folders != expected_folders:
        missing = set(expected_folders) - set(actual_folders)
        extra = set(actual_folders) - set(expected_folders)
        raise ValueError(
            "crop folders do not match classes.txt mapping. "
            f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
        )


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)

            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            for target, pred in zip(targets.view(-1), preds.view(-1), strict=True):
                confusion[int(target), int(pred)] += 1

    elapsed = time.perf_counter() - start
    accuracy = correct / max(total_samples, 1)

    precisions: list[float] = []
    recalls: list[float] = []
    f1_per_class: list[float] = []
    for class_idx in range(num_classes):
        tp = confusion[class_idx, class_idx].item()
        fp = confusion[:, class_idx].sum().item() - tp
        fn = confusion[class_idx, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_per_class.append(f1)

    f1_macro = sum(f1_per_class) / max(num_classes, 1)
    precision_macro = sum(precisions) / max(num_classes, 1)
    recall_macro = sum(recalls) / max(num_classes, 1)
    ms_per_image = (elapsed / max(total_samples, 1)) * 1000.0

    return {
        "loss": round(total_loss / max(total_samples, 1), 6),
        "train_loss": None,
        "val_loss": round(total_loss / max(total_samples, 1), 6),
        "accuracy": round(accuracy, 6),
        "precision": round(precision_macro, 6),
        "recall": round(recall_macro, 6),
        "f1_macro": round(f1_macro, 6),
        "f1_per_class": [round(v, 6) for v in f1_per_class],
        "confusion_matrix": confusion.tolist(),
        "latency_ms_per_image": round(ms_per_image, 4),
        "num_samples": total_samples,
    }


def save_confusion_matrix_png(matrix: list[list[int]], class_names: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.array(matrix)
    fig_w = max(8, len(class_names) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    tick_positions = list(range(len(class_names)))
    ax.set(
        xticks=tick_positions,
        yticks=tick_positions,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)

    return total_loss / max(total_samples, 1)


def train_cnn_run(
    *,
    params: dict[str, Any],
    run_dir: Path,
    weights_path: Path,
    crops_root: Path,
    classes_path: Path,
    device: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    set_seed(int(params["seed"]))

    expected_class_names = get_class_names(classes_path)
    train_loader, val_loader, folder_class_names = build_dataloaders(
        crops_root=crops_root,
        input_size=int(params["input_size"]),
        batch_size=int(params["batch"]),
        workers=int(params["workers"]),
    )
    validate_folder_class_order(folder_class_names, expected_class_names)
    validate_folder_class_order(val_loader.dataset.classes, expected_class_names)
    num_classes = len(folder_class_names)

    model = build_classifier(params["model"], num_classes=num_classes, pretrained=True)
    apply_transfer_strategy(model, str(params["strategy"]))
    model = model.to(resolved_device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    epochs_without_improve = 0
    best_state: dict[str, Any] | None = None

    try:
        for epoch in range(1, int(params["epochs"]) + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, resolved_device)
            val_metrics = compute_metrics(model, val_loader, resolved_device, num_classes)
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = round(train_loss, 6)
            history.append(val_metrics)

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_metrics = dict(val_metrics)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= int(params["patience"]):
                break

        if best_state is None:
            raise RuntimeError("training finished without a valid checkpoint")

        model.load_state_dict(best_state)
        save_checkpoint(
            weights_path,
            model=model,
            class_names=expected_class_names,
            model_name=params["model"],
            input_size=int(params["input_size"]),
            metrics=best_metrics,
        )

        save_confusion_matrix_png(
            best_metrics["confusion_matrix"],
            expected_class_names,
            run_dir / "confusion_matrix.png",
        )
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = dict(best_metrics)
        summary["model_size_mb"] = round(weights_path.stat().st_size / (1024 * 1024), 4)
        return summary, history
    finally:
        release_runtime_memory(model)


def evaluate_cnn_run(
    *,
    weights_path: Path,
    crops_root: Path,
    classes_path: Path,
    params: dict[str, Any],
    device: str | None,
) -> dict[str, Any]:
    from src.common.cnn_model import load_checkpoint

    resolved_device = resolve_device(device)
    expected_class_names = get_class_names(classes_path)
    _, val_loader, folder_class_names = build_dataloaders(
        crops_root=crops_root,
        input_size=int(params["input_size"]),
        batch_size=int(params["batch"]),
        workers=int(params["workers"]),
    )
    validate_folder_class_order(folder_class_names, expected_class_names)

    model = None
    try:
        model, _ = load_checkpoint(weights_path, device=resolved_device)
        metrics = compute_metrics(model, val_loader, resolved_device, len(folder_class_names))
        metrics["model_size_mb"] = round(weights_path.stat().st_size / (1024 * 1024), 4)
        return metrics
    finally:
        release_runtime_memory(model)


def write_run_config(
    run_dir: Path,
    *,
    run_id: str,
    experiment: str,
    params: dict[str, Any],
    device: str | None,
    status: str,
    mode: str,
    started_at: str,
    finished_at: str | None = None,
    reused: bool = False,
) -> None:
    payload = {
        "run_id": run_id,
        "experiment": experiment,
        "pipeline": "pipeline2",
        "mode": mode,
        "reused": reused,
        "hyperparams": params,
        "device": device,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
    }
    path = run_dir / "run_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_metrics(
    run_dir: Path,
    *,
    run_id: str,
    experiment: str,
    weights_file: str,
    metrics: dict[str, Any],
    reused: bool = False,
) -> Path:
    payload = {
        "run_id": run_id,
        "experiment": experiment,
        "weights_file": weights_file,
        "reused": reused,
        "metrics": metrics,
    }
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
