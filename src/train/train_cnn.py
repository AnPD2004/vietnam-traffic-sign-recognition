from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.common.cnn_model import build_classifier, save_checkpoint, save_class_index
from src.common.image_ops import build_cnn_transform
from src.common.labels import get_class_names


def parse_args(defaults: dict[str, Any] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN classifier for Flow 2.")
    if defaults:
        parser.set_defaults(**defaults)
    parser.add_argument(
        "--data-dir",
        default="data/vn-traffic-signs/crops",
        help="Root with train/ and val/ ImageFolder structure.",
    )
    parser.add_argument(
        "--classes",
        default="data/vn-traffic-signs/classes.txt",
        help="Shared class names file.",
    )
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help='Device, e.g. "cuda", "cpu" (default: auto).',
    )
    parser.add_argument(
        "--output",
        default="artifacts/cnn_classifier",
        help="Directory for best.pth, metrics, and class index.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config (CLI args override config values).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {path}")
    return cfg


def parser_defaults_from_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    cfg = load_config(config_path)
    mapping = {
        "model": "model",
        "input_size": "input_size",
        "batch_size": "batch",
        "epochs": "epochs",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "patience": "patience",
        "num_workers": "workers",
        "seed": "seed",
    }
    defaults: dict[str, Any] = {}
    for cfg_key, arg_attr in mapping.items():
        if cfg_key in cfg:
            defaults[arg_attr] = cfg[cfg_key]
    return defaults


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    data_dir: Path,
    input_size: int,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"expected {train_dir} and {val_dir}; run build_crops_for_cnn first"
        )

    train_ds = ImageFolder(train_dir, transform=build_cnn_transform(input_size, train=True))
    val_ds = ImageFolder(val_dir, transform=build_cnn_transform(input_size, train=False))
    class_names = train_ds.classes

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
    return train_loader, val_loader, class_names


def validate_folder_class_order(folder_class_names: list[str], expected_class_names: list[str]) -> None:
    from src.common.labels import class_dir_name

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
            for t, p in zip(targets.view(-1), preds.view(-1), strict=True):
                confusion[int(t), int(p)] += 1

    elapsed = time.perf_counter() - start
    accuracy = correct / max(total_samples, 1)
    f1_per_class: list[float] = []
    for class_idx in range(num_classes):
        tp = confusion[class_idx, class_idx].item()
        fp = confusion[:, class_idx].sum().item() - tp
        fn = confusion[class_idx, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)

    f1_macro = sum(f1_per_class) / max(num_classes, 1)
    ms_per_image = (elapsed / max(total_samples, 1)) * 1000.0

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "confusion_matrix": confusion.tolist(),
        "latency_ms_per_image": ms_per_image,
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


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = parser_defaults_from_config(
        Path(pre_args.config) if pre_args.config else None
    )
    args = parse_args(config_defaults)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_class_names = get_class_names(args.classes)
    train_loader, val_loader, folder_class_names = build_dataloaders(
        data_dir=data_dir,
        input_size=args.input_size,
        batch_size=args.batch,
        workers=args.workers,
    )

    val_folder_class_names = val_loader.dataset.classes
    validate_folder_class_order(folder_class_names, expected_class_names)
    validate_folder_class_order(val_folder_class_names, expected_class_names)
    num_classes = len(folder_class_names)

    model = build_classifier(args.model, num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = compute_metrics(model, val_loader, device, num_classes)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = train_loss
        history.append(val_metrics)

        improved = val_metrics["f1_macro"] > best_f1
        if improved:
            best_f1 = val_metrics["f1_macro"]
            best_metrics = val_metrics
            epochs_without_improve = 0
            save_checkpoint(
                output_dir / "best.pth",
                model=model,
                class_names=expected_class_names,
                model_name=args.model,
                input_size=args.input_size,
                metrics=val_metrics,
            )
        else:
            epochs_without_improve += 1

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if epochs_without_improve >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    save_checkpoint(
        output_dir / "last.pth",
        model=model,
        class_names=expected_class_names,
        model_name=args.model,
        input_size=args.input_size,
        metrics=history[-1] if history else {},
    )
    save_class_index(output_dir / "class_to_idx.json", expected_class_names)

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "best": best_metrics,
        "history": history,
        "model": args.model,
        "input_size": args.input_size,
        "class_names": expected_class_names,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if best_metrics.get("confusion_matrix"):
        save_confusion_matrix_png(
            best_metrics["confusion_matrix"],
            expected_class_names,
            output_dir / "confusion_matrix.png",
        )


if __name__ == "__main__":
    main()
