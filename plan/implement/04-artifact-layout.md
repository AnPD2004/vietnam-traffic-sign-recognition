# Artifact Layout — Cấu trúc folder & đặt tên model theo tham số

> Mỗi lượt run export **một model file có tên = toàn bộ hyperparameter** của run đó.
> Pipeline 1: **15 runs** · Pipeline 2: **7 runs** → cần cấu trúc dễ duyệt, tra cứu, và không trùng tên.

---

## Nguyên tắc thiết kế

| # | Nguyên tắc | Lý do |
|---|-----------|-------|
| 1 | **Tên model = run_id** | Nhìn tên file biết ngay config, không cần mở metadata |
| 2 | **run_id deterministic** | Cùng params → cùng tên → skip/re-run idempotent |
| 3 | **Nhóm theo experiment** | `runs/{experiment}/{run_id}/` — báo cáo theo stage, không lẫn run |
| 4 | **Registry tập trung** | `registry.json` index toàn bộ run, không phải scan FS |
| 5 | **Chỉ giữ weights cần thiết** | Mỗi run 1 file `.pt`/`.pth` (best); bỏ checkpoint Ultralytics thừa |
| 6 | **best/ là alias** | Copy hoặc symlink model winner, không duplicate logic path |

---

## Hàm encode tham số (dùng chung P1 & P2)

```python
# src/common/run_naming.py

PARAM_ORDER_P1 = ["model", "epochs", "imgsz", "lr", "batch", "mosaic"]
PARAM_ORDER_P2 = ["model", "epochs", "input_size", "lr", "batch", "strategy"]

def fmt_lr(v: float) -> str:
    """0.005 → lr0.005 | 0.0001 → lr0.0001"""
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return f"lr{s}"

def build_run_id(params: dict, pipeline: str) -> str:
    parts = []
    if pipeline == "pipeline1":
        parts = [
            params["model"],                    # yolov8n
            f"e{params['epochs']}",             # e100
            f"sz{params['imgsz']}",             # sz640
            fmt_lr(params["lr"]),               # lr0.005
            f"b{params['batch']}",              # b16
            f"mos{params['mosaic']}",           # mos1
        ]
    elif pipeline == "pipeline2":
        strategy = params["strategy"]           # frozen | finetune
        strat_tag = "frz" if strategy == "frozen" else "ft"
        parts = [
            params["model"],                    # resnet50
            f"e{params['epochs']}",             # e50
            f"isz{params['input_size']}",       # isz224
            fmt_lr(params["lr"]),
            f"b{params['batch']}",
            strat_tag,                          # frz | ft
        ]
    return "_".join(parts)
```

### Ví dụ run_id

| Pipeline | run_id |
|----------|--------|
| P1 Exp1 | `yolov5n_e100_sz640_lr0.005_b16_mos0` |
| P1 Exp2 | `yolov8n_e100_sz416_lr0.005_b16_mos1` |
| P1 Exp4 | `yolov8n_e100_sz640_lr0.005_b8_mos1` |
| P2 Exp1 | `resnet50_e50_isz224_lr0.001_b32_ft` |
| P2 Exp3 | `efficientnet_b0_e50_isz224_lr0.001_b32_frz` |

### Tên file model

```
{run_id}.pt    # Pipeline 1 — YOLO
{run_id}.pth   # Pipeline 2 — CNN
```

Model file **nằm trong thư mục run**, tên file **trùng** run_id (không dùng `best.pt` generic).

---

## Cấu trúc folder đầy đủ

### Pipeline 1

```
artifacts/pipeline1/
│
├── registry.json                 # master index — mọi run, mọi experiment
├── registry.jsonl                # append-only log (optional, dễ audit)
│
├── runs/
│   ├── exp1_model_compare/
│   │   ├── yolov5n_e100_sz640_lr0.005_b16_mos0/
│   │   │   ├── yolov5n_e100_sz640_lr0.005_b16_mos0.pt   ← MODEL
│   │   │   ├── metrics.json
│   │   │   ├── run_config.yaml          # hyperparams đầy đủ + seed, device
│   │   │   └── train_log.csv            # copy từ Ultralytics results.csv
│   │   ├── yolov5n_e100_sz640_lr0.005_b16_mos1/
│   │   ├── yolov8n_e100_sz640_lr0.005_b16_mos0/
│   │   ├── yolov8n_e100_sz640_lr0.005_b16_mos1/
│   │   ├── yolov11n_e100_sz640_lr0.005_b16_mos0/
│   │   ├── yolov11n_e100_sz640_lr0.005_b16_mos1/
│   │   └── _best.json                   # winner của experiment này
│   │
│   ├── exp2_imgsz/
│   │   ├── yolov8n_e100_sz416_lr0.005_b16_mos1/
│   │   ├── yolov8n_e100_sz640_lr0.005_b16_mos1/
│   │   ├── yolov8n_e100_sz800_lr0.005_b16_mos1/
│   │   └── _best.json
│   │
│   ├── exp3_lr/
│   │   ├── yolov8n_e100_sz640_lr0.001_b16_mos1/
│   │   ├── yolov8n_e100_sz640_lr0.005_b16_mos1/
│   │   ├── yolov8n_e100_sz640_lr0.01_b16_mos1/
│   │   └── _best.json
│   │
│   └── exp4_batch/
│       ├── yolov8n_e100_sz640_lr0.005_b8_mos1/
│       ├── yolov8n_e100_sz640_lr0.005_b16_mos1/
│       ├── yolov8n_e100_sz640_lr0.005_b32_mos1/
│       └── _best.json
│
├── best/
│   ├── yolov8n_e100_sz640_lr0.005_b16_mos1.pt   # copy từ winner Exp4
│   └── best.json                                 # metadata + pointer run_id
│
└── report/
    ├── summary.md                                # bảng so sánh từng experiment
    ├── summary.json
    └── by_experiment/
        ├── exp1_model_compare.md
        ├── exp2_imgsz.md
        ├── exp3_lr.md
        └── exp4_batch.md
```

### Pipeline 2

```
artifacts/pipeline2/
│
├── registry.json
│
├── runs/
│   ├── exp1_model_compare/
│   │   ├── resnet50_e50_isz224_lr0.001_b32_ft/
│   │   │   ├── resnet50_e50_isz224_lr0.001_b32_ft.pth   ← MODEL
│   │   │   ├── metrics.json
│   │   │   ├── run_config.yaml
│   │   │   ├── history.json               # loss per epoch
│   │   │   └── confusion_matrix.png
│   │   ├── efficientnet_b0_e50_isz224_lr0.001_b32_ft/
│   │   └── _best.json
│   │
│   ├── exp2_lr/
│   │   ├── efficientnet_b0_e50_isz224_lr0.0001_b32_ft/
│   │   ├── efficientnet_b0_e50_isz224_lr0.001_b32_ft/
│   │   ├── efficientnet_b0_e50_isz224_lr0.01_b32_ft/
│   │   └── _best.json
│   │
│   └── exp3_transfer/
│       ├── efficientnet_b0_e50_isz224_lr0.001_b32_frz/
│       ├── efficientnet_b0_e50_isz224_lr0.001_b32_ft/
│       └── _best.json
│
├── best/
│   ├── efficientnet_b0_e50_isz224_lr0.001_b32_ft.pth
│   ├── yolo_ref.json                    # pointer → pipeline1/best/best.json
│   └── best.json
│
└── report/
    ├── summary.md
    └── by_experiment/
        ├── exp1_model_compare.md
        ├── exp2_lr.md
        └── exp3_transfer.md
```

> **Crop dataset** vẫn ở `data/vn-traffic-signs/crops/` (data, không phải artifact).
> Pipeline 2 không copy crops vào `artifacts/`.

---

## Nội dung từng file trong run folder

### `run_config.yaml` (bắt buộc)

```yaml
run_id: yolov8n_e100_sz640_lr0.005_b16_mos1
experiment: exp1_model_compare
pipeline: pipeline1
hyperparams:
  model: yolov8n
  epochs: 100
  imgsz: 640
  lr: 0.005
  batch: 16
  mosaic: 1
  patience: 30
  seed: 42
device: "0"
started_at: "2026-06-15T10:30:00"
finished_at: "2026-06-15T14:22:00"
status: completed   # completed | failed | skipped
```

### `metrics.json` (bắt buộc — dùng để skip run)

```json
{
  "run_id": "yolov8n_e100_sz640_lr0.005_b16_mos1",
  "experiment": "exp1_model_compare",
  "weights_file": "yolov8n_e100_sz640_lr0.005_b16_mos1.pt",
  "metrics": {
    "precision": 0.95,
    "recall": 0.93,
    "map50": 0.98,
    "map50_95": 0.72,
    "fps": 45.2,
    "inference_ms": 22.1,
    "model_size_mb": 6.2
  }
}
```

### `_best.json` (per experiment)

```json
{
  "experiment": "exp1_model_compare",
  "run_id": "yolov8n_e100_sz640_lr0.005_b16_mos1",
  "weights": "runs/exp1_model_compare/yolov8n_e100_sz640_lr0.005_b16_mos1/yolov8n_e100_sz640_lr0.005_b16_mos1.pt",
  "metric_used": "map50_95",
  "metric_value": 0.72
}
```

### `registry.json` (master index)

```json
{
  "pipeline": "pipeline1",
  "updated_at": "2026-06-15T14:22:00",
  "total_runs": 15,
  "runs": {
    "yolov8n_e100_sz640_lr0.005_b16_mos1": {
      "experiment": "exp1_model_compare",
      "dir": "runs/exp1_model_compare/yolov8n_e100_sz640_lr0.005_b16_mos1",
      "weights": "runs/exp1_model_compare/yolov8n_e100_sz640_lr0.005_b16_mos1/yolov8n_e100_sz640_lr0.005_b16_mos1.pt",
      "status": "completed",
      "metrics": { "map50_95": 0.72 }
    }
  },
  "best_per_experiment": {
    "exp1_model_compare": "yolov8n_e100_sz640_lr0.005_b16_mos1"
  },
  "best_overall": "yolov8n_e100_sz640_lr0.005_b16_mos1"
}
```

---

## Luồng export sau mỗi run

```
train (Ultralytics / PyTorch)
  ↓
ultralytics tạm: {run_dir}/_ultra/weights/best.pt
  ↓
rename + move → {run_dir}/{run_id}.pt
  ↓
xóa {run_dir}/_ultra/          # chỉ giữ 1 file model
  ↓
ghi metrics.json + run_config.yaml
  ↓
append/update registry.json
```

### Tiết kiệm disk

| Hành động | Cấu hình |
|-----------|----------|
| Không lưu `last.pt` | Ultralytics: `save_period=-1`, chỉ copy best |
| Không lưu epoch checkpoints | `save=False` sau khi đã copy best |
| Xóa thư mục `_ultra/` | Sau khi rename model |
| Confusion matrix | Chỉ PNG, không lưu raw tensor |

Ước tính: 15 × ~6 MB (YOLOn) ≈ **90 MB** + 7 × ~25 MB (CNN) ≈ **175 MB** — chấp nhận được.

---

## Tra cứu & CLI tiện ích

```bash
# Liệt kê mọi run đã hoàn thành
python -m src.pipeline1 --list-runs

# Liệt kê run trong 1 experiment
python -m src.pipeline1 --list-runs --experiment exp1_model_compare

# Dry-run: in ra run_id + path sẽ tạo
python -m src.pipeline1 --dry-run

# Test: full flow, 1 epoch mỗi run (smoke test trước chạy chính)
python -m src.pipeline1 --test

# Chạy chính
python -m src.pipeline1
```

---

## Quy tắc tránh trùng / conflict

1. **run_id unique toàn pipeline** — không chỉ trong experiment.
   Nếu Exp2 và Exp3 tình cờ cùng params (hiếm), vẫn khác experiment folder → OK.
2. **Không train 2 run cùng run_id** — orchestrator check `registry.json` trước.
3. **`--force`** — ghi đè model + metrics, cập nhật `status: completed` + timestamp mới.
4. **Failed run** — ghi `status: failed` + `error.txt`, **không** tạo file `.pt`/`.pth`.

---

## Inference load path

```python
# Đọc best overall
best = json.load(open("artifacts/pipeline1/best/best.json"))
model_path = best["weights"]   # artifacts/pipeline1/best/yolov8n_e100_sz640_lr0.005_b16_mos1.pt

# Hoặc load run bất kỳ bằng run_id
entry = registry["runs"]["yolov5n_e100_sz640_lr0.005_b16_mos0"]
model_path = entry["weights"]
```

`best.json` schema cập nhật:

```json
{
  "pipeline": "pipeline1",
  "run_id": "yolov8n_e100_sz640_lr0.005_b16_mos1",
  "weights": "artifacts/pipeline1/best/yolov8n_e100_sz640_lr0.005_b16_mos1.pt",
  "source_run_dir": "artifacts/pipeline1/runs/exp4_batch/yolov8n_e100_sz640_lr0.005_b16_mos1",
  "hyperparams": { "model": "yolov8n", "epochs": 100, "imgsz": 640, "lr": 0.005, "batch": 16, "mosaic": 1 },
  "metrics": { "map50_95": 0.72 }
}
```

---

## Module cần thêm

```
src/common/run_naming.py     # build_run_id(), fmt_lr(), run_dir(), weights_filename()
src/common/registry.py       # load/save/update registry.json, is_complete(run_id)
```

Orchestrator gọi:

```python
run_id = build_run_id(params, pipeline="pipeline1")
run_dir = output_root / "runs" / experiment / run_id
weights_path = run_dir / f"{run_id}.pt"

if registry.is_complete(run_id) and not force:
    skip
```
