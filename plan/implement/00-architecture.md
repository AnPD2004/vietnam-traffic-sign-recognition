# Architecture — Cấu trúc sau refactor

## Cấu trúc thư mục đề xuất

```
src/
├── common/                    # Giữ nguyên — utilities dùng chung
│   ├── cnn_model.py           # resnet50 (bỏ resnet18)
│   ├── image_ops.py
│   ├── labels.py
│   ├── splits.py
│   ├── metrics.py             # MỚI — parse YOLO results.csv, tổng hợp CNN metrics
│   ├── report.py              # MỚI — sinh bảng so sánh Markdown/JSON
│   ├── run_naming.py          # MỚI — build_run_id(), encode hyperparams → tên file
│   └── registry.py            # MỚI — registry.json CRUD, skip-if-complete
│
├── pipeline1/                 # MỚI — YOLO End-to-End
│   ├── __init__.py
│   ├── __main__.py            # entry: python -m src.pipeline1
│   ├── run.py                 # orchestrator chính
│   ├── config.yaml            # experiment matrix + baseline
│   ├── train.py               # refactor từ train/train_yolo.py
│   ├── evaluate.py            # val + FPS + model size
│   └── select_best.py         # chọn winner mỗi stage
│
├── pipeline2/                 # MỚI — YOLO + CNN Hybrid
│   ├── __init__.py
│   ├── __main__.py            # entry: python -m src.pipeline2
│   ├── run.py
│   ├── config.yaml
│   ├── prepare_data.py        # wrap build_crops_for_cnn.py
│   ├── train.py               # refactor từ train/train_cnn.py
│   ├── evaluate.py
│   └── select_best.py
│
├── infer/                     # MỚI — inference CLI (sau benchmark)
│   ├── infer_flow1.py
│   └── infer_flow2.py
│
└── pipelines/                 # GIỮ — inference classes (API contract)
    ├── flow1_pipeline.py
    └── flow2_pipeline.py

configs/                       # GIỮ + mở rộng
├── pipeline1.yaml             # default config P1 (copy từ src/pipeline1/config.yaml)
└── pipeline2.yaml

artifacts/
├── pipeline1/
│   ├── registry.json          # master index tất cả runs
│   ├── runs/
│   │   ├── exp1_model_compare/
│   │   │   └── {run_id}/      # vd: yolov8n_e100_sz640_lr0.005_b16_mos1/
│   │   │       ├── {run_id}.pt          ← model file = tên tham số
│   │   │       ├── metrics.json
│   │   │       └── run_config.yaml
│   │   ├── exp2_imgsz/
│   │   ├── exp3_lr/
│   │   └── exp4_batch/
│   ├── best/
│   │   ├── {run_id}.pt        # copy winner
│   │   └── best.json
│   └── report/
│
└── pipeline2/
    ├── registry.json
    ├── runs/
    │   ├── exp1_model_compare/
    │   │   └── {run_id}/      # vd: resnet50_e50_isz224_lr0.001_b32_ft/
    │   │       └── {run_id}.pth
    │   ├── exp2_lr/
    │   └── exp3_transfer/
    ├── best/
    │   ├── {run_id}.pth
    │   ├── yolo_ref.json
    │   └── best.json
    └── report/

# Chi tiết naming, registry schema, disk policy → plan/implement/04-artifact-layout.md
```

## Xóa / deprecate sau refactor

| Cũ | Hành động |
|----|-----------|
| `src/train/train_yolo.py` | Di chuyển logic → `src/pipeline1/train.py` |
| `src/train/train_cnn.py` | Di chuyển logic → `src/pipeline2/train.py` |
| `src/build_crops/` | Wrap trong `pipeline2/prepare_data.py`, giữ module gốc |
| `artifacts/yolo_detect_cls/` | Migrate sang `artifacts/pipeline1/` |

## Config schema (YAML)

### `configs/pipeline1.yaml`

```yaml
data: data/vn-traffic-signs/data.yaml

baseline:
  epochs: 100
  imgsz: 640
  lr: 0.005
  batch: 16
  mosaic: 1
  patience: 30
  seed: 42

models: [yolov5n, yolov8n, yolov11n]

experiments:
  exp1_model_compare:
    grid:
      mosaic: [0, 1]
    metric: map50_95          # primary metric để chọn best model
  exp2_imgsz:
    grid:
      imgsz: [416, 640, 800]
    fixed: { lr: 0.005, batch: 16, mosaic: 1 }
    metric: map50_95
  exp3_lr:
    grid:
      lr: [0.001, 0.005, 0.01]
    fixed: { imgsz: 640, batch: 16, mosaic: 1 }
    metric: map50_95
  exp4_batch:
    grid:
      batch: [8, 16, 32]
    fixed: { imgsz: 640, lr: 0.005, mosaic: 1 }
    metric: map50_95

output: artifacts/pipeline1
```

### `configs/pipeline2.yaml`

```yaml
data:
  yolo_yaml: data/vn-traffic-signs/data.yaml
  crops_dir: data/vn-traffic-signs/crops
  classes: data/vn-traffic-signs/classes.txt

yolo:
  source: artifacts/pipeline1/best/best.json   # auto-load best YOLO

baseline:
  epochs: 50
  batch: 32
  lr: 0.001
  input_size: 224
  patience: 10
  seed: 42

models: [resnet50, efficientnet_b0]

experiments:
  exp1_model_compare:
    metric: f1_macro
  exp2_lr:
    grid:
      lr: [0.0001, 0.001, 0.01]
    fixed: { batch: 32 }
    metric: f1_macro
  exp3_transfer:
    grid:
      strategy: [frozen, finetune]
    metric: f1_macro

output: artifacts/pipeline2
```

## CLI interface

### Pipeline 1

```
python -m src.pipeline1 [OPTIONS]

Options:
  --config PATH       default: configs/pipeline1.yaml
  --test              full flow, 1 epoch/run → artifacts/pipeline1_test/
  --experiment NAME   chỉ chạy 1 experiment (exp1_model_compare, ...)
  --dry-run           in ra danh sách run, không train
  --force             train lại dù đã có metrics
  --device DEVICE     0 | cpu | auto
  --skip-eval         chỉ train, bỏ qua FPS/size
  --run-id ID         chỉ chạy 1 run cụ thể
  --list-runs         liệt kê runs trong registry
```

### Pipeline 2

```
python -m src.pipeline2 [OPTIONS]

Options:
  --config PATH
  --test              full flow, 1 epoch/run → artifacts/pipeline2_test/
  --experiment NAME
  --dry-run
  --force
  --device DEVICE
  --skip-crops        bỏ qua bước build crops nếu đã có
```

## Orchestrator flow (chung)

```
load config
  ↓
resolve best from previous experiment (nếu exp2+)
  ↓
build run list (cartesian product của grid × best model)
  ↓
for each run:
  ├── run_id = build_run_id(params)           # vd: yolov8n_e100_sz640_lr0.005_b16_mos1
  ├── run_dir = runs/{experiment}/{run_id}/
  ├── check registry + metrics.json → skip nếu complete & !force
  ├── train → export {run_dir}/{run_id}.pt
  ├── evaluate → metrics.json + run_config.yaml
  └── update registry.json
  ↓
select_best(experiment) → runs/{exp}/_best.json
  ↓
final: merge best across all experiments → artifacts/.../best/best.json
  ↓
generate report (summary.md + summary.json)
```

## Metric chính để chọn best

| Pipeline | Primary | Tie-breaker |
|----------|---------|-------------|
| P1 YOLO | mAP@0.5:0.95 | mAP@0.5 → FPS |
| P2 CNN | F1-macro | Accuracy → latency |

## Inference contract (không đổi)

`Flow1YoloYoloPipeline` và `Flow2YoloCnnPipeline` load từ `best.json` → path tới `{run_id}.pt` / `{run_id}.pth`.

Ví dụ: `artifacts/pipeline1/best/yolov8n_e100_sz640_lr0.005_b16_mos1.pt`

Chi tiết: [04-artifact-layout.md](04-artifact-layout.md)
