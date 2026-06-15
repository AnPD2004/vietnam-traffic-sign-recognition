# Pipeline 1 — YOLO End-to-End

> Design: `plan/design/logic-pipeline1.txt`

## Lệnh duy nhất

```bash
python -m src.pipeline1
# hoặc
python -m src.pipeline1 --config configs/pipeline1.yaml
```

Chạy **4 thực nghiệm tuần tự**, mỗi thực nghiệm dùng best model/config từ thực nghiệm trước.

---

## Baseline (cố định toàn pipeline)

| Param | Giá trị |
|-------|---------|
| Epochs | 100 |
| Image Size | 640 |
| Learning Rate | 0.005 |
| Batch Size | 16 |
| Mosaic | 1 (default; Exp1 override) |

Models: `yolov5n`, `yolov8n`, `yolov11n`

---

## Thực nghiệm 1 — So sánh mô hình YOLO

### Mục tiêu
Chọn architecture YOLO tốt nhất.

### Matrix

| Case | Mosaic | Models |
|------|--------|--------|
| 1 | 0 | v5n, v8n, v11n |
| 2 | 1 | v5n, v8n, v11n |

**6 runs** → output: `artifacts/pipeline1/runs/exp1_model_compare/`

### Naming convention

> Chi tiết đầy đủ: [04-artifact-layout.md](04-artifact-layout.md)

Mỗi run có **run_id** encode toàn bộ hyperparams, model file trùng tên:

```
runs/exp1_model_compare/
├── yolov5n_e100_sz640_lr0.005_b16_mos0/
│   └── yolov5n_e100_sz640_lr0.005_b16_mos0.pt
├── yolov8n_e100_sz640_lr0.005_b16_mos1/
│   └── yolov8n_e100_sz640_lr0.005_b16_mos1.pt
└── _best.json
```

Format: `{model}_e{epochs}_sz{imgsz}_lr{lr}_b{batch}_mos{mosaic}`

### Metrics thu thập

- Precision, Recall, mAP@0.5, mAP@0.5:0.95
- FPS, Inference Time, Model Size (MB)

### Select best

```
best_model = argmax(mAP@0.5:0.95) across 6 runs
best_mosaic = mosaic value của winner
→ lưu `artifacts/pipeline1/runs/exp1_model_compare/_best.json` + cập nhật `registry.json`
```

---

## Thực nghiệm 2 — Image Size

### Input
Best model + best mosaic từ Exp1.

### Fixed
`lr=0.005, batch=16, mosaic={best_mosaic}`

### Grid
`imgsz ∈ {416, 640, 800}` — **3 runs**

### Output
`artifacts/pipeline1/runs/exp2_imgsz/`

Ví dụ run_id (model + mosaic cố định từ Exp1, chỉ imgsz thay đổi):
`yolov8n_e100_sz416_lr0.005_b16_mos1`

### Select best
`best_imgsz = argmax(mAP@0.5:0.95)`

---

## Thực nghiệm 3 — Learning Rate

### Input
Best model, mosaic, imgsz từ Exp1+2.

### Fixed
`imgsz={best_imgsz}, batch=16, mosaic={best_mosaic}`

### Grid
`lr ∈ {0.001, 0.005, 0.01}` — **3 runs**

### Metrics
Precision, Recall, mAP@0.5, mAP@0.5:0.95

---

## Thực nghiệm 4 — Batch Size

### Input
Best model, mosaic, imgsz, lr từ Exp1+2+3.

### Fixed
`imgsz={best_imgsz}, lr={best_lr}, mosaic={best_mosaic}`

### Grid
`batch ∈ {8, 16, 32}` — **3 runs**

### Metrics bổ sung
Training Time, GPU Memory Usage (peak VRAM nếu có CUDA)

---

## Final output

```
artifacts/pipeline1/
├── registry.json
├── runs/ ... (15 run folders, mỗi folder 1 file .pt đặt tên theo params)
├── best/
│   ├── yolov8n_e100_sz640_lr0.005_b16_mos1.pt   # copy winner Exp4
│   └── best.json
└── report/
```

### `best.json` schema

```json
{
  "pipeline": "pipeline1",
  "run_id": "yolov8n_e100_sz640_lr0.005_b16_mos1",
  "weights": "artifacts/pipeline1/best/yolov8n_e100_sz640_lr0.005_b16_mos1.pt",
  "source_run_dir": "artifacts/pipeline1/runs/exp4_batch/yolov8n_e100_sz640_lr0.005_b16_mos1",
  "hyperparams": {
    "model": "yolov8n",
    "epochs": 100,
    "imgsz": 640,
    "lr": 0.005,
    "batch": 16,
    "mosaic": 1
  },
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

---

## Module implement

### `src/pipeline1/run.py`

```python
def main(config_path: Path, **flags) -> None:
    cfg = load_config(config_path)
    state = {}  # accumulated best params

    for exp_name in ordered_experiments(cfg):
        if flags.experiment and exp_name != flags.experiment:
            continue
        runs = build_run_list(cfg, exp_name, state)
        if flags.dry_run:
            print_run_plan(runs); continue
        for run in runs:
            run_id = run.output_dir.name
            if run.metrics_exists() and not flags.force:
                continue
            train_yolo(run)
            metrics = evaluate_yolo(run)
            save_run_artifacts(run, metrics)
        state.update(select_best(cfg, exp_name))
        save_experiment_best(exp_name, state)

    finalize_best(cfg, state)
    generate_report(cfg)
```

### `src/pipeline1/train.py`

Refactor từ `src/train/train_yolo.py`:

- Thêm param `lr` (hiện thiếu — Ultralytics: `lr0`)
- Thêm param `mosaic` (Ultralytics: `mosaic`)
- Map model name → checkpoint:
  - `yolov5n` → `yolov5n.pt` (hoặc `ultralytics/yolov5/yolov5n.pt`)
  - `yolov8n` → `yolov8n.pt`
  - `yolov11n` → `yolo11n.pt`
- Output cố định: `runs/{experiment}/{run_id}/{run_id}.pt`
- Sau train: rename `best.pt` → `{run_id}.pt`, xóa thư mục Ultralytics tạm

### `src/pipeline1/evaluate.py`

- Gọi `model.val()` → parse `results.csv` / `results_dict`
- Đo FPS: predict N ảnh test set, warmup 10 frames
- Model size: `os.path.getsize(best.pt) / 1e6`
- GPU memory: `torch.cuda.max_memory_allocated()` sau 1 epoch (Exp4 only)

### `src/common/metrics.py`

```python
def parse_yolo_val_results(run_dir: Path) -> dict: ...
def measure_yolo_fps(model, images: list, imgsz: int) -> dict: ...
def select_best_run(runs: list[dict], metric: str) -> dict: ...
```

---

## Gap so với code hiện tại

| Vấn đề | Giải pháp |
|--------|-----------|
| `train_yolo.py` không có `lr`, `mosaic` | Thêm vào train wrapper |
| Chỉ train được YOLOv8n | Hỗ trợ v5n, v11n qua model map |
| Output rải rác `runs/detect/` | Chuẩn hóa `artifacts/pipeline1/` |
| Không có orchestrator | `run.py` điều phối 15 runs |
| Không chọn best tự động | `select_best.py` |
| LR design = 0.005, README = 0.01 | Dùng **0.005** theo design |

---

## Checklist implement

- [ ] Tạo `configs/pipeline1.yaml`
- [ ] Tạo `src/pipeline1/` (run, train, evaluate, select_best, __main__)
- [ ] Thêm `lr0`, `mosaic` vào YOLO train
- [ ] Model map yolov5n / yolov8n / yolov11n
- [ ] `src/common/run_naming.py` + `registry.py`
- [ ] `src/common/metrics.py` — parse + FPS
- [ ] `src/common/report.py` — summary Markdown
- [ ] Test dry-run: `python -m src.pipeline1 --dry-run`
- [ ] Test 1 run: `python -m src.pipeline1 --experiment exp1_model_compare --force`
- [ ] Verify `Flow1YoloYoloPipeline` load từ `artifacts/pipeline1/best/`
