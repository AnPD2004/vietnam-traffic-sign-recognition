# Pipeline 2 — YOLO + CNN Hybrid

> Design: `plan/design/logic-pipeline2.txt`

## Lệnh duy nhất

```bash
python -m src.pipeline2
# hoặc
python -m src.pipeline2 --config configs/pipeline2.yaml
```

**Điều kiện tiên quyết:** Pipeline 1 đã chạy xong, có `artifacts/pipeline1/best/best.json`.

---

## Luồng tổng thể

```
Load best YOLO từ Pipeline 1 (frozen — không train lại)
  ↓
Build crop dataset (nếu chưa có)
  ↓
Exp1: So sánh ResNet50 vs EfficientNet-B0
  ↓
Exp2: Learning rate study (best CNN từ Exp1)
  ↓
Exp3: Frozen backbone vs Fine-tuning
  ↓
Chọn best CNN → artifacts/pipeline2/best/
  ↓
Generate report
```

---

## YOLO reference (cố định)

Load từ `artifacts/pipeline1/best/best.json`:

```yaml
yolo:
  weights: artifacts/pipeline1/best/weights/best.pt
  imgsz: 640        # từ best hyperparams P1
  # dùng cho: crop padding context, inference imgsz
```

> Pipeline 2 **không train YOLO**. YOLO chỉ dùng để detect + crop khi inference và khi build crops.

---

## Bước 0 — Chuẩn bị dữ liệu

### Tự động trong `python -m src.pipeline2`

```python
if not crops_exist() and not flags.skip_crops:
    prepare_data.build_crops(
        data_yaml=cfg.data.yolo_yaml,
        output=cfg.data.crops_dir,
        padding=0.05,
    )
```

Wrap `src/build_crops/build_crops_for_cnn.py` → `src/pipeline2/prepare_data.py`.

### Output structure (giữ nguyên)

```
data/vn-traffic-signs/crops/
├── train/{class_name}/*.jpg
├── val/{class_name}/*.jpg
└── test/{class_name}/*.jpg
```

---

## Baseline CNN

| Param | Giá trị |
|-------|---------|
| Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Input Size | 224 |

Models: `resnet50`, `efficientnet_b0`

---

## Cach dem run / reuse policy

Pipeline 2 co **7 benchmark cases** trong experiment matrix, nhung thuong chi
co **5 unique training/evaluation runs**.

Ly do: `run_id` cua CNN duoc tao tu model, epochs, input size, learning rate,
batch size va transfer strategy. Neu mot experiment sau co baseline value
trung voi best config cua experiment truoc, case do se reuse metrics da co
thay vi train/evaluate lai dung cung mot CNN configuration.

| Experiment | Benchmark cases | New unique runs | Reused baseline case |
|------------|-----------------|-----------------|----------------------|
| Exp1 CNN model compare | 2 | 2 | 0 |
| Exp2 learning rate | 3 | 2 | 1 (`lr` = previous best) |
| Exp3 transfer strategy | 2 | 1 | 1 (`strategy` = previous best) |
| **Total** | **7 cases** | **5 runs** | **2 reused** |

Khi viet report, ghi theo cach: **7 benchmark cases, 5 unique
trained/evaluated configurations, 2 reused baseline cases**.

---

## Thực nghiệm 1 — So sánh mô hình CNN

### Grid
`model ∈ {resnet50, efficientnet_b0}` — **2 unique runs**

### Metrics
- Accuracy, Precision, Recall, F1-macro
- Confusion Matrix (PNG)
- Inference Time (ms/image)
- Model Size (MB)

### Select best
`best_cnn_arch = argmax(F1-macro)`

---

## Thực nghiệm 2 — Learning Rate

### Input
Best CNN architecture từ Exp1.

### Fixed
`epochs=50, batch=32`

### Grid
`lr ∈ {0.0001, 0.001, 0.01}` — **3 benchmark cases**

Mot case trung voi learning rate cua best config tu Exp1, nen stage nay thuong
chi train/evaluate **2 unique runs moi** va **reuse 1 baseline case**.

### Metrics bổ sung
Training Loss, Validation Loss (per epoch trong history)

---

## Thực nghiệm 3 — Transfer Learning

### Input
Best architecture + best lr từ Exp1+2.

### Grid

| Case | Strategy | Mô tả |
|------|----------|-------|
| 1 | `frozen` | Freeze backbone, chỉ train head |
| 2 | `finetune` | Train toàn bộ model |

**2 benchmark cases**

Mot case trung voi transfer strategy cua best config tu Exp2, nen stage nay
thuong chi train/evaluate **1 unique run moi** va **reuse 1 baseline case**.

### Implement frozen backbone

```python
def apply_transfer_strategy(model, strategy: str) -> None:
    if strategy == "frozen":
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
    elif strategy == "finetune":
        for param in model.parameters():
            param.requires_grad = True
```

Optimizer chỉ nhận `filter(lambda p: p.requires_grad, model.parameters())`.

---

## Final output

```
artifacts/pipeline2/
├── registry.json
├── runs/ ... (5 unique trained/evaluated configs + reused case records)
├── best/
│   ├── efficientnet_b0_e50_isz224_lr0.001_b32_ft.pth
│   ├── yolo_ref.json
│   └── best.json
└── report/
```

### `best.json` schema

```json
{
  "pipeline": "pipeline2",
  "run_id": "efficientnet_b0_e50_isz224_lr0.001_b32_ft",
  "weights": "artifacts/pipeline2/best/efficientnet_b0_e50_isz224_lr0.001_b32_ft.pth",
  "source_run_dir": "artifacts/pipeline2/runs/exp3_transfer/efficientnet_b0_e50_isz224_lr0.001_b32_ft",
  "cnn": {
    "model": "efficientnet_b0",
    "hyperparams": {
      "epochs": 50,
      "batch": 32,
      "lr": 0.001,
      "input_size": 224,
      "strategy": "finetune"
    },
    "metrics": {
      "accuracy": 0.97,
      "f1_macro": 0.95,
      "latency_ms_per_image": 3.2,
      "model_size_mb": 21.5
    }
  },
  "yolo": {
    "ref": "artifacts/pipeline1/best/best.json"
  }
}
```

---

## Module implement

### `src/pipeline2/run.py`

Cùng pattern orchestrator như Pipeline 1:

```python
def main(config_path, **flags):
    cfg = load_config(config_path)
    yolo_best = load_pipeline1_best(cfg.yolo.source)

    if not flags.skip_crops:
        ensure_crops(cfg.data)

    state = {"yolo": yolo_best}
    for exp_name in ordered_experiments(cfg):
        ...
    finalize_best(cfg, state)
    generate_report(cfg, pipeline="pipeline2")
```

### `src/pipeline2/train.py`

Refactor từ `src/train/train_cnn.py`:

| Thay đổi | Chi tiết |
|----------|----------|
| **Bỏ `resnet18`, thay `resnet50`** | Xóa branch resnet18; chỉ hỗ trợ `resnet50`, `efficientnet_b0` |
| Thêm `strategy` | frozen / finetune |
| Output path | `runs/{experiment}/{run_id}/{run_id}.pth` |
| Per-run artifacts | `{run_id}.pth`, `metrics.json`, `run_config.yaml`, `confusion_matrix.png` |

### `src/pipeline2/evaluate.py`

- Reuse `compute_metrics()` từ train_cnn
- Thêm model size: `best.pth` file size
- End-to-end latency (optional): YOLO detect + CNN classify trên val images

### `src/common/cnn_model.py` — thay ResNet18 → ResNet50

**Bỏ hoàn toàn** branch `resnet18`. Chỉ giữ:

```python
if model_name == "resnet50":
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if model_name == "efficientnet_b0":
    ...
```

`choices` trong CLI/config: `["resnet50", "efficientnet_b0"]` — không còn `resnet18`.

---

## Inference sau benchmark

```bash
python -m src.infer.infer_flow2 \
  --source path/to/image.jpg \
  --config artifacts/pipeline2/best/best.json
```

Hoặc auto-resolve từ `artifacts/pipeline2/best/best.json`.

`Flow2YoloCnnPipeline` giữ nguyên — chỉ đổi default paths.

---

## Gap so với code hiện tại

| Vấn đề | Giải pháp |
|--------|-----------|
| `train_cnn.py` default `resnet18` | Đổi default → `resnet50`, xóa `resnet18` khỏi choices |
| `configs/cnn.yaml` ghi `resnet18` | Đổi → `resnet50` |
| Artifact cũ `resnet18` best.pth | Không migrate — train lại qua `python -m src.pipeline2` |
| EfficientNet chưa train | Exp1 tự động train |
| Không có frozen/finetune | Exp3 strategy param |
| Build crops là script riêng | Tích hợp vào `pipeline2` step 0 |
| Không link YOLO best từ P1 | `yolo_ref.json` + config pointer |

---

## Checklist implement

- [ ] Thay `resnet18` → `resnet50` trong `src/common/cnn_model.py` (xóa branch cũ)
- [ ] Cập nhật `configs/cnn.yaml`: `model: resnet50`
- [ ] Thêm `apply_transfer_strategy()` 
- [ ] Tạo `configs/pipeline2.yaml`
- [ ] Tạo `src/pipeline2/` (run, prepare_data, train, evaluate, select_best, __main__)
- [ ] Validate P1 best exists trước khi chạy
- [ ] Test dry-run
- [ ] Test full: `python -m src.pipeline2`
- [ ] Verify `Flow2YoloCnnPipeline` với best artifacts
