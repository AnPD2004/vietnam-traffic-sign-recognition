# Migration — Từ code hiện tại sang refactor

## Map module cũ → mới

```
src/train/train_yolo.py          →  src/pipeline1/train.py
src/train/train_cnn.py           →  src/pipeline2/train.py
src/build_crops/build_crops_for_cnn.py  →  src/pipeline2/prepare_data.py (import/re-export)
src/pipelines/flow1_pipeline.py  →  GIỮ (inference only)
src/pipelines/flow2_pipeline.py  →  GIỮ (inference only)
configs/cnn.yaml                 →  configs/pipeline2.yaml (merge + mở rộng)
```

## Artifact migration

| Cũ | Mới |
|----|-----|
| `runs/detect/.../weights/best.pt` | `artifacts/pipeline1/runs/{exp}/{run_id}/{run_id}.pt` |
| `artifacts/cnn_classifier/best.pth` | `artifacts/pipeline2/runs/{exp}/{run_id}/{run_id}.pth` |

Script migration một lần (optional):

```bash
python -m src.tools.migrate_artifacts   # tạo sau nếu cần giữ kết quả cũ
```

## Lệnh trước vs sau refactor

### Trước (nhiều lệnh thủ công)

```bash
# Pipeline 1 — phải chạy từng model, từng config
python -m src.train.train_yolo --data data/.../data.yaml --model yolov8n.pt --epochs 100
# ... lặp lại 15 lần với params khác nhau
# ... đánh giá thủ công, chọn best thủ công

# Pipeline 2
python -m src.build_crops.build_crops_for_cnn
python -m src.train.train_cnn --config configs/cnn.yaml
# ... lặp lại cho EfficientNet, lr, frozen/finetune
```

### Sau (1 lệnh / pipeline)

```bash
python -m src.pipeline1
python -m src.pipeline2
```

## Thứ tự refactor (đề xuất từng PR/commit)

### PR 1 — Foundation
- [ ] `src/common/run_naming.py` — `build_run_id()`, encode params
- [ ] `src/common/registry.py` — index, skip-if-complete
- [ ] `src/common/metrics.py`
- [ ] `src/common/report.py`
- [ ] `configs/pipeline1.yaml`, `configs/pipeline2.yaml`
- [ ] Thay `resnet18` → `resnet50` trong `cnn_model.py`, `train_cnn.py`, `configs/cnn.yaml`

### PR 2 — Pipeline 1
- [ ] `src/pipeline1/` full module
- [ ] Test `--dry-run` + 1 experiment nhỏ
- [ ] Không xóa `train_yolo.py` ngay — deprecate comment

### PR 3 — Pipeline 2
- [ ] `src/pipeline2/` full module
- [ ] `prepare_data.py` wrap build_crops
- [ ] Test với YOLO best từ PR2 hoặc artifact cũ

### PR 4 — Cleanup
- [ ] `src/infer/infer_flow1.py`, `infer_flow2.py`
- [ ] Xóa `src/train/` (hoặc re-export shim với warning)
- [ ] Cập nhật `README.md`, `flow.md`
- [ ] Cập nhật Development Checklist

## Cập nhật flow.md checklist (sau refactor)

```markdown
[ ] python -m src.pipeline1          # thay cho train/eval YOLO thủ công
[ ] python -m src.pipeline2          # thay cho build_crops + train CNN thủ công
[ ] python -m src.infer.infer_flow1  # inference CLI
[ ] python -m src.infer.infer_flow2
[ ] Build Web Benchmark System       # Phase sau — đọc artifacts/pipeline*/best/
```

## Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|--------|------------|
| 15 YOLO runs × 100 epochs = rất lâu | `--experiment` chạy từng phần; skip nếu có metrics |
| YOLOv5/v11 không tương thích Ultralytics API | Test từng model trước khi full grid |
| OOM batch=32 trên GPU nhỏ | Catch OOM → ghi `status: failed` vào metrics, không crash pipeline |
| P2 chạy khi P1 chưa xong | Validate `artifacts/pipeline1/best/best.json` exists, exit rõ ràng |

## Stage 3 — So sánh Pipeline 1 vs 2 (Phase E — ngoài scope refactor P1/P2)

Sau khi cả 2 pipeline chạy xong:

```bash
python -m src.benchmark   # so sánh flow1 vs flow2 trên cùng test set
```

**Chi tiết implement:** [06-stage3-benchmark.md](06-stage3-benchmark.md)

Metrics: detection + classification performance, FPS, model size tổng (YOLO + CNN).
Output: `artifacts/benchmark/best_pipeline.json`.

## Definition of Done

- [ ] `python -m src.pipeline1` chạy 4 experiments, sinh `artifacts/pipeline1/best/best.json`
- [ ] `python -m src.pipeline2` chạy 3 experiments, sinh `artifacts/pipeline2/best/best.json`
- [ ] `--dry-run`, `--force`, `--experiment` hoạt động
- [ ] Report Markdown có bảng so sánh từng experiment
- [ ] `Flow1YoloYoloPipeline` + `Flow2YoloCnnPipeline` load được từ best artifacts
- [ ] README có 2 lệnh chính, không còn hướng dẫn train từng model riêng lẻ
