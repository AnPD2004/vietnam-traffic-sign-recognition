# Kế hoạch Implement — Refactor Pipeline

## Mục tiêu

Refactor toàn bộ dự án để **mỗi pipeline chạy bằng một lệnh duy nhất**, tự động hóa toàn bộ thực nghiệm theo `plan/design/`.

```bash
# Pipeline 1 — YOLO End-to-End (4 thực nghiệm, chọn best YOLO)
python -m src.pipeline1

# Pipeline 2 — YOLO + CNN Hybrid (3 thực nghiệm, chọn best CNN)
python -m src.pipeline2
```

## Tài liệu

| File | Nội dung |
|------|----------|
| [00-architecture.md](00-architecture.md) | Cấu trúc thư mục mới, module chia sẻ, artifact layout |
| [01-pipeline1.md](01-pipeline1.md) | Chi tiết implement Pipeline 1 |
| [02-pipeline2.md](02-pipeline2.md) | Chi tiết implement Pipeline 2 |
| [03-migration.md](03-migration.md) | Map code cũ → mới, checklist refactor |
| [04-artifact-layout.md](04-artifact-layout.md) | **Cấu trúc folder, đặt tên model theo tham số, registry** |
| [05-test-mode.md](05-test-mode.md) | **`--test` — full flow, 1 epoch, artifact tách biệt** |

## Tham chiếu design

- `plan/design/logic-pipeline1.txt` — 4 thực nghiệm YOLO
- `plan/design/logic-pipeline2.txt` — 3 thực nghiệm CNN (dùng YOLO best từ P1)

## Nguyên tắc refactor

1. **Một entry point / pipeline** — orchestrator điều phối toàn bộ experiment matrix.
2. **Config-driven** — hyperparameter grid khai báo trong YAML, không hard-code trong script.
3. **Idempotent artifacts** — skip run đã hoàn thành (có `metrics.json`), hỗ trợ `--force`.
4. **Best model tự động** — sau mỗi stage chọn winner theo metric chính, ghi `best.json`.
5. **Tách inference khỏi training** — `Flow1YoloYoloPipeline` / `Flow2YoloCnnPipeline` giữ nguyên contract, chỉ đổi path load model.
6. **CNN: bỏ ResNet18** — chỉ dùng `resnet50` + `efficientnet_b0` theo design; artifact ResNet18 cũ không migrate.
7. **Model đặt tên theo tham số** — mỗi run export `{run_id}.pt`/`.pth`; xem [04-artifact-layout.md](04-artifact-layout.md).
8. **`--test` trước chạy chính** — full flow với 1 epoch → `artifacts/pipeline*_test/`; xem [05-test-mode.md](05-test-mode.md).

## Thứ tự thực hiện

```
Phase A  →  Cấu trúc thư mục + config YAML + shared metrics/report
Phase B  →  Pipeline 1 (YOLO benchmark end-to-end)
Phase C  →  Pipeline 2 (phụ thuộc best YOLO từ P1)
Phase D  →  Xóa/deprecate code cũ, cập nhật README + flow.md
```

## Ước lượng số run training

| Pipeline | Thực nghiệm | Số run |
|----------|-------------|--------|
| P1 Exp1 | 3 model × 2 mosaic | 6 |
| P1 Exp2 | 3 image sizes | 3 |
| P1 Exp3 | 3 learning rates | 3 |
| P1 Exp4 | 3 batch sizes | 3 |
| **P1 tổng** | | **15** |
| P2 Exp1 | 2 CNN models | 2 |
| P2 Exp2 | 3 learning rates | 3 |
| P2 Exp3 | 2 transfer strategies | 2 |
| **P2 tổng** | | **7** |

> YOLO trong P2 **không train lại** — load checkpoint best từ P1.
