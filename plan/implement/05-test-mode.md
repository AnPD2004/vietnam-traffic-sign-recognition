# Test Mode — Chạy thử trước khi chạy chính

## Mục đích

Xác minh **toàn bộ luồng pipeline** (mọi experiment → train → evaluate → chọn best → report) hoạt động đúng, **không cần train 100 epoch**.

Khác với `--dry-run` (chỉ in kế hoạch, không train).

---

## Lệnh

```bash
# Pipeline 1 — full flow, 1 epoch mỗi run
python -m src.pipeline1 --test

# Pipeline 2 — full flow, 1 epoch mỗi run
python -m src.pipeline2 --test
```

---

## Hành vi `--test`

| Thuộc tính | Chạy chính | `--test` |
|------------|------------|----------|
| Experiments | 4 (P1) / 3 (P2) | **Giữ nguyên** — chạy đủ matrix |
| Runs mỗi experiment | Đầy đủ (15 / 7) | **Giữ nguyên** |
| Epochs | 100 (P1) / 50 (P2) | **1** |
| Patience | 30 / 10 | **1** |
| Output dir | `artifacts/pipeline1` | `artifacts/pipeline1_test` |
| run_id | `..._e100_...` | `..._e1_...` (không đè artifact chính) |
| Select best + report | Có | **Có** — validate logic end-to-end |
| Metrics | Đáng tin cho benchmark | Chỉ để smoke test, **không dùng chọn model production** |

---

## Luồng

```
python -m src.pipeline1 --test
  ↓
override: epochs=1, patience=1, output=artifacts/pipeline1_test
  ↓
Exp1 (6 runs × 1 epoch) → _best.json
  ↓
Exp2 (3 runs) → dùng best model/mosaic từ Exp1
  ↓
Exp3 (3 runs) → dùng best từ Exp1+2
  ↓
Exp4 (3 runs) → dùng best từ Exp1+2+3
  ↓
best/best.json + report/summary.md
  ↓
in ra: "TEST MODE complete — N runs, output: artifacts/pipeline1_test"
```

---

## Artifact tách biệt

```
artifacts/
├── pipeline1/           ← chạy chính (epochs=100)
└── pipeline1_test/        ← --test (epochs=1, run_id có e1)
```

Pipeline 2 tương tự: `pipeline2_test/`.

Sau khi `--test` pass, chạy chính:

```bash
python -m src.pipeline1          # không có --test
python -m src.pipeline2
```

---

## Kết hợp với flag khác

```bash
# Test chỉ 1 experiment
python -m src.pipeline1 --test --experiment exp1_model_compare

# Test + ép train lại
python -m src.pipeline1 --test --force

# Test 1 run cụ thể (debug nhanh)
python -m src.pipeline1 --test --run-id yolov8n_e1_sz640_lr0.005_b16_mos0
```

`--dry-run` và `--test` **không dùng cùng lúc** — dry-run thắng (chỉ in plan).

---

## Checklist

- [x] `--test` override epochs + output dir
- [x] run_id phản ánh epochs thực (e1 vs e100)
- [x] Full orchestrator path: train → eval → registry → select_best → report
- [x] Log rõ ràng banner `=== TEST MODE ===` khi bắt đầu
- [x] Pipeline 2 `--test` cũng build crops nếu chưa có (hoặc `--skip-crops`)
- [x] **RAM: mỗi run chạy subprocess riêng** (`run_worker.py`) — OS giải phóng RAM khi process exit
- [x] `release_runtime_memory()` sau train/eval + xóa `runs/detect/` tạm
