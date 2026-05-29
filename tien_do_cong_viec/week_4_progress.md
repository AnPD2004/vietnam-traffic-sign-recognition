# Tuần 4 - Tiến độ (2026-05-29)

## Tổng quan
- **Flow 1 (YOLO end-to-end):** Hoàn thành và có demo web cơ bản.
- Người thực hiện: bạn (đã implement và kiểm tra local).

## Những gì đã hoàn thành
- Đã train/chuẩn bị model YOLO và có weights: `yolov8n.pt`.
- Script inference Flow 1: `src/infer/infer_flow1_yolo_yolo.py`.
- Demo frontend cơ bản: `fe/` (ví dụ `fe/index.html`, `fe/src/`).
- Codebase cấu trúc theo `flow.md` đề xuất (đang có `src/`, `runs/`, `data/`).

## Lệnh chạy nhanh (Flow 1)

```bash
python -m src.infer.infer_flow1_yolo_yolo --input path_or_video
```

## Vấn đề/ghi chú
- Kiểm tra lại vị trí của `class_map` chung; nếu chưa có, cần tạo: `src/common/labels.py` hoặc `data/vn-traffic-signs/classes.txt` làm nguồn chuẩn.
- Xác nhận checkpoint tốt nhất được lưu tại `artifacts/yolo_detect_cls/` hoặc `runs/detect/exp*/weights/best.pt`.

## Bước tiếp theo (Ưu tiên)
1. Viết script crop dataset cho CNN: `src/data/build_crops_for_cnn.py`.
2. Tạo pipeline huấn luyện CNN: `src/train/train_cnn.py` (ResNet/EfficientNet, transfer learning).
3. Viết file infer cho Flow 2: `src/infer/infer_flow2_yolo_cnn.py`.
4. Đồng bộ `class_map` giữa YOLO và CNN.
5. Thực hiện benchmark so sánh `flow1` vs `flow2` (mAP, accuracy, FPS, model size).

## Checklist (Tuần 4)
- [x] Xác nhận Flow 1 hoạt động (inference + demo web).
- [ ] Viết script crop dataset cho CNN.
- [ ] Triển khai và train CNN classifier.
- [ ] Viết infer cho Flow 2.
- [ ] Chạy benchmark so sánh.

---

Nếu bạn đồng ý, tôi có thể tiếp tục và tạo skeleton cho `src/data/build_crops_for_cnn.py` và `src/infer/infer_flow2_yolo_cnn.py` ngay bây giờ.