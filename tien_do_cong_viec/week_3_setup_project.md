# Tuần 3 - Cài đặt và triển khai dự án

## Mục tiêu

- Thiết lập cấu trúc dự án và các file hỗ trợ.
- Bắt đầu triển khai YOLOv8n và CNN.

## Cấu trúc dự án đề xuất

- `data/` — chứa dataset và file cấu hình.
- `models/` — lưu checkpoint, weights.
- `scripts/` hoặc `src/` — chứa script train/infer.
- `tien_do_cong_viec/` — lưu tiến độ, ghi chú.
- `README.md` và `project.md` — mô tả mục tiêu, cách chạy.

## Những gì đã làm

- Đã xác định dataset và cấu trúc dữ liệu.
- Đã cài đặt môi trường `ultralytics`, `torch`, `torchvision`.
- Đã lên kế hoạch nghiên cứu YOLOv8n và CNN.

## Công việc nên thực hiện

1. Tạo `data/vn-traffic-signs/data.yaml`.
2. Viết script/command train YOLOv8n.
3. Kiểm tra kết quả đầu tiên của YOLO.
4. Chuẩn bị data pipeline crop ảnh cho CNN.
5. Bắt đầu xây dựng mô hình CNN và đánh giá.

## Ghi chú

- Dự án đang ở giai đoạn `setup và chuẩn bị triển khai`.
- Tiếp tục ưu tiên YOLOv8n vì bộ dữ liệu đã phù hợp nhất với detection.
