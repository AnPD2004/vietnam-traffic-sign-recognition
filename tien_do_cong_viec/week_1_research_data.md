# Tuần 1 - Nghiên cứu và chuẩn bị dữ liệu

## Mục tiêu

- Xác định bộ dữ liệu và cấu trúc hiện có.
- Hiểu định dạng nhãn YOLO và kiểm tra tính phù hợp với YOLOv8.
- Chuẩn bị môi trường để bắt đầu huấn luyện.

## Những gì đã hoàn thành

- Phân tích bộ dữ liệu `data/vn-traffic-signs`:
  - 3.216 ảnh `.jpg`
  - 3.216 file nhãn `.txt`
  - 52 lớp biển báo
  - Nhãn chuẩn YOLO: `class_id x_center y_center width height`
  - Split sẵn: `train_files.txt` (2.552 ảnh), `test_files.txt` (639 ảnh)
- Xác định các file chính:
  - `data/vn-traffic-signs/images/`
  - `data/vn-traffic-signs/labels/`
  - `data/vn-traffic-signs/classes.txt`
  - `data/vn-traffic-signs/classes_en.txt`
  - `data/vn-traffic-signs/classes_vie.txt`
- Đã cập nhật `project.md` với thông tin dataset và cấu trúc dữ liệu.
- Kiểm tra môi trường và cài `ultralytics`, `torch`, `torchvision`.
