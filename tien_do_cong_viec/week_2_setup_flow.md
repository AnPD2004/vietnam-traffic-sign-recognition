# Tuần 2 - Thiết lập quy trình làm việc

## Mục tiêu

- Xây dựng quy trình dữ liệu và pipeline huấn luyện.
- Chuẩn bị các file, script, và cấu hình cần thiết.

## Quy trình dữ liệu

1. Tạo `data/vn-traffic-signs/data.yaml` chứa:
   - `train:` đường dẫn đến thư mục hoặc danh sách ảnh train
   - `val:` đường dẫn đến thư mục hoặc danh sách ảnh val/test
   - `nc: 52`
   - `names:` danh sách tên 52 lớp
2. Kiểm tra dataset:
   - mọi ảnh đều có file nhãn tương ứng
   - train/test split hợp lệ
   - nhãn có class_id trong khoảng 0..51

## Quy trình YOLOv8n

1. Chuẩn bị môi trường Python với `ultralytics`.
2. Chạy thử lệnh training cơ bản.
3. Theo dõi kết quả: loss, mAP, precision, recall.
4. Lưu model tốt nhất và đánh giá bằng `yolo val`.

## Quy trình CNN

1. Crop biển báo từ ảnh dựa trên file nhãn YOLO.
2. Tạo dataset classification với nhãn `class_id`.
3. Xây dựng và huấn luyện mô hình CNN.
4. Đánh giá accuracy và so sánh với YOLO.

## Kết quả mong đợi

- Có pipeline YOLOv8n hoàn chỉnh.
- Có pipeline CNN classification để so sánh.
- Có tài liệu đánh giá rõ ràng.
