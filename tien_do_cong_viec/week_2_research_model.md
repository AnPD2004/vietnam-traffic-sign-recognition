# Tuần 2 - Nghiên cứu mô hình

## Mục tiêu

- So sánh hai hướng: YOLOv8n và CNN.
- Chọn chiến lược huấn luyện phù hợp với dataset.

## Nghiên cứu YOLOv8n

- YOLOv8n phù hợp cho detection + classification.
- Dùng nhãn YOLO hiện có để train trực tiếp.
- Model nhỏ, tốc độ nhanh, dễ triển khai trên thiết bị yếu.
- Cần lệnh train cơ bản:
  - `yolo detect train model=yolov8n.pt data=data/vn-traffic-signs/data.yaml epochs=100 imgsz=640`

## Nghiên cứu CNN

- CNN làm classification trên ảnh crop biển báo.
- Cần chuyển dữ liệu detection sang classification:
  - crop vùng bounding box theo nhãn
  - tạo dataset ảnh và nhãn lớp
- Có thể dùng:
  - CNN tự xây dựng nhỏ
  - transfer learning với ResNet/EfficientNet
- Mục tiêu: so sánh với YOLO về độ chính xác classification và inference speed.

## Tiêu chí so sánh

- YOLOv8n:
  - mAP, precision, recall
  - inference FPS
  - model size
- CNN:
  - accuracy, F1 score
  - confusion matrix
  - inference speed với ảnh crop

## Ghi chú

- Dataset đã sẵn định dạng YOLO nên ưu tiên thực hiện YOLOv8n trước.
- CNN là bước bổ sung để so sánh khả năng phân loại biển báo sau khi detect.
