# Project Flow - Vietnamese Traffic Sign Recognition

## Research Workflow

The project consists of three benchmarking stages and two independent inference pipelines.

---

# Stage 1 - YOLO Benchmark

Objective:

Select the best YOLO model for traffic sign detection.

Models:

* YOLOv5n
* YOLOv8n
* YOLOv11n

Evaluation Metrics:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95
* FPS
* Model Size

Output:

Best YOLO model

---

# Stage 2 - CNN Benchmark

Objective:

Select the best CNN classifier.

Models:

* ResNet50
* EfficientNet-B0

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Output:

Best CNN model

---

# Stage 3 - Pipeline Benchmark

Compare:

Pipeline 1

YOLO End-to-End

vs

Pipeline 2

YOLO + CNN

Evaluation Metrics:

* Detection Performance
* Classification Performance
* FPS
* Latency
* Model Size

---

# Data Preparation Flow

Raw Dataset

↓

YOLO Annotation Verification

↓

YOLO Detection Dataset

↓

Train YOLO Models

↓

Crop Bounding Boxes

↓

Classification Dataset

↓

Train CNN Models

---

# Pipeline 1

YOLO End-to-End

Image

↓

YOLO

↓

Bounding Box

*

Class

↓

Output

Responsibilities:

* Detection: YOLO
* Classification: YOLO

---

# Pipeline 2

YOLO + CNN Hybrid

Image

↓

YOLO Detector

↓

Bounding Box

↓

Crop Image

↓

CNN Classifier

↓

Class

↓

Output

Responsibilities:

* Detection: YOLO
* Classification: CNN

---

# Folder Structure

src/

├── build_crops/

├── train/

│ ├── train_yolo.py

│ └── train_cnn.py

├── infer/

│ ├── infer_flow1.py

│ └── infer_flow2.py

├── pipelines/

│ ├── flow1_pipeline.py

│ └── flow2_pipeline.py

├── web/

└── common/

---

# API Endpoints

POST /predict/flow1

YOLO End-to-End

POST /predict/flow2

YOLO + CNN

POST /benchmark

Compare model outputs

---

# Development Checklist

[x] Verify YOLO dataset — 3.216 ảnh/nhãn, 52 lớp, split train (2.552) / test (639); `data/vn-traffic-signs/data.yaml`

[x] Build crop dataset — script `src/build_crops/build_crops_for_cnn.py`; đã chạy (CNN train val: 1.645 mẫu)

[ ] Train YOLOv5n

[x] Train YOLOv8n — 100 epochs; `runs/detect/artifacts/yolo_detect_cls/flow1_train/` (mAP50≈0.98)

[ ] Train YOLOv11n

[x] Evaluate YOLO models — chỉ YOLOv8n (`results.csv`, confusion matrix); chưa so sánh v5/v11

[ ] Train ResNet50 — code hiện dùng ResNet18, chưa hỗ trợ ResNet50

[ ] Train EfficientNet-B0 — hỗ trợ trong `train_cnn.py`, chưa train

[x] Evaluate CNN models — chỉ ResNet18 (`metrics.json`, acc≈96.1%, F1≈0.93); chưa so sánh EfficientNet

[ ] Select best models — chưa benchmark đủ model để chọn

[x] Implement Flow 1 — `src/pipelines/flow1_pipeline.py`

[x] Implement Flow 2 — `src/pipelines/flow2_pipeline.py`

[ ] Build Web Benchmark System — chưa có `src/web/`, `infer/`, Flask API

[ ] Final Evaluation Report — chưa có `reports/`
