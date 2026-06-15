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

[ ] Verify YOLO dataset

[ ] Build crop dataset

[ ] Train YOLOv5n

[ ] Train YOLOv8n

[ ] Train YOLOv11n

[ ] Evaluate YOLO models

[ ] Train ResNet50

[ ] Train EfficientNet-B0

[ ] Evaluate CNN models

[ ] Select best models

[ ] Implement Flow 1

[ ] Implement Flow 2

[ ] Build Web Benchmark System

[ ] Final Evaluation Report
