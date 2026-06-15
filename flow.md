# Project Flow - Vietnamese Traffic Sign Recognition

## Research Workflow

The project consists of three benchmarking stages and two independent inference pipelines.

---

# Stage 1 - YOLO Benchmark

## Objective

Select the best YOLO model for traffic sign detection.

## Models

* YOLOv5n
* YOLOv8n
* YOLOv11n

## Fixed Configuration

* Epochs = 100
* Batch Size = 16
* Image Size = 640
* Learning Rate = 0.01
* Mosaic = 1

## Evaluation Metrics

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95
* FPS
* Inference Time
* Model Size

## Output

Best YOLO model

---

# Stage 1.1 - YOLO Hyperparameter Study

## Objective

Analyze the impact of training hyperparameters on YOLO performance.

## Parameters

### Image Size

* 416
* 640
* 800

### Learning Rate

* 0.001
* 0.005
* 0.01

### Batch Size

* 8
* 16
* 32

### Mosaic Augmentation

* Mosaic = 0
* Mosaic = 1

## Evaluation Metrics

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95
* FPS
* Inference Time

## Output

Optimal YOLO configuration

---

# Stage 2 - CNN Benchmark

## Objective

Select the best CNN classifier.

## Models

* ResNet50
* EfficientNet-B0

## Fixed Configuration

* Epochs = 50
* Batch Size = 32
* Learning Rate = 0.001

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Inference Time
* Model Size

## Output

Best CNN model

---

# Stage 2.1 - CNN Hyperparameter Study

## Objective

Analyze the impact of CNN training strategies.

## Parameters

### Learning Rate

* 0.0001
* 0.001
* 0.01

### Transfer Learning Strategy

* Frozen Backbone
* Fine-Tuning

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Training Loss
* Validation Loss
* Inference Time

## Output

Optimal CNN configuration

---

# Stage 3 - Pipeline Benchmark

## Objective

Compare the two recognition pipelines.

## Compare

### Pipeline 1

YOLO End-to-End

### Pipeline 2

YOLO + CNN Hybrid

## Evaluation Metrics

* Detection Performance
* Classification Performance
* Precision
* Recall
* F1-score
* mAP
* FPS
* Inference Time
* Model Size

## Output

Best Overall Pipeline

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
