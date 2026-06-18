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
* Learning Rate = 0.005
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

Run counting:

* 15 benchmark cases
* 12 unique training/evaluation runs
* 3 reused baseline cases

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

Run counting:

* 7 benchmark cases
* 5 unique training/evaluation runs
* 2 reused baseline cases

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

├── pipeline1/

├── pipeline2/

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

## Completed

- [x] Dataset config exists: `data/vn-traffic-signs/data.yaml`
- [x] Class mapping exists: `data/vn-traffic-signs/classes.txt`
- [x] Pipeline 1 config exists: `configs/pipeline1.yaml`
- [x] Pipeline 1 refactor module exists: `src/pipeline1/`
- [x] Pipeline 1 CLI exists: `python -m src.pipeline1`
- [x] Pipeline 1 supports `--dry-run`, `--test`, `--force`, `--experiment`, `--run-id`, `--list-runs`
- [x] Pipeline 1 YOLO training wrapper supports model map, `lr0`, `mosaic`, named run artifacts
- [x] Pipeline 1 evaluation collects precision, recall, mAP@0.5, mAP@0.5:0.95, FPS, inference time, model size
- [x] Pipeline 1 best-run selection uses `mAP@0.5:0.95`, then `mAP@0.5`, then FPS
- [x] Pipeline 1 registry/report metadata exists under `artifacts/pipeline1/`
- [x] Pipeline 1 benchmark metadata covers 15 benchmark cases / 12 unique runs / 3 reused baseline cases
- [x] Crop builder script exists: `src/build_crops/build_crops_for_cnn.py`
- [x] Flow 1 inference class exists: `src/pipelines/flow1_pipeline.py`
- [x] Flow 2 inference class exists: `src/pipelines/flow2_pipeline.py`
- [x] Pipeline 2 config exists: `configs/pipeline2.yaml`
- [x] Pipeline 2 refactor module exists: `src/pipeline2/`
- [x] Pipeline 2 orchestrator exists: `python -m src.pipeline2`
- [x] CNN models use `resnet50` + `efficientnet_b0` (no `resnet18`)
- [x] Pipeline 1 weight files (`*.pt`) present; `artifacts/pipeline1/best/best.json` points to valid weights
- [x] Classification crop dataset present at `data/vn-traffic-signs/crops`
- [x] Pipeline 2 benchmark completed under `artifacts/pipeline2/` (registry, runs, report)

## Pending / Partial

- [ ] Inference CLI folder is not implemented: `src/infer/`
- [ ] Stage 3 benchmark comparing Pipeline 1 vs Pipeline 2 is not implemented
- [ ] Flask/web API endpoints are not implemented: `/predict/flow1`, `/predict/flow2`, `/benchmark`
