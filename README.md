# Vietnamese Traffic Sign Recognition System

## Overview

This project focuses on building a Vietnamese Traffic Sign Recognition System using Deep Learning.

The system is designed to detect and classify 52 Vietnamese traffic sign classes from images and videos.

The research evaluates two different approaches:

### Pipeline 1 – YOLO End-to-End

A single YOLO model performs:

* Traffic sign detection
* Traffic sign classification

### Pipeline 2 – YOLO + CNN Hybrid

YOLO is responsible for object localization, while CNN performs the final classification.

The objective is to compare the trade-off between:

* Accuracy
* Inference speed
* Model size
* Real-world deployment capability

---

## Research Objectives

The project aims to:

* Build a traffic sign recognition system for Vietnam.
* Train and evaluate multiple YOLO models.
* Train and evaluate multiple CNN classifiers.
* Compare detection and classification performance.
* Compare End-to-End and Hybrid pipelines.
* Deploy a web-based benchmark system.

---

## Models

### Detection Models

The following YOLO models are evaluated:

* YOLOv5n
* YOLOv8n
* YOLOv11n

Metrics:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95
* FPS
* Model Size

---

### Classification Models

The following CNN models are evaluated:

* ResNet50
* EfficientNet-B0

Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Inference Time

---

## Dataset

### Detection Dataset

Format:

images/
labels/

Number of classes:

* 52 Vietnamese traffic sign classes

Dataset size:

* Approximately 3,216 images

---

### Classification Dataset

Generated from detection labels by cropping traffic sign regions.

Structure:

crops/
├── train/
├── val/
└── test/

Each class is stored in an independent directory.

---

## System Architecture

### Pipeline 1

Image

↓

YOLO

↓

Bounding Box + Class

---

### Pipeline 2

Image

↓

YOLO Detector

↓

Crop Traffic Sign

↓

CNN Classifier

↓

Final Class

---

## Technology Stack

Programming Language:

* Python

Deep Learning Framework:

* PyTorch
* Torchvision

Object Detection:

* Ultralytics YOLO

Web Framework:

* Flask

Utilities:

* OpenCV
* NumPy
* Pandas
* Matplotlib

---

## Project Structure

src/
├── build_crops/
├── pipeline1/
├── pipeline2/
├── infer/
├── pipelines/
├── web/
└── common/

artifacts/
├── pipeline1/
└── pipeline2/

configs/
├── pipeline1.yaml
└── pipeline2.yaml

datasets/

reports/

---

## Benchmark Strategy

Stage 1:

Compare:

* YOLOv5n
* YOLOv8n
* YOLOv11n

Stage 2:

Compare:

* ResNet50
* EfficientNet-B0

Stage 3:

Compare:

* Best YOLO End-to-End model
* Best YOLO + CNN Hybrid model

---

## Expected Outputs

* Trained YOLO models
* Trained CNN models
* Classification dataset
* Web benchmark system
* Experimental report
* Performance comparison report

---

## Applications

* ADAS
* Autonomous Vehicles
* Intelligent Transportation Systems
* Traffic Monitoring Systems
* Real-Time Traffic Sign Recognition
