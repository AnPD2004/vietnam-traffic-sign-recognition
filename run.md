uv pip uninstall torch torchvision torchaudio
uv add torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130

python -m src.train.train_yolo --data data/vn-traffic-signs/data.yaml --epochs 100 --imgsz 640 --batch 16

python -m src.infer.infer_flow1_yolo_yolo --model artifacts/yolo_detect_cls/flow1_train/weights/best.pt --input path/to/image_or_video --save --json-out artifacts/yolo_detect_cls/flow1_predict/result.json

uvicorn api.main:app --host 0.0.0.0 --port 8000