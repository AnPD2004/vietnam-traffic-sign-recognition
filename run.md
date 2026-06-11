python -m src.train.train_yolo --data data/vn-traffic-signs/data.yaml --epochs 100 --imgsz 640 --batch 16
uv run python -m src.build_crops.build_crops_for_cnn --overwrite
uv run python -m src.train.train_cnn --config configs/cnn.yaml