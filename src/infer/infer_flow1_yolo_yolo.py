from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for Flow 1 (YOLO detect + YOLO predict)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO model, usually best.pt.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image, folder, video, webcam id, or stream URL.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image.")
    parser.add_argument(
        "--device",
        default=None,
        help='Device, e.g. "0", "cpu" (default: auto).',
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualized prediction outputs.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save txt labels in YOLO format.",
    )
    parser.add_argument(
        "--project",
        default="artifacts/yolo_detect_cls",
        help="Output directory for save options.",
    )
    parser.add_argument(
        "--name",
        default="flow1_predict",
        help="Run name under output directory.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write prediction JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    from src.pipelines.flow1_pipeline import Flow1YoloYoloPipeline

    args = parse_args()
    pipeline = Flow1YoloYoloPipeline(model_path=args.model, device=args.device)
    result = pipeline.predict(
        source=args.input,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        save=args.save,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
    )

    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(pretty_json)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(pretty_json, encoding="utf-8")


if __name__ == "__main__":
    main()

