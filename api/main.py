from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io, base64, tempfile, os
import numpy as np

app = FastAPI(title="Vn TS Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the exported model (update path if needed)
MODEL_PATH = os.path.join("runs", "detect", "artifacts", "yolo_detect_cls", "flow1_train", "weights", "best.pt")
model = YOLO(MODEL_PATH)

# Load Vietnamese class names from data files (indexed order)
VI_CLASSES = []
EN_CLASSES = []
try:
    en_path = os.path.join('data', 'vn-traffic-signs', 'classes_en.txt')
    vi_path = os.path.join('data', 'vn-traffic-signs', 'classes_vie.txt')
    if os.path.exists(vi_path):
        with open(vi_path, 'r', encoding='utf-8') as f:
            VI_CLASSES = [l.strip() for l in f.readlines() if l.strip()]
    if os.path.exists(en_path):
        with open(en_path, 'r', encoding='utf-8') as f:
            EN_CLASSES = [l.strip() for l in f.readlines() if l.strip()]
except Exception:
    VI_CLASSES = []
    EN_CLASSES = []


def results_to_contract(results):
    detections = []
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = results.names if hasattr(results, "names") else model.model.names

    for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "class_id": int(cl),
            "class_name": names.get(cl, str(cl)),
            "confidence": float(c),
            "source": "yolo",
        })
    return detections


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        res_list = model.predict(source=tmp_path, imgsz=640, conf=0.25, device='cpu')
        # model.predict returns a list (one per image)
        results = res_list[0]

        detections = results_to_contract(results)

        # create annotated image with Vietnamese labels
        # open original image to draw on
        pil_img = Image.open(tmp_path).convert('RGB')
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except Exception:
            font = ImageFont.load_default()

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                # draw rectangle
                draw.rectangle([x1i, y1i, x2i, y2i], outline=(0,255,0), width=2)
                # label text: Vietnamese if available else English
                vi_label = None
                if 0 <= int(cl) < len(VI_CLASSES):
                    vi_label = VI_CLASSES[int(cl)]
                en_label = None
                if 0 <= int(cl) < len(EN_CLASSES):
                    en_label = EN_CLASSES[int(cl)]
                if vi_label:
                    text = f"{vi_label} {c:.2f}"
                else:
                    text = f"{en_label or str(cl)} {c:.2f}"

                # compute text size with textbbox if available, else fallback to font.getsize
                try:
                    bbox_txt = draw.textbbox((0, 0), text, font=font)
                    text_w = bbox_txt[2] - bbox_txt[0]
                    text_h = bbox_txt[3] - bbox_txt[1]
                except Exception:
                    try:
                        text_w, text_h = font.getsize(text)
                    except Exception:
                        text_w, text_h = (len(text) * 6, 12)

                y0 = max(0, y1i - text_h - 4)
                text_bg = [x1i, y0, x1i + text_w + 4, y1i]
                draw.rectangle(text_bg, fill=(0,0,255))
                draw.text((x1i + 2, y0 + 2), text, fill=(255,255,255), font=font)

        img = pil_img
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        annotated_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        payload = {
            "flow": "flow1_yolo_yolo",
            "detections": detections,
            "annotated_image_b64": annotated_b64,
        }
        return JSONResponse(payload)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
