from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import io, base64, tempfile, os, shutil
import numpy as np
import asyncio
import time
from uuid import uuid4
from functools import lru_cache

app = FastAPI(title="Vn TS Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the exported model (update path if needed)
MODEL_PATH = os.path.join("runs", "detect", "artifacts", "yolo_detect_cls", "flow1_train", "weights", "best.pt")
# choose default device: use GPU if available
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {DEFAULT_DEVICE}")
if DEFAULT_DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
model = YOLO(MODEL_PATH)
if DEFAULT_DEVICE != 'cpu':
    try:
        model.to(DEFAULT_DEVICE)
    except Exception:
        pass
    try:
        model.fuse()
    except Exception:
        pass

# Load Vietnamese class names from data files (indexed order)
VI_CLASSES = []
EN_CLASSES = []
VIDEO_JOBS = {}
VIDEO_BATCH_SIZE = max(1, int(os.getenv('VIDEO_BATCH_SIZE', '1')))
VIDEO_IMGSZ = max(640, int(os.getenv('VIDEO_IMGSZ', '640')))
USE_HALF = DEFAULT_DEVICE != 'cpu'
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


def get_result_arrays(results):
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None, None

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = results.names if hasattr(results, "names") else model.model.names
    return xyxy, conf, cls, names


def results_to_contract(results):
    detections = []
    xyxy, conf, cls, names = get_result_arrays(results)
    if xyxy is None:
        return detections

    for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "class_id": int(cl),
            "class_name": names.get(cl, str(cl)),
            "confidence": float(c),
            "source": "yolo",
        })
    return detections


def frame_to_b64(frame_bgr, quality=80, max_width=640):
    # Resize to reduce encoding cost and bandwidth
    try:
        h, w = frame_bgr.shape[:2]
    except Exception:
        return None
    if w > max_width:
        scale = max_width / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        frame = frame_bgr

    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def annotate_bgr_frame(img_bgr, results):
    xyxy, conf, cls, names = get_result_arrays(results)
    if xyxy is None:
        return img_bgr

    for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

        vi_label = VI_CLASSES[int(cl)] if 0 <= int(cl) < len(VI_CLASSES) else None
        en_label = EN_CLASSES[int(cl)] if 0 <= int(cl) < len(EN_CLASSES) else None
        text = f"{(vi_label or en_label or str(cl))} {c:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(0, y1i - text_h - baseline - 4)
        cv2.rectangle(img_bgr, (x1i, y0), (x1i + text_w + 4, y1i), (0, 0, 255), -1)
        cv2.putText(img_bgr, text, (x1i + 2, y1i - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img_bgr


def extract_detections(results):
    return results_to_contract(results)


def predict_frame_batch(batch_frames_rgb):
    with torch.inference_mode():
        return model.predict(
            source=batch_frames_rgb,
            imgsz=VIDEO_IMGSZ,
            conf=0.25,
            device=DEFAULT_DEVICE,
            half=USE_HALF,
            verbose=False,
        )


@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.file.seek(0)
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    job_id = str(uuid4())
    VIDEO_JOBS[job_id] = {
        "path": tmp_path,
        "filename": file.filename,
    }
    print(f"[HTTP] uploaded video job_id={job_id} path={tmp_path} bytes={os.path.getsize(tmp_path)}", flush=True)
    return JSONResponse({"job_id": job_id, "filename": file.filename})


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.file.seek(0)
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        with torch.inference_mode():
            res_list = model.predict(source=tmp_path, imgsz=640, conf=0.25, device=DEFAULT_DEVICE)
        # model.predict returns a list (one per image)
        results = res_list[0]

        detections = results_to_contract(results)

        img = cv2.imread(tmp_path)
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        img = annotate_bgr_frame(img, results)
        buf = io.BytesIO()
        ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError('failed to encode annotated image')
        buf.write(enc.tobytes())
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


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.file.seek(0)
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    out_tmp = None
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return JSONResponse({"error": "cannot open video"}, status_code=400)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fd, out_tmp = tempfile.mkstemp(suffix='.mp4')
        os.close(out_fd)
        writer = cv2.VideoWriter(out_tmp, fourcc, fps, (w, h))

        frame_idx = 0
        batch_frames_rgb = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames_rgb.append(img_rgb)

            if len(batch_frames_rgb) < VIDEO_BATCH_SIZE:
                continue

            results_list = predict_frame_batch(batch_frames_rgb)
            for frame_rgb, results in zip(batch_frames_rgb, results_list):
                out_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out_frame = annotate_bgr_frame(out_frame, results)
                writer.write(out_frame)
                frame_idx += 1

            batch_frames_rgb = []

        if batch_frames_rgb:
            results_list = predict_frame_batch(batch_frames_rgb)
            for frame_rgb, results in zip(batch_frames_rgb, results_list):
                out_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out_frame = annotate_bgr_frame(out_frame, results)
                writer.write(out_frame)
                frame_idx += 1

        cap.release()
        writer.release()

        # return annotated video as base64
        with open(out_tmp, 'rb') as f:
            b = f.read()
        annotated_b64 = base64.b64encode(b).decode('utf-8')
        return JSONResponse({"annotated_video_b64": annotated_b64, "frames": frame_idx})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        if out_tmp and os.path.exists(out_tmp):
            try:
                os.remove(out_tmp)
            except Exception:
                pass


@app.websocket("/ws/video_stream/{job_id}")
async def websocket_video_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        job = VIDEO_JOBS.get(job_id)
        if not job or not os.path.exists(job["path"]):
            await websocket.send_json({"type": "error", "message": f"job not found: {job_id}"})
            await websocket.close()
            return

        tmp_path = job["path"]
        print(f"[WS] start stream job_id={job_id} path={tmp_path}", flush=True)

        # send an immediate placeholder so the frontend shows something right away
        placeholder = Image.new('RGB', (640, 360), (20, 20, 20))
        placeholder_draw = ImageDraw.Draw(placeholder)
        try:
            placeholder_font = ImageFont.truetype("arial.ttf", size=24)
        except Exception:
            placeholder_font = ImageFont.load_default()
        placeholder_draw.text((24, 24), "Video received. Decoding...", fill=(255, 255, 255), font=placeholder_font)
        buf = io.BytesIO()
        placeholder.save(buf, format='JPEG')
        await websocket.send_json({"type": "started_preview", "frame": 0, "image_b64": base64.b64encode(buf.getvalue()).decode('utf-8')})

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            print("[WS] cv2.VideoCapture failed to open video", flush=True)
            await websocket.send_json({"type": "error", "message": "cannot open video"})
            await websocket.close()
            return

        await websocket.send_json({"type": "started"})
        print("[WS] video opened", flush=True)

        frame_idx = 0
        batch_frames_rgb = []
        batch_frame_indices = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[WS] cap.read() ended at frame {frame_idx}", flush=True)
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # send a quick preview of the raw frame before inference
                if frame_idx == 0:
                    # offload quick preview encoding to threadpool to avoid blocking
                    preview_b64 = await asyncio.to_thread(frame_to_b64, frame, 70, 640)
                    if preview_b64:
                        await websocket.send_json({"type": "preview", "frame": frame_idx, "image_b64": preview_b64})
                        await asyncio.sleep(0)

                batch_frames_rgb.append(img_rgb)
                batch_frame_indices.append(frame_idx)
                frame_idx += 1

                if len(batch_frames_rgb) < VIDEO_BATCH_SIZE:
                    continue
                # measure inference and encoding times
                t0 = time.time()
                results_list = predict_frame_batch(batch_frames_rgb)
                # ensure GPU finished (if used) before measuring
                try:
                    if DEFAULT_DEVICE == 'cuda':
                        torch.cuda.synchronize()
                except Exception:
                    pass
                t1 = time.time()
                predict_time = t1 - t0

                enc_total = 0.0
                for batch_frame_idx, frame_rgb, results in zip(batch_frame_indices, batch_frames_rgb, results_list):
                    annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    annotated_bgr = annotate_bgr_frame(annotated_bgr, results)
                    t_enc0 = time.time()
                    annotated_b64 = await asyncio.to_thread(frame_to_b64, annotated_bgr, 60, 640)
                    t_enc1 = time.time()
                    enc_time = t_enc1 - t_enc0
                    enc_total += enc_time
                    if not annotated_b64:
                        continue
                    await websocket.send_json({
                        "type": "frame",
                        "frame": batch_frame_idx,
                        "image_b64": annotated_b64,
                        "detections": extract_detections(results),
                    })
                    if batch_frame_idx % 10 == 0:
                        print(f"[WS] sent frame {batch_frame_idx}", flush=True)

                # log batch timings
                try:
                    n = len(results_list) or 1
                    print(f"[WS] batch frames={n} predict_time={predict_time:.3f}s total_encode={enc_total:.3f}s avg_encode={enc_total/n:.3f}s", flush=True)
                except Exception:
                    pass

                batch_frames_rgb = []
                batch_frame_indices = []

            if batch_frames_rgb:
                t0 = time.time()
                results_list = predict_frame_batch(batch_frames_rgb)
                try:
                    if DEFAULT_DEVICE == 'cuda':
                        torch.cuda.synchronize()
                except Exception:
                    pass
                t1 = time.time()
                predict_time = t1 - t0

                enc_total = 0.0
                for batch_frame_idx, frame_rgb, results in zip(batch_frame_indices, batch_frames_rgb, results_list):
                    annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    annotated_bgr = annotate_bgr_frame(annotated_bgr, results)
                    t_enc0 = time.time()
                    annotated_b64 = await asyncio.to_thread(frame_to_b64, annotated_bgr, 60, 640)
                    t_enc1 = time.time()
                    enc_time = t_enc1 - t_enc0
                    enc_total += enc_time
                    if not annotated_b64:
                        continue
                    await websocket.send_json({
                        "type": "frame",
                        "frame": batch_frame_idx,
                        "image_b64": annotated_b64,
                        "detections": extract_detections(results),
                    })
                    if batch_frame_idx % 10 == 0:
                        print(f"[WS] sent frame {batch_frame_idx}", flush=True)

                try:
                    n = len(results_list) or 1
                    print(f"[WS] final-batch frames={n} predict_time={predict_time:.3f}s total_encode={enc_total:.3f}s avg_encode={enc_total/n:.3f}s", flush=True)
                except Exception:
                    pass

            await websocket.send_json({"type": "done", "frames": frame_idx})
            print(f"[WS] done, frames={frame_idx}", flush=True)
        finally:
            cap.release()
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            VIDEO_JOBS.pop(job_id, None)

    except WebSocketDisconnect:
        return
    except Exception as e:
        print(f"[WS] error: {e}", flush=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
