# detector/utils/predictor.py
import os
import io
import uuid
from PIL import Image, ImageDraw, ImageFont
from django.conf import settings
from ultralytics import YOLO

# paths
BASE_APP = os.path.dirname(os.path.dirname(__file__))   # detector/
MODEL_PATH = os.path.abspath(os.path.join(BASE_APP, 'weights', 'mask_detector.pt'))
RESULTS_DIR = os.path.join(settings.MEDIA_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# load YOLO model once (no GitHub, local only)
_model = YOLO(MODEL_PATH)

def draw_boxes(img_pil, detections, names=None):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for x1, y1, x2, y2, conf, cls in detections:
        label = names[int(cls)] if names and int(cls) in names else str(int(cls))
        text = f"{label} {conf:.2f}"

        # Color logic
        if label.lower() in ["mask", "with_mask", "wearing_mask"]:
            color = "green"
        else:
            color = "red"

        # Get text size safely
        try:
            bbox = font.getbbox(text)  # (x0, y0, x1, y1)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textlength(text, font=font), font.size

        # Draw rectangle + label
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
        draw.text((x1, y1 - th), text, fill="white", font=font)

    return img_pil



def predict_and_save(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    detections = []
    names = _model.names  # class names from YOLO model

    # run YOLO inference
    results = _model.predict(img, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            detections.append([x1, y1, x2, y2, conf, cls])

    # draw and save
    result_img = img.copy()
    draw_boxes(result_img, detections, names)
    fn = f"result_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(RESULTS_DIR, fn)
    result_img.save(save_path, quality=90)

    return os.path.join('results', fn)   # relative to MEDIA_ROOT
