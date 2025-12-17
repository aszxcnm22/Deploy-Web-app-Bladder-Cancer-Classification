import os
import numpy as np
import torch
import cv2
import base64
from io import BytesIO
from PIL import Image
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from torchvision import transforms
from ultralytics import YOLO

# ======================
# FastAPI setup
# ======================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ======================
# Load models
# ======================
yolo_model = YOLO("Model/best.pt")

cls_model_path = "Model/resnet101_web_model_v.1.pt"
class_names = ["T1", "T2", "T3", "T4"]

cls_model = torch.load(cls_model_path, map_location="cpu")
cls_model.eval()

# ======================
# Image transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ======================
# Stage colors (BGR)
# ======================
stage_colors = {
    "T1": (0, 200, 0),      # Green
    "T2": (0, 255, 255),    # Yellow
    "T3": (0, 128, 255),    # Orange
    "T4": (0, 0, 255),      # Red
}

# ======================
# Detection + Classification
# ======================
def detect_and_predict(image_np: np.ndarray):
    orig_image = image_np.copy()

    results = yolo_model(image_np)
    boxes = (
        results[0].boxes.xyxy.cpu().numpy()
        if results[0].boxes is not None
        else []
    )

    predicted_labels = []
    all_probs = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = image_np[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_pil = Image.fromarray(roi)
        roi_tensor = transform(roi_pil).unsqueeze(0)

        with torch.no_grad():
            logits = cls_model(roi_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        all_probs.append(probs.tolist())

        idx = int(np.argmax(probs))
        stage = class_names[idx]
        conf = float(probs[idx])
        predicted_labels.append((stage, conf))

        # ===== Draw bounding box (FIXED) =====
        color = stage_colors.get(stage, (255, 255, 255))
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 2)

        # ===== Label with background =====
        label = f"{stage}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            orig_image,
            (x1, y1 - th - 8),
            (x1 + tw + 6, y1),
            color,
            -1
        )

        cv2.putText(
            orig_image,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # ===== Convert result image to base64 =====
    _, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
    )
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return predicted_labels, all_probs, img_base64

# ======================
# Routes
# ======================
# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/index.html", response_class=HTMLResponse) 
async def read_index(request: Request): 
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/contact.html", response_class=HTMLResponse)
async def read_contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.post("/contact.html", response_class=HTMLResponse)
async def process_contact(request: Request, file: UploadFile = File(...)):

    image_bytes = await file.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)

    predictions, all_probs, result_img_b64 = detect_and_predict(image_np)

    # ===== Average probabilities =====
    if all_probs:
        avg_probs = np.mean(np.array(all_probs), axis=0)
        percentages = {
            cls: float(p) * 100
            for cls, p in zip(class_names, avg_probs)
        }
        dominant_stage = class_names[int(np.argmax(avg_probs))]
    else:
        percentages = {cls: 0.0 for cls in class_names}
        dominant_stage = None

    html = f"""
    <style>
        .preview-image {{
            max-width: 500px;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .center {{
            text-align: center;
            font-size: 2.5rem;
        }}
        .badge {{
            display: inline-block;
            padding: 8px 18px;
            border-radius: 14px;
            font-weight: bold;
            margin: 6px 0;
            color: white;
        }}
        .t1 {{ background: #2ecc71; }}
        .t2 {{ background: #f1c40f; }}
        .t3 {{ background: #e67e22; }}
        .t4 {{ background: #e74c3c; }}
    </style>

    <div class="center">
        <h4>Result Image</h4>
        <img src="data:image/jpeg;base64,{result_img_b64}" class="preview-image">
    </div>

    <div class="center" style="margin-top:20px;">
        <h5>Predictions per bounding box:</h5>
    """

    if predictions:
        for cls, conf in predictions:
            html += f"""
            <div class="badge {cls.lower()}">
                {cls} — {conf:.2f}
            </div>
            """
    else:
        html += "<p>No detection</p>"

    html += "<h5 style='margin-top:25px;'>Stage Probabilities:</h5>"

    for cls in class_names:
        html += f"""
        <div class="badge {cls.lower()}">
            {cls}: {percentages[cls]:.1f}%
        </div>
        """

    if dominant_stage:
        html += f"""
        <div style="margin-top:30px;">
            <h5>Dominant Stage</h5>
            <div class="badge {dominant_stage.lower()}"
                 style="font-size:2.5rem; padding:12px 28px;">
                {dominant_stage}
            </div>
        </div>
        """

    return HTMLResponse(content=html)
