from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
from PIL import Image
import io
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

try:
    from torchvision import models, transforms
except Exception as e:
    models = None
    transforms = None
    _tv_error = str(e)

DATA_ROOT = Path('artifacts/data_ingestion')
MODEL_PATH_PTH = Path('artifacts/training/model.pth')
MODEL_PATH = Path('artifacts/training/model.h5')  # legacy fallback path
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.75  # predictions below this are returned as "unknown"

app = FastAPI(title="Cats vs Dogs Classifier (PyTorch)")

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
transform = None
if transforms is not None:
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

model: Optional[nn.Module] = None


def get_class_names():
    """Read class names from the data directory; fall back to ['cats_set', 'dogs_set'] if unavailable."""
    if not DATA_ROOT.exists():
        return ["cats_set", "dogs_set"]
    classes = sorted([d for d in os.listdir(DATA_ROOT) if (DATA_ROOT / d).is_dir()])
    return classes if classes else ["cats_set", "dogs_set"]


def build_model(num_classes: int = 2):
    """Instantiate VGG16 with the final classifier layer replaced to match the number of classes."""
    if models is None:
        return None
    base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    in_features = base.classifier[6].in_features
    base.classifier[6] = nn.Linear(in_features, num_classes)
    return base


def try_load_state_dict(m: nn.Module):
    """Attempt to load trained weights into the model, trying weights_only=True first then False."""
    ckpt_path = MODEL_PATH_PTH if MODEL_PATH_PTH.exists() else (MODEL_PATH if MODEL_PATH.exists() else None)
    if ckpt_path is None:
        return m
    sd = None
    try:
        sd = torch.load(ckpt_path, weights_only=True, map_location=device)
    except Exception:
        try:
            try:
                import torchvision
                torch.serialization.add_safe_globals([torchvision.models.vgg.VGG])
            except Exception:
                pass
            sd = torch.load(ckpt_path, weights_only=False, map_location=device)
        except Exception:
            sd = None
    if sd is not None:
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        # Try strict first (should match training architecture), then relax
        try:
            m.load_state_dict(sd, strict=True)
        except Exception:
            try:
                m.load_state_dict(sd, strict=False)
            except Exception:
                pass
    return m


def load_model():
    """Build the model, load trained weights, set eval mode, and move to the active device."""
    global model
    if models is None:
        model = None
        return
    classes = get_class_names()
    num_classes = len(classes)
    m = build_model(num_classes=num_classes)
    m = try_load_state_dict(m)
    m.eval()
    model = m.to(device)


load_model()


@app.get("/health")
async def health():
    """Return server liveness, model status, device, and class names."""
    return {
        "status": "ok",
        "model_loaded": bool(model),
        "device": str(device),
        "classes": get_class_names(),
        "torchvision_error": None if models else _tv_error
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the HTML upload form for browser-based inference."""
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Cats vs Dogs Classifier</title>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; }
          .card { max-width: 560px; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
          img { max-width: 300px; border: 1px solid #eee; border-radius: 6px; }
          .result { margin-top: 1rem; white-space: pre-wrap; font-family: monospace; }
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Upload an image</h2>
          <form id="form" enctype="multipart/form-data" method="post" action="/predict">
            <input id="file" type="file" name="file" accept="image/*" required />
            <button type="submit">Classify</button>
          </form>
          <img id="preview" alt="preview" />
          <div id="out" class="result"></div>
        </div>
        <script>
          const form = document.getElementById('form');
          const out = document.getElementById('out');
          const fileInput = document.getElementById('file');
          const preview = document.getElementById('preview');
          fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
              const reader = new FileReader();
              reader.onload = ev => { preview.src = ev.target.result; };
              reader.readAsDataURL(e.target.files[0]);
            }
          });
          form.addEventListener('submit', async (e) => {
            e.preventDefault();
            out.textContent = 'Predicting...';
            const fd = new FormData(form);
            try {
              const res = await fetch('/predict', { method: 'POST', body: fd });
              const json = await res.json();
              out.textContent = JSON.stringify(json, null, 2);
            } catch (err) {
              out.textContent = String(err);
            }
          });
        </script>
      </body>
    </html>
    """


@app.get("/reload")
async def reload_model():
    """Hot-reload the trained weights from disk without restarting the server."""
    load_model()
    return {"model_loaded": bool(model)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run inference on an uploaded image; return label and per-class probabilities, or 'unknown' if confidence is low."""
    try:
        if models is None or transform is None:
            return JSONResponse({"error": "torchvision not installed: " + _tv_error}, status_code=500)
        mdl = model
        if not isinstance(mdl, nn.Module):
            return JSONResponse({"error": "Model not loaded."}, status_code=500)
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = mdl(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # Constant outputs indicate the model has not been trained
        if np.allclose(probs, probs[0], atol=1e-6):
            return JSONResponse({
                "error": "Model outputs appear constant. Ensure a trained .pth is saved and loaded.",
                "probs": {"class_0": float(probs[0]), "class_1": float(probs[-1])}
            }, status_code=500)
        classes = get_class_names()
        if len(classes) != len(probs):
            classes = ["cat", "dog"][:len(probs)]
        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])
        label = classes[pred_idx]
        if pred_conf < CONFIDENCE_THRESHOLD:
            return JSONResponse({
                "label": "unknown",
                "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
                "note": f"low confidence < {CONFIDENCE_THRESHOLD}"
            })
        return JSONResponse({
            "label": label,
            "probs": {classes[i]: float(probs[i]) for i in range(len(classes))}
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
