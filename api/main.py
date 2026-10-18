"""
FastAPI backend for SlumNet.

Endpoints:
    POST /upload      -> stores an uploaded image, returns an image_id
    POST /predict      -> runs inference on a stored (or directly uploaded) image
    GET  /report/{id}  -> returns the JSON area/cluster report for a prior prediction
    GET  /health        -> liveness check

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
import io
import json
import uuid

import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

from utils.common import load_config, get_device, load_checkpoint, overlay_mask
from dataset.augmentation import build_inference_augmentations
from models.slumnet import build_model
from utils.gis_utils import compute_area_statistics, save_geojson
from predict import run_inference

app = FastAPI(title="SlumNet API", version="1.0.0")

CFG_PATH = os.environ.get("SLUMNET_CONFIG", "configs/config.yaml")
CKPT_PATH = os.environ.get("SLUMNET_CHECKPOINT", "checkpoints/best_model.pth")
UPLOAD_DIR = "outputs/api_uploads"
RESULT_DIR = "outputs/api_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

_state = {"model": None, "cfg": None, "device": None}


def _lazy_load_model():
    if _state["model"] is None:
        cfg = load_config(CFG_PATH)
        device = get_device(cfg)
        model = build_model(cfg).to(device)
        if os.path.exists(CKPT_PATH):
            load_checkpoint(CKPT_PATH, model, map_location=device)
        else:
            print(f"[warn] Checkpoint not found at {CKPT_PATH}; using randomly-initialized weights.")
        model.eval()
        _state.update({"model": model, "cfg": cfg, "device": device})
    return _state["model"], _state["cfg"], _state["device"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".png"
    path = os.path.join(UPLOAD_DIR, image_id + ext)
    contents = await file.read()
    with open(path, "wb") as f:
        f.write(contents)
    return {"image_id": image_id, "path": path}


@app.post("/predict")
async def predict(file: UploadFile = File(...), pixel_area_m2: float | None = None):
    model, cfg, device = _lazy_load_model()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_rgb = np.array(image)

    pred_mask, confidence, _ = run_inference(model, image_rgb, cfg, device)

    result_id = str(uuid.uuid4())
    result_dir = os.path.join(RESULT_DIR, result_id)
    os.makedirs(result_dir, exist_ok=True)

    mask_path = os.path.join(result_dir, "mask.png")
    overlay_path = os.path.join(result_dir, "overlay.png")
    cv2.imwrite(mask_path, pred_mask * 255 if cfg["data"]["num_classes"] == 2 else pred_mask)
    overlay = overlay_mask(image_rgb, pred_mask)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    stats = compute_area_statistics(pred_mask, pixel_area_m2=pixel_area_m2,
                                     min_cluster_pixels=cfg["gis"]["min_cluster_pixels"])
    report_path = os.path.join(result_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)

    try:
        save_geojson(pred_mask, os.path.join(result_dir, "prediction.geojson"),
                      min_cluster_pixels=cfg["gis"]["min_cluster_pixels"])
        geojson_available = True
    except ImportError:
        geojson_available = False

    return JSONResponse({
        "result_id": result_id,
        "mean_confidence": float(np.mean(confidence)),
        "area_statistics": stats,
        "geojson_available": geojson_available,
        "download": {
            "mask": f"/report/{result_id}/mask",
            "overlay": f"/report/{result_id}/overlay",
            "geojson": f"/report/{result_id}/geojson" if geojson_available else None,
        },
    })


@app.get("/report/{result_id}")
def get_report(result_id: str):
    report_path = os.path.join(RESULT_DIR, result_id, "report.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Result not found")
    with open(report_path) as f:
        return json.load(f)


@app.get("/report/{result_id}/mask")
def get_mask(result_id: str):
    path = os.path.join(RESULT_DIR, result_id, "mask.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Mask not found")
    return FileResponse(path, media_type="image/png")


@app.get("/report/{result_id}/overlay")
def get_overlay(result_id: str):
    path = os.path.join(RESULT_DIR, result_id, "overlay.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(path, media_type="image/png")


@app.get("/report/{result_id}/geojson")
def get_geojson(result_id: str):
    path = os.path.join(RESULT_DIR, result_id, "prediction.geojson")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="GeoJSON not found")
    return FileResponse(path, media_type="application/geo+json")
