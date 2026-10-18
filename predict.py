"""
Run inference on a single satellite image: produces a segmentation mask,
confidence scores, an overlay visualization, GIS exports (GeoTIFF /
Shapefile / GeoJSON), area/cluster statistics, and an optional Grad-CAM
explanation heatmap.

Usage:
    python predict.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth \
        --image path/to/scene.tif --out-dir outputs/prediction_001
"""

import argparse
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from utils.common import load_config, get_device, load_checkpoint, denormalize, overlay_mask
from dataset.augmentation import build_inference_augmentations
from dataset.dataset import _read_image
from models.slumnet import build_model
from utils.gis_utils import save_geotiff, save_geojson, compute_area_statistics
from utils.xai import GradCAM, overlay_heatmap


def run_inference(model, image_rgb: np.ndarray, cfg: dict, device):
    patch_size = cfg["data"]["patch_size"]
    tf = build_inference_augmentations(cfg["data"]["mean"], cfg["data"]["std"], patch_size)
    transformed = tf(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(transformed)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        confidence = torch.max(probs, dim=1).values.squeeze(0).cpu().numpy()

    pred_resized = cv2.resize(pred, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    confidence_resized = cv2.resize(confidence, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return pred_resized, confidence_resized, transformed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default="outputs/prediction")
    parser.add_argument("--pixel-area-m2", type=float, default=None,
                         help="Ground sample area per pixel in m^2, if known, for real-world area stats")
    parser.add_argument("--gradcam", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)
    device = get_device(cfg)

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    image_rgb, meta = _read_image(args.image)
    pred_mask, confidence, input_tensor = run_inference(model, image_rgb, cfg, device)

    # Save raw mask + overlay
    cv2.imwrite(os.path.join(args.out_dir, "mask.png"), pred_mask * 255 if cfg["data"]["num_classes"] == 2 else pred_mask)
    overlay = overlay_mask(image_rgb, pred_mask, color=(255, 0, 0))
    cv2.imwrite(os.path.join(args.out_dir, "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Confidence map
    conf_u8 = (confidence * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, "confidence.png"), conf_u8)

    # GIS exports (only meaningful with a real transform/CRS; falls back to pixel space otherwise)
    transform = meta.get("transform")
    crs = str(meta.get("crs")) if meta.get("crs") else "EPSG:4326"
    try:
        save_geotiff(pred_mask, os.path.join(args.out_dir, "prediction.tif"), transform=transform, crs=crs)
        save_geojson(pred_mask, os.path.join(args.out_dir, "prediction.geojson"), transform=transform, crs=crs,
                      min_cluster_pixels=cfg["gis"]["min_cluster_pixels"])
    except ImportError as e:
        print(f"[warn] Skipping vector/raster GIS export: {e}")

    # Area / cluster statistics
    stats = compute_area_statistics(pred_mask, pixel_area_m2=args.pixel_area_m2,
                                     min_cluster_pixels=cfg["gis"]["min_cluster_pixels"])
    with open(os.path.join(args.out_dir, "area_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Explainability
    if args.gradcam:
        target_layer = model.backbone.model.feature_info  # placeholder resolution below
        try:
            layer = dict(model.backbone.named_modules())["model.layer4"]
        except KeyError:
            layer = list(model.backbone.model.children())[-1]
        cam = GradCAM(model, layer)
        heatmap = cam(input_tensor, target_class=1 if cfg["data"]["num_classes"] == 2 else int(pred_mask.max()))
        heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
        cam_overlay = overlay_heatmap(image_rgb, heatmap_resized)
        cv2.imwrite(os.path.join(args.out_dir, "gradcam.png"), cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

    expert_usage = model.get_expert_usage()
    if expert_usage:
        with open(os.path.join(args.out_dir, "expert_usage.json"), "w") as f:
            json.dump(expert_usage, f, indent=2)

    print(f"Prediction complete. Outputs written to {args.out_dir}")
    print(json.dumps({k: v for k, v in stats.items() if k != "clusters"}, indent=2))


if __name__ == "__main__":
    main()
