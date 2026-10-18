"""
Evaluate a trained SlumNet checkpoint on a held-out test set and produce
a metrics report (mIoU, Dice, precision/recall/F1, boundary IoU, and
per-class confusion matrix).

Usage:
    python test.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import load_config, get_device, load_checkpoint
from dataset.dataset import SlumSegmentationDataset
from dataset.augmentation import build_val_augmentations
from models.slumnet import build_model
from metrics.metrics import SegmentationMetrics, boundary_iou
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--test-images-dir", default=None, help="Override cfg data.images_dir for a held-out test split")
    parser.add_argument("--test-masks-dir", default=None)
    parser.add_argument("--out", default="outputs/test_report.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)

    images_dir = args.test_images_dir or cfg["data"]["images_dir"]
    masks_dir = args.test_masks_dir or cfg["data"]["masks_dir"]

    test_tf = build_val_augmentations(cfg, cfg["data"]["patch_size"])
    test_set = SlumSegmentationDataset(images_dir, masks_dir, cfg, transform=test_tf, split="test")
    test_loader = DataLoader(test_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
                              num_workers=cfg["training"]["num_workers"])

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    metric_tracker = SegmentationMetrics(cfg["data"]["num_classes"], cfg["data"]["class_names"])
    b_ious = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            metric_tracker.update(logits, masks)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = masks.cpu().numpy()
            for p, g in zip(preds, gts):
                b_ious.append(boundary_iou(p.astype(np.uint8), g.astype(np.uint8)))

    report = metric_tracker.compute()
    report["boundary_iou"] = float(np.mean(b_ious)) if b_ious else 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "confusion_matrix"}, indent=2))
    print(f"Full report saved to {args.out}")


if __name__ == "__main__":
    main()
