"""General-purpose utilities: config loading, reproducibility, checkpointing, overlay visualization."""

import os
import random
import yaml
import numpy as np
import torch
import cv2


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    want = cfg.get("project", {}).get("device", "cuda")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str = "best_model.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def denormalize(image: np.ndarray, mean, std) -> np.ndarray:
    """Reverse channel-wise normalization for visualization. image: (H, W, 3) float."""
    mean = np.array(mean)
    std = np.array(std)
    img = image * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    """Overlay a binary/class mask on an RGB image with a solid color blend."""
    overlay = image_rgb.copy()
    colored = np.zeros_like(image_rgb)
    colored[mask > 0] = color
    return cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)
