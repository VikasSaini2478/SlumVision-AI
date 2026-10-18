"""
SlumNet training entry point.

Usage:
    python train.py --config configs/config.yaml
"""

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils.common import load_config, set_seed, get_device, save_checkpoint
from dataset.dataset import SlumSegmentationDataset
from dataset.augmentation import build_train_augmentations, build_val_augmentations
from models.slumnet import build_model
from losses.losses import CombinedLoss, compute_class_weights
from metrics.metrics import SegmentationMetrics


def build_dataloaders(cfg):
    patch_size = cfg["data"]["patch_size"]

    train_tf = build_train_augmentations(cfg, patch_size)
    val_tf = build_val_augmentations(cfg, patch_size)

    full_dataset = SlumSegmentationDataset(
        cfg["data"]["images_dir"],
        cfg["data"]["masks_dir"],
        cfg,
        transform=None,
        split="train"
    )

    total_size = len(full_dataset)
    n_val = int(total_size * cfg["data"]["val_split"])
    n_train = total_size - n_val

    indices = torch.randperm(
        total_size,
        generator=torch.Generator().manual_seed(cfg["project"]["seed"])
    ).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip, use_amp):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss, _ = criterion(logits, masks)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, class_names, use_amp):
    model.eval()
    running_loss = 0.0
    metric_tracker = SegmentationMetrics(num_classes, class_names)
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss, _ = criterion(logits, masks)
        running_loss += loss.item()
        metric_tracker.update(logits, masks)

    metrics = metric_tracker.compute()
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)
    print(f"Using device: {device}")

    train_loader, val_loader, train_dataset = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(cfg).to(device)

    class_weights = None
    if cfg["loss"].get("class_weights") is None:
        print("Computing class weights from training data...")
        class_weights = compute_class_weights(train_dataset, cfg["data"]["num_classes"]).to(device)
    else:
        class_weights = torch.tensor(cfg["loss"]["class_weights"], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    criterion = CombinedLoss(cfg, class_weights=class_weights)

    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"] and device.type == "cuda")

    log_dir = cfg["training"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_miou = 0.0
    patience_counter = 0

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cfg["training"]["grad_clip"], cfg["training"]["amp"] and device.type == "cuda",
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            cfg["data"]["num_classes"], cfg["data"]["class_names"],
            cfg["training"]["amp"] and device.type == "cuda",
        )
        scheduler.step()

        print(f"[Epoch {epoch+1}/{cfg['training']['epochs']}] "
              f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
              f"mIoU={val_metrics['miou']:.4f} F1={val_metrics['mean_f1']:.4f}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("metrics/mIoU", val_metrics["miou"], epoch)
        writer.add_scalar("metrics/F1", val_metrics["mean_f1"], epoch)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            patience_counter = 0
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": best_miou,
                "config": cfg,
            }, cfg["training"]["checkpoint_dir"], "best_model.pth")
            print(f"  -> New best mIoU {best_miou:.4f}, checkpoint saved.")
        else:
            patience_counter += 1
            if not cfg["training"]["save_best_only"]:
                save_checkpoint({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                }, cfg["training"]["checkpoint_dir"], f"epoch_{epoch+1}.pth")

        if patience_counter >= cfg["training"]["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch+1} (no improvement for "
                  f"{cfg['training']['early_stopping_patience']} epochs).")
            break

    writer.close()
    print(f"Training complete. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
