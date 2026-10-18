"""
Segmentation evaluation metrics for SlumNet: mIoU, Dice, pixel accuracy,
precision/recall/F1, boundary IoU, and per-class breakdowns / confusion
matrices for reporting.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def _to_numpy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t


class SegmentationMetrics:
    """Accumulates confusion-matrix stats across batches, then reports final metrics."""

    def __init__(self, num_classes: int, class_names: list[str] | None = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        preds_np = _to_numpy(preds).reshape(-1)
        target_np = _to_numpy(target).reshape(-1)
        cm = confusion_matrix(target_np, preds_np, labels=list(range(self.num_classes)))
        self.confusion += cm

    def _iou_per_class(self):
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom = tp + fp + fn
        return np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)

    def _dice_per_class(self):
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)

    def _precision_recall_f1(self):
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                        out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
        return precision, recall, f1

    def compute(self) -> dict:
        iou = self._iou_per_class()
        dice = self._dice_per_class()
        precision, recall, f1 = self._precision_recall_f1()
        pixel_acc = np.diag(self.confusion).sum() / max(self.confusion.sum(), 1)

        per_class = {
            self.class_names[i]: {
                "iou": float(iou[i]), "dice": float(dice[i]),
                "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
            } for i in range(self.num_classes)
        }

        return {
            "miou": float(np.mean(iou)),
            "mean_dice": float(np.mean(dice)),
            "pixel_accuracy": float(pixel_acc),
            "mean_precision": float(np.mean(precision)),
            "mean_recall": float(np.mean(recall)),
            "mean_f1": float(np.mean(f1)),
            "per_class": per_class,
            "confusion_matrix": self.confusion.tolist(),
        }


def boundary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, dilation: int = 2) -> float:
    """
    Boundary IoU (Cheng et al., 2021): restricts IoU computation to a thin
    band around object boundaries, which is more sensitive to boundary
    quality than standard IoU — relevant for accurate slum-area estimates.
    """
    import cv2

    def _boundary(mask):
        mask_u8 = (mask > 0).astype(np.uint8)
        eroded = cv2.erode(mask_u8, np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8))
        return mask_u8 - eroded

    pred_b = _boundary(pred_mask)
    gt_b = _boundary(gt_mask)
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0
