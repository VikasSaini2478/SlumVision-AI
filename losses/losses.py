
"""
Loss functions for SlumNet semantic segmentation.

Includes:
- Dice Loss
- Focal Loss
- Cross Entropy Loss
- AMP-safe Boundary Loss
- Combined Loss
- Class-weight computation

Designed for CUDA + mixed precision training on low-VRAM GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def _to_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert segmentation target:

        [B, H, W]

    into:

        [B, C, H, W]
    """
    return (
        F.one_hot(target.long(), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        probs = F.softmax(logits, dim=1)

        target_oh = _to_one_hot(
            target,
            self.num_classes,
        ).to(
            device=logits.device,
            dtype=probs.dtype,
        )

        dims = (0, 2, 3)

        intersection = torch.sum(
            probs * target_oh,
            dims,
        )

        cardinality = torch.sum(
            probs + target_oh,
            dims,
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            cardinality + self.smooth
        )

        return 1.0 - dice.mean()


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()

        self.gamma = gamma

        if weight is not None:
            self.register_buffer(
                "weight",
                weight.detach().clone().float(),
            )
        else:
            self.weight = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        weight = self.weight

        if weight is not None:
            weight = weight.to(
                device=logits.device,
                dtype=torch.float32,
            )

        # CE in float32 is more stable with AMP
        ce = F.cross_entropy(
            logits.float(),
            target.long(),
            weight=weight,
            reduction="none",
        )

        pt = torch.exp(-ce)

        focal = (
            (1.0 - pt) ** self.gamma
        ) * ce

        return focal.mean()


# ============================================================
# Boundary Loss
# ============================================================

class BoundaryLoss(nn.Module):
    """
    AMP-safe Sobel boundary loss.

    Important:
    The Sobel kernels are registered as buffers and are explicitly
    moved to the same device and dtype as the input.

    The actual Sobel computation is performed in float32 to avoid
    CUDA/AMP dtype mismatch and improve numerical stability.
    """

    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        self.num_classes = num_classes

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        sobel_y = sobel_x.t().contiguous()

        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3),
        )

        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3),
        )

    def _gradient_magnitude(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # IMPORTANT:
        # Always perform boundary computation in float32.
        # This prevents:
        #
        # CUDA HalfTensor vs CPU FloatTensor
        # CUDA FloatTensor vs CPU FloatTensor
        #
        # dtype/device mismatch errors.
        # ----------------------------------------------------

        x = x.float()

        device = x.device

        sobel_x = self.sobel_x.to(
            device=device,
            dtype=torch.float32,
        )

        sobel_y = self.sobel_y.to(
            device=device,
            dtype=torch.float32,
        )

        B, C, H, W = x.shape

        # Process every channel independently.
        x = x.reshape(
            B * C,
            1,
            H,
            W,
        )

        gx = F.conv2d(
            x,
            sobel_x,
            padding=1,
        )

        gy = F.conv2d(
            x,
            sobel_y,
            padding=1,
        )

        magnitude = torch.sqrt(
            gx * gx
            + gy * gy
            + 1e-6
        )

        return magnitude.reshape(
            B,
            C,
            H,
            W,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Disable autocast for this section.
        #
        # The model can still use AMP everywhere else.
        # ----------------------------------------------------

        device = logits.device

        with torch.autocast(
            device_type=device.type,
            enabled=False,
        ):

            probs = F.softmax(
                logits.float(),
                dim=1,
            )

            target_oh = _to_one_hot(
                target,
                self.num_classes,
            ).to(
                device=device,
                dtype=torch.float32,
            )

            pred_edges = self._gradient_magnitude(
                probs
            )

            gt_edges = self._gradient_magnitude(
                target_oh
            )

            loss = F.l1_loss(
                pred_edges,
                gt_edges,
            )

        return loss


# ============================================================
# Combined Loss
# ============================================================

class CombinedLoss(nn.Module):
    """
    Weighted combination of:

        Dice
        Focal
        Cross Entropy
        Boundary

    Example:

        total =
            dice_weight * dice
            + focal_weight * focal
            + ce_weight * ce
            + boundary_weight * boundary
    """

    def __init__(
        self,
        cfg: dict,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        loss_cfg = cfg["loss"]

        num_classes = cfg["data"]["num_classes"]

        self.weights = loss_cfg.get(
            "weights",
            {},
        )

        self.use_dice = loss_cfg.get(
            "use_dice",
            True,
        )

        self.use_focal = loss_cfg.get(
            "use_focal",
            True,
        )

        self.use_ce = loss_cfg.get(
            "use_ce",
            True,
        )

        self.use_boundary = loss_cfg.get(
            "use_boundary",
            True,
        )

        # ----------------------------------------------------
        # Loss modules
        # ----------------------------------------------------

        if self.use_dice:
            self.dice = DiceLoss(
                num_classes=num_classes,
            )

        if self.use_focal:
            self.focal = FocalLoss(
                gamma=2.0,
                weight=class_weights,
            )

        if self.use_ce:

            if class_weights is not None:
                self.register_buffer(
                    "class_weights",
                    class_weights.detach()
                    .clone()
                    .float(),
                )
            else:
                self.class_weights = None

        if self.use_boundary:
            self.boundary = BoundaryLoss(
                num_classes=num_classes,
            )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ):

        total = logits.new_tensor(0.0)

        parts = {}

        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        if self.use_dice:

            loss = self.dice(
                logits,
                target,
            )

            weight = self.weights.get(
                "dice",
                1.0,
            )

            total = total + weight * loss

            parts["dice"] = loss.detach().item()

        # ----------------------------------------------------
        # Focal
        # ----------------------------------------------------

        if self.use_focal:

            loss = self.focal(
                logits,
                target,
            )

            weight = self.weights.get(
                "focal",
                1.0,
            )

            total = total + weight * loss

            parts["focal"] = loss.detach().item()

        # ----------------------------------------------------
        # Cross Entropy
        # ----------------------------------------------------

        if self.use_ce:

            if self.class_weights is not None:

                class_weights = self.class_weights.to(
                    device=logits.device,
                    dtype=torch.float32,
                )

            else:
                class_weights = None

            # Float32 CE improves AMP stability.
            loss = F.cross_entropy(
                logits.float(),
                target.long(),
                weight=class_weights,
            )

            weight = self.weights.get(
                "ce",
                0.5,
            )

            total = total + weight * loss

            parts["ce"] = loss.detach().item()

        # ----------------------------------------------------
        # Boundary
        # ----------------------------------------------------

        if self.use_boundary:

            loss = self.boundary(
                logits,
                target,
            )

            weight = self.weights.get(
                "boundary",
                0.5,
            )

            total = total + weight * loss

            parts["boundary"] = loss.detach().item()

        parts["total"] = total.detach().item()

        return total, parts


# ============================================================
# Class Weight Calculation
# ============================================================

@torch.no_grad()
def compute_class_weights(
    dataset,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute normalized inverse-frequency class weights.

    IMPORTANT:
    The returned tensor stays on CPU.
    train.py is responsible for moving it to CUDA.

    This avoids unnecessary CUDA allocations while scanning
    potentially tens of thousands of mask patches.
    """

    counts = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )

    print(
        f"Computing class weights from "
        f"{len(dataset)} training patches..."
    )

    for i in range(len(dataset)):

        sample = dataset[i]

        mask = sample["mask"]

        if not torch.is_tensor(mask):
            mask = torch.as_tensor(
                mask,
                dtype=torch.long,
            )

        mask = mask.detach().cpu().long()

        # bincount is significantly faster than
        # looping over every class.
        batch_counts = torch.bincount(
            mask.reshape(-1),
            minlength=num_classes,
        )

        counts += batch_counts.to(
            dtype=torch.float64
        )

    # Prevent division by zero.
    counts = torch.clamp(
        counts,
        min=1.0,
    )

    total_pixels = counts.sum()

    weights = (
        total_pixels
        / (
            num_classes
            * counts
        )
    )

    # Normalize so average weight = 1.
    weights = weights / weights.mean()

    weights = weights.float()

    print(
        f"Pixel counts: "
        f"{counts.tolist()}"
    )

    print(
        f"Class weights: "
        f"{weights.tolist()}"
    )

    # Sanity check.
    minority_class = int(
        torch.argmin(counts)
    )

    print(
        f"Minority class: "
        f"{minority_class}"
    )

    print(
        f"Minority class weight: "
        f"{weights[minority_class].item():.4f}"
    )

    return weights
