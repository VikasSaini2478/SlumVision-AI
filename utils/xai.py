"""
Explainability utilities: Grad-CAM over a chosen backbone layer, attention
heatmap extraction from the model's CBAM/MoE gates, and feature-map
visualization — so a user can see *why* a region was labeled a slum.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    """
    Standard Grad-CAM: hooks a target layer's forward activations and
    backward gradients, then combines them into a class-discriminative
    localization map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        image: (1, C, H, W) preprocessed tensor, requires_grad not needed on input.
        Returns a (H, W) heatmap normalized to [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(image)
        score = logits[:, target_class].mean()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image_rgb: (H, W, 3) uint8. heatmap: (H, W) float in [0,1]."""
    heatmap_u8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, colored, alpha, 0)
    return overlay


def moe_gate_heatmap(gates: torch.Tensor, expert_idx: int, target_size) -> np.ndarray:
    """
    Extract a single expert's spatial activation map from the MoE gating
    tensor (B, E, H, W) at a given pyramid level, resized to target_size.
    """
    gate = gates[0, expert_idx].detach().cpu().numpy()
    gate = cv2.resize(gate, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    gate = (gate - gate.min()) / (gate.max() - gate.min() + 1e-8)
    return gate


def visualize_feature_maps(feature_map: torch.Tensor, num_channels: int = 16) -> np.ndarray:
    """
    Build a grid image of the top-activating channels of a feature map,
    (1, C, H, W) -> grid uint8 RGB image, for qualitative feature inspection.
    """
    fmap = feature_map[0].detach().cpu()
    activations = fmap.mean(dim=(1, 2))
    top_idx = torch.topk(activations, min(num_channels, fmap.shape[0])).indices

    tiles = []
    for idx in top_idx:
        ch = fmap[idx].numpy()
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        ch_u8 = np.uint8(255 * ch)
        ch_color = cv2.applyColorMap(ch_u8, cv2.COLORMAP_VIRIDIS)
        tiles.append(ch_color)

    grid_cols = int(np.ceil(np.sqrt(len(tiles))))
    grid_rows = int(np.ceil(len(tiles) / grid_cols))
    h, w = tiles[0].shape[:2]
    grid = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, grid_cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    return cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
