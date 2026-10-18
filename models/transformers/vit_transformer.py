"""
Transformer branch to capture long-range spatial relationships and
neighborhood layout patterns (e.g. informal settlement sprawl vs. planned
grid layouts) that a purely convolutional receptive field misses.

Wraps a timm Swin/ViT model in features_only mode and projects its output
to a fixed embedding dimension for fusion with the CNN backbone features.
"""

import torch
import torch.nn as nn
import timm


class TransformerBranch(nn.Module):

    SUPPORTED = {
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "vit_small": "vit_small_patch16_224",
        "segformer_b0": "mit_b0",  # SegFormer's MixTransformer encoder, exposed in timm as mit_*
    }

    def __init__(self, name: str = "swin_tiny", pretrained: bool = True,
                 out_dim: int = 256, img_size: int = 256):
        super().__init__()
        if name not in self.SUPPORTED:
            raise ValueError(f"Unsupported transformer '{name}'. Choose from {list(self.SUPPORTED)}")

        model_name = self.SUPPORTED[name]
        try:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, features_only=True, img_size=img_size,
            )
        except TypeError:
            # Some timm models (e.g. mit_b0) don't accept img_size for features_only
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        feat_channels = self.model.feature_info.channels()
        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_dim, kernel_size=1) for c in feat_channels
        ])
        self.out_dim = out_dim

    def forward(self, x):
        feats = self.model(x)
        # Swin returns (B, H, W, C) in some timm versions -> normalize to (B, C, H, W)
        norm_feats = []
        for f in feats:
            if f.dim() == 4 and f.shape[1] not in (f.shape[-1],) and f.shape[-1] > f.shape[1]:
                pass
            if f.dim() == 4 and f.shape[-1] == self._infer_channel_last(f):
                f = f.permute(0, 3, 1, 2).contiguous()
            norm_feats.append(f)

        projected = [proj(f) for proj, f in zip(self.proj, norm_feats)]
        return projected  # list of feature maps, shallow -> deep, all with out_dim channels

    @staticmethod
    def _infer_channel_last(f: torch.Tensor) -> int:
        # Heuristic: Swin's raw output is (B, H, W, C) with C typically >> H, W spatial dims
        return f.shape[-1] if f.shape[-1] > f.shape[1] else -1
