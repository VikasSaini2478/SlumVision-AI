"""
CNN backbone for local texture extraction (roof patterns, narrow roads,
dense building clusters). Wraps timm feature-extractor backbones and
exposes a consistent multi-scale feature-map API.
"""

import timm
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    Returns a list of 4 feature maps at strides [4, 8, 16, 32], each shaped
    (B, C_i, H_i, W_i), consistent regardless of the underlying timm model.
    """

    SUPPORTED = {
        "resnet50": "resnet50",
        "efficientnet_b4": "efficientnet_b4",
        "convnext_tiny": "convnext_tiny",
    }

    def __init__(self, name: str = "resnet50", pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        if name not in self.SUPPORTED:
            raise ValueError(f"Unsupported backbone '{name}'. Choose from {list(self.SUPPORTED)}")

        self.model = timm.create_model(
            self.SUPPORTED[name],
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=(1, 2, 3, 4),  # strides 4, 8, 16, 32
        )
        self.out_channels = self.model.feature_info.channels()

    def forward(self, x):
        return self.model(x)  # list of 4 tensors, shallow -> deep
