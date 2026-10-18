"""
Multi-scale feature fusion. Combines the CNN backbone's shallow, high-
resolution features (fine building/roof texture) with the transformer
branch's deep, context-rich features (neighborhood-scale layout), across
4 pyramid levels, so both small and large slum regions are detected well.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FPNFusion(nn.Module):
    """Classic top-down Feature Pyramid Network fusion."""

    def __init__(self, in_channels_list, out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.smooth = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels) for _ in in_channels_list
        ])

    def forward(self, feats):
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up
        return [smooth(l) for smooth, l in zip(self.smooth, laterals)]


class BiFPNBlock(nn.Module):
    """
    Simplified single-layer Bidirectional FPN: top-down pass then
    bottom-up pass with learnable weighted feature fusion, so both large
    dense slum clusters and small isolated ones survive the fusion.
    """

    def __init__(self, channels: int, num_levels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.num_levels = num_levels
        self.td_weights = nn.Parameter(torch.ones(num_levels - 1, 2))
        self.bu_weights = nn.Parameter(torch.ones(num_levels - 1, 3))
        self.td_conv = nn.ModuleList([ConvBNReLU(channels, channels) for _ in range(num_levels - 1)])
        self.bu_conv = nn.ModuleList([ConvBNReLU(channels, channels) for _ in range(num_levels - 1)])
        self.down = nn.ModuleList([nn.MaxPool2d(2) for _ in range(num_levels - 1)])

    def forward(self, feats):
        # Top-down
        td = [feats[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + self.eps)
            up = F.interpolate(td[0], size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            fused = w[0] * feats[i] + w[1] * up
            td.insert(0, self.td_conv[i](fused))

        # Bottom-up
        bu = [td[0]]
        for i in range(1, self.num_levels):
            w = F.relu(self.bu_weights[i - 1])
            w = w / (w.sum() + self.eps)
            down = self.down[i - 1](bu[-1])
            down = F.interpolate(down, size=td[i].shape[-2:], mode="bilinear", align_corners=False)
            fused = w[0] * feats[i] + w[1] * td[i] + w[2] * down
            bu.append(self.bu_conv[i - 1](fused))

        return bu


class BiFPNFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels: int = 256, num_layers: int = 2):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.blocks = nn.ModuleList([
            BiFPNBlock(out_channels, len(in_channels_list)) for _ in range(num_layers)
        ])

    def forward(self, feats):
        x = [lat(f) for lat, f in zip(self.lateral, feats)]
        for block in self.blocks:
            x = block(x)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Fuses CNN features (queries) with transformer features (keys/values)
    at each pyramid level via cross-attention, letting local texture cues
    attend to long-range context.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, cnn_feat, trans_feat):
        B, C, H, W = cnn_feat.shape
        trans_feat = F.interpolate(trans_feat, size=(H, W), mode="bilinear", align_corners=False)

        q = self.norm_q(cnn_feat.flatten(2).transpose(1, 2))
        kv = self.norm_kv(trans_feat.flatten(2).transpose(1, 2))
        out, _ = self.attn(q, kv, kv)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return self.proj_out(out + cnn_feat)


def build_fusion(name: str, in_channels_list, out_channels: int = 256):
    name = name.lower()
    if name == "fpn":
        return FPNFusion(in_channels_list, out_channels)
    if name == "bifpn":
        return BiFPNFusion(in_channels_list, out_channels)
    if name == "cross_attention":
        # Note: cross-attention fusion is applied per-level between two
        # equal-length feature lists (see SlumNet.forward for usage).
        return nn.ModuleList([CrossAttentionFusion(out_channels) for _ in in_channels_list])
    raise ValueError(f"Unknown fusion type: {name}")
