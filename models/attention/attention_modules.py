"""
Attention modules used to focus the network on dense, irregular built-up
areas (slums) and suppress irrelevant background such as open fields,
water, or well-planned road grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Applies MHSA over the flattened spatial tokens of a feature map. Useful
    on the deepest, lowest-resolution feature map where the token count is
    manageable.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mhsa = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        tokens_norm = self.norm(tokens)
        attn_out, _ = self.mhsa(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out
        return tokens.transpose(1, 2).reshape(B, C, H, W)


def build_attention(name: str, channels: int):
    name = name.lower()
    if name == "cbam":
        return CBAM(channels)
    if name == "channel":
        return ChannelAttention(channels)
    if name == "spatial":
        return SpatialAttention()
    if name == "mhsa":
        return MultiHeadSelfAttention(channels)
    raise ValueError(f"Unknown attention type: {name}")
