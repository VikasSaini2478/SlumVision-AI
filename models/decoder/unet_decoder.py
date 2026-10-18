"""
U-Net-style decoder that upsamples fused multi-scale features back to
full resolution, using skip connections, residual conv blocks, and
attention gates (to suppress irrelevant skip-connection activations,
e.g. water bodies or shadow regions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al., 2018): uses the coarser decoder
    signal `g` to gate the finer skip-connection features `x`, so the
    decoder only propagates skip activations relevant to the current
    reconstruction context.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_resized = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.relu(self.W_g(g_resized) + self.W_x(x))
        attn = self.psi(attn)
        return x * attn


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 use_attention_gate: bool = True, use_residual: bool = True):
        super().__init__()
        self.use_attention_gate = use_attention_gate and skip_channels > 0
        if self.use_attention_gate:
            self.attn_gate = AttentionGate(in_channels, skip_channels, out_channels // 2)

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        fuse_channels = out_channels + skip_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock(out_channels) if use_residual else nn.Identity()

    def forward(self, x, skip=None):
        x_up = self.upconv(x)
        if skip is not None:
            if self.use_attention_gate:
                skip = self.attn_gate(x, skip)
            if x_up.shape[-2:] != skip.shape[-2:]:
                x_up = F.interpolate(x_up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x_up = torch.cat([x_up, skip], dim=1)
        return self.residual(self.fuse_conv(x_up))


class UNetDecoder(nn.Module):
    """
    Consumes the fused pyramid (shallow -> deep, all `pyramid_channels`
    wide) and produces a full-resolution feature map + a segmentation
    logits head.
    """

    def __init__(self, pyramid_channels: int, num_classes: int,
                 decoder_channels=(256, 128, 64, 32),
                 use_attention_gates: bool = True, use_residual_blocks: bool = True):
        super().__init__()
        levels = len(decoder_channels)
        in_ch = pyramid_channels
        self.blocks = nn.ModuleList()
        for i in range(levels):
            skip_ch = pyramid_channels if i < levels - 1 else 0
            out_ch = decoder_channels[i]
            self.blocks.append(DecoderBlock(
                in_ch, skip_ch, out_ch,
                use_attention_gate=use_attention_gates,
                use_residual=use_residual_blocks,
            ))
            in_ch = out_ch

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(in_ch // 2, num_classes, kernel_size=1)

    def forward(self, pyramid_feats, target_size):
        """pyramid_feats: list shallow -> deep. Decoding proceeds deep -> shallow."""
        feats = list(reversed(pyramid_feats))  # deep -> shallow
        x = feats[0]
        for i, block in enumerate(self.blocks):
            skip = feats[i + 1] if i + 1 < len(feats) else None
            x = block(x, skip)

        x = self.final_upsample(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        logits = self.head(x)
        return logits, x  # logits for loss, x as pre-head features for XAI
