"""
SlumNet: hybrid CNN + Transformer segmentation network with attention,
multi-scale fusion, a Mixture-of-Experts refinement stage, and a
U-Net-style decoder with attention gates.

Pipeline:
    image
      -> CNN backbone (4 levels)         [local texture: roofs, roads]
      -> Transformer branch (4 levels)   [long-range context: layout]
      -> per-level fuse(CNN, Transformer): FPN | BiFPN | CrossAttention
      -> per-level attention (CBAM/etc.) [focus on dense settlements]
      -> per-level Mixture-of-Experts     [specialized cue extraction]
      -> U-Net decoder w/ attention gates -> segmentation logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.cnn_backbone import CNNBackbone
from models.transformers.vit_transformer import TransformerBranch
from models.attention.attention_modules import build_attention
from models.decoder.fusion import build_fusion, CrossAttentionFusion
from models.experts.moe import MixtureOfExperts
from models.decoder.unet_decoder import UNetDecoder


class SlumNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.num_classes = cfg["data"]["num_classes"]
        self.use_transformer = m.get("use_transformer", True)
        self.use_moe = m.get("use_moe", True)
        fusion_channels = m.get("fusion_channels", 256)

        self.backbone = CNNBackbone(m["backbone"], pretrained=m.get("backbone_pretrained", True))
        cnn_channels = self.backbone.out_channels  # 4 levels, shallow->deep

        if self.use_transformer:
            self.transformer = TransformerBranch(m["transformer"], out_dim=fusion_channels)

        # Project CNN features to fusion_channels so both branches match before fusion
        self.cnn_proj = nn.ModuleList([
            nn.Conv2d(c, fusion_channels, 1) for c in cnn_channels
        ])

        self.fusion_type = m.get("fusion", "bifpn").lower()
        fused_in_channels = [fusion_channels] * len(cnn_channels)
        if self.fusion_type == "cross_attention":
            self.cross_fusion = build_fusion("cross_attention", fused_in_channels, fusion_channels)
        else:
            self.pyramid_fusion = build_fusion(self.fusion_type, fused_in_channels, fusion_channels)

        self.attentions = nn.ModuleList([
            build_attention(m.get("attention", "cbam"), fusion_channels) for _ in cnn_channels
        ])

        if self.use_moe:
            moe_cfg = m.get("moe", {})
            self.moe_blocks = nn.ModuleList([
                MixtureOfExperts(
                    fusion_channels,
                    num_experts=moe_cfg.get("num_experts", 5),
                    expert_names=moe_cfg.get("expert_names"),
                    top_k=moe_cfg.get("top_k", 2),
                ) for _ in cnn_channels
            ])

        dec_cfg = m.get("decoder", {})
        self.decoder = UNetDecoder(
            pyramid_channels=fusion_channels,
            num_classes=self.num_classes,
            decoder_channels=tuple(dec_cfg.get("decoder_channels", [256, 128, 64, 32])),
            use_attention_gates=dec_cfg.get("use_attention_gates", True),
            use_residual_blocks=dec_cfg.get("use_residual_blocks", True),
        )

        self._last_moe_gates = None  # cached for XAI / expert-usage reporting

    def forward(self, x, return_aux: bool = False):
        target_size = x.shape[-2:]

        cnn_feats = self.backbone(x)  # shallow -> deep
        cnn_feats = [proj(f) for proj, f in zip(self.cnn_proj, cnn_feats)]

        if self.use_transformer:
            trans_feats = self.transformer(x)
            trans_feats = [
                F.interpolate(tf, size=cf.shape[-2:], mode="bilinear", align_corners=False)
                for tf, cf in zip(trans_feats, cnn_feats)
            ]
        else:
            trans_feats = cnn_feats

        if self.fusion_type == "cross_attention":
            fused = [ca(cf, tf) for ca, cf, tf in zip(self.cross_fusion, cnn_feats, trans_feats)]
        else:
            combined = [cf + tf for cf, tf in zip(cnn_feats, trans_feats)]
            fused = self.pyramid_fusion(combined)

        attended = [attn(f) for attn, f in zip(self.attentions, fused)]

        if self.use_moe:
            moe_out, gates = [], []
            for moe, f in zip(self.moe_blocks, attended):
                out, g = moe(f, return_gates=True)
                moe_out.append(out)
                gates.append(g)
            self._last_moe_gates = gates
            pyramid = moe_out
        else:
            pyramid = attended

        logits, pre_head_feats = self.decoder(pyramid, target_size)

        if return_aux:
            aux = {
                "pyramid_feats": pyramid,
                "pre_head_feats": pre_head_feats,
                "moe_gates": self._last_moe_gates,
            }
            return logits, aux
        return logits

    def get_expert_usage(self) -> list[dict] | None:
        """Average per-expert activation for the last forward pass (for the XAI dashboard)."""
        if not self.use_moe or self._last_moe_gates is None:
            return None
        return [moe.expert_activation_summary(g) for moe, g in zip(self.moe_blocks, self._last_moe_gates)]


def build_model(cfg: dict) -> SlumNet:
    return SlumNet(cfg)
