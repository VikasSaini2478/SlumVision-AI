"""
Mixture-of-Experts (MoE) block. Each expert is a lightweight conv branch
specialized (by training signal, not architecture constraint) toward a
particular visual cue relevant to informal settlements:

    - dense_roof_pattern:       tight, irregular roof textures
    - informal_road_network:    narrow, non-grid pathways
    - building_density:         structure packing / footprint ratio
    - vegetation_separation:    distinguishing green space from built-up
    - open_space:               planned open areas / non-slum voids

A learnable gating network produces per-pixel (or per-token) soft weights
over the experts; a top-k sparse mode is supported to keep compute in
check on large feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, channels: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or channels
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))  # residual expert


class GatingNetwork(nn.Module):
    """Per-pixel soft gate producing a distribution over experts."""

    def __init__(self, channels: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, num_experts, 1),
        )
        self.spatial_gate = nn.Conv2d(channels, num_experts, 1)

    def forward(self, x):
        # combine a global (per-image) prior with a per-pixel spatial gate
        global_logits = self.gate(x)                      # (B, E, 1, 1)
        spatial_logits = self.spatial_gate(x)               # (B, E, H, W)
        logits = spatial_logits + global_logits
        return F.softmax(logits, dim=1)                     # (B, E, H, W)


class MixtureOfExperts(nn.Module):
    def __init__(self, channels: int, num_experts: int = 5,
                 expert_names: list[str] | None = None, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_names = expert_names or [f"expert_{i}" for i in range(num_experts)]
        self.experts = nn.ModuleList([Expert(channels) for _ in range(num_experts)])
        self.gating = GatingNetwork(channels, num_experts)

    def forward(self, x, return_gates: bool = False):
        gates = self.gating(x)  # (B, E, H, W), softmax over experts per pixel

        if self.top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(gates, self.top_k, dim=1)
            sparse_gates = torch.zeros_like(gates).scatter_(1, topk_idx, topk_vals)
            gates = sparse_gates / (sparse_gates.sum(dim=1, keepdim=True) + 1e-8)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, C, H, W)
        gates_expanded = gates.unsqueeze(2)  # (B, E, 1, H, W)
        fused = (expert_outputs * gates_expanded).sum(dim=1)  # (B, C, H, W)

        if return_gates:
            return fused, gates
        return fused

    def expert_activation_summary(self, gates: torch.Tensor) -> dict:
        """Average activation per expert across the batch/spatial dims — useful for XAI reporting."""
        avg = gates.mean(dim=(0, 2, 3))  # (E,)
        return {name: float(avg[i]) for i, name in enumerate(self.expert_names)}
