"""
models/gam.py — Geometric Attention Module (GAM).

v5.0.0: Residual connection + bidirectional tanh gating (identity-safe).

Uses a geometry embedding to reweight SA layer features via a learned attention delta.
The delta values (alpha) are computed per-forward-pass — not fixed model weights.
"""

import torch
import torch.nn as nn


class GeometricAttentionModule(nn.Module):
    """
    Reweight PointNet++ SA features using a geometry embedding.

    v5.0.0 changes (fixed):
      - Residual connection: output = sa_feat + alpha * sa_feat
      - Bidirectional gating: α ∈ [-0.5, 0.5] via 0.5 * tanh(...)
      - Net effect: output = sa_feat * (1 + alpha) ∈ [0.5x, 1.5x] sa_feat
        * α =  0  → identity (safe default kalau geom tidak informatif)
        * α < 0  → suppress channel (kalau geom indikasi channel kurang relevan)
        * α > 0  → amplify channel (kalau geom indikasi channel sangat relevan)

    Args:
        sa_ch   : number of channels in the SA feature (C)
        geom_ch : dim of the geometry embedding (default 64)

    Input:
        sa_feat  : (B, N, C)  — output of a SA layer
        geom_emb : (B, geom_ch) — output of GeometryEncoder

    Output:
        (B, N, C) — reweighted features
    """

    def __init__(self, sa_ch: int, geom_ch: int = 64):
        super().__init__()
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_ch, sa_ch),
            nn.ReLU(inplace=True),
        )
        self.attn_gate = nn.Sequential(
            nn.Linear(sa_ch * 2, sa_ch),
            nn.ReLU(inplace=True),
            nn.Linear(sa_ch, sa_ch),
            nn.Tanh(),  # v5.0.0: tanh instead of sigmoid
        )

    def forward(self, sa_feat: torch.Tensor, geom_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sa_feat  : (B, N, C)
            geom_emb : (B, geom_ch)

        Returns:
            reweighted : (B, N, C)
        """
        B, N, C = sa_feat.shape

        # Project geometry to SA channel dim
        geom_proj = self.geom_proj(geom_emb)               # (B, C)
        geom_exp  = geom_proj.unsqueeze(1).expand(B, N, C) # (B, N, C)

        # Compute bidirectional attention delta
        concat = torch.cat([sa_feat, geom_exp], dim=-1)    # (B, N, 2C)
        tanh_out = self.attn_gate(concat)                  # (B, N, C) ∈ [-1, 1]

        # v5.0.0 (fixed): bidirectional scaling
        # α ∈ [-0.5, 0.5] → output ∈ [0.5x, 1.5x] sa_feat
        # α = 0 = identity safe (kalau Tanh saturate ke 0 → output tetap = sa_feat)
        alpha = 0.5 * tanh_out

        return sa_feat + alpha * sa_feat
