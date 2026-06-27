"""
models/geometry_encoder.py — MLP encoder untuk 13-dim fitur geometri.

Input: vektor fitur geometri (GEOMETRY_DIM=13) yang sudah di-z-score normalize.
Fitur: finger_lengths×5, palm_width, palm_height, palm_depth_std,
       finger_widths×4 (skip thumb), scan_distance_mm.

v5.0.0: Tambah LayerNorm di akhir untuk stabilisasi skala geom_emb sebelum GAM/fusion.
"""

import torch
import torch.nn as nn

from utils.dataset import GEOMETRY_DIM


class GeometryEncoder(nn.Module):
    """
    Encode vektor geometri (13-dim) menjadi embedding 64-dim.

    Architecture:
        Linear(13→64) → BN → ReLU
        Linear(64→64) → BN → ReLU
        Linear(64→64) → ReLU
        LayerNorm(64)   ← v5.0.0: stabilisasi skala sebelum GAM/fusion

    Output TIDAK di-L2-normalize di sini — normalisasi dilakukan di encoder
    penuh setelah fusion dengan PointNet++ embedding.
    """

    def __init__(self, in_dim: int = GEOMETRY_DIM, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_dim),  # v5.0.0: stabilisasi skala geom_emb
        )

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geom : (B, GEOMETRY_DIM) float32 — fitur geometri yang sudah di-normalize

        Returns:
            emb  : (B, 64) float32
        """
        return self.net(geom)
