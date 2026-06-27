"""
models/siamese.py — Siamese network wrapper for palm verification.

One shared encoder called twice — NOT two separate encoders.

v0.3.0: Added optional ArcFace head for supervised pretraining.
"""

import torch
import torch.nn as nn

from .encoder import GeoAttPointNetEncoder


class SiamesePalmNet(nn.Module):
    """
    Siamese network for pairwise palm identity verification.

    Uses ONE shared encoder. Both sessions pass through the same weights.

    Optional ArcFace head (num_classes > 0) untuk supervised pretraining.
    Saat inference/evaluasi, ArcFace head diabaikan — hanya encoder yang dipakai.

    Forward output:
        emb_a : (B, 128) — L2-normalized embedding for session A
        emb_b : (B, 128) — L2-normalized embedding for session B
        sim   : (B,)     — cosine similarity ∈ [-1, 1]
                           (dot product of L2-normalized embeddings)
    """

    def __init__(self, geom_dim: int = 33, use_geom: bool = True,
                 num_classes: int = 0, arc_margin: float = 0.50,
                 arc_scale: float = 30.0,
                 use_gam: bool | None = None,
                 use_geom_fusion: bool | None = None):
        super().__init__()
        self.encoder = GeoAttPointNetEncoder(
            geom_dim=geom_dim,
            use_geom=use_geom,
            use_gam=use_gam,
            use_geom_fusion=use_geom_fusion,
        )
        self.num_classes = num_classes
        # gabungkan flag agar `self.use_geom` mengikuti varian aktif encoder
        self.use_geom = self.encoder.use_geom
        self.use_gam = self.encoder.use_gam
        self.use_geom_fusion = self.encoder.use_geom_fusion

        # Optional ArcFace head untuk supervised pretraining
        if num_classes > 0:
            from losses.arcface import ArcMarginProduct
            self.arcface = ArcMarginProduct(
                in_features=128,
                out_features=num_classes,
                s=arc_scale,
                m=arc_margin,
            )
        else:
            self.arcface = None

    def encode(self, pts: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        """
        Encode satu batch frame → embedding (B, 128) L2-normalized.

        Digunakan untuk:
          - Training dengan OnlineTripletLoss (forward satu batch frame)
          - Ekstraksi gallery / probe embedding di evaluate.ipynb
        """
        return self.encoder(pts, geom)

    def forward(
        self,
        pts_a:  torch.Tensor,
        geom_a: torch.Tensor,
        pts_b:  torch.Tensor,
        geom_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pts_a  : (B, N, 6)
            geom_a : (B, 33)
            pts_b  : (B, N, 6)
            geom_b : (B, 33)

        Returns:
            emb_a : (B, 128)
            emb_b : (B, 128)
            sim   : (B,) cosine similarity
        """
        emb_a = self.encoder(pts_a, geom_a)           # (B, 128) L2-normed
        emb_b = self.encoder(pts_b, geom_b)           # (B, 128) L2-normed
        sim   = (emb_a * emb_b).sum(dim=1)            # (B,) cosine similarity
        return emb_a, emb_b, sim

    def forward_arcface(self, pts: torch.Tensor, geom: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """
        Forward untuk ArcFace training.

        Args:
            pts    : (B, N, 6)
            geom   : (B, geom_dim)
            labels : (B,) long

        Returns:
            logits : (B, num_classes)
        """
        if self.arcface is None:
            raise RuntimeError("ArcFace head tidak diinisialisasi. "
                               "Gunakan num_classes > 0 saat membuat model.")
        emb = self.encoder(pts, geom)  # (B, 128) L2-normalized
        return self.arcface(emb, labels)
