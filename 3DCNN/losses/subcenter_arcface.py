"""
losses/subcenter_arcface.py — Sub-Center ArcFace untuk v7.0.0.

Sub-Center ArcFace memperluas ArcFace dengan K sub-centers per kelas.
Setiap kelas diwakili K prototype; similarity ke kelas = similarity ke prototype terdekat.
Ini memberikan toleransi terhadap label noise dan intra-class variation tinggi
(relevan untuk capture burst dengan variasi minor antar-frame).

Formula:
  Untuk kelas c: s_c(x) = max_k cos(x, w_ck)
  Loss: ArcFace margin diterapkan pada s_c (max similarity sub-center)

Ref: Deng et al. (2020), "Sub-center ArcFace: Boosting Face Recognition
     by Large-Scale Noisy Web Faces"

Args default yang direkomendasikan untuk N=11 subjek:
  K=3, margin=0.5, scale=30
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCenterArcMarginProduct(nn.Module):
    """
    Sub-Center ArcFace: K sub-centers per kelas.

    Shape weight: (out_features * K, in_features)
    → reshape ke (out_features, K, in_features) untuk max-pool antar sub-center.

    Args:
        in_features  : dimensi embedding (default 128)
        out_features : jumlah kelas (default 11)
        K            : jumlah sub-centers per kelas (default 3)
        s            : scale factor (default 30.0)
        m            : angular margin (default 0.50)
    """

    def __init__(self, in_features: int = 128, out_features: int = 11,
                 K: int = 3, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.K  = K
        self.s  = s
        self.m  = m

        # (out_features * K, in_features) → K sub-centers per class
        self.weight = nn.Parameter(torch.FloatTensor(out_features * K, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input : (B, in_features) — embedding
            label : (B,) long

        Returns:
            logits : (B, out_features)
        """
        # Normalize
        x = F.normalize(input)                          # (B, D)
        w = F.normalize(self.weight)                    # (C*K, D)

        # Cosine similarity ke semua sub-centers
        cosine_all = F.linear(x, w)                     # (B, C*K)
        cosine_all = cosine_all.view(-1, self.out_features, self.K)  # (B, C, K)

        # Max sub-center similarity per class
        cosine, _ = cosine_all.max(dim=-1)              # (B, C)

        # ArcFace margin pada kelas target
        phi = cosine - self.m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SubCenterArcFaceLoss(nn.Module):
    """
    Wrapper Sub-Center ArcFace = SubCenterArcMarginProduct + CrossEntropyLoss.

    Penggunaan:
        criterion = SubCenterArcFaceLoss(num_classes=11, K=3, margin=0.5, scale=30)
        loss = criterion(embeddings, labels)
    """

    def __init__(self, num_classes: int = 11, K: int = 3,
                 margin: float = 0.50, scale: float = 30.0,
                 embedding_dim: int = 128):
        super().__init__()
        self.arc_margin = SubCenterArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            K=K,
            s=scale,
            m=margin,
        )
        self.ce = nn.CrossEntropyLoss()
        self.K  = K

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        return self.arc_margin(embeddings, labels)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Convenience: langsung return loss."""
        logits = self.forward(embeddings, labels)
        return self.compute_loss(logits, labels)
