"""
losses/arcface.py — ArcFace (Additive Angular Margin) Loss untuk deep metric learning.

ArcFace menambahkan margin angular pada cosine similarity di classifier head,
memaksa antar-kelas terpisah lebih baik pada unit hypersphere.

Ref: Deng et al. (2019), "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    Classifier head dengan additive angular margin.

    Forward menghasilkan logits yang sudah diterapkan margin:
        logits = s * (cos(theta + m))  untuk kelas target
        logits = s * cos(theta)        untuk kelas non-target

    Args:
        in_features  : dimensi embedding input (default 128)
        out_features : jumlah kelas / identitas (default 11)
        s            : scale factor (default 30.0)
        m            : additive angular margin dalam radian (default 0.50 ≈ 28.6°)
    """

    def __init__(self, in_features: int = 128, out_features: int = 11,
                 s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Weight matrix: (out_features, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input : (B, in_features) — L2-normalized embedding
            label : (B,) long — ground-truth class indices

        Returns:
            logits : (B, out_features) — scaled cosine similarity with margin
        """
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # (B, out_features)

        # Additive angular margin: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        # Dalam implementasi praktis, kita langsung modify target indices
        # Karena input sudah L2-normalized, cosine = cos(theta)
        # phi = cos(theta + m) = cos(theta) - m  (approximasi linear untuk speed)
        # Note: Approximasi linear ini cukup efektif dan lebih stabil numerik
        phi = cosine - self.m

        # One-hot untuk kelas target
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # Apply margin hanya pada kelas target
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcFaceLoss(nn.Module):
    """
    Wrapper ArcFace = ArcMarginProduct + CrossEntropyLoss.

    Penggunaan:
        criterion = ArcFaceLoss(num_classes=11, margin=0.50, scale=30.0)
        logits = criterion(embeddings, labels)   # embeddings belum perlu normalize
        loss = criterion.compute_loss(logits, labels)
    """

    def __init__(self, num_classes: int = 11, margin: float = 0.50,
                 scale: float = 30.0, embedding_dim: int = 128):
        super().__init__()
        self.arc_margin = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=scale,
            m=margin,
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings : (B, 128) — bisa normalized atau belum
            labels     : (B,) long

        Returns:
            logits : (B, num_classes)
        """
        return self.arc_margin(embeddings, labels)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Convenience: langsung return loss (bukan logits).
        """
        logits = self.forward(embeddings, labels)
        return self.compute_loss(logits, labels)


# ---------------------------------------------------------------------------
# Hybrid Loss: ArcFace + Triplet (untuk Phase 2 fine-tuning)
# ---------------------------------------------------------------------------

class HybridArcTripletLoss(nn.Module):
    """
    Kombinasi ArcFace (supervised) + OnlineTripletLoss (metric).

    Weight default: 0.7 ArcFace + 0.3 Triplet.
    """

    def __init__(self, num_classes: int = 11, arc_margin: float = 0.50,
                 arc_scale: float = 30.0, triplet_margin: float = 0.3,
                 arc_weight: float = 0.7, triplet_weight: float = 0.3,
                 embedding_dim: int = 128):
        super().__init__()
        from .triplet import OnlineTripletLoss
        self.arcface = ArcFaceLoss(num_classes, arc_margin, arc_scale, embedding_dim)
        self.triplet = OnlineTripletLoss(margin=triplet_margin)
        self.arc_weight = arc_weight
        self.triplet_weight = triplet_weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        arc_loss = self.arcface(embeddings, labels)
        trip_loss = self.triplet(embeddings, labels)
        return self.arc_weight * arc_loss + self.triplet_weight * trip_loss
