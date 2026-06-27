"""
losses/cosface.py — CosFace (Large Margin Cosine Loss) untuk v7.0.0.

CosFace menambahkan margin pada cosine similarity (bukan angular seperti ArcFace),
menghasilkan decision boundary yang lebih sederhana dan stabil secara numerik.

Formula:
  L = -log(e^{s(cos θ_yi - m)} / (e^{s(cos θ_yi - m)} + Σ_{j≠yi} e^{s·cos θ_j}))

Ref: Wang et al. (2018), "CosFace: Large Margin Cosine Loss for Deep Face Recognition"

Perbandingan vs ArcFace:
  - ArcFace : margin pada sudut (angular margin) → cos(θ + m)
  - CosFace : margin pada cosine value           → cos(θ) - m
  - CosFace lebih mudah di-tune; ArcFace sedikit lebih kuat di N besar
  - Pada N kecil (11 subjek) performa keduanya sering setara
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosMarginProduct(nn.Module):
    """
    Classifier head dengan large margin cosine loss (CosFace).

    Forward menghasilkan logits:
        logits = s * (cos θ_yi - m)  untuk kelas target
        logits = s * cos θ_j         untuk kelas non-target

    Args:
        in_features  : dimensi embedding input (default 128)
        out_features : jumlah kelas / identitas
        s            : scale factor (default 30.0)
        m            : cosine margin (default 0.35)
    """

    def __init__(self, in_features: int = 128, out_features: int = 11,
                 s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input : (B, in_features) — embedding (akan dinormalize di sini)
            label : (B,) long — ground-truth class indices

        Returns:
            logits : (B, out_features) — scaled cosine similarity with margin
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # (B, C)
        phi    = cosine - self.m                                           # margin hanya untuk target

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class CosFaceLoss(nn.Module):
    """
    Wrapper CosFace = CosMarginProduct + CrossEntropyLoss.

    Penggunaan:
        criterion = CosFaceLoss(num_classes=11, margin=0.35, scale=30.0)
        loss = criterion(embeddings, labels)
    """

    def __init__(self, num_classes: int = 11, margin: float = 0.35,
                 scale: float = 30.0, embedding_dim: int = 128):
        super().__init__()
        self.cos_margin = CosMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=scale,
            m=margin,
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        return self.cos_margin(embeddings, labels)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Convenience: langsung return loss."""
        logits = self.forward(embeddings, labels)
        return self.compute_loss(logits, labels)
