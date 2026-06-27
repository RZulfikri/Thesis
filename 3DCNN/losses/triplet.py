"""
losses/triplet.py — Online Triplet Loss dengan batch-hard mining dan cross-session variant (v7 C2).

Untuk setiap anchor di dalam batch, pilih:
  - hardest positive : sample sesama subjek dengan jarak L2 terbesar
  - hardest negative : sample subjek lain dengan jarak L2 terkecil

Loss = mean( relu(d_pos - d_neg + margin) ) atas anchor yang valid.

Berbeda dari ContrastiveLoss yang menggunakan random pairs (kebanyakan "easy"),
TripletLoss + hard mining memaksa model belajar dari kasus paling sulit di setiap
batch — sangat efektif untuk dataset kecil seperti palm recognition 11 subjek.

Distance derivation (sama dengan ContrastiveLoss):
    ||a - b||² = 2 - 2*cos(a,b) = 2*(1 - sim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTripletLoss(nn.Module):
    """
    Args:
        margin : float — minimum gap (d_neg - d_pos) yang diinginkan (default 0.3)
        mining : str   — 'batch_hard' (default) atau 'batch_all'
    """

    def __init__(self, margin: float = 0.3, mining: str = "batch_hard"):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings : (B, D) — L2-normalized embeddings
            labels     : (B,) long — identity label index

        Returns:
            loss : scalar
        """
        # Pairwise L2 distance pada unit sphere
        # ||a - b||² = 2 - 2*cos(a,b) → distance = sqrt(2*(1 - sim))
        sim = embeddings @ embeddings.T                         # (B, B)
        dist = torch.clamp(2.0 * (1.0 - sim), min=1e-8).sqrt()  # (B, B)

        # Mask genuine (sama subjek) dan impostor
        eq = labels.unsqueeze(0) == labels.unsqueeze(1)         # (B, B) bool
        pos_mask = eq.clone()
        pos_mask.fill_diagonal_(False)                          # exclude self-pair
        neg_mask = ~eq

        if self.mining == "batch_hard":
            return self._batch_hard(dist, pos_mask, neg_mask, embeddings)
        elif self.mining == "batch_all":
            return self._batch_all(dist, pos_mask, neg_mask, embeddings)
        else:
            raise ValueError(f"mining tidak dikenal: {self.mining}")

    def _batch_hard(self, dist, pos_mask, neg_mask, embeddings):
        """
        Batch-hard mining: tiap anchor pilih hardest positive + hardest negative.
        """
        # Hardest positive: jarak terbesar antar same-label (mask non-positive ke -inf)
        d_pos_masked = dist.masked_fill(~pos_mask, float("-inf"))
        d_pos = d_pos_masked.max(dim=1).values                  # (B,)

        # Hardest negative: jarak terkecil antar different-label (mask non-negative ke +inf)
        d_neg_masked = dist.masked_fill(~neg_mask, float("inf"))
        d_neg = d_neg_masked.min(dim=1).values                  # (B,)

        # Hanya anchor yang punya minimal 1 positive DAN 1 negative di batch
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0   # zero loss, gradient path tetap ada

        loss = F.relu(d_pos[valid] - d_neg[valid] + self.margin)
        return loss.mean()

    def _batch_all(self, dist, pos_mask, neg_mask, embeddings):
        """
        Batch-all mining: jumlahkan semua triplet (anchor, positive, negative).
        Lebih lambat, tapi mencakup lebih banyak signal.
        """
        # (B, B, 1) − (B, 1, B) + margin → (B, B, B)
        d_anchor_pos = dist.unsqueeze(2)
        d_anchor_neg = dist.unsqueeze(1)
        triplet_loss = F.relu(d_anchor_pos - d_anchor_neg + self.margin)

        # Mask: pos[i,j] AND neg[i,k]
        mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)    # (B, B, B)
        triplet_loss = triplet_loss * mask.float()

        n_valid = mask.sum().clamp(min=1)
        return triplet_loss.sum() / n_valid


class CrossSessionTripletLoss(nn.Module):
    """
    Batch-hard Triplet Loss dengan cross-session constraint (v7 C2).

    Positive pair hanya valid jika anchor dan positive berasal dari sesi BERBEDA.
    Ini mencegah model menghafalkan sesi-specific noise dan memaksa
    generalisasi lintas sesi yang lebih kuat.

    Args:
        margin          : float — minimum gap d_pos - d_neg (default 0.3)
        fallback_intra  : jika True, fallback ke intra-session positive ketika
                          tidak ada cross-session positive di batch (default True)
    """

    def __init__(self, margin: float = 0.3, fallback_intra: bool = True):
        super().__init__()
        self.margin         = margin
        self.fallback_intra = fallback_intra

    def forward(
        self,
        embeddings:   torch.Tensor,   # (B, D)
        labels:       torch.Tensor,   # (B,) long — identity index
        session_ids:  torch.Tensor,   # (B,) long — session index
    ) -> torch.Tensor:
        """
        Args:
            embeddings  : (B, D) L2-normalized
            labels      : (B,) identity labels
            session_ids : (B,) session indices (dari PalmFrameDataset batch["session_idx"])

        Returns:
            loss : scalar
        """
        sim  = embeddings @ embeddings.T
        dist = torch.clamp(2.0 * (1.0 - sim), min=1e-8).sqrt()

        same_label   = labels.unsqueeze(0) == labels.unsqueeze(1)       # (B, B)
        diff_session = session_ids.unsqueeze(0) != session_ids.unsqueeze(1)  # (B, B)

        # Cross-session positive: sama label, sesi berbeda, bukan self
        cross_pos_mask = same_label & diff_session
        cross_pos_mask.fill_diagonal_(False)

        # Standard positive (intra-session): sama label, bukan self
        std_pos_mask = same_label.clone()
        std_pos_mask.fill_diagonal_(False)

        neg_mask = ~same_label

        # Gunakan cross-session positive jika tersedia; fallback ke intra-session
        has_cross_pos = cross_pos_mask.any(dim=1)
        if self.fallback_intra:
            pos_mask = torch.where(
                has_cross_pos.unsqueeze(1).expand_as(cross_pos_mask),
                cross_pos_mask,
                std_pos_mask,
            )
        else:
            pos_mask = cross_pos_mask

        # Batch-hard mining
        d_pos = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
        d_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0

        loss = F.relu(d_pos[valid] - d_neg[valid] + self.margin)
        return loss.mean()
