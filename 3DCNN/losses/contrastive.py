"""
losses/contrastive.py — Contrastive loss for siamese palm verification.

Formula:
    d    = sqrt( clamp(2*(1 - sim), min=1e-8) )
    loss = y * d^2  +  (1-y) * relu(margin - d)^2

    y=1 → genuine pair  → minimize d (pull embeddings together)
    y=0 → impostor pair → maximize d beyond margin (push apart)

Distance formula derives from L2 distance on unit-sphere:
    ||a - b||^2 = 2 - 2*cos(a,b) = 2*(1 - sim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Args:
        margin : float — minimum L2 distance for impostor pairs (default 0.5)
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, sim: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sim   : (B,) cosine similarity ∈ [-1, 1]
            label : (B,) float — 1.0=genuine, 0.0=impostor

        Returns:
            loss : scalar
        """
        # Convert cosine similarity to L2 distance (on unit sphere)
        d = torch.sqrt(torch.clamp(2.0 * (1.0 - sim), min=1e-8))

        genuine_loss  = label * d.pow(2)
        impostor_loss = (1.0 - label) * F.relu(self.margin - d).pow(2)

        return (genuine_loss + impostor_loss).mean()
