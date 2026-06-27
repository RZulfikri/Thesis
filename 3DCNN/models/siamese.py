"""
models/siamese.py — Siamese network wrapper for palm verification.

One shared encoder called twice — NOT two separate encoders.

v0.3.0: Added optional ArcFace head for supervised pretraining.
v5.0.0: Added optional auxiliary classification loss (aux_classifier) untuk
         forcing geom branch belajar representasi diskriminatif di low-data regime.
"""

import torch
import torch.nn as nn

from .encoder import GeoAttPointNetEncoder


class SiamesePalmNet(nn.Module):
    """
    Siamese network for pairwise palm identity verification.

    Uses ONE shared encoder. Both sessions pass through the same weights.

    Optional ArcFace head (num_classes > 0) untuk supervised pretraining.
    Optional auxiliary classifier (use_aux_loss=True) untuk direct supervision
    ke geom branch (v5.0.0).

    Saat inference/evaluasi, ArcFace head dan aux_classifier diabaikan —
    hanya encoder yang dipakai.

    Forward output:
        emb_a : (B, 128) — L2-normalized embedding for session A
        emb_b : (B, 128) — L2-normalized embedding for session B
        sim   : (B,)     — cosine similarity ∈ [-1, 1]
                           (dot product of L2-normalized embeddings)
    """

    def __init__(self, geom_dim: int = 13, use_geom: bool = True,
                 num_classes: int = 0, arc_margin: float = 0.50,
                 arc_scale: float = 30.0,
                 use_gam: bool | None = None,
                 use_geom_fusion: bool | None = None,
                 siamese_mode: str = "concat",
                 use_aux_loss: bool = False,
                 n_subjects: int = 10,
                 loss_type: str = "arcface",
                 arcface_variant: str = "linear",
                 subcenter_k: int = 3,
                 qa_floor: float = 0.3):
        """
        Args:
            siamese_mode: "concat" (default, v0.4.0-optimize) atau "split" (v0.3.0
                / v0.4.0-baseline). Lihat docstring forward() untuk detail.
            use_aux_loss: v5.0.0 — aktifkan auxiliary classifier dari geom_emb.
            n_subjects  : v5.0.0 — jumlah subjek untuk aux_classifier (default 10, gede dropped).
        """
        super().__init__()
        self.encoder = GeoAttPointNetEncoder(
            geom_dim=geom_dim,
            use_geom=use_geom,
            use_gam=use_gam,
            use_geom_fusion=use_geom_fusion,
        )
        self.num_classes = num_classes
        self.use_aux_loss = use_aux_loss
        assert siamese_mode in ("concat", "split"), f"siamese_mode harus 'concat' atau 'split', got '{siamese_mode}'"
        self.siamese_mode = siamese_mode
        # gabungkan flag agar `self.use_geom` mengikuti varian aktif encoder
        self.use_geom = self.encoder.use_geom
        self.use_gam = self.encoder.use_gam
        self.use_geom_fusion = self.encoder.use_geom_fusion

        # Margin head (v8): dipilih sesuai loss_type via factory terpadu.
        # Default loss_type='arcface' variant='linear' → identik ArcMarginProduct v7.x
        # (rumus cosθ−m) sehingga reuse checkpoint v7.2.0 tetap valid.
        self.loss_type = loss_type
        if num_classes > 0:
            from losses.margin_heads import build_margin_head
            self.arcface = build_margin_head(
                loss_type, num_classes, embedding_dim=128,
                margin=arc_margin, scale=arc_scale, subcenter_k=subcenter_k,
                arcface_variant=arcface_variant, qa_floor=qa_floor,
            )
        else:
            self.arcface = None

        # v5.0.0: Optional auxiliary classifier untuk geom branch
        if use_aux_loss:
            self.aux_classifier = nn.Linear(64, n_subjects)
        else:
            self.aux_classifier = None

    def encode(self, pts: torch.Tensor, geom: torch.Tensor,
               return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode satu batch frame → embedding (B, 128) L2-normalized.

        Digunakan untuk:
          - Training dengan OnlineTripletLoss (forward satu batch frame)
          - Ekstraksi gallery / probe embedding di evaluate.ipynb

        Args:
            pts        : (B, N, 6)
            geom       : (B, geom_dim)
            return_aux : v5.0.0 — jika True, juga return aux_logits dari geom branch

        Returns:
            emb              : (B, 128) — L2-normalized embedding
            (emb, aux_logits): jika return_aux=True, aux_logits = (B, n_subjects)
        """
        emb = self.encoder(pts, geom)  # (B, 128)
        if return_aux and self.aux_classifier is not None:
            # Extract geom_emb dari encoder untuk aux classification
            geom_emb = self.encoder.geom_encoder(geom)  # (B, 64)
            aux_logits = self.aux_classifier(geom_emb)  # (B, n_subjects)
            return emb, aux_logits
        return emb

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
            geom_a : (B, geom_dim)
            pts_b  : (B, N, 6)
            geom_b : (B, geom_dim)

        Returns:
            emb_a : (B, 128)
            emb_b : (B, 128)
            sim   : (B,) cosine similarity
        """
        # Dua mode:
        # - "concat" (v0.4.0-optimize): cat dua branch → 1 forward call. ~15-25%
        #   faster, BN compute statistics over 2B sampel. Standar untuk Siamese
        #   contrastive (SimCLR/MoCo).
        # - "split" (v0.3.0 / v0.4.0-baseline): 2 panggilan encoder terpisah.
        #   BN per-branch (B sampel). Lebih lambat tapi sesuai baseline original.
        if self.siamese_mode == "concat":
            B = pts_a.size(0)
            pts  = torch.cat((pts_a,  pts_b),  dim=0)     # (2B, N, 6)
            geom = torch.cat((geom_a, geom_b), dim=0)     # (2B, geom_dim)
            emb  = self.encoder(pts, geom)                # (2B, 128) L2-normed
            emb_a, emb_b = emb[:B], emb[B:]
        else:  # "split"
            emb_a = self.encoder(pts_a, geom_a)           # (B, 128) L2-normed
            emb_b = self.encoder(pts_b, geom_b)           # (B, 128) L2-normed
        sim = (emb_a * emb_b).sum(dim=1)                  # (B,) cosine similarity
        return emb_a, emb_b, sim

    def forward_arcface(self, pts: torch.Tensor, geom: torch.Tensor,
                        labels: torch.Tensor,
                        quality: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward untuk margin-loss training (ArcFace/CosFace/SubCenter/AdaCos/Curricular/QA).

        Args:
            pts     : (B, N, 6)
            geom    : (B, geom_dim)
            labels  : (B,) long
            quality : (B,) float ∈ [0,1] — hanya dipakai head quality-aware (QA-ArcFace); else diabaikan.

        Returns:
            logits : (B, num_classes)
        """
        if self.arcface is None:
            raise RuntimeError("Margin head tidak diinisialisasi. "
                               "Gunakan num_classes > 0 saat membuat model.")
        emb = self.encoder(pts, geom)  # (B, 128) L2-normalized
        return self.arcface(emb, labels, quality)
