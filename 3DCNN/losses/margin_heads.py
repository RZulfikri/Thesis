"""
losses/margin_heads.py — v8: keluarga MARGIN HEAD terpadu untuk classification-based metric learning.

Latar: di v7.x, `SiamesePalmNet.forward_arcface` SELALU memakai `ArcMarginProduct`
(rumus `cosθ − m`, yakni CosFace) sementara `criterion` (CosFace/SubCenter) hanya
dipakai cross-entropy-nya → head margin alternatif TAK PERNAH aktif. Untuk paper IEEE
yang membandingkan loss secara jujur, semua head disatukan di sini dgn antarmuka seragam:

    head(embeddings_L2, labels, quality=None) -> logits (B, num_classes)

Head tersedia (loss_type):
  - "arcface"           : ArcFace LINEAR (cosθ − m). DIPERTAHANKAN identik dgn v7.x agar
                          reuse checkpoint v7.2.0 valid. (Catatan: rumus ini == CosFace.)
  - "arcface_true"      : ArcFace SEJATI cos(θ+m) = cosθ·cos m − sinθ·sin m (easy-margin).
  - "cosface"           : CosFace cosθ − m (additive cosine margin).
  - "subcenter_arcface" : K sub-center per kelas (max over K) + margin ArcFace sejati.
  - "adacos"            : AdaCos — scale s adaptif otomatis (tanpa tuning s), margin 0.
  - "curricularface"    : CurricularFace — modulasi negatif sulit via t (EMA).
  - "qa_arcface"        : USULAN — ArcFace sejati dgn margin diadaptasi KUALITAS scan 3D
                          (quality∈[0,1] dari geometry.json): m_eff = m * (q_floor + (1-q_floor)*q).
                          Kualitas rendah → margin longgar (tak over-penalize), tinggi → ketat.

Semua head: weight (C[,K], 128) di-L2-normalize; embeddings diasumsikan sudah L2-normalized
(encoder.py:168). Logits = s * (cosine dgn margin pada kelas target).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _onehot(labels: torch.Tensor, num_classes: int, like: torch.Tensor) -> torch.Tensor:
    oh = torch.zeros_like(like)
    oh.scatter_(1, labels.view(-1, 1), 1.0)
    return oh


# ---------------------------------------------------------------------------
# ArcFace (linear == v7.x) & ArcFace sejati
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """ArcFace. variant='linear' → cosθ−m (v7.x, == CosFace); variant='true' → cos(θ+m) easy-margin."""

    def __init__(self, in_features=128, out_features=11, s=30.0, m=0.50, variant="linear"):
        super().__init__()
        assert variant in ("linear", "true")
        self.s, self.m, self.variant = s, m, variant
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # konstanta utk varian sejati
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)        # ambang easy-margin
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels, quality=None):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.variant == "linear":
            phi = cosine - self.m
        else:  # true ArcFace: cos(theta + m), easy-margin
            sine = torch.sqrt((1.0 - cosine ** 2).clamp_min(1e-12))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh = _onehot(labels, cosine.size(1), cosine)
        return self.s * (oh * phi + (1.0 - oh) * cosine)


class CosFaceHead(nn.Module):
    """CosFace: target logit = cosθ − m (additive cosine margin)."""

    def __init__(self, in_features=128, out_features=11, s=30.0, m=0.35):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels, quality=None):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        oh = _onehot(labels, cosine.size(1), cosine)
        return self.s * (cosine - oh * self.m)


class SubCenterArcFaceHead(nn.Module):
    """SubCenter-ArcFace: K sub-center/kelas (max over K) + margin ArcFace sejati."""

    def __init__(self, in_features=128, out_features=11, K=3, s=30.0, m=0.50):
        super().__init__()
        self.s, self.m, self.K, self.C = s, m, K, out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features * K, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, emb, labels, quality=None):
        cos_all = F.linear(F.normalize(emb), F.normalize(self.weight))   # (B, C*K)
        cos_all = cos_all.view(-1, self.C, self.K)
        cosine = cos_all.max(dim=2).values.clamp(-1 + 1e-7, 1 - 1e-7)    # (B, C)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp_min(1e-12))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh = _onehot(labels, self.C, cosine)
        return self.s * (oh * phi + (1.0 - oh) * cosine)


class AdaCosHead(nn.Module):
    """AdaCos (Zhang 2019): scale s adaptif otomatis dari statistik sudut; tanpa margin & tuning s."""

    def __init__(self, in_features=128, out_features=11, m=0.0):
        super().__init__()
        self.C = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = math.sqrt(2.0) * math.log(out_features - 1) if out_features > 2 else 10.0
        self.register_buffer("s_dyn", torch.tensor(float(self.s)))

    def forward(self, emb, labels, quality=None):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.training:
            with torch.no_grad():
                theta = torch.acos(cosine)
                oh = _onehot(labels, self.C, cosine)
                # B_avg: rata-rata exp(s*cos) pada kelas non-target
                B_avg = torch.exp(self.s_dyn * cosine) * (1.0 - oh)
                B_avg = B_avg.sum(dim=1).mean()
                theta_med = torch.median(theta[oh.bool()])
                theta_med = torch.clamp(theta_med, max=math.pi / 4)
                self.s_dyn = torch.log(B_avg.clamp_min(1e-12)) / torch.cos(theta_med).clamp_min(1e-6)
        return self.s_dyn * cosine


class CurricularFaceHead(nn.Module):
    """CurricularFace (Huang 2020): negatif sulit dimodulasi adaptif via t (EMA)."""

    def __init__(self, in_features=128, out_features=11, s=30.0, m=0.50, momentum=0.99):
        super().__init__()
        self.s, self.m, self.C, self.mom = s, m, out_features, momentum
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
        self.register_buffer("t", torch.zeros(1))

    def forward(self, emb, labels, quality=None):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp_min(1e-12))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh = _onehot(labels, self.C, cosine)
        target = (oh * phi).sum(dim=1, keepdim=True)
        if self.training:
            with torch.no_grad():
                self.t = self.mom * self.t + (1 - self.mom) * cosine[oh.bool()].mean()
        # modulasi negatif sulit (cos_j > target) dgn faktor (t + cos_j)
        hard = cosine > target
        cosine_mod = torch.where(hard, cosine * (self.t + cosine), cosine)
        return self.s * (oh * phi + (1.0 - oh) * cosine_mod)


class QAArcFaceHead(nn.Module):
    """
    USULAN — Quality-Adaptive ArcFace untuk telapak: margin sudut diadaptasi KUALITAS scan 3D.
        m_eff = m * (q_floor + (1 - q_floor) * quality),  quality∈[0,1] dari geometry.json.
    Kualitas rendah → margin longgar (tak over-penalize sampel buruk); tinggi → margin penuh.
    Bila quality=None → fallback ArcFace sejati (m konstan).
    """

    def __init__(self, in_features=128, out_features=11, s=30.0, m=0.50, q_floor=0.3):
        super().__init__()
        self.s, self.m, self.C, self.q_floor = s, m, out_features, q_floor
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels, quality=None):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if quality is None:
            m_eff = torch.full((cosine.size(0), 1), self.m, device=cosine.device)
        else:
            q = quality.to(cosine.device).float().clamp(0, 1).view(-1, 1)
            m_eff = self.m * (self.q_floor + (1.0 - self.q_floor) * q)   # (B,1)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp_min(1e-12))
        phi = cosine * torch.cos(m_eff) - sine * torch.sin(m_eff)        # cos(θ+m_eff), broadcast
        th = torch.cos(math.pi - m_eff); mm = torch.sin(math.pi - m_eff) * m_eff
        phi = torch.where(cosine > th, phi, cosine - mm)
        oh = _onehot(labels, self.C, cosine)
        return self.s * (oh * phi + (1.0 - oh) * cosine)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_margin_head(loss_type: str, num_classes: int, *, embedding_dim: int = 128,
                      margin: float = 0.50, scale: float = 30.0, subcenter_k: int = 3,
                      arcface_variant: str = "linear", qa_floor: float = 0.3) -> nn.Module:
    """Bangun margin head sesuai loss_type. 'arcface'(+variant), 'arcface_true', 'cosface',
    'subcenter_arcface', 'adacos', 'curricularface', 'qa_arcface'."""
    lt = loss_type
    if lt == "arcface":
        return ArcFaceHead(embedding_dim, num_classes, s=scale, m=margin, variant=arcface_variant)
    if lt == "arcface_true":
        return ArcFaceHead(embedding_dim, num_classes, s=scale, m=margin, variant="true")
    if lt == "cosface":
        return CosFaceHead(embedding_dim, num_classes, s=scale, m=margin)
    if lt == "subcenter_arcface":
        return SubCenterArcFaceHead(embedding_dim, num_classes, K=subcenter_k, s=scale, m=margin)
    if lt == "adacos":
        return AdaCosHead(embedding_dim, num_classes)
    if lt == "curricularface":
        return CurricularFaceHead(embedding_dim, num_classes, s=scale, m=margin)
    if lt == "qa_arcface":
        return QAArcFaceHead(embedding_dim, num_classes, s=scale, m=margin, q_floor=qa_floor)
    raise ValueError(f"loss_type margin tidak dikenal: {lt!r}")


QUALITY_AWARE_HEADS = {"qa_arcface"}
MARGIN_LOSS_TYPES = {"arcface", "arcface_true", "cosface", "subcenter_arcface",
                     "adacos", "curricularface", "qa_arcface"}
