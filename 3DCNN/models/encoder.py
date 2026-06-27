"""
models/encoder.py — GeoAtt-PointNet++ encoder.

Combines:
  - PointNet++ backbone (3x SA layers)
  - Geometric Attention Module after each SA layer (opsional)
  - Geometry MLP branch (opsional, parallel)
  - Fusion head → 128-dim L2-normalized embedding

v0.3.0 — diagnostik:
  - Pemecahan flag `use_geom` menjadi dua flag independen `use_gam` dan
    `use_geom_fusion` untuk ablasi terpisah (Plan §D4).
  - **RNG parity**: GeometryEncoder & GAM selalu dibangun di __init__ dengan
    urutan identik sehingga konsumsi RNG global sama di semua varian
    (Plan §D1). Forward path tetap mematuhi flag (modul tidak dipakai jika
    nonaktif), jadi parameter ekstra tidak berkontribusi pada output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gam import GeometricAttentionModule
from .geometry_encoder import GeometryEncoder
from .pointnet_utils import SetAbstraction


class GeoAttPointNetEncoder(nn.Module):
    """
    Full GeoAtt-PointNet++ encoder.

    Input:
        pts  : (B, N, 6)  — XYZ + normals, N titik hasil sampling on-the-fly
        geom : (B, geom_dim) — geometric features (z-score normalized); diabaikan
                               jika `use_gam=False` DAN `use_geom_fusion=False`

    Output:
        embedding : (B, 128) — L2-normalized identity embedding

    Args:
        geom_dim         : dimensi vektor geom mentah (default 14, lihat
                           utils.dataset.GEOMETRY_DIM)
        use_geom         : alias kompatibel — True ⇒ use_gam=True dan
                           use_geom_fusion=True; False ⇒ keduanya False.
                           Boleh `None` jika ingin set use_gam/use_geom_fusion
                           secara independen.
        use_gam          : aktifkan Geometric Attention Module di forward path.
        use_geom_fusion  : aktifkan concatenation geom_emb ke fusion head.

    Catatan parity inisialisasi (Plan §D1):
        Semua sub-modul (GeometryEncoder, GAM1, GAM2, fusion head) selalu dibangun
        agar urutan konsumsi RNG global identik untuk semua varian. Hanya forward
        path yang berubah berdasarkan flag.
    """

    def __init__(
        self,
        geom_dim: int = 13,
        use_geom: bool | None = True,
        use_gam: bool | None = None,
        use_geom_fusion: bool | None = None,
    ):
        super().__init__()

        # Resolusi flag — kompat dengan checkpoint lama yang hanya pakai use_geom.
        if use_gam is None and use_geom_fusion is None:
            use_gam = bool(use_geom)
            use_geom_fusion = bool(use_geom)
        else:
            if use_gam is None:
                use_gam = bool(use_geom)
            if use_geom_fusion is None:
                use_geom_fusion = bool(use_geom)
        self.use_geom = bool(use_gam or use_geom_fusion)
        self.use_gam = bool(use_gam)
        self.use_geom_fusion = bool(use_geom_fusion)

        # ---- RNG parity: bangun SEMUA sub-modul terlebih dulu dengan urutan tetap ----
        # Urutan kreasi modul harus identik antar varian agar bobot SA1/SA2/SA3/proj
        # mendapat seed yang sama. Modul "tambahan" yang nonaktif tetap dibangun, hanya
        # saja tidak digunakan di forward.

        # 1) Geometry encoder — selalu dibangun
        self.geom_encoder = GeometryEncoder(in_dim=geom_dim, hidden=64, out_dim=64)

        # 2) SA1
        self.sa1 = SetAbstraction(n_point=512, radius=0.05, n_sample=32,
                                  in_ch=6, mlp_dims=[32, 32, 64])
        # 3) GAM1 — selalu dibangun (skipped di forward kalau use_gam=False)
        self.gam1 = GeometricAttentionModule(sa_ch=64, geom_ch=64)

        # 4) SA2
        self.sa2 = SetAbstraction(n_point=128, radius=0.15, n_sample=64,
                                  in_ch=64, mlp_dims=[64, 64, 128])
        # 5) GAM2 — selalu dibangun
        self.gam2 = GeometricAttentionModule(sa_ch=128, geom_ch=64)

        # 6) SA3 (global)
        self.sa3 = SetAbstraction(n_point=1, radius=5.0, n_sample=128,
                                  in_ch=128, mlp_dims=[128, 256, 256])

        # 7) Fusion head — dimensinya bergantung pada flag use_geom_fusion.
        # Untuk menjaga RNG identik, kita selalu kreasi proj head dengan dimensi
        # 256+64 → 256 → 128 (mode "with fusion"), dan jika fusion dimatikan
        # kita tambahkan adapter linear (256→320) yang berperan sebagai
        # pass-through deterministik nol-padding ekuivalen. Pendekatan praktis:
        # bangun head **dua** versi dengan urutan tetap dan pilih saat forward.
        # Pendekatan ini menjaga konsumsi RNG identik antar varian.
        proj_in_full = 256 + 64
        self.proj_with_geom = nn.Sequential(
            nn.Linear(proj_in_full, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
        )
        self.proj_no_geom = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
        )

    # ---- Kompatibilitas: alias `proj` mengikuti varian aktif ------------------
    @property
    def proj(self) -> nn.Sequential:  # for backward-compat introspection / hooks
        return self.proj_with_geom if self.use_geom_fusion else self.proj_no_geom

    def forward(self, pts: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts  : (B, N, 6)
            geom : (B, geom_dim) — diabaikan jika use_gam=False DAN use_geom_fusion=False

        Returns:
            embedding : (B, 128) L2-normalized
        """
        xyz = pts[:, :, :3]
        feat = pts

        # geom_emb dihitung jika dibutuhkan oleh GAM atau fusion
        geom_emb: torch.Tensor | None = None
        if self.use_gam or self.use_geom_fusion:
            geom_emb = self.geom_encoder(geom)               # (B, 64)

        # SA 1
        xyz1, feat1 = self.sa1(xyz, feat)
        if self.use_gam:
            feat1 = self.gam1(feat1, geom_emb)

        # SA 2
        xyz2, feat2 = self.sa2(xyz1, feat1)
        if self.use_gam:
            feat2 = self.gam2(feat2, geom_emb)

        # SA 3 — global
        _, feat3 = self.sa3(xyz2, feat2)
        global_feat = feat3.squeeze(1)                       # (B, 256)

        # Fusion
        if self.use_geom_fusion:
            combined = torch.cat([global_feat, geom_emb], dim=1)
            embedding = self.proj_with_geom(combined)
        else:
            embedding = self.proj_no_geom(global_feat)

        return F.normalize(embedding, p=2, dim=1)
