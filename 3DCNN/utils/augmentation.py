"""
utils/augmentation.py — Augmentasi point cloud dan fitur geometri untuk training set.

Point cloud augmentation: rotasi Z, jitter, skala acak, point dropout.
Geometry augmentation: Gaussian noise kecil pada fitur mm — simulasi variasi
  pengukuran antar sesi (tangan sedikit bengkak, pose berbeda, noise sensor).
  Diterapkan SETELAH z-score normalization agar skala noise konsisten (σ relatif terhadap std).

NEW (v0.3.0):
  - OriginalSpaceAugmentor: augmentasi di original camera space (sebelum PCA-align).
    Rotasi/tilt/translate merepresentasikan variasi pose NYATA saat scanning,
    bukan variasi artificial di canonical frame.
"""

import numpy as np


class PointCloudAugmentor:
    """
    Apply random augmentations to a (N, 6) point cloud [XYZ + Nx, Ny, Nz].

    IMPORTANT: cnn_input.npy sudah PCA-aligned dan unit-sphere normalized.
    Artinya rotasi/tilt tidak merepresentasikan pose asli, melainkan
    variasi artificial pada canonical frame.  Oleh karena itu rotasi besar
    dibatasi agar tidak menghancurkan struktur biometrik.

    All transforms preserve the channel layout:
        cols 0:3 → XYZ coordinates
        cols 3:6 → surface normals

    Usage:
        aug = PointCloudAugmentor()
        pts_aug = aug(pts)   # (N, 6)
    """

    def __init__(
        self,
        rot_range_deg: float = 45.0,          # v5.0.0: naik dari 15° → 45°
        jitter_sigma: float = 0.001,          # v5.0.0: turun 2mm → 1mm (TrueDepth relatif clean)
        scale_range: tuple[float, float] = (0.95, 1.05),  # v5.0.0: lebih ketat 0.95–1.05
        dropout_ratio: float = 0.15,
        large_rotation_prob: float = 0.2,
        large_rotation_deg: float = 45.0,
        tilt_prob: float = 0.3,
        tilt_range_deg: float = 20.0,         # v5.0.0: naik dari 15° → 20°
        translate_prob: float = 0.3,
        translate_range: float = 0.03,        # v5.0.0: naik 2cm → 3cm
        z_translate_prob: float = 0.3,        # v5.0.0: BARU — translate Z ±3cm
        z_translate_range: float = 0.03,      # v5.0.0: BARU
        seed: int | None = None,
    ):
        self.rot_range_deg      = rot_range_deg
        self.jitter_sigma       = jitter_sigma
        self.scale_range        = scale_range
        self.dropout_ratio      = dropout_ratio
        self.large_rotation_prob = large_rotation_prob
        self.large_rotation_deg  = large_rotation_deg
        self.tilt_prob           = tilt_prob
        self.tilt_range_deg      = tilt_range_deg
        self.translate_prob      = translate_prob
        self.translate_range     = translate_range
        self.z_translate_prob    = z_translate_prob
        self.z_translate_range   = z_translate_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """
        Args:
            pts: (N, 6) float32 — XYZ + normals

        Returns:
            augmented pts: (N, 6) float32
        """
        pts = pts.copy()
        xyz = pts[:, :3]
        normals = pts[:, 3:6]

        # 1a. Z-rotation: in-plane (±rot_range_deg kecil ATAU ±large_rotation_deg besar)
        if self.rng.random() < self.large_rotation_prob:
            # Rotasi besar ±90°: simulasi tangan diputar saat scan
            sign = self.rng.choice([-1, 1])
            angle_z = np.radians(sign * self.large_rotation_deg)
        else:
            angle_z = np.radians(
                self.rng.uniform(-self.rot_range_deg, self.rot_range_deg)
            )
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz     = xyz @ Rz.T
        normals = normals @ Rz.T

        # 1b. X-tilt: depan/belakang (ujung jari mendekati/menjauhi kamera)
        if self.rng.random() < self.tilt_prob:
            ax = np.radians(self.rng.uniform(-self.tilt_range_deg, self.tilt_range_deg))
            cx, sx = np.cos(ax), np.sin(ax)
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
            xyz     = xyz @ Rx.T
            normals = normals @ Rx.T

        # 1c. Y-tilt: kiri/kanan (tepi jempol/kelingking mendekati kamera)
        if self.rng.random() < self.tilt_prob:
            ay = np.radians(self.rng.uniform(-self.tilt_range_deg, self.tilt_range_deg))
            cy, sy = np.cos(ay), np.sin(ay)
            Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
            xyz     = xyz @ Ry.T
            normals = normals @ Ry.T

        # 2. Gaussian jitter on XYZ
        xyz = xyz + self.rng.normal(0.0, self.jitter_sigma, xyz.shape).astype(np.float32)

        # 3. Random uniform scale on XYZ
        scale = self.rng.uniform(*self.scale_range)
        xyz = xyz * scale

        # 4. Random point dropout — remove 5%, re-sample to original size
        n = len(xyz)
        n_drop = int(n * self.dropout_ratio)
        keep_idx = self.rng.choice(n, n - n_drop, replace=False)
        xyz = xyz[keep_idx]
        normals = normals[keep_idx]

        # Re-sample to N points
        resample_idx = self.rng.choice(len(xyz), n, replace=(len(xyz) < n))
        xyz = xyz[resample_idx]
        normals = normals[resample_idx]

        # 5. Random XY translation — simulasi palm tidak tepat di tengah frame
        if self.rng.random() < self.translate_prob:
            shift = self.rng.uniform(
                -self.translate_range, self.translate_range, size=2
            ).astype(np.float32)
            xyz[:, 0] += shift[0]
            xyz[:, 1] += shift[1]

        # 5b. Random Z translation — v5.0.0: simulasi variasi jarak ke sensor
        if self.rng.random() < self.z_translate_prob:
            # rng.uniform tanpa size= mengembalikan Python float → cast manual ke np.float32
            dz = np.float32(self.rng.uniform(
                -self.z_translate_range, self.z_translate_range
            ))
            xyz[:, 2] += dz

        pts_aug = np.concatenate([xyz, normals], axis=1).astype(np.float32)
        return pts_aug


class GeometryAugmentor:
    """
    Augmentasi pada vektor fitur geometri (sudah di-z-score normalize).

    Mensimulasikan variabilitas estimasi geometry antar sesi akibat:
    - Pose tangan yang berbeda (palm_height CV=27%, palm_depth CV=17%)
    - Noise sensor depth
    - Inkonsistensi landmark detection

    Diterapkan SETELAH z-score normalization — noise_sigma dalam unit std.
    Hanya digunakan pada training set, TIDAK pada val/test.
    """

    def __init__(self, noise_sigma: float = 0.05, drop_prob: float = 0.1,
                 seed: int | None = None):
        """
        Args:
            noise_sigma : std noise dalam unit z-score (default 0.05 = ±5% std)
            drop_prob   : probabilitas drop satu fitur geometri (simulasi missing measurement)
        """
        self.noise_sigma = noise_sigma
        self.drop_prob = drop_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, geom: np.ndarray) -> np.ndarray:
        """
        Args:
            geom : (GEOMETRY_DIM,) float32 — sudah di-z-score normalize

        Returns:
            geom dengan noise dan occasional dropout
        """
        noise = self.rng.normal(0.0, self.noise_sigma, geom.shape).astype(np.float32)
        geom = geom + noise
        # Occasional feature dropout (simulate missing measurement)
        if self.drop_prob > 0:
            mask = self.rng.random(geom.shape) > self.drop_prob
            geom = geom * mask.astype(np.float32)
        return geom


# ---------------------------------------------------------------------------
# v0.3.0 — OriginalSpaceAugmentor
# ---------------------------------------------------------------------------

class OriginalSpaceAugmentor:
    """
    Augmentasi di original camera space (sebelum PCA-align).

    Berbeda dari PointCloudAugmentor yang bekerja di canonical frame,
    augmentasi ini merepresentasikan variasi pose NYATA saat scanning:
      - Rotasi Z ±30°  → tangan diputar di bidang meja
      - Tilt X/Y ±15°  → tangan dimiringkan saat scanning
      - Translate ±2cm → posisi tangan berbeda di FOV kamera
      - Scale 0.9–1.1  → jarak tangan ke kamera berbeda
      - Jitter σ=2mm   → sensor noise
      - Dropout 10–15% → occlusion / missing depth

    Karena augmentasi terjadi SEBELUM PCA-align, jitter dan dropout
    mengubah principal components → canonical frame yang dihasilkan
    lebih variatif dan realistis.

    Usage:
        aug = OriginalSpaceAugmentor(seed=42)
        pts_aug, normals_aug = aug(pts, normals)
        # pts_aug dan normals_aug kemudian di-PCA-align
    """

    def __init__(
        self,
        rot_z: float = 30.0,
        tilt: float = 15.0,
        translate: float = 0.02,
        scale: tuple[float, float] = (0.9, 1.1),
        jitter: float = 0.002,
        dropout: float = 0.15,
        seed: int | None = None,
    ):
        self.rot_z = rot_z
        self.tilt = tilt
        self.translate = translate
        self.scale = scale
        self.jitter = jitter
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        pts: np.ndarray,
        normals: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Args:
            pts     : (N, 3) float32 — XYZ original (belum PCA-align)
            normals : (N, 3) float32 opsional — surface normals

        Returns:
            pts_aug     : (N', 3) float32 — N' ≤ N jika dropout aktif
            normals_aug : (N', 3) float32 atau None
        """
        pts = pts.copy().astype(np.float32)
        if normals is not None:
            normals = normals.copy().astype(np.float32)

        xyz = pts

        # 1. Rotasi Z (tangan diputar di bidang meja)
        angle_z = np.radians(self.rng.uniform(-self.rot_z, self.rot_z))
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = xyz @ Rz.T
        if normals is not None:
            normals = normals @ Rz.T

        # 2. Tilt X (tangan dimiringkan depan/belakang)
        angle_x = np.radians(self.rng.uniform(-self.tilt, self.tilt))
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
        xyz = xyz @ Rx.T
        if normals is not None:
            normals = normals @ Rx.T

        # 3. Tilt Y (tangan dimiringkan kiri/kanan)
        angle_y = np.radians(self.rng.uniform(-self.tilt, self.tilt))
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        xyz = xyz @ Ry.T
        if normals is not None:
            normals = normals @ Ry.T

        # 4. Translate (posisi berbeda di FOV kamera)
        shift = self.rng.uniform(-self.translate, self.translate, size=3).astype(np.float32)
        xyz += shift

        # 5. Scale (jarak ke kamera berbeda)
        s = self.rng.uniform(*self.scale)
        xyz *= s

        # 6. Jitter Gaussian (sensor noise)
        xyz += self.rng.normal(0.0, self.jitter, xyz.shape).astype(np.float32)

        # 7. Point dropout (occlusion / missing depth)
        #    Tidak di-resample — biarkan N' ≤ N, sampling dilakukan oleh dataset
        if self.dropout > 0:
            n = len(xyz)
            n_keep = int(n * (1.0 - self.dropout))
            keep_idx = self.rng.choice(n, max(n_keep, 1), replace=False)
            xyz = xyz[keep_idx]
            if normals is not None:
                normals = normals[keep_idx]

        return xyz, normals
