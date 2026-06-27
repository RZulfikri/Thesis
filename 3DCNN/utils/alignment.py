"""
utils/alignment.py — v8: Point-cloud alignment / normalization variants (single source of truth).

Dipakai OFFLINE (3DRegistration/make_align_variants.py meng-generate file .npy per varian)
DAN RUNTIME (notebook rotation-robustness test) → parity dijamin karena satu implementasi.

Mode alignment (input: xyz (N,3), nrm (N,3) koordinat kamera; output: (xyz', nrm') float32):
  - "raw"          : apa adanya (tanpa transform).
  - "center"       : translasi centroid → origin.
  - "centerscale"  : center + unit-sphere (TANPA rotasi).
  - "pca"          : kanonikalisasi PCA v7.2.0 (range-Y, median-Y flip, right-handed) + unit-sphere.
                     Mereplikasi 3DRegistration/preprocess_for_cnn.py:pca_align (R2/cnn_input.npy).
  - "pca_robust"   : PCA deterministik utk FIX rotasi 90° — pilih Y by range dgn tie-break variance,
                     sign tiap sumbu via skewness (momen-3), + unit-sphere. Tanpa landmark.
  - "anatomical"   : kanonikalisasi berbasis landmark tangan (analog face-landmark→align).
                     Y = wrist→jari-tengah, Z = normal telapak, X = cross dgn sign dari handedness.
                     Deterministik & rotation-robust (fix 90°). Reuse logika detektor inline.

Catatan: semua mode yg melakukan unit-sphere → scale-invarian; mode "pca"/"pca_robust"/"anatomical"
juga rotation-invarian (output sama untuk input yg dirotasi rigid) — itulah inti perbaikan 90°.
"""

from __future__ import annotations

import numpy as np

ALIGN_MODES = ("raw", "center", "centerscale", "pca", "pca_robust", "anatomical")


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _unit_sphere(xyz: np.ndarray) -> np.ndarray:
    """Scale agar semua titik masuk unit sphere (radius maks = 1). Normals tak diubah."""
    scale = float(np.max(np.linalg.norm(xyz, axis=1))) if len(xyz) else 0.0
    if scale < 1e-8:
        return xyz.astype(np.float32)
    return (xyz / scale).astype(np.float32)


def _skewness_sign(proj: np.ndarray) -> float:
    """
    Sign deterministik dari sebaran 1-D: kembalikan +1/-1 sehingga skewness (momen-3) >= 0.
    Membuat orientasi sumbu unik (hapus ambiguitas ±) secara rotation-invarian.
    """
    p = proj - proj.mean()
    denom = float(np.mean(p * p)) ** 1.5
    if denom < 1e-12:
        return 1.0
    skew = float(np.mean(p ** 3)) / denom
    return 1.0 if skew >= 0 else -1.0


def _apply_R(xyz_c: np.ndarray, nrm: np.ndarray, R: np.ndarray):
    """Rotasi titik (sudah ter-center) & normals dgn R (baris = sumbu baru)."""
    return (xyz_c @ R.T).astype(np.float32), (nrm @ R.T).astype(np.float32)


# ---------------------------------------------------------------------------
# PCA (v7.2.0 replica) — mode "pca"
# ---------------------------------------------------------------------------

def _pca_align(xyz: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replika preprocess_for_cnn.py:pca_align (range-Y, median-Y flip, right-handed). Tanpa scale."""
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    r0 = float(np.ptp(centered @ Vt[0]))
    r1 = float(np.ptp(centered @ Vt[1]))
    if r0 >= r1:
        y_axis, x_axis = Vt[0], Vt[1]
    else:
        y_axis, x_axis = Vt[1], Vt[0]
    z_axis = np.cross(x_axis, y_axis); z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis); x_axis /= np.linalg.norm(x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    al, aln = _apply_R(centered, nrm, R)
    # flip agar mayoritas titik di atas median-Y (jari → +Y)
    if np.sum(al[:, 1] > np.median(al[:, 1])) < len(al) // 2:
        al[:, 0] *= -1; al[:, 1] *= -1
        aln[:, 0] *= -1; aln[:, 1] *= -1
    return al.astype(np.float32), aln.astype(np.float32)


# ---------------------------------------------------------------------------
# PCA-robust — mode "pca_robust" (FIX 90° tanpa landmark)
# ---------------------------------------------------------------------------

def _pca_robust(xyz: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Kanonikalisasi PCA deterministik:
      - Z = sumbu varians terkecil (normal telapak) — stabil (bukan tie).
      - Y = di antara 2 sumbu in-plane, yg RANGE terbesar; bila |range0-range1| < 2% → tie-break
            pakai VARIANCE terbesar (lebih stabil thd rotasi 90°).
      - Sign tiap sumbu via SKEWNESS (momen-3) → orientasi unik, hapus ambiguitas ± (sumber swap 90°).
    """
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    p0 = centered @ Vt[0]; p1 = centered @ Vt[1]
    r0, r1 = float(np.ptp(p0)), float(np.ptp(p1))
    tie = abs(r0 - r1) < 0.02 * max(r0, r1, 1e-9)
    if tie:
        # tie-break: variance terbesar (S terurut menurun → Vt[0] var lebih besar) → stabil di ~90°
        y_axis, x_axis = Vt[0], Vt[1]
    else:
        y_axis, x_axis = (Vt[0], Vt[1]) if r0 >= r1 else (Vt[1], Vt[0])
    # right-handed R0
    z_axis = np.cross(x_axis, y_axis); z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis); x_axis /= np.linalg.norm(x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    al, aln = _apply_R(centered, nrm, R)
    # sign deterministik via skewness (momen-3) tiap sumbu; jaga rotasi proper (det=+1)
    def _abs_skew(proj):
        p = proj - proj.mean(); d = float(np.mean(p * p)) ** 1.5
        return 0.0 if d < 1e-12 else abs(float(np.mean(p ** 3)) / d)
    sgn = np.array([_skewness_sign(al[:, k]) for k in range(3)], dtype=np.float64)
    if sgn[0] * sgn[1] * sgn[2] < 0:  # refleksi → balik sumbu paling tidak-yakin (|skew| terkecil)
        j = int(np.argmin([_abs_skew(al[:, k]) for k in range(3)]))
        sgn[j] *= -1
    al = al * sgn; aln = aln * sgn
    return al.astype(np.float32), aln.astype(np.float32)


# ---------------------------------------------------------------------------
# Landmark detectors (inline, self-contained) — asumsi frame dgn jari ≈ +Y
# Replika ringkas dari 3DRegistration/extract_geometry.py
# ---------------------------------------------------------------------------

def _detect_wrist_center(pts: np.ndarray, fraction: float = 0.18) -> np.ndarray:
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    thr = y_min + (y_max - y_min) * fraction
    wrist = pts[pts[:, 1] <= thr]
    if len(wrist) < 30:
        thr = y_min + (y_max - y_min) * 0.28
        wrist = pts[pts[:, 1] <= thr]
    return wrist.mean(axis=0)


def _detect_fingertips(pts: np.ndarray, n_fingers: int = 5):
    """Return (tips (5,3), n_fallbacks) di frame dgn jari ≈ +Y; urutan kiri→kanan (belum thumb-order)."""
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    bw = (x_max - x_min) / n_fingers
    tips, nfb = [], 0
    for i in range(n_fingers):
        lo = x_min + i * bw; hi = lo + bw
        band = pts[(pts[:, 0] >= lo) & (pts[:, 0] < hi)]
        if len(band) < 10:
            tips.append(np.array([(lo + hi) / 2, pts[:, 1].max(), 0.0])); nfb += 1
        else:
            tips.append(band[np.argmax(band[:, 1])])
    return np.array(tips), nfb


def _orientation_score(pts: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Skor seberapa 'tangan-dgn-jari-ke-+Y' sebuah frame. Tinggi = lebih cocok.
    Skor = (5 - n_fallbacks) + bonus jika fingertips berada di atas wrist (struktur anatomis benar).
    Return (score, wrist_center, tips).
    """
    wrist = _detect_wrist_center(pts)
    tips, nfb = _detect_fingertips(pts)
    structure = 1.0 if (tips[:, 1].mean() > wrist[1]) else 0.0
    spread = float(np.ptp(tips[:, 0]))  # jari menyebar di X → struktur jari
    score = (5 - nfb) + structure + 0.001 * spread
    return score, wrist, tips


# ---------------------------------------------------------------------------
# Anatomical — mode "anatomical" (FIX 90° berbasis landmark, pilihan user)
# ---------------------------------------------------------------------------

def _anatomical(xyz: np.ndarray, nrm: np.ndarray, handedness: str | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Kanonikalisasi anatomis deterministik:
      1. Center; PCA → Z0 = normal telapak (varians terkecil), basis in-plane (u,v).
      2. Coba 4 orientasi (Y∈{+u,-u,+v,-v}); pilih yg paling 'tangan' (jari ke +Y) via _orientation_score.
      3. Refine: Y_exact = normalize(mean(fingertips) - wrist) → rotasikan agar tepat +Y.
      4. X-sign dari handedness (right: thumb di -X), else skewness.
      5. unit-sphere.
    Karena pemilihan orientasi & refine berbasis anatomi (bukan tie-break statistik), output
    invarian thd rotasi rigid input → kurva rotasi datar (termasuk 90°).
    """
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    z0 = Vt[2]
    u, v = Vt[0], Vt[1]
    cands = [u, -u, v, -v]
    best = None
    for y_try in cands:
        z_try = z0 / np.linalg.norm(z0)
        x_try = np.cross(y_try, z_try);
        nx = np.linalg.norm(x_try)
        if nx < 1e-8:
            continue
        x_try = x_try / nx
        y_n = np.cross(z_try, x_try); y_n /= np.linalg.norm(y_n)
        R = np.stack([x_try, y_n, z_try], axis=0)
        pts = (centered @ R.T).astype(np.float32)
        score, wrist, tips = _orientation_score(pts)
        if best is None or score > best[0]:
            best = (score, R, pts, wrist, tips)
    score, R, pts, wrist, tips = best
    al, aln = _apply_R(centered, nrm, R)
    # refine Y ke arah wrist→mean(tips)
    fdir = tips.mean(axis=0) - wrist
    fdir[2] = 0.0  # proyeksi ke bidang telapak (XY frame ini)
    n = np.linalg.norm(fdir)
    if n > 1e-6:
        fdir = fdir / n
        # rotasi 2D di bidang XY agar fdir → +Y
        cos_t, sin_t = float(fdir[1]), float(fdir[0])  # sudut thd +Y
        # matriks rotasi z: bawa (sin_t, cos_t) → (0,1)
        Rz = np.array([[cos_t, -sin_t, 0.0],
                       [sin_t,  cos_t, 0.0],
                       [0.0,    0.0,   1.0]], dtype=np.float32)
        al, aln = al @ Rz.T, aln @ Rz.T
        # rebuild tips/wrist tak diperlukan lagi
    # Z-sign: normal telapak menghadap +Z deterministik via skewness Z (flip X+Z = rotasi 180° thd Y, proper)
    if _skewness_sign(al[:, 2]) < 0:
        al[:, 2] *= -1; aln[:, 2] *= -1
        al[:, 0] *= -1; aln[:, 0] *= -1
    # X-sign dari handedness: tangan kanan → thumb di sisi -X. Koreksi via rotasi 180° thd Y (flip X+Z),
    # BUKAN mirror (agar chirality/identitas tangan tak rusak). Else fallback skewness-X (flip X+Z).
    def _flip_xz():
        al[:, 0] *= -1; aln[:, 0] *= -1
        al[:, 2] *= -1; aln[:, 2] *= -1
    if handedness in ("right", "left"):
        want = -1.0 if handedness == "right" else 1.0
        tips_now = _detect_fingertips(al)[0]
        thumb_side = float(np.sign(tips_now[int(np.argmax(np.abs(tips_now[:, 0]))), 0])) or 1.0
        if thumb_side != want:
            _flip_xz()
    else:
        if _skewness_sign(al[:, 0]) < 0:
            _flip_xz()
    return al.astype(np.float32), aln.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_points(xyz: np.ndarray, nrm: np.ndarray | None, mode: str,
                 handedness: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Terapkan alignment `mode` pada point cloud. Return (xyz', nrm') float32 (N,3).
    nrm boleh None → diisi nol.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if nrm is None:
        nrm = np.zeros_like(xyz)
    nrm = np.asarray(nrm, dtype=np.float64)
    if mode == "raw":
        return xyz.astype(np.float32), nrm.astype(np.float32)
    if mode == "center":
        return (xyz - xyz.mean(axis=0)).astype(np.float32), nrm.astype(np.float32)
    if mode == "centerscale":
        c = xyz - xyz.mean(axis=0)
        return _unit_sphere(c), nrm.astype(np.float32)
    if mode == "pca":
        al, aln = _pca_align(xyz, nrm); return _unit_sphere(al), aln
    if mode == "pca_robust":
        al, aln = _pca_robust(xyz, nrm); return _unit_sphere(al), aln
    if mode == "anatomical":
        al, aln = _anatomical(xyz, nrm, handedness); return _unit_sphere(al), aln
    raise ValueError(f"mode alignment tidak dikenal: {mode!r} (pilih dari {ALIGN_MODES})")


def align_cloud6(cloud: np.ndarray, mode: str, handedness: str | None = None) -> np.ndarray:
    """Convenience: cloud (N,6)=xyz+normals → (N,6) ter-align."""
    cloud = np.asarray(cloud)
    xyz, nrm = cloud[:, :3], cloud[:, 3:6]
    a, an = align_points(xyz, nrm, mode, handedness)
    return np.concatenate([a, an], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test: rotation-invariance (pca/pca_robust/anatomical harus invarian rotasi rigid)
# ---------------------------------------------------------------------------

def _rotz(xyz, deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return xyz @ R.T


def _synthetic_hand(seed=0):
    """Palm + 4 jari (ke +Y) + thumb asimetris (sisi -X, lebih pendek) → X-skewness ≠ 0 (realistis)."""
    rng = np.random.default_rng(seed)
    palm = rng.normal([0, 0, 0], [30, 18, 4], size=(4000, 3))
    parts = [palm]
    for fx in (-12, 0, 12, 24):           # index..pinky (ke +Y)
        ln = 70 - (fx - (-12)) * 1.0
        f = rng.normal([fx, 0, 0], [3.5, 1, 2], size=(700, 3))
        f[:, 1] += np.linspace(20, 20 + ln, 700)
        parts.append(f)
    # thumb: di -X, lebih rendah & menyamping → memecah simetri X
    th = rng.normal([-34, 8, 0], [4, 2, 2], size=(500, 3))
    th[:, 0] -= np.linspace(0, 22, 500)   # menjulur ke -X
    th[:, 1] += np.linspace(0, 18, 500)
    parts.append(th)
    xyz = np.concatenate(parts, axis=0)
    nrm = np.tile([0, 0, 1.0], (len(xyz), 1))
    return xyz, nrm


def _selftest():
    xyz, nrm = _synthetic_hand()
    print("self-test rotation-invariance (allclose output vs θ=0):")
    for mode in ("pca", "pca_robust", "anatomical"):
        base, _ = align_points(xyz, nrm, mode, handedness="right")
        ok = True; maxd = 0.0
        for deg in (30, 60, 90, 180):
            rot, _ = align_points(_rotz(xyz, deg), nrm, mode, handedness="right")
            if rot.shape != base.shape:
                ok = False; break
            d = float(np.max(np.linalg.norm(rot - base, axis=1)))
            maxd = max(maxd, d)
            if d > 0.05:  # toleransi (unit-sphere → ~5% radius)
                ok = False
        print(f"  {mode:11s}: {'INVARIAN' if ok else 'TIDAK invarian'}  (maxΔ={maxd:.4f})")
    # parity cloud6
    c6 = np.concatenate([xyz, nrm], axis=1)
    out = align_cloud6(c6, "pca_robust", "right")
    assert out.shape == (len(xyz), 6) and out.dtype == np.float32
    print("  align_cloud6: OK shape", out.shape)


if __name__ == "__main__":
    _selftest()
