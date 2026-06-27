"""
extract_geometry.py — Extract geometric biometric features from a registered palm point cloud.

Pendekatan:
  1. Deteksi Wrist ROI (bagian bawah tangan) sebagai anchor pengukuran
  2. Deteksi knuckle row (baris MCP joint) sebagai batas telapak–jari
  3. Fingertip detection via X-band maxima
  4. Semua panjang diukur dari wrist center ke fingertip
  5. Lebar jari dan celah antar jari diukur di zona jari (atas knuckle)

Features extracted — satuan mm, tanpa rasio:

  Fitur CNN (14 nilai — anatomis, stabil lintas pose):
  - finger_lengths_mm[5]  : panjang tiap jari dari wrist center ke ujung (thumb→pinky)
  - palm_width_mm         : lebar telapak di knuckle row (MCP) — X-extent
  - palm_height_mm        : tinggi telapak dari wrist top ke knuckle row — Y-extent
  - palm_depth_std_mm     : std dev Z di area palm (PCA-aligned) — kelengkungan permukaan
  - finger_widths_mm[5]   : lebar tiap jari di zona atas knuckle (p5–p95 X-extent)
  - mean_palm_curvature   : mean |1 - |nz|| di area telapak (0=rata, 1=melengkung)

  Metadata / QC (tidak masuk fitur CNN):
  - inter_finger_gaps_mm[4]: celah antar jari — pose-dependent, hanya untuk quality check
  - scan_distance_mm       : jarak kamera → telapak (~200–450mm) — QC gate saja

Validitas:
  - quality_issues: list masalah deteksi (fingertip fallback, fingers too close, knuckle gagal)
  - is_valid: True jika quality_issues kosong

Usage:
  python extract_geometry.py <ply_file> [--output geometry.json] [--handedness right|left|unknown]

Example:
  python extract_geometry.py result/rahmat/20260401_200613/output.ply --handedness right
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from preprocess_for_cnn import pca_align


# ---------------------------------------------------------------------------
# ROI Detection
# ---------------------------------------------------------------------------

def detect_wrist_roi(pts_mm: np.ndarray, fraction: float = 0.18):
    """
    Deteksi ROI pergelangan tangan: ambil titik-titik di bagian bawah tangan
    (nilai Y paling kecil setelah PCA alignment, karena jari mengarah ke +Y).

    Wrist ROI digunakan sebagai anchor reference untuk semua pengukuran:
    - wrist_center[1] (Y) = baseline untuk menghitung panjang jari
    - y_top = batas bawah telapak tangan

    Returns:
        wrist_center : (3,) centroid wrist ROI dalam mm
        y_top        : Y tertinggi dari wrist ROI (= batas bawah telapak)
        wrist_pts    : (N,3) semua titik dalam wrist ROI
    """
    y_min, y_max = pts_mm[:, 1].min(), pts_mm[:, 1].max()
    y_thresh = y_min + (y_max - y_min) * fraction
    wrist_pts = pts_mm[pts_mm[:, 1] <= y_thresh]

    # Fallback: jika terlalu sedikit titik, perlebar threshold
    if len(wrist_pts) < 30:
        y_thresh = y_min + (y_max - y_min) * 0.28
        wrist_pts = pts_mm[pts_mm[:, 1] <= y_thresh]

    wrist_center = wrist_pts.mean(axis=0)
    y_top = float(wrist_pts[:, 1].max())
    return wrist_center, y_top, wrist_pts


def detect_knuckle_y(pts_mm: np.ndarray, wrist_y_top: float) -> float:
    """
    Estimasi posisi Y baris MCP joint (knuckle row).

    Metode: scan level Y dari wrist_y_top ke 55% total ketinggian.
    Knuckle row = Y dimana lebar X cross-section maksimum — karena di sini
    kelima metacarpal head berjejer membentuk titik terlebar pada telapak.

    Returns:
        knuckle_y : Y-koordinat estimasi knuckle row (mm)
    """
    y_min, y_max = pts_mm[:, 1].min(), pts_mm[:, 1].max()
    y_scan_max = y_min + (y_max - y_min) * 0.55
    n_slices = 40
    y_levels = np.linspace(wrist_y_top, y_scan_max, n_slices)
    slice_h = float(y_levels[1] - y_levels[0]) if n_slices > 1 else 5.0

    max_width = 0.0
    knuckle_y = float(wrist_y_top)
    for y in y_levels:
        mask = (pts_mm[:, 1] >= y) & (pts_mm[:, 1] < y + slice_h)
        slice_pts = pts_mm[mask]
        if len(slice_pts) < 20:
            continue
        width = float(slice_pts[:, 0].max() - slice_pts[:, 0].min())
        if width > max_width:
            max_width = width
            knuckle_y = float(y)

    return knuckle_y


def _validate_knuckle_y(knuckle_y: float, wrist_y_top: float, y_max: float) -> bool:
    """
    Sanity check: knuckle row harus berada di posisi anatomis yang masuk akal.

    Kriteria:
      - palm_height (knuckle_y - wrist_y_top) >= 20 mm
      - palm_height <= 75% total tinggi tangan dari wrist

    Jika gagal, kemungkinan besar max(X-width) tertangkap di area wrist
    atau di pose ekstrem yang tidak memiliki knuckle row terdefinisi.
    """
    palm_height = knuckle_y - wrist_y_top
    total_height = y_max - wrist_y_top
    if palm_height < 20.0:
        return False
    if total_height > 0 and palm_height > total_height * 0.75:
        return False
    return True


# ---------------------------------------------------------------------------
# Fingertip Detection
# ---------------------------------------------------------------------------

def detect_fingertips(pts: np.ndarray, n_fingers: int = 5,
                      handedness: str = "unknown") -> tuple:
    """
    Deteksi posisi ujung jari dengan mencari Y-maksimum di setiap X-band,
    kemudian mengurutkan secara konsisten sebagai [thumb, index, middle, ring, pinky].

    Dua sinyal digunakan untuk menentukan sisi mana yang thumb:
      Signal A (height) : ujung thumb lebih tinggi Y-nya dari pinky (~95% akurat,
                          bisa gagal jika thumb bengkok).
      Signal B (gap)    : web space thumb–index lebih lebar dari ring–pinky
                          (~90% akurat, komplementer terhadap A).

    Jika handedness diketahui ("right"/"left"):
      - Kedua sinyal dievaluasi. Jika setuju → pakai itu.
      - Jika tidak setuju → Signal B menang (lebih tahan terhadap pose jari).

    Jika handedness "unknown":
      - Hanya Signal A yang digunakan (perilaku asli).

    Returns:
        tips        : (n_fingers, 3) XYZ ujung jari, diurutkan [thumb, index, middle, ring, pinky]
        n_fallbacks : jumlah band yang menggunakan fallback (< 10 titik) — indikator deteksi lemah
    """
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    band_w = (x_max - x_min) / n_fingers
    tips = []
    n_fallbacks = 0

    for i in range(n_fingers):
        bx_lo = x_min + i * band_w
        bx_hi = bx_lo + band_w
        mask = (pts[:, 0] >= bx_lo) & (pts[:, 0] < bx_hi)
        band_pts = pts[mask]
        if len(band_pts) < 10:
            tips.append(np.array([(bx_lo + bx_hi) / 2, pts[:, 1].max(), 0.0]))
            n_fallbacks += 1
        else:
            tips.append(band_pts[np.argmax(band_pts[:, 1])])

    tips = np.array(tips)  # (5, 3), berurutan min-X ke max-X

    # Signal A: sisi mana yang ujungnya lebih tinggi? (thumb > pinky dalam Y)
    signal_a_left_is_thumb = tips[0, 1] >= tips[4, 1]

    # Signal B: sisi mana yang celah adjacent-nya lebih lebar? (thumb–index > ring–pinky)
    left_gap  = tips[1, 0] - tips[0, 0]
    right_gap = tips[4, 0] - tips[3, 0]
    signal_b_left_is_thumb = left_gap >= right_gap

    if handedness in ("right", "left"):
        if signal_a_left_is_thumb == signal_b_left_is_thumb:
            left_is_thumb = signal_a_left_is_thumb
        else:
            left_is_thumb = signal_b_left_is_thumb  # gap wins on disagreement
    else:
        left_is_thumb = signal_a_left_is_thumb

    if not left_is_thumb:
        tips = tips[::-1].copy()

    return tips, n_fallbacks  # (5, 3), [thumb, index, middle, ring, pinky]


# ---------------------------------------------------------------------------
# Finger Width and Inter-finger Gap
# ---------------------------------------------------------------------------

def compute_finger_widths_and_gaps(pts_mm: np.ndarray, tips_mm: np.ndarray,
                                   knuckle_y: float) -> tuple:
    """
    Hitung lebar tiap jari dan lebar celah kosong antar jari, di zona jari
    (Y > knuckle_y, di bawah 90% total ketinggian).

    Lebar jari  = X-extent (p5–p95) dari titik-titik finger band di zona jari.
                  Menggunakan persentil untuk menghindari outlier.

    Celah antar jari = jarak antara tepi kanan band-i (p95 X) dan tepi kiri
                       band-(i+1) (p5 X) di zona jari. Ini adalah ruang KOSONG
                       yang sebenarnya antara dua jari berdekatan.

    Tips_mm sudah dalam urutan [thumb...pinky]. Jika thumb berada di sisi kanan
    (X besar), hasil dibalik agar tetap sesuai urutan [thumb...pinky].

    Returns:
        finger_widths_mm : list[5] lebar tiap jari dalam mm
        gap_widths_mm    : list[4] lebar celah antar jari dalam mm
    """
    y_min, y_max = pts_mm[:, 1].min(), pts_mm[:, 1].max()
    x_min, x_max = pts_mm[:, 0].min(), pts_mm[:, 0].max()
    band_w = (x_max - x_min) / 5

    # Zona jari: dari knuckle ke 90% total tinggi
    y_zone_hi = y_min + (y_max - y_min) * 0.90
    zone_mask = (pts_mm[:, 1] > knuckle_y) & (pts_mm[:, 1] <= y_zone_hi)
    zone_pts = pts_mm[zone_mask]

    # Hitung per X-band (berurutan kiri ke kanan dalam PCA space)
    band_right_edges = []  # p95 X untuk tiap band
    band_left_edges  = []  # p5  X untuk tiap band
    finger_widths    = []

    for i in range(5):
        bx_lo = x_min + i * band_w
        bx_hi = bx_lo + band_w
        mask = (zone_pts[:, 0] >= bx_lo) & (zone_pts[:, 0] < bx_hi)
        band_pts = zone_pts[mask]

        if len(band_pts) < 15:
            # Fallback ke band boundary jika titik terlalu sedikit
            finger_widths.append(0.0)
            band_left_edges.append(bx_lo)
            band_right_edges.append(bx_hi)
            continue

        x_p5  = float(np.percentile(band_pts[:, 0], 5))
        x_p95 = float(np.percentile(band_pts[:, 0], 95))
        finger_widths.append(round(x_p95 - x_p5, 2))
        band_left_edges.append(x_p5)
        band_right_edges.append(x_p95)

    # Celah antar jari = ruang kosong antara tepi kanan band-i dan tepi kiri band-(i+1)
    gap_widths = []
    for i in range(4):
        gap = max(0.0, band_left_edges[i + 1] - band_right_edges[i])
        gap_widths.append(round(gap, 2))

    # Jika thumb ada di kanan (X besar), balik agar urutan = [thumb...pinky]
    thumb_is_rightmost = tips_mm[0, 0] > tips_mm[4, 0]
    if thumb_is_rightmost:
        finger_widths = finger_widths[::-1]
        gap_widths    = gap_widths[::-1]

    return finger_widths, gap_widths


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_geometry(ply_path: str, output_path: str, handedness: str = "unknown") -> dict:
    pcd = o3d.io.read_point_cloud(ply_path)

    if len(pcd.points) == 0:
        sys.exit(f"Error: {ply_path} tidak mengandung titik")

    print(f"Loaded {len(pcd.points):,} points from {ply_path}")

    # Estimasi normal jika belum ada (dibutuhkan untuk curvature)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.orient_normals_towards_camera_location()

    # Scan distance: median Z palm area di koordinat ASLI (sebelum PCA).
    # Ini adalah jarak fisik kamera → telapak saat scan (nilai khas 200–400mm).
    # Digunakan sebagai scale reference — jarak yang berbeda memengaruhi skala XY.
    pts_orig_mm = np.asarray(pcd.points) * 1000.0
    scan_distance_mm = float(np.median(pts_orig_mm[:, 2]))

    # Canonical alignment: jari → +Y, depth → +Z
    pcd_aligned = pca_align(pcd)
    pts = np.asarray(pcd_aligned.points)       # metre
    pts_mm = pts * 1000.0                       # konversi ke mm
    nrm = np.asarray(pcd_aligned.normals) if pcd_aligned.has_normals() else None

    if handedness != "unknown":
        print(f"  Handedness: {handedness} (multi-signal thumb detection aktif)")

    # --- Step 1: Wrist ROI ---
    wrist_center, wrist_y_top, wrist_pts = detect_wrist_roi(pts_mm)
    base_y = float(wrist_center[1])  # referensi Y untuk panjang jari
    print(f"  Wrist ROI: y_center={base_y:.1f}mm, y_top={wrist_y_top:.1f}mm "
          f"({len(wrist_pts)} pts)")

    # --- Step 2: Knuckle row ---
    quality_issues = []  # gate is_valid — hanya isu yang merusak point cloud (lihat Step 8)
    warnings = []        # non-gating — isu ekstraksi fitur geometri, tidak memblokir frame
    y_min, y_max = pts_mm[:, 1].min(), pts_mm[:, 1].max()
    knuckle_y_detected = detect_knuckle_y(pts_mm, wrist_y_top)
    knuckle_y = knuckle_y_detected
    if not _validate_knuckle_y(knuckle_y_detected, wrist_y_top, y_max):
        # Fallback: estimasi anatomis ~50% total tinggi dari wrist.
        # knuckle_fallback adalah WARNING (fitur geometri), BUKAN quality gate —
        # point cloud tetap valid untuk CNN. Lihat catatan Step 8 baris 372-375.
        total_height = y_max - wrist_y_top
        knuckle_y = wrist_y_top + total_height * 0.50
        warnings.append(
            f"knuckle_fallback:detected={knuckle_y_detected:.1f}mm,used_fallback={knuckle_y:.1f}mm"
        )
        print(f"  [WARN] Knuckle detection failed (detected={knuckle_y_detected:.1f}mm), "
              f"using anatomical fallback={knuckle_y:.1f}mm")
    print(f"  Knuckle Y: {knuckle_y:.1f}mm")

    # --- Step 3: Fingertip detection ---
    tips_mm, n_fallbacks = detect_fingertips(pts_mm, n_fingers=5, handedness=handedness)
    if n_fallbacks > 0:
        print(f"  [WARN] Fingertip fallback: {n_fallbacks} band(s) < 10 pts")

    # --- Step 4: Finger lengths (wrist center → fingertip) ---
    finger_lengths = np.maximum(0.0, tips_mm[:, 1] - base_y)  # (5,)

    # --- Step 5: Palm dimensions ---
    # palm_height: dari wrist_y_top ke knuckle_y (tinggi telapak yang sebenarnya)
    palm_height_mm = max(0.0, knuckle_y - wrist_y_top)

    # palm_width: X-extent di irisan knuckle row (±10mm di sekitar knuckle_y)
    slice_h = 10.0
    knuckle_slice = pts_mm[
        (pts_mm[:, 1] >= knuckle_y - slice_h) &
        (pts_mm[:, 1] <= knuckle_y + slice_h)
    ]
    if len(knuckle_slice) > 10:
        palm_width_mm = float(knuckle_slice[:, 0].max() - knuckle_slice[:, 0].min())
    else:
        # Fallback: X-extent wrist ROI
        palm_width_mm = float(wrist_pts[:, 0].max() - wrist_pts[:, 0].min())

    # palm_depth_std: std dev Z di area telapak (PCA-aligned, wrist → knuckle).
    # Mengukur seberapa rata/melengkung permukaan telapak dalam arah depth kamera.
    # - Nilai kecil → telapak relatif rata (flat palm)
    # - Nilai besar → telapak melengkung atau cekung (hollow palm)
    # Menggunakan std dev (bukan Z-extent) agar robust terhadap outlier satu titik.
    palm_area_mask = (pts_mm[:, 1] >= float(wrist_pts[:, 1].min())) & \
                     (pts_mm[:, 1] <= knuckle_y)
    palm_area_pts = pts_mm[palm_area_mask]
    palm_depth_std_mm = float(np.std(palm_area_pts[:, 2])) if len(palm_area_pts) > 10 else 0.0

    # --- Step 6: Finger widths dan celah antar jari ---
    finger_widths_mm, inter_finger_gaps_mm = compute_finger_widths_and_gaps(
        pts_mm, tips_mm, knuckle_y)

    # --- Step 7: Palm surface curvature ---
    # Mean |1 - |nz|| di area telapak (wrist ke knuckle)
    mean_curvature = 0.0
    if nrm is not None:
        palm_mask = (pts_mm[:, 1] >= float(wrist_pts[:, 1].min())) & \
                    (pts_mm[:, 1] <= knuckle_y)
        palm_nz = np.abs(nrm[palm_mask, 2])
        mean_curvature = float(np.mean(1.0 - palm_nz)) if len(palm_nz) > 0 else 0.0

    # --- Step 8: Validasi kualitas ---
    # Catatan: knuckle_detection TIDAK lagi menjadi quality gate karena iOS Vision
    # sudah memfilter depth ke area wrist–fingertip. palm_height_mm disimpan sebagai
    # informasi saja tapi tidak memblokir frame yang valid.
    #
    # scan_distance digunakan sebagai metadata kualitas, BUKAN untuk normalisasi ukuran.
    # Alasan: point cloud sudah menggunakan koordinat 3D riil (mm) via unprojection dengan
    # camera intrinsics — ukuran terukur tidak berubah dengan jarak scan. Yang berubah
    # adalah kepadatan titik dan noise sensor (semakin jauh, semakin sparse & noisy).
    #
    # Range yang dapat diterima: 180–450mm.
    # Threshold sebelumnya 200mm terlalu konservatif — inspeksi empiris menunjukkan frame
    # pada 192–199mm memiliki geometri lengkap dan nilai fitur yang konsisten dengan frame
    # di atas 200mm (point_count, finger_lengths, palm_width sebanding).
    # iOS TrueDepth secara fisik dapat mengumpulkan data mulai ~100mm; batas bawah 180mm
    # memberikan margin aman tanpa membuang frame yang sebenarnya valid.
    SCAN_DIST_MIN_MM = 150.0
    SCAN_DIST_MAX_MM = 450.0
    # quality_issues sudah diinisialisasi di Step 2
    if n_fallbacks >= 2:
        quality_issues.append(f"fingertip_fallback:{n_fallbacks}_fingers")
    if inter_finger_gaps_mm and min(inter_finger_gaps_mm) < 1.0:
        quality_issues.append(
            f"fingers_too_close:min_gap={min(inter_finger_gaps_mm):.1f}mm")
    if not (SCAN_DIST_MIN_MM <= scan_distance_mm <= SCAN_DIST_MAX_MM):
        quality_issues.append(
            f"scan_distance_out_of_range:{scan_distance_mm:.0f}mm "
            f"(optimal {SCAN_DIST_MIN_MM:.0f}–{SCAN_DIST_MAX_MM:.0f}mm)")

    scan_id = Path(ply_path).parent.name

    result = {
        "scan_id":     scan_id,
        "point_count": len(pcd.points),
        "handedness":  handedness,
        # Panjang jari: wrist center → ujung jari (mm)
        "finger_lengths_mm":   [round(float(v), 2) for v in finger_lengths],
        # Dimensi telapak (mm)
        "palm_width_mm":       round(palm_width_mm, 2),
        "palm_height_mm":      round(palm_height_mm, 2),
        # Kelengkungan permukaan telapak dalam arah depth (std Z PCA-aligned)
        "palm_depth_std_mm":   round(palm_depth_std_mm, 2),
        # Jarak fisik kamera → telapak saat scan (median Z koordinat asli)
        "scan_distance_mm":    round(scan_distance_mm, 2),
        # Celah kosong antar jari (mm): [thumb–index, index–middle, middle–ring, ring–pinky]
        "inter_finger_gaps_mm": inter_finger_gaps_mm,
        # Lebar tiap jari di zona jari (mm)
        "finger_widths_mm":    finger_widths_mm,
        # Kelengkungan permukaan telapak
        "mean_palm_curvature": round(mean_curvature, 4),
        # Validitas deteksi
        "quality_issues": quality_issues,
        # Peringatan non-gating (mis. knuckle_fallback) — fitur geometri kurang andal
        # tapi point cloud tetap valid untuk CNN
        "warnings":       warnings,
        "is_valid":       len(quality_issues) == 0,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    valid_str = "OK" if result["is_valid"] else f"INVALID: {'; '.join(quality_issues)}"
    if warnings:
        valid_str += f" (warnings: {'; '.join(warnings)})"
    print(f"Saved geometry features → {output_path}  [{valid_str}]")
    print(f"  Finger lengths (mm)    : {result['finger_lengths_mm']}")
    print(f"  Palm: {palm_width_mm:.1f} x {palm_height_mm:.1f} mm (W×H), depth_std={palm_depth_std_mm:.1f} mm")
    print(f"  Scan distance          : {scan_distance_mm:.1f} mm")
    print(f"  Inter-finger gaps (mm) : {result['inter_finger_gaps_mm']}")
    print(f"  Finger widths (mm)     : {result['finger_widths_mm']}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstrak fitur geometri dari PLY telapak tangan")
    parser.add_argument("ply", help="input PLY file (registered point cloud)")
    parser.add_argument("--output", default=None,
                        help="output JSON path (default: <ply_dir>/geometry.json)")
    parser.add_argument("--handedness", default=None,
                        choices=["right", "left", "unknown"],
                        help="handedness dari scan. Jika tidak diisi, otomatis dibaca dari "
                             "metadata.json di folder yang sama dengan PLY.")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    output_path = args.output or str(ply_path.parent / "geometry.json")

    # Resolve handedness: CLI arg → metadata.json sibling → "unknown"
    handedness = "unknown"
    if args.handedness:
        handedness = args.handedness
    else:
        meta_path = ply_path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            handedness = meta.get("handedness", "unknown")
            if handedness != "unknown":
                print(f"Handedness dibaca dari {meta_path}: {handedness}")

    extract_geometry(str(ply_path), output_path, handedness=handedness)
