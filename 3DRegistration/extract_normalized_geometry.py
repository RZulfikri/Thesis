"""
[DEPRECATED] extract_normalized_geometry.py

File ini tidak lagi digunakan dalam pipeline CNN.
Normalisasi fitur geometri (StandardScaler) sekarang dilakukan di dataset.py
saat build dataset training — bukan per-frame di sini.

Alasan: normalisasi harus di-fit dari distribusi seluruh training set
(bukan tiap sampel sendiri), agar tidak ada data leakage dan ukuran absolut
tangan sebagai fitur biometrik tetap terjaga.

File ini dipertahankan hanya untuk keperluan analisis/visualisasi manual.
Gunakan dataset.py untuk pipeline training.
----------------------------------------------------------------------
extract_normalized_geometry.py — Normalisasi fitur geometri ke skala canonical 100mm.

Semua ukuran absolut (mm) dinormalisasi terhadap palm_width_mm sebagai scale reference:
    normalized = (x_mm / palm_width_mm) * 100

Hasilnya adalah "normalized mm" — scale-invariant tapi tetap dalam satuan mm-like.
palm_width_mm sendiri menjadi 100.0 (by definition).
palm_depth_std_mm dinormalisasi dengan formula yang sama → rasio kelengkungan/width telapak.

scan_distance_mm TIDAK dinormalisasi dan bukan digunakan sebagai scale reference.
Penjelasan: point cloud TrueDepth sudah menggunakan koordinat 3D riil (mm) melalui
unprojection dengan camera intrinsics — ukuran terukur (finger_lengths, palm_width, dll.)
tidak berubah dengan jarak scan. scan_distance disimpan sebagai metadata kualitas scan
(kepadatan titik dan noise sensor berkorelasi dengan jarak; optimal 200–450mm).

mean_palm_curvature tidak diubah karena sudah dimensionless.

Key name output SAMA dengan geometry.json (field yang sama, nilai sudah dinormalisasi).
Ini memudahkan dataset.py untuk menggunakan GEOMETRY_KEYS yang sama.

Kondisi INVALID:
  - scale_valid=False jika palm_width_mm < 40mm → semua field mm diisi None
  - is_valid=False (dari geometry.json) → diteruskan apa adanya

Output: normalized_geometry.json per frame di folder yang sama dengan geometry.json.

Usage:
  # Satu scan:
  python extract_normalized_geometry.py result_frames/rahmat/20260401_200613/frame_00/geometry.json

  # Semua scan (frame layout):
  python extract_normalized_geometry.py --all [--result_dir result_frames]
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------

def extract_normalized_geometry(geo: dict, output_path: str) -> dict:
    """
    Normalisasi geometry.json: semua ukuran mm dinormalisasi ke canonical 100mm palm.

    formula: normalized = (x_mm / palm_width_mm) * 100
    palm_width_mm sendiri → 100.0 (scale reference)
    mean_palm_curvature → tidak berubah (sudah scale-invariant)

    scale_valid=False jika palm_width_mm < 40mm (skala tidak bisa dipercaya).
    Dalam kasus itu, semua field mm diisi None.

    Returns dict yang juga disimpan ke output_path.
    """
    scale_ref   = float(geo.get("palm_width_mm", 0.0))
    scale_valid = scale_ref >= 40.0
    factor      = (100.0 / scale_ref) if scale_valid and scale_ref > 0 else None

    def norm(v):
        if factor is None or v is None:
            return None
        return round(float(v) * factor, 4)

    def norm_list(lst):
        if lst is None:
            return None
        return [norm(v) for v in lst]

    result = {
        "scan_id":            geo.get("scan_id"),
        # Nilai mm asli sebelum normalisasi — acuan dimensi fisik scan
        "scale_ref_mm":       round(scale_ref, 2),          # palm_width asli (X, mm)
        "scan_distance_mm":   round(float(geo.get("scan_distance_mm", 0.0)), 2),  # jarak kamera asli
        "scale_valid":        scale_valid,
        # --- Fitur dinormalisasi (key name sama dengan geometry.json) ---
        # Panjang jari / palm_width × 100
        "finger_lengths_mm":    norm_list(geo.get("finger_lengths_mm")),
        # palm_width → selalu 100.0 (reference)
        "palm_width_mm":        100.0 if scale_valid else None,
        # palm_height / palm_width × 100
        "palm_height_mm":       norm(geo.get("palm_height_mm")),
        # palm_depth_std / palm_width × 100 — rasio kelengkungan permukaan terhadap lebar
        "palm_depth_std_mm":    norm(geo.get("palm_depth_std_mm")),
        # scan_distance tidak dinormalisasi — ini nilai referensi absolut, bukan fitur relatif
        # Celah antar jari / palm_width × 100
        "inter_finger_gaps_mm": norm_list(geo.get("inter_finger_gaps_mm")),
        # Lebar jari / palm_width × 100
        "finger_widths_mm":     norm_list(geo.get("finger_widths_mm")),
        # Kelengkungan telapak — tidak berubah (dimensionless)
        "mean_palm_curvature":  geo.get("mean_palm_curvature"),
        # --- Validitas dari geometry.json ---
        "quality_issues": geo.get("quality_issues", []),
        "is_valid":       geo.get("is_valid", True),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "OK" if scale_valid else "scale_valid=False (palm_width terlalu kecil)"
    print(f"Saved normalized_geometry.json → {output_path}  [{status}]")
    return result


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_all(result_dir: Path, force: bool):
    # Dukung frame layout (result_frames/*/*/frame_*/geometry.json)
    # dan session layout (result/*/*/geometry.json)
    frame_geo_files = sorted(result_dir.glob("*/*/frame_*/geometry.json"))
    if frame_geo_files:
        geo_files = frame_geo_files
        print(f"Frame layout: {len(geo_files)} file ditemukan")
    else:
        geo_files = sorted(result_dir.glob("*/*/geometry.json"))
        print(f"Session layout: {len(geo_files)} file ditemukan")

    if not geo_files:
        print(f"Tidak ada geometry.json di {result_dir}")
        return

    ok = warn = skip = 0
    for gf in geo_files:
        out = gf.parent / "normalized_geometry.json"
        if out.exists() and not force:
            skip += 1
            continue
        geo = json.load(gf.open())
        result = extract_normalized_geometry(geo, str(out))
        if result["scale_valid"]:
            ok += 1
        else:
            warn += 1

    print(f"\nSelesai: {ok} OK, {warn} scale_valid=False, {skip} dilewati (sudah ada)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalisasi fitur geometri ke skala canonical 100mm palm")
    parser.add_argument("geometry_json", nargs="?", default=None,
                        help="path satu geometry.json (opsional jika --all)")
    parser.add_argument("--all", action="store_true",
                        help="proses semua geometry.json di result_dir")
    parser.add_argument("--result_dir", default="result_frames",
                        help="direktori hasil untuk mode --all (default: result_frames)")
    parser.add_argument("--force", action="store_true",
                        help="timpa normalized_geometry.json yang sudah ada")
    args = parser.parse_args()

    if args.all:
        run_all(Path(args.result_dir), force=args.force)
    elif args.geometry_json:
        geo_path = Path(args.geometry_json)
        out_path = geo_path.parent / "normalized_geometry.json"
        geo = json.load(geo_path.open())
        extract_normalized_geometry(geo, str(out_path))
    else:
        parser.print_help()
