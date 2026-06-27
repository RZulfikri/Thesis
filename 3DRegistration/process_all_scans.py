"""
process_all_scans.py — Batch pipeline: register + extract features untuk setiap scan di dataset/.

Format nama folder input: [label]_YYYYMMDD_HHMMSS  (contoh: rahmat_20260401_200613)

Untuk setiap folder yang cocok, semua output disimpan ke:
  result/[label]/[timestamp]/
      output.ply           ← registered point cloud
      geometry.json        ← fitur geometri biometrik (33 nilai)
      texture.npy          ← tekstur kanonis
      cnn_input.npy        ← full PCA-aligned + unit-sphere cloud, (N, 6) float32
      cnn_input_fps.npy    ← FPS-downsampled ke n_points tetap, (n_points, 6) float32 [backup novelty]

Lewati langkah jika output sudah ada (gunakan --force untuk proses ulang).

Usage:
  python process_all_scans.py [--data_dir dataset] [--force] [--skip_registration]
  python process_all_scans.py --data_dir dataset/rahmat_20260401_200613  # satu scan
  python process_all_scans.py --skip_fps    # lewati FPS (hanya full cloud)
  python process_all_scans.py --skip_full   # lewati full cloud (hanya FPS)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from extract_geometry import extract_geometry
from extract_texture import extract_texture
from preprocess_for_cnn import preprocess_full, preprocess_fps
from validate_dataset import check_scan


# ---------------------------------------------------------------------------
# Parse label & timestamp dari nama folder
# ---------------------------------------------------------------------------

def parse_scan_folder(folder_name: str):
    """
    Ekstrak label dan timestamp dari nama folder [label]_YYYYMMDD_HHMMSS.
    Contoh: 'rahmat_20260401_200613' → label='rahmat', timestamp='20260401_200613'
    Contoh: 'rahmat_zulfikri_20260401_200613' → label='rahmat_zulfikri', timestamp='20260401_200613'

    Return: (label, timestamp) atau (None, None) jika format tidak cocok.
    """
    m = re.match(r'^(.+)_(\d{8}_\d{6})$', folder_name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Registration (calls run.py as subprocess to keep environment clean)
# ---------------------------------------------------------------------------

PALM_RUN_ARGS = [
    "--method", "0",
    "--min_depth", "0.10",
    "--max_depth", "0.50",
    "--max_point_dist", "0.015",
    "--normal_radius", "0.008",
    "--voxel_size", "0.001",
    "--keep_nearest_cluster", "1",
    "--foreground_depth_range", "0.12",
    "--cluster_connectivity_eps", "0.008",
    "--cluster_connectivity_min_points", "20",
    # Statistical outlier removal — std_ratio 1.0 lebih agresif dari 1.5
    "--outlier_nb_neighbors", "30",
    "--outlier_std_ratio", "1.0",
    # Radius outlier removal — minta lebih banyak neighbor dalam radius lebih ketat
    "--radius_outlier_nb_points", "20",
    "--radius_outlier_radius", "0.008",
    "--xy_clip", "3.0",
    "--viz", "0",
]


def result_folder(scan_folder: Path) -> Path:
    """Returns result/[label]/[timestamp]/, creating it if needed."""
    label, timestamp = parse_scan_folder(scan_folder.name)
    out = Path("result") / label / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_registration(scan_folder: Path, out_folder: Path, force: bool) -> bool:
    output_ply = out_folder / "output.ply"
    if output_ply.exists() and not force:
        print(f"  [skip] output.ply sudah ada")
        return True

    print(f"  [register] Menjalankan registrasi ICP...")
    cmd = [sys.executable, "run.py", str(scan_folder),
           "--output", str(output_ply)] + PALM_RUN_ARGS
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [error] Registrasi gagal untuk {scan_folder.name}")
        return False
    return output_ply.exists()


# ---------------------------------------------------------------------------
# Per-scan processing
# ---------------------------------------------------------------------------

def process_scan(scan_folder: Path, force: bool, skip_registration: bool,
                 n_points: int, texture_size: int,
                 skip_full: bool, skip_fps: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"Processing: {scan_folder.name}")
    print(f"{'='*60}")

    label, timestamp = parse_scan_folder(scan_folder.name)
    if label is None:
        print(f"  [skip] Nama folder tidak cocok format [label]_YYYYMMDD_HHMMSS: {scan_folder.name}")
        return {"scan_id": scan_folder.name, "status": "skip", "errors": ["format nama tidak cocok"]}

    print(f"  Label     : {label}")
    print(f"  Timestamp : {timestamp}")

    summary = {"scan_id": scan_folder.name, "label": label, "timestamp": timestamp,
               "status": "ok", "errors": []}

    out_folder = result_folder(scan_folder)
    print(f"  Output    : {out_folder}")

    # Step 1: Registrasi ICP → output.ply
    if not skip_registration:
        ok = run_registration(scan_folder, out_folder, force)
        if not ok:
            summary["status"] = "failed"
            summary["errors"].append("registrasi gagal")
            return summary
    else:
        print("  [skip] Registrasi dilewati (--skip_registration)")

    ply_path = out_folder / "output.ply"
    if not ply_path.exists():
        summary["status"] = "failed"
        summary["errors"].append("output.ply tidak ditemukan")
        return summary

    # Baca handedness dari metadata.json di folder scan (bukan output folder)
    handedness = "unknown"
    meta_path = scan_folder / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        handedness = meta.get("handedness", "unknown")
        if handedness != "unknown":
            print(f"  Handedness : {handedness} (dari metadata.json)")

    # Step 2: Fitur geometri → geometry.json
    geo_path = out_folder / "geometry.json"
    geo = {}
    if geo_path.exists() and not force:
        print(f"  [skip] geometry.json sudah ada")
        with open(geo_path) as f:
            geo = json.load(f)
    else:
        try:
            print("  [geometry] Ekstrak fitur geometri...")
            geo = extract_geometry(str(ply_path), str(geo_path), handedness=handedness)
        except Exception as e:
            summary["errors"].append(f"geometry: {e}")

    if geo:
        summary["finger_lengths_mm"] = geo.get("finger_lengths_mm", [])
        summary["palm_width_mm"]     = geo.get("palm_width_mm", 0)
        summary["point_count"]       = geo.get("point_count", 0)

        # QC check — tandai scan yang tidak layak training
        quality_status, quality_reasons = check_scan(geo)
        summary["quality_status"]  = quality_status
        summary["quality_reasons"] = quality_reasons
        if quality_status == "FAIL":
            print(f"  [QC] FAIL: {'; '.join(quality_reasons)}")
        elif quality_status == "WARN":
            print(f"  [QC] WARN: {'; '.join(quality_reasons)}")

    # Step 3: Tekstur → texture.npy
    tex_path = out_folder / "texture.npy"
    if tex_path.exists() and not force:
        print(f"  [skip] texture.npy sudah ada")
    else:
        try:
            print(f"  [texture] Ekstrak tekstur {texture_size}x{texture_size}...")
            extract_texture(str(ply_path), texture_size, str(tex_path))
        except Exception as e:
            summary["errors"].append(f"texture: {e}")

    # Step 4: Full PCA-aligned cloud → cnn_input.npy  (input utama GeoAtt-PointNet++)
    if not skip_full:
        cnn_path = out_folder / "cnn_input.npy"
        if cnn_path.exists() and not force:
            print(f"  [skip] cnn_input.npy sudah ada")
        else:
            try:
                print(f"  [cnn_full] Simpan full cloud (PCA + unit sphere)...")
                preprocess_full(str(ply_path), str(cnn_path))
            except Exception as e:
                summary["errors"].append(f"cnn_full: {e}")
    else:
        print("  [skip] cnn_input.npy dilewati (--skip_full)")

    # Step 5: FPS downsample → cnn_input_fps.npy  (backup novelty, ablation study)
    if not skip_fps:
        fps_path = out_folder / "cnn_input_fps.npy"
        if fps_path.exists() and not force:
            print(f"  [skip] cnn_input_fps.npy sudah ada")
        else:
            try:
                print(f"  [cnn_fps] FPS downsample → {n_points} titik...")
                preprocess_fps(str(ply_path), n_points, str(fps_path))
            except Exception as e:
                summary["errors"].append(f"cnn_fps: {e}")
    else:
        print("  [skip] cnn_input_fps.npy dilewati (--skip_fps)")

    # Step 6: (dihapus) normalized_geometry.json tidak lagi dibuat per-scan.
    # Normalisasi (StandardScaler) dilakukan di dataset.py saat build dataset training.

    if summary["errors"]:
        summary["status"] = "partial"

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-process all palm scans in dataset/")
    parser.add_argument("--data_dir", default="dataset",
                        help="root data directory or single scan folder (default: dataset)")
    parser.add_argument("--force", action="store_true",
                        help="reprocess even if output files already exist")
    parser.add_argument("--skip_registration", action="store_true",
                        help="skip ICP registration step (assume output.ply exists)")
    parser.add_argument("--n_points", type=int, default=1024,
                        help="jumlah titik untuk FPS backup (default: 1024)")
    parser.add_argument("--texture_size", type=int, default=256,
                        help="texture grid size HxW (default: 256)")
    parser.add_argument("--skip_full", action="store_true",
                        help="skip cnn_input.npy (full cloud)")
    parser.add_argument("--skip_fps",  action="store_true",
                        help="skip cnn_input_fps.npy (FPS backup)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        sys.exit(f"Error: '{data_dir}' not found")

    # Collect scan folders — format: [label]_YYYYMMDD_HHMMSS
    if (data_dir / "calibration.json").exists():
        # Single scan folder passed directly
        scan_folders = [data_dir]
    else:
        scan_folders = sorted(
            p for p in data_dir.iterdir()
            if p.is_dir() and re.search(r'_\d{8}_\d{6}$', p.name)
        )

    if not scan_folders:
        sys.exit(f"No scan_* folders found in '{data_dir}'")

    print(f"Found {len(scan_folders)} scan folder(s)")

    t_start = time.time()
    summaries = []

    for scan_folder in scan_folders:
        summary = process_scan(
            scan_folder,
            force=args.force,
            skip_registration=args.skip_registration,
            n_points=args.n_points,
            texture_size=args.texture_size,
            skip_full=args.skip_full,
            skip_fps=args.skip_fps,
        )
        summaries.append(summary)

    # Print summary table
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"SUMMARY  ({len(summaries)} scans,  {elapsed:.1f}s total)")
    print(f"{'='*80}")
    print(f"{'Label':<18} {'Timestamp':<17} {'Status':<10} {'Points':>7}  {'Quality':<8}")
    print(f"{'-'*80}")
    for s in summaries:
        pts  = s.get("point_count", "—")
        lbl  = s.get("label", s["scan_id"])
        ts   = s.get("timestamp", "—")
        qst  = s.get("quality_status", "—")
        print(f"{lbl:<18} {ts:<17} {s['status']:<10} {str(pts):>7}  {qst:<8}")
        for err in s.get("errors", []):
            print(f"  ⚠ {err}")
        for reason in s.get("quality_reasons", []):
            print(f"  ✗ {reason}")

    print(f"\nOutput disimpan di: {Path('result').resolve()}")


if __name__ == "__main__":
    main()
