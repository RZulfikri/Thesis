"""
process_single_frames.py — Batch pipeline: proses setiap depth frame sebagai
satu data sample mandiri (tanpa ICP registration).

Pendekatan ini menggantikan pipeline ICP multi-frame di process_all_scans.py.
Setiap depth[NN].bin dalam folder sesi diproses secara independen.

Output per frame:
  result_frames/[label]/[timestamp]/frame_[NN]/
      output.ply              ← single-frame PLY setelah DBSCAN isolation
      geometry.json           ← fitur biometrik (nilai mm absolut)
      cnn_input.npy           ← (N, 6) float32, PCA-aligned + unit-sphere

Normalisasi fitur geometri (StandardScaler) dilakukan di dataset.py saat
build dataset training — bukan per-frame di sini.

Usage:
  python process_single_frames.py [--data_dir dataset] [--out_dir result_frames]
  python process_single_frames.py --data_dir dataset/rahmat_20260401_200613  # satu sesi
  python process_single_frames.py --force           # proses ulang semua
  python process_single_frames.py --skip_geometry   # lewati geometry.json
  python process_single_frames.py --skip_cnn        # lewati cnn_input.npy
  python process_single_frames.py --min_points 2000 # filter frame sparse
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from extract_geometry import extract_geometry
from preprocess_for_cnn import preprocess_full
from validate_dataset import check_scan
from lib.single_frame import process_single_frame, DEFAULT_ARGS


# ---------------------------------------------------------------------------
# Parse label & timestamp dari nama folder
# ---------------------------------------------------------------------------

def parse_scan_folder(folder_name: str):
    m = re.match(r'^(.+)_(\d{8}_\d{6})$', folder_name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(
    scan_folder: Path,
    depth_file: Path,
    out_dir: Path,
    label: str,
    timestamp: str,
    handedness: str,
    force: bool,
    min_points: int,
    skip_geometry: bool,
    skip_cnn: bool,
) -> dict:
    frame_num  = re.sub(r'\D', '', depth_file.stem).zfill(2)  # "depth00" → "00"
    frame_name = f"frame_{frame_num}"
    frame_dir  = out_dir / label / timestamp / frame_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    ply_path = frame_dir / "output.ply"
    result   = {"frame_id": frame_name, "status": "ok", "errors": []}

    # Step 1: Single-frame PLY
    if ply_path.exists() and not force:
        print(f"    [skip] output.ply sudah ada")
    else:
        pcd = process_single_frame(
            calibration_file=str(scan_folder / "calibration.json"),
            depth_file=str(depth_file),
            out_ply=str(ply_path),
            args=DEFAULT_ARGS,
            min_points=min_points,
        )
        if pcd is None or not ply_path.exists():
            result["status"] = "failed"
            result["errors"].append("isolasi foreground gagal atau terlalu sparse")
            return result

    # Step 2: Geometry features
    geo      = {}
    geo_path = frame_dir / "geometry.json"
    if not skip_geometry:
        if geo_path.exists() and not force:
            print(f"    [skip] geometry.json sudah ada")
            with open(geo_path) as f:
                geo = json.load(f)
        else:
            try:
                print(f"    [geometry] Ekstrak fitur geometri...")
                geo = extract_geometry(str(ply_path), str(geo_path), handedness=handedness)
            except Exception as e:
                result["errors"].append(f"geometry: {e}")

    # Cek validitas frame sebelum lanjut ke step 3 & 4
    if geo and not geo.get("is_valid", True):
        quality_issues = geo.get("quality_issues", [])
        print(f"    [INVALID] {'; '.join(quality_issues)}")
        invalid_info = {
            "frame_id":      frame_name,
            "quality_issues": quality_issues,
        }
        with open(frame_dir / "invalid_frame.json", "w") as f:
            json.dump(invalid_info, f, indent=2)
        result["status"] = "invalid"
        result["errors"].extend(quality_issues)
        result["quality_status"]  = "FAIL"
        result["quality_reasons"] = quality_issues
        result["point_count"]     = geo.get("point_count", 0)
        return result

    # Step 3: QC
    # normalized_geometry.json tidak lagi dibuat per-frame.
    # Normalisasi (StandardScaler) dilakukan di dataset.py saat build dataset training.
    if geo and not skip_geometry:
        quality_status, quality_reasons = check_scan(geo, point_count_warn=10_000)
        result["quality_status"]  = quality_status
        result["quality_reasons"] = quality_reasons
        result["point_count"]     = geo.get("point_count", 0)

        if quality_status == "FAIL":
            print(f"    [QC] FAIL: {'; '.join(quality_reasons)}")
        elif quality_status == "WARN":
            print(f"    [QC] WARN: {'; '.join(quality_reasons)}")

    # Step 4: CNN input (hanya untuk frame yang valid)
    if not skip_cnn:
        cnn_path = frame_dir / "cnn_input.npy"
        if cnn_path.exists() and not force:
            print(f"    [skip] cnn_input.npy sudah ada")
        else:
            try:
                print(f"    [cnn] Simpan full cloud (PCA + unit sphere)...")
                preprocess_full(str(ply_path), str(cnn_path))
            except Exception as e:
                result["errors"].append(f"cnn: {e}")

    if result["errors"]:
        result["status"] = "partial"

    return result


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def process_session(
    scan_folder: Path,
    out_dir: Path,
    force: bool,
    min_points: int,
    skip_geometry: bool,
    skip_cnn: bool,
) -> dict:
    label, timestamp = parse_scan_folder(scan_folder.name)
    if label is None:
        return {
            "session_id": scan_folder.name,
            "status": "skip",
            "errors": ["format nama tidak cocok [label]_YYYYMMDD_HHMMSS"],
        }

    print(f"\n{'='*60}")
    print(f"Session : {scan_folder.name}")
    print(f"Label   : {label}   Timestamp : {timestamp}")
    print(f"{'='*60}")

    # Baca handedness dari metadata.json
    handedness = "unknown"
    meta_path  = scan_folder / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        handedness = meta.get("handedness", "unknown")
        if handedness != "unknown":
            print(f"  Handedness : {handedness}")

    # Temukan semua depth*.bin
    depth_files = sorted(scan_folder.glob("depth*.bin"))
    if not depth_files:
        print(f"  [skip] Tidak ada depth*.bin")
        return {
            "session_id": scan_folder.name,
            "label": label,
            "timestamp": timestamp,
            "status": "skip",
            "errors": ["tidak ada depth*.bin"],
        }

    print(f"  Frames : {len(depth_files)}")

    frame_results = []
    for depth_file in depth_files:
        print(f"\n  Frame: {depth_file.name}")
        fr = process_frame(
            scan_folder=scan_folder,
            depth_file=depth_file,
            out_dir=out_dir,
            label=label,
            timestamp=timestamp,
            handedness=handedness,
            force=force,
            min_points=min_points,
            skip_geometry=skip_geometry,
            skip_cnn=skip_cnn,
        )
        frame_results.append(fr)

    n_ok      = sum(1 for fr in frame_results if fr["status"] in ("ok", "partial"))
    n_fail    = sum(1 for fr in frame_results if fr["status"] == "failed")
    n_qc_fail = sum(1 for fr in frame_results if fr.get("quality_status") == "FAIL")

    print(f"\n  Hasil sesi: {n_ok}/{len(depth_files)} frame berhasil"
          f" ({n_qc_fail} QC FAIL)")

    return {
        "session_id":      scan_folder.name,
        "label":           label,
        "timestamp":       timestamp,
        "status":          "ok" if n_fail == 0 else "partial",
        "frames_total":    len(depth_files),
        "frames_ok":       n_ok,
        "frames_failed":   n_fail,
        "frames_qc_fail":  n_qc_fail,
        "frame_results":   frame_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Proses setiap depth frame sebagai satu data sample mandiri (tanpa ICP)")
    parser.add_argument("--data_dir",     default="dataset",
                        help="root dataset atau satu folder sesi (default: dataset)")
    parser.add_argument("--out_dir",      default="result_frames",
                        help="output root directory (default: result_frames)")
    parser.add_argument("--force",        action="store_true",
                        help="proses ulang meski output sudah ada")
    parser.add_argument("--min_points",   type=int, default=1000,
                        help="buang frame dengan titik < N setelah isolasi (default: 1000)")
    parser.add_argument("--skip_geometry", action="store_true",
                        help="lewati ekstraksi geometry.json dan normalized_geometry.json")
    parser.add_argument("--skip_cnn",     action="store_true",
                        help="lewati pembuatan cnn_input.npy")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    if not data_dir.exists():
        sys.exit(f"Error: '{data_dir}' tidak ditemukan")

    # Kumpulkan folder sesi
    if (data_dir / "calibration.json").exists():
        scan_folders = [data_dir]
    else:
        scan_folders = sorted(
            p for p in data_dir.iterdir()
            if p.is_dir() and re.search(r'_\d{8}_\d{6}$', p.name)
        )

    if not scan_folders:
        sys.exit(f"Tidak ada folder sesi di '{data_dir}'")

    print(f"Ditemukan {len(scan_folders)} folder sesi")
    print(f"Output   : {out_dir.resolve()}\n")

    t_start    = time.time()
    summaries  = []

    for scan_folder in scan_folders:
        summary = process_session(
            scan_folder=scan_folder,
            out_dir=out_dir,
            force=args.force,
            min_points=args.min_points,
            skip_geometry=args.skip_geometry,
            skip_cnn=args.skip_cnn,
        )
        summaries.append(summary)

    # ---- Summary table ----
    elapsed      = time.time() - t_start
    total_frames = sum(s.get("frames_total", 0) for s in summaries)
    total_ok     = sum(s.get("frames_ok",    0) for s in summaries)
    total_fail   = sum(s.get("frames_failed", 0) for s in summaries)
    total_qc     = sum(s.get("frames_qc_fail", 0) for s in summaries)

    print(f"\n{'='*80}")
    print(f"SUMMARY  ({len(summaries)} sesi,  {total_frames} frame,  {elapsed:.1f}s total)")
    print(f"  Berhasil  : {total_ok} frame")
    print(f"  Gagal     : {total_fail} frame")
    print(f"  QC FAIL   : {total_qc} frame")
    print(f"{'='*80}")
    print(f"{'Label':<18} {'Timestamp':<17} {'Total':>6} {'OK':>5} {'Fail':>5} {'QCFail':>7}")
    print(f"{'-'*80}")
    for s in summaries:
        if s.get("status") == "skip":
            continue
        lbl = s.get("label", s["session_id"])
        ts  = s.get("timestamp", "—")
        tot = s.get("frames_total", 0)
        ok  = s.get("frames_ok",    0)
        fl  = s.get("frames_failed", 0)
        qf  = s.get("frames_qc_fail", 0)
        print(f"{lbl:<18} {ts:<17} {tot:>6} {ok:>5} {fl:>5} {qf:>7}")

    print(f"\nOutput disimpan di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
