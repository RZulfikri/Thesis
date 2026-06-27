"""
reextract_all_geometry.py — Re-extract geometry.json untuk SELURUH dataset.

Menggunakan extract_geometry.py yang sudah di-patch (hotfix knuckle fallback).
Memastikan konsistensi preprocessing untuk seluruh penelitian.

Usage:
    cd 3DRegistration
    python3 reextract_all_geometry.py [--data_dir ../3DCNN/dataset] [--force] [--workers N]

Features:
  - Resume-able: lewati frame yang sudah diproses (kecuali --force)
  - Preserve handedness: baca dari geometry.json lama
  - Log file: reextract_<timestamp>.log
  - Progress report setiap 100 frame
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from extract_geometry import extract_geometry


def _find_all_frames(data_dir: Path) -> list[Path]:
    """Temukan semua output.ply di dataset."""
    return sorted(data_dir.rglob("frame_*/output.ply"))


def _get_handedness(ply_path: Path) -> str:
    """Baca handedness dari geometry.json yang sudah ada, atau unknown."""
    geo_path = ply_path.parent / "geometry.json"
    if geo_path.exists():
        try:
            with open(geo_path) as f:
                geo = json.load(f)
            return geo.get("handedness", "unknown")
        except Exception:
            pass
    return "unknown"


def _process_one(ply_path: Path, force: bool) -> dict:
    """
    Proses satu frame. Return dict dengan status.
    """
    geo_path = ply_path.parent / "geometry.json"

    if geo_path.exists() and not force:
        # Cek apakah sudah hasil dari hotfix (ada field knuckle_fallback di quality_issues)
        try:
            with open(geo_path) as f:
                geo = json.load(f)
            # Jika sudah ada knuckle_fallback, anggap sudah diproses dengan hotfix
            issues = geo.get("quality_issues", [])
            if any("knuckle_fallback" in i for i in issues):
                return {
                    "ply": str(ply_path),
                    "status": "skipped_already_hotfix",
                    "error": None,
                }
        except Exception:
            pass

    handedness = _get_handedness(ply_path)
    try:
        result = extract_geometry(str(ply_path), str(geo_path), handedness=handedness)
        return {
            "ply": str(ply_path),
            "status": "ok",
            "is_valid": result.get("is_valid", True),
            "quality_issues": result.get("quality_issues", []),
            "error": None,
        }
    except Exception as e:
        return {
            "ply": str(ply_path),
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description="Re-extract geometry.json untuk seluruh dataset")
    parser.add_argument("--data_dir", default="../3DCNN/dataset", help="Root dataset directory")
    parser.add_argument("--force", action="store_true", help="Reprocess semua frame, termasuk yang sudah ada")
    parser.add_argument("--workers", type=int, default=1, help="Jumlah parallel workers (default: 1, sequential)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Error: data_dir '{data_dir}' tidak ditemukan")

    frames = _find_all_frames(data_dir)
    if not frames:
        sys.exit(f"Tidak ada output.ply ditemukan di {data_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"reextract_{timestamp}.log")
    report_path = Path(f"reextract_{timestamp}_report.json")

    print(f"Re-extract Geometry — {len(frames)} frame ditemukan")
    print(f"  Dataset: {data_dir.resolve()}")
    print(f"  Mode: {'FORCE' if args.force else 'SKIP existing'}")
    print(f"  Workers: {args.workers}")
    print(f"  Log: {log_path}")
    print()

    results = []
    ok_count = skip_count = error_count = 0
    t_start = time.time()

    with open(log_path, "w") as log_f:
        log_f.write(f"# Re-extract Geometry Log — {datetime.now().isoformat()}\n")
        log_f.write(f"# Frames: {len(frames)}\n")
        log_f.write(f"# Force: {args.force}\n")
        log_f.write(f"# Workers: {args.workers}\n\n")

        if args.workers <= 1:
            # Sequential mode
            for i, ply_path in enumerate(frames, 1):
                res = _process_one(ply_path, args.force)
                results.append(res)

                status = res["status"]
                if status == "ok":
                    ok_count += 1
                elif status == "skipped_already_hotfix":
                    skip_count += 1
                else:
                    error_count += 1
                    log_f.write(f"[ERROR] {res['ply']}\n{res.get('traceback', res['error'])}\n\n")

                if i % 100 == 0 or i == len(frames):
                    elapsed = time.time() - t_start
                    fps = i / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {i}/{len(frames)}  OK={ok_count} Skip={skip_count} Err={error_count}  "
                          f"({fps:.1f} frame/s)")
        else:
            # Parallel mode
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_process_one, p, args.force): p for p in frames}
                for i, future in enumerate(as_completed(futures), 1):
                    res = future.result()
                    results.append(res)

                    status = res["status"]
                    if status == "ok":
                        ok_count += 1
                    elif status == "skipped_already_hotfix":
                        skip_count += 1
                    else:
                        error_count += 1
                        log_f.write(f"[ERROR] {res['ply']}\n{res.get('traceback', res['error'])}\n\n")

                    if i % 100 == 0 or i == len(frames):
                        elapsed = time.time() - t_start
                        fps = i / elapsed if elapsed > 0 else 0
                        print(f"  Progress: {i}/{len(frames)}  OK={ok_count} Skip={skip_count} Err={error_count}  "
                              f"({fps:.1f} frame/s)")

    # Summary
    elapsed = time.time() - t_start
    summary = {
        "timestamp": timestamp,
        "total_frames": len(frames),
        "ok": ok_count,
        "skipped": skip_count,
        "error": error_count,
        "elapsed_seconds": round(elapsed, 2),
        "frames_per_second": round(len(frames) / elapsed, 2) if elapsed > 0 else 0,
    }

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SELESAI")
    print(f"  Total : {len(frames)}")
    print(f"  OK    : {ok_count}")
    print(f"  Skip  : {skip_count}")
    print(f"  Error : {error_count}")
    print(f"  Waktu : {elapsed:.1f}s ({summary['frames_per_second']:.1f} frame/s)")
    print(f"  Log   : {log_path}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")

    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
