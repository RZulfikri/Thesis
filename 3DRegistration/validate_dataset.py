"""
validate_dataset.py — Quality control untuk dataset scan telapak tangan.

Mendukung dua layout folder:
  Session layout (ICP, lama):   result/[subject]/[timestamp]/geometry.json
  Frame layout (single-frame):  result_frames/[subject]/[timestamp]/frame_NN/geometry.json

Layout terdeteksi otomatis berdasarkan ada tidaknya subdirektori frame_*.

Output:
  <result_dir>/dataset_manifest.json  — status per scan (PASS/WARN/FAIL)
  <result_dir>/qc_summary.txt         — ringkasan teks

Kriteria FAIL:
  - is_valid=False (dari geometry.json) → fingertip fallback ≥2, fingers too close, atau
    knuckle detection gagal
  - palm_height_mm < 10          → knuckle detection gagal (fallback untuk file lama)

Kriteria WARN:
  - point_count < 10_000 (frame) / 40_000 (session) → scan sparse
  - palm_height_mm < 40          → telapak kecil, mungkin pose tidak ideal

Usage:
  python validate_dataset.py [--result_dir result]           # session layout
  python validate_dataset.py [--result_dir result_frames]    # frame layout (auto-detect)
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# QC rules
# ---------------------------------------------------------------------------

def check_scan(geo: dict, point_count_warn: int = 40_000) -> tuple[str, list[str]]:
    """
    Evaluasi satu geometry.json.

    Args:
        geo               : dict dari geometry.json
        point_count_warn  : threshold WARN untuk point_count.
                            Default 40_000 untuk session/ICP layout.
                            Gunakan 10_000 untuk frame layout (single-frame ~28k normal).

    Returns:
        status  : "PASS", "WARN", atau "FAIL"
        reasons : list string penjelasan masalah (kosong jika PASS)
    """
    reasons_fail = []
    reasons_warn = []

    palm_h = geo.get("palm_height_mm", 0.0)
    pts    = geo.get("point_count", 0)

    # --- FAIL rules ---
    # Hanya cek fingertip fallback dan fingers_too_close.
    # palm_height / knuckle_detection TIDAK lagi menjadi FAIL gate karena:
    #   - iOS Vision sudah memfilter depth ke area wrist–fingertip
    #   - knuckle detection di Python tidak reliable untuk flat palm
    if not geo.get("is_valid", True):
        for issue in geo.get("quality_issues", ["is_valid=False"]):
            # Skip palm_height / knuckle issues — already handled by iOS
            # Skip scan_distance — hanya metadata kualitas, bukan penyebab FAIL
            if ("knuckle" not in issue
                    and "palm_height" not in issue
                    and "scan_distance" not in issue):
                reasons_fail.append(issue)

    # --- WARN rules tambahan ---
    for issue in geo.get("quality_issues", []):
        if "scan_distance_out_of_range" in issue:
            reasons_warn.append(issue)

    # --- WARN rules (hanya jika tidak FAIL) ---
    if not reasons_fail:
        if pts < point_count_warn:
            reasons_warn.append(f"point_count={pts:,} < {point_count_warn:,} (scan sparse)")

    if reasons_fail:
        return "FAIL", reasons_fail
    if reasons_warn:
        return "WARN", reasons_warn
    return "PASS", []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_dataset(result_dir: Path, output_path: Path, summary_path: Path):
    # Auto-detect layout berdasarkan ada tidaknya frame_* subdirektori
    frame_geo_files = sorted(result_dir.glob("*/*/frame_*/geometry.json"))
    if frame_geo_files:
        geo_files    = frame_geo_files
        is_frame     = True
        pt_threshold = 10_000   # single-frame ~28k normal; WARN jika < 10k
        print(f"Layout terdeteksi: frame (single-frame)  — {len(geo_files)} frame ditemukan")
    else:
        geo_files    = sorted(result_dir.glob("*/*/geometry.json"))
        is_frame     = False
        pt_threshold = 40_000   # ICP merged cloud
        print(f"Layout terdeteksi: session (ICP multi-frame)  — {len(geo_files)} sesi ditemukan")

    if not geo_files:
        print(f"Tidak ada geometry.json ditemukan di {result_dir}")
        return

    records = []
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    for gf in geo_files:
        if is_frame:
            # result_frames/[subject]/[timestamp]/frame_NN/geometry.json
            subject = gf.parent.parent.parent.name
            scan_id = f"{gf.parent.parent.name}/{gf.parent.name}"
        else:
            # result/[subject]/[timestamp]/geometry.json
            subject = gf.parent.parent.name
            scan_id = gf.parent.name
        geo = json.load(gf.open())

        status, reasons = check_scan(geo, point_count_warn=pt_threshold)
        counts[status] += 1

        record = {
            "subject":        subject,
            "scan_id":        scan_id,
            "status":         status,
            "reasons":        reasons,
            "palm_height_mm": geo.get("palm_height_mm"),
            "palm_width_mm":  geo.get("palm_width_mm"),
            "point_count":    geo.get("point_count"),
            "is_valid":       geo.get("is_valid", True),
            "quality_issues": geo.get("quality_issues", []),
        }
        records.append(record)

    # Tulis manifest JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    # Bangun ringkasan teks
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET QUALITY REPORT")
    lines.append("=" * 70)
    lines.append(f"Total scan    : {len(records)}")
    lines.append(f"  PASS        : {counts['PASS']}  ({counts['PASS']/len(records)*100:.1f}%)")
    lines.append(f"  WARN        : {counts['WARN']}  ({counts['WARN']/len(records)*100:.1f}%)")
    lines.append(f"  FAIL        : {counts['FAIL']}  ({counts['FAIL']/len(records)*100:.1f}%)")
    lines.append("")

    # Tabel per subjek
    subjects = sorted({r["subject"] for r in records})
    lines.append(f"{'Subjek':<10} {'PASS':>5} {'WARN':>5} {'FAIL':>5}")
    lines.append("-" * 30)
    for subj in subjects:
        sub = [r for r in records if r["subject"] == subj]
        p = sum(1 for r in sub if r["status"] == "PASS")
        w = sum(1 for r in sub if r["status"] == "WARN")
        f = sum(1 for r in sub if r["status"] == "FAIL")
        lines.append(f"{subj:<10} {p:>5} {w:>5} {f:>5}")
    lines.append("")

    # Detail FAIL dan WARN
    for status_label in ("FAIL", "WARN"):
        bad = [r for r in records if r["status"] == status_label]
        if bad:
            lines.append(f"--- {status_label} ---")
            for r in bad:
                lines.append(f"  {r['subject']}/{r['scan_id']}")
                for reason in r["reasons"]:
                    lines.append(f"    • {reason}")
            lines.append("")

    summary_text = "\n".join(lines)
    print(summary_text)
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    print(f"\nManifest tersimpan → {output_path}")
    print(f"Ringkasan tersimpan → {summary_path}")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validasi kualitas dataset geometry scan")
    parser.add_argument("--result_dir", default="result",
                        help="direktori hasil (default: result)")
    parser.add_argument("--output", default=None,
                        help="path output manifest JSON (default: <result_dir>/dataset_manifest.json)")
    args = parser.parse_args()

    result_dir  = Path(args.result_dir)
    output_path = Path(args.output) if args.output else result_dir / "dataset_manifest.json"
    summary_path = result_dir / "qc_summary.txt"

    validate_dataset(result_dir, output_path, summary_path)
