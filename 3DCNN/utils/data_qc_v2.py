"""
utils/data_qc_v2.py — Data Quality Control v2 (Universal Objective Criteria).

Mendeteksi dan mengarantina sesi-sesi dengan within-session geometric instability.
Kriteria objektif, diterapkan uniform ke SEMUA subjek — bukan cherry-pick.

Cara kerja:
  1. Untuk tiap sesi (timestamp), hitung within-session std per fitur geometri
     (14 fitur, dihitung dari 10 frame per sesi).
  2. Hitung median within-session std GLOBAL (lintas semua subjek-sesi).
  3. Flag sesi sebagai "outlier" jika within-session std fitur mana pun
     melebihi k × median_global.
  4. Rename folder sesi yang ter-flag dengan prefix `_QC2_` (reversibel).
  5. Generate laporan markdown `qc_report.md` dengan justifikasi numerik.

Usage:
    python utils/data_qc_v2.py --data_dir dataset --k 7 --dry-run
    python utils/data_qc_v2.py --data_dir dataset --k 7  # execute

Parameter:
    --k       threshold multiplier (default: 7). Dari empiris:
              k=5  → ~23% sesi flagged (terlalu agresif untuk dataset ini)
              k=7  → ~21% sesi flagged
              k=10 → ~13% sesi flagged (default rekomendasi)
              k=15 → ~3% sesi flagged (terlalu konservatif)

Output:
    - Folder ter-flag di-rename dengan prefix `_QC2_`
    - Laporan: `eval_results/qc_v2/<timestamp>/qc_report.md`
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fitur geometri — harus cocok dengan geometry.json
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "finger_len_1",
    "finger_len_2",
    "finger_len_3",
    "finger_len_4",
    "finger_len_5",
    "palm_width",
    "palm_height",
    "palm_depth_std",
    "finger_width_1",
    "finger_width_2",
    "finger_width_3",
    "finger_width_4",
    "finger_width_5",
    "mean_palm_curvature",
]


def _load_geometry(geo_path: Path) -> list[float] | None:
    """Load geometry.json dan flatten ke list 14 nilai. Return None jika invalid."""
    try:
        with open(geo_path) as f:
            geo = json.load(f)
        fl = geo.get("finger_lengths_mm", [])
        fw = geo.get("finger_widths_mm", [])
        if len(fl) != 5 or len(fw) != 5:
            return None
        return [
            float(fl[0]),
            float(fl[1]),
            float(fl[2]),
            float(fl[3]),
            float(fl[4]),
            float(geo.get("palm_width_mm", 0)),
            float(geo.get("palm_height_mm", 0)),
            float(geo.get("palm_depth_std_mm", 0)),
            float(fw[0]),
            float(fw[1]),
            float(fw[2]),
            float(fw[3]),
            float(fw[4]),
            float(geo.get("mean_palm_curvature", 0)),
        ]
    except Exception:
        return None


def _scan_sessions(data_dir: Path) -> list[dict]:
    """
    Scan dataset dan kembalikan list session dict:
        {subject, session, frames: list[list[float]]}
    """
    sessions = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith(".") or subj_dir.name.startswith("_"):
            continue
        for ts_dir in sorted(subj_dir.iterdir()):
            if not ts_dir.is_dir() or ts_dir.name.startswith("_"):
                continue
            frames = []
            for frame_dir in sorted(ts_dir.iterdir()):
                if not frame_dir.is_dir() or not frame_dir.name.startswith("frame_"):
                    continue
                geo_path = frame_dir / "geometry.json"
                vals = _load_geometry(geo_path)
                if vals is not None:
                    frames.append(vals)
            if len(frames) >= 3:  # minimal 3 frame untuk std yang masuk akal
                sessions.append(
                    {
                        "subject": subj_dir.name,
                        "session": ts_dir.name,
                        "session_dir": ts_dir,
                        "frames": frames,
                    }
                )
    return sessions


def _compute_stats(sessions: list[dict]) -> tuple[dict[str, float], list[dict]]:
    """
    Hitung within-session std untuk setiap fitur dan setiap sesi.
    Return: (median_std_global, enriched_sessions)
    """
    session_stds = defaultdict(list)
    enriched = []

    for sess in sessions:
        arr = np.array(sess["frames"])  # (N_frames, 14)
        stds = arr.std(axis=0).tolist()
        means = arr.mean(axis=0).tolist()
        sess_stats = {
            "subject": sess["subject"],
            "session": sess["session"],
            "session_dir": sess["session_dir"],
            "stds": stds,
            "means": means,
            "n_frames": len(sess["frames"]),
        }
        enriched.append(sess_stats)
        for i, name in enumerate(FEATURE_NAMES):
            session_stds[name].append(stds[i])

    median_std = {name: float(np.median(session_stds[name])) for name in FEATURE_NAMES}
    return median_std, enriched


def _flag_sessions(enriched: list[dict], median_std: dict[str, float], k: float) -> list[dict]:
    """Flag sesi yang within-session std-nya melebihi k × median_global."""
    flagged = []
    for sess in enriched:
        reasons = []
        for i, name in enumerate(FEATURE_NAMES):
            threshold = k * median_std[name]
            if sess["stds"][i] > threshold:
                reasons.append(
                    {
                        "feature": name,
                        "std": round(sess["stds"][i], 4),
                        "threshold": round(threshold, 4),
                        "mean": round(sess["means"][i], 4),
                    }
                )
        if reasons:
            flagged.append({**sess, "reasons": reasons})
    return flagged


def _apply_quarantine(flagged: list[dict], dry_run: bool = False) -> list[tuple[Path, Path]]:
    """Rename folder sesi yang ter-flag dengan prefix `_QC2_`. Return list (src, dst)."""
    moved = []
    for item in flagged:
        src = item["session_dir"]
        dst_name = f"_QC2_{src.name}"
        dst = src.parent / dst_name

        if dst.exists():
            # Sudah pernah di-quarantine
            continue

        if dry_run:
            moved.append((src, dst))
            continue

        shutil.move(str(src), str(dst))
        moved.append((src, dst))
    return moved


def _generate_report(
    enriched: list[dict],
    flagged: list[dict],
    median_std: dict[str, float],
    k: float,
    data_dir: Path,
    output_dir: Path,
    dry_run: bool,
) -> str:
    """Generate laporan markdown. Return path ke file report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "qc_report.md"

    lines = [
        "# QC v2 Report — Universal Objective Criteria",
        "",
        f"**Timestamp:** {datetime.now().isoformat()}",
        f"**Dataset:** `{data_dir}`",
        f"**Threshold k:** {k}",
        f"**Mode:** {'DRY-RUN (no files moved)' if dry_run else 'EXECUTED (folders renamed)'}",
        "",
        "---",
        "",
        "## 1. Global Median Within-Session Std",
        "",
        "| Fitur | Median Std Global | Threshold (k×median) |",
        "|-------|-------------------|----------------------|",
    ]
    for name in FEATURE_NAMES:
        lines.append(
            f"| {name:<20} | {median_std[name]:>17.4f} | {k * median_std[name]:>20.4f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 2. Ringkasan",
            "",
            f"- Total sesi dianalisis: **{len(enriched)}**",
            f"- Sesi ter-flag: **{len(flagged)}** ({len(flagged) / len(enriched) * 100:.1f}%)",
            "",
            "---",
            "",
            "## 3. Detail Sesi Ter-Flag",
            "",
        ]
    )

    if not flagged:
        lines.append("Tidak ada sesi yang ter-flag.\n")
    else:
        # Group by subject
        by_subject = defaultdict(list)
        for item in flagged:
            by_subject[item["subject"]].append(item)

        for subject in sorted(by_subject.keys()):
            lines.append(f"### {subject}")
            lines.append("")
            for item in by_subject[subject]:
                lines.append(f"**{item['session']}**  (n_frames={item['n_frames']})")
                lines.append("")
                lines.append("| Fitur | Mean | Std | Threshold |")
                lines.append("|-------|------|-----|-----------|")
                for r in item["reasons"]:
                    lines.append(
                        f"| {r['feature']:<20} | {r['mean']:>8.2f} | {r['std']:>8.2f} | {r['threshold']:>9.2f} |"
                    )
                lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 4. Keputusan & Restorasi",
            "",
            "Folder yang ter-flag di-rename dengan prefix `_QC2_`. "
            "Untuk me-restore sesi tertentu:",
            "",
            "```bash",
            "# Contoh restore satu sesi",
            "mv dataset/nola/_QC2_20260513_112503 dataset/nola/20260513_112503",
            "```",
            "",
            "---",
            "",
            "*Laporan ini di-generate secara otomatis oleh `utils/data_qc_v2.py`. "
            "Kriteria objektif (within-session stability) diterapkan uniform ke semua subjek.*",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="QC v2: Universal objective criteria for geometric session stability"
    )
    parser.add_argument("--data_dir", default="dataset", help="Root dataset directory")
    parser.add_argument("--k", type=float, default=10.0, help="Threshold multiplier (default: 10)")
    parser.add_argument("--dry_run", action="store_true", help="Analyze only, do not rename folders")
    parser.add_argument("--output_dir", default=None, help="Report output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Error: data_dir '{data_dir}' tidak ditemukan")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("eval_results/qc_v2") / timestamp

    print(f"QC v2 — Dataset: {data_dir.resolve()}")
    print(f"  Threshold k = {args.k}")
    print(f"  Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    print()

    # Step 1: Scan
    print("[1/4] Scanning sessions...")
    sessions = _scan_sessions(data_dir)
    print(f"      Found {len(sessions)} sessions with ≥3 valid frames")

    # Step 2: Compute stats
    print("[2/4] Computing within-session std...")
    median_std, enriched = _compute_stats(sessions)

    # Step 3: Flag
    print(f"[3/4] Flagging sessions with std > {args.k} × median_global...")
    flagged = _flag_sessions(enriched, median_std, args.k)
    print(f"      {len(flagged)} sessions flagged ({len(flagged) / len(enriched) * 100:.1f}%)")

    # Step 4: Apply / Report
    print("[4/4] Applying quarantine + generating report...")
    moved = _apply_quarantine(flagged, dry_run=args.dry_run)
    if moved:
        print(f"      {len(moved)} folders {'would be' if args.dry_run else ''} renamed:")
        for src, dst in moved[:10]:
            print(f"        {src.name} → {dst.name}")
        if len(moved) > 10:
            print(f"        ... and {len(moved) - 10} more")

    report_path = _generate_report(
        enriched, flagged, median_std, args.k, data_dir, output_dir, args.dry_run
    )
    print(f"      Report saved to: {report_path}")

    # Exit code: 0 if no flags, 1 if flags found (useful for CI)
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
