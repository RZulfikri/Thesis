"""
utils/audit_temporal_gap.py — Audit time-gap antar sesi per split.

Tujuan: dokumentasi eksplisit limitation temporal gap di laporan thesis.

Output:
  - Per subjek: min/mean/max time-gap antar sesi dalam split
  - Per split:  min/mean/max time-gap train→val→test→holdout
  - Peringatan kalau gap < 120 detik (indikasi capture burst)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_lowdata import build_lowdata_splits


def _parse_ts(session_dir: Path) -> datetime:
    """Parse timestamp dari nama folder session: YYYYMMDD_HHMMSS."""
    name = session_dir.name
    return datetime.strptime(name, "%Y%m%d_%H%M%S")


def _session_gap_seconds(session_dirs: list[Path]) -> list[float]:
    """Hitung pairwise time-gap (dalam detik) antar sesi yang berurutan."""
    if len(session_dirs) < 2:
        return []
    dts = [_parse_ts(d) for d in session_dirs]
    gaps = [(dts[i+1] - dts[i]).total_seconds() for i in range(len(dts) - 1)]
    return gaps


def audit_temporal_gap(dataset_root: Path) -> dict:
    """
    Audit time-gap untuk seluruh split.

    Returns:
        dict dengan struktur:
        {
            "per_subject": {
                "subject_name": {
                    "session_count": int,
                    "gaps_all": [float],       # detik, antar sesi berurutan (15 sesi)
                    "train_gaps": [float],
                    "val_gaps": [float],
                    "test_gaps": [float],
                    "holdout_gaps": [float],
                    "train_to_holdout_gap": float,  # detik, s15 - s1
                }
            },
            "summary": {
                "min_gap_sec": float,
                "mean_gap_sec": float,
                "max_gap_sec": float,
                "train_to_holdout_min": float,
                "train_to_holdout_mean": float,
                "train_to_holdout_max": float,
            }
        }
    """
    dataset_root = Path(dataset_root)
    splits = build_lowdata_splits(dataset_root)

    per_subject: dict[str, dict] = {}
    all_gaps: list[float] = []
    train_to_holdout_gaps: list[float] = []

    for label in sorted(splits["train"].keys()):
        # Reconstruct ordered 15 sessions untuk subjek ini
        # (karena build_lowdata_splits sudah chronological, kita scan ulang)
        label_dir = dataset_root / label
        sessions = sorted(
            p for p in label_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )[:15]

        gaps_all = _session_gap_seconds(sessions)
        all_gaps.extend(gaps_all)

        # Split ke train/val/test/holdout berdasarkan index
        train_sess = sessions[0:8]
        val_sess   = sessions[8:10]
        test_sess  = sessions[10:12]
        hold_sess  = sessions[12:15]

        train_gaps = _session_gap_seconds(train_sess)
        val_gaps   = _session_gap_seconds(val_sess)
        test_gaps  = _session_gap_seconds(test_sess)
        hold_gaps  = _session_gap_seconds(hold_sess)

        # Gap dari train terakhir ke holdout pertama
        th_gap = None
        if train_sess and hold_sess:
            th_gap = (_parse_ts(hold_sess[0]) - _parse_ts(train_sess[-1])).total_seconds()
            train_to_holdout_gaps.append(th_gap)

        per_subject[label] = {
            "session_count": len(sessions),
            "gaps_all": gaps_all,
            "train_gaps": train_gaps,
            "val_gaps": val_gaps,
            "test_gaps": test_gaps,
            "holdout_gaps": hold_gaps,
            "train_to_holdout_gap": th_gap,
        }

    summary = {}
    if all_gaps:
        summary = {
            "min_gap_sec": float(np.min(all_gaps)),
            "mean_gap_sec": float(np.mean(all_gaps)),
            "max_gap_sec": float(np.max(all_gaps)),
        }
    if train_to_holdout_gaps:
        summary.update({
            "train_to_holdout_min": float(np.min(train_to_holdout_gaps)),
            "train_to_holdout_mean": float(np.mean(train_to_holdout_gaps)),
            "train_to_holdout_max": float(np.max(train_to_holdout_gaps)),
        })

    return {"per_subject": per_subject, "summary": summary}


def print_audit(report: dict):
    """Cetak laporan audit dalam format human-readable."""
    print("=" * 70)
    print("TEMPORAL GAP AUDIT — v5.0.0 Low-Data Regime")
    print("=" * 70)

    per_subject = report["per_subject"]
    summary = report["summary"]

    for label, data in sorted(per_subject.items()):
        print(f"\nSubjek: {label}")
        print(f"  Sessions       : {data['session_count']}")
        if data["gaps_all"]:
            print(f"  All gaps       : min={min(data['gaps_all']):.1f}s  "
                  f"mean={np.mean(data['gaps_all']):.1f}s  "
                  f"max={max(data['gaps_all']):.1f}s")
        if data["train_to_holdout_gap"] is not None:
            print(f"  Train→Holdout  : {data['train_to_holdout_gap']:.1f}s")
            if data["train_to_holdout_gap"] < 120:
                print(f"    ⚠️  WARNING: gap < 120 detik — capture burst detected")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if "min_gap_sec" in summary:
        print(f"  All-session gap   : min={summary['min_gap_sec']:.1f}s  "
              f"mean={summary['mean_gap_sec']:.1f}s  "
              f"max={summary['max_gap_sec']:.1f}s")
    if "train_to_holdout_min" in summary:
        print(f"  Train→Holdout gap : min={summary['train_to_holdout_min']:.1f}s  "
              f"mean={summary['train_to_holdout_mean']:.1f}s  "
              f"max={summary['train_to_holdout_max']:.1f}s")
        if summary["train_to_holdout_max"] < 120:
            print("\n  ⚠️  CRITICAL: max train→holdout gap < 120 detik")
            print("      → Klaim 'generalization to future time' BELUM tervalidasi.")
            print("      → Dokumentasikan eksplisit di laporan thesis.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Audit temporal gap antar sesi")
    p.add_argument("--dataset_root", default="../dataset", help="Dataset root")
    p.add_argument("--output", default=None, help="Simpan report ke JSON")
    args = p.parse_args()

    report = audit_temporal_gap(args.dataset_root)
    print_audit(report)

    if args.output:
        # Convert Path objects ke string untuk JSON serialization
        out = {
            "per_subject": {
                k: {kk: (vv if not isinstance(vv, list) or not vv or not isinstance(vv[0], Path) else [str(x) for x in vv])
                    for kk, vv in v.items()}
                for k, v in report["per_subject"].items()
            },
            "summary": report["summary"],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nReport disimpan di: {args.output}")
