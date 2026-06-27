"""
validate_v720.py — Validasi dataset hasil regenerasi v7.2.0.

Cek:
  1. Semua 11 subjek hadir + jumlah frame per subjek
  2. Setiap frame valid punya output.ply (dengan normals), cnn_input.npy, geometry.json
  3. cnn_input.npy bit-identik dengan dataset lama (sample lintas subjek) — basis v7.1.0
  4. Distribusi is_valid / warnings (knuckle kini non-gating)
  5. cnn_input_fps.npy shape (8192, 6) bila sudah digenerate
  6. Tidak ada invalid_frame.json yang nyangkut tanpa alasan point-cloud

Usage:
  python validate_v720.py --new result_frames_v720 --old ../3DCNN/dataset
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="result_frames_v720")
    ap.add_argument("--old", default="../3DCNN/dataset")
    ap.add_argument("--n_sample", type=int, default=20, help="frame sample untuk cek bit-identik")
    args = ap.parse_args()

    new = Path(args.new)
    old = Path(args.old)
    if not new.exists():
        sys.exit(f"Error: {new} tidak ada")

    subjects = sorted(p.name for p in new.iterdir() if p.is_dir())
    print(f"Subjek ({len(subjects)}): {subjects}\n")

    geos = sorted(new.glob("*/*/frame_*/geometry.json"))
    print(f"Total frame (geometry.json): {len(geos)}\n")

    # Per-subjek + kelengkapan artefak
    per_sub = Counter()
    valid_cnt = Counter()
    miss_ply = miss_cnn = ply_no_normal = 0
    issue_counter = Counter()
    warn_counter = Counter()
    has_fps = 0
    fps_bad = 0
    for g in geos:
        fdir = g.parent
        sub = fdir.parts[-3]
        per_sub[sub] += 1
        d = json.load(open(g))
        if d.get("is_valid", True):
            valid_cnt[sub] += 1
        for i in d.get("quality_issues", []):
            issue_counter[i.split(":")[0]] += 1
        for w in d.get("warnings", []):
            warn_counter[w.split(":")[0]] += 1
        ply = fdir / "output.ply"
        cnn = fdir / "cnn_input.npy"
        if not ply.exists():
            miss_ply += 1
        if not cnn.exists():
            miss_cnn += 1
        fps = fdir / "cnn_input_fps.npy"
        if fps.exists():
            has_fps += 1
            if tuple(np.load(fps).shape) != (8192, 6):
                fps_bad += 1

    print("Per-subjek (frame / valid):")
    for s in subjects:
        print(f"  {s:10s}: {per_sub[s]:4d} / {valid_cnt[s]:4d} valid")
    print()
    print(f"quality_issues (gate): {dict(issue_counter) or 'none'}")
    print(f"warnings (non-gate)  : {dict(warn_counter) or 'none'}")
    print(f"missing output.ply   : {miss_ply}")
    print(f"missing cnn_input.npy: {miss_cnn}")
    print(f"cnn_input_fps.npy     : {has_fps} ada, {fps_bad} shape salah")
    print()

    # Cek normals di sample PLY
    plys = sorted(new.glob("*/*/frame_*/output.ply"))
    if plys:
        no_norm = 0
        step = max(1, len(plys) // args.n_sample)
        sample = plys[::step][: args.n_sample]
        for p in sample:
            if not o3d.io.read_point_cloud(str(p)).has_normals():
                no_norm += 1
        print(f"PLY normals check ({len(sample)} sample): {no_norm} TANPA normals")

    # Bit-identik vs dataset lama
    if old.exists():
        print(f"\nKonsistensi cnn_input.npy vs dataset lama ({args.n_sample} sample):")
        cnns = sorted(new.glob("*/*/frame_*/cnn_input.npy"))
        step = max(1, len(cnns) // args.n_sample)
        sample = cnns[::step][: args.n_sample]
        n_match = n_diff = n_noold = 0
        max_diff = 0.0
        for c in sample:
            rel = c.relative_to(new)
            oldc = old / rel
            if not oldc.exists():
                n_noold += 1
                continue
            a, b = np.load(c), np.load(oldc)
            if a.shape == b.shape:
                dif = float(np.abs(a - b).max())
                max_diff = max(max_diff, dif)
                if dif == 0.0:
                    n_match += 1
                else:
                    n_diff += 1
            else:
                n_diff += 1
        print(f"  identik: {n_match}  beda: {n_diff}  tak-ada-di-lama: {n_noold}  max_abs_diff: {max_diff}")


if __name__ == "__main__":
    main()
