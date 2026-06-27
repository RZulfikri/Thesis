"""
make_align_variants.py — v8: generate varian alignment/normalisasi (TANPA FPS) per frame.

Untuk Study A (alignment ablation). Dari tiap `output.ply` (koordinat kamera + normals)
hasilkan point cloud **full** (tanpa downsampling) untuk tiap mode alignment, disimpan .npy
(N,6)=xyz+normals — sejajar dgn `cnn_input.npy` (R2) sehingga loader memperlakukannya identik
(runtime random-sample ke n_points saat training/eval).

Mode & file output (lihat 3DCNN/utils/alignment.py — sumber-tunggal kebenaran):
  align_center.npy        (A1)  center saja
  align_centerscale.npy   (A2)  center + unit-sphere (tanpa rotasi)
  align_pca_robust.npy    (A4)  PCA deterministik — FIX rotasi 90° (tanpa landmark)
  align_anatomical.npy    (A5)  alignment anatomis berbasis landmark — FIX 90° (pilihan user)

A0 (raw) = output.ply langsung; A3 (pca canonical) = cnn_input.npy yang sudah ada → tidak digenerate ulang.

handedness diambil dari geometry.json (atau metadata.json) sibling — untuk disambiguasi
sumbu-X pada mode anatomical.

Usage:
  python make_align_variants.py --data_dir ../3DCNN/dataset
  python make_align_variants.py --data_dir ../3DCNN/dataset --modes align_anatomical --force
  python make_align_variants.py --data_dir ../3DCNN/dataset --limit 5   # uji cepat
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

# utils/alignment.py ada di subproject 3DCNN — tambahkan ke path (relatif thd file ini)
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "3DCNN"))
from utils.alignment import align_cloud6, ALIGN_MODES  # noqa: E402

# pemetaan mode → nama file (subset yg di-generate; raw & pca canonical sudah ada)
MODE_FILE = {
    "center":       "align_center.npy",
    "centerscale":  "align_centerscale.npy",
    "pca_robust":   "align_pca_robust.npy",
    "anatomical":   "align_anatomical.npy",
}


def _load_ply_xyz_normals(ply_path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if len(xyz) == 0:
        raise ValueError(f"PLY kosong: {ply_path}")
    if pcd.has_normals():
        nrm = np.asarray(pcd.normals, dtype=np.float32)
    else:
        pcd.estimate_normals()
        nrm = np.asarray(pcd.normals, dtype=np.float32)
    if len(nrm) != len(xyz):
        nrm = np.zeros_like(xyz)
    return np.concatenate([xyz, nrm], axis=1).astype(np.float32)


def _read_handedness(frame_dir: Path) -> str:
    for name in ("geometry.json", "metadata.json"):
        f = frame_dir / name
        if f.exists():
            try:
                with open(f) as fp:
                    d = json.load(fp)
                h = d.get("handedness")
                if h in ("right", "left"):
                    return h
            except Exception:
                pass
    # metadata.json kadang di level sesi (parent)
    f = frame_dir.parent / "metadata.json"
    if f.exists():
        try:
            with open(f) as fp:
                h = json.load(fp).get("handedness")
            if h in ("right", "left"):
                return h
        except Exception:
            pass
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Generate varian alignment (v8 Study A) dari output.ply")
    ap.add_argument("--data_dir", default="../3DCNN/dataset")
    ap.add_argument("--modes", nargs="+", default=list(MODE_FILE.keys()),
                    choices=list(MODE_FILE.keys()), help="mode yg digenerate (default: semua)")
    ap.add_argument("--force", action="store_true", help="timpa file yg sudah ada")
    ap.add_argument("--limit", type=int, default=0, help="batasi N frame (0=semua) — utk uji cepat")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Error: '{data_dir}' tidak ada")
    plys = sorted(data_dir.glob("*/*/frame_*/output.ply"))
    if args.limit:
        plys = plys[:args.limit]
    if not plys:
        sys.exit(f"Tidak ada output.ply di '{data_dir}'")
    print(f"Ditemukan {len(plys)} output.ply | modes={args.modes}\n")

    t0 = time.time()
    done = skip = fail = 0
    for i, ply in enumerate(plys, 1):
        fdir = ply.parent
        targets = {m: fdir / MODE_FILE[m] for m in args.modes}
        if not args.force and all(p.exists() for p in targets.values()):
            skip += 1
            continue
        try:
            cloud6 = _load_ply_xyz_normals(ply)
            hand = _read_handedness(fdir)
            for m in args.modes:
                out = targets[m]
                if out.exists() and not args.force:
                    continue
                aligned = align_cloud6(cloud6, m, handedness=hand)  # (N,6)
                np.save(out, aligned)
            done += 1
        except Exception as e:
            print(f"  [FAIL] {ply}: {e}")
            fail += 1
        if i % 200 == 0:
            print(f"  {i}/{len(plys)}  done={done} skip={skip} fail={fail}")

    print(f"\n{'='*60}\nSELESAI ({time.time()-t0:.1f}s)  done={done} skip={skip} fail={fail}\n{'='*60}")


if __name__ == "__main__":
    main()
