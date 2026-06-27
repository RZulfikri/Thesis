"""
generate_dataset.py — Regenerasi 3DCNN/dataset dari "Raw Depth Data" (REPRODUCIBLE).

Repo ini TIDAK menyimpan dataset turunan; hanya raw scan iPhone (ZIP, ~91 MB) +
kode. Dataset (output.ply / geometry.json / cnn_input.npy / cnn_input_fps.npy /
align_*.npy) DIBANGUN ULANG dari raw via skrip ini. Jadi dataset = fungsi(raw + kode)
→ sepenuhnya reproducible, repo tetap ramping.

Tahapan (semua memakai skrip yang sudah ada — TANPA build C++):
  0. unzip "Raw Depth Data"/*.zip → staging (folder sesi <label>_<timestamp>/)
  1. process_single_frames.py  → output.ply + geometry.json + cnn_input.npy   (per frame)
  2. make_fps.py               → cnn_input_fps.npy (R3)
  3. make_align_variants.py    → align_center/centerscale/pca_robust/anatomical (A1/A2/A4/A5)

Idempotent: skrip per-tahap melewati file yang sudah ada (kecuali --force). Aman
diulang / lanjut setelah restart. Mencetak ringkasan + timing tiap tahap.

Usage (dari folder 3DRegistration/):
  python generate_dataset.py
  python generate_dataset.py --raw_dir "../Raw Depth Data" --data_dir ../3DCNN/dataset
  python generate_dataset.py --skip_align          # hanya ply+geometry+cnn+fps
  python generate_dataset.py --force               # bangun ulang semua
"""

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

_THIS = Path(__file__).resolve().parent          # .../3DRegistration


def _run(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        sys.exit(f"GAGAL (rc={r.returncode}): {' '.join(str(c) for c in cmd)}")


def _count(data_dir: Path, name: str) -> int:
    return len(list(data_dir.glob(f"*/*/frame_*/{name}")))


def main():
    ap = argparse.ArgumentParser(description="Regenerasi dataset dari Raw Depth Data (reproducible)")
    ap.add_argument("--raw_dir", default=str(_THIS.parent / "Raw Depth Data"),
                    help='folder berisi ZIP scan iPhone (default: "<repo>/Raw Depth Data")')
    ap.add_argument("--data_dir", default=str(_THIS.parent / "3DCNN" / "dataset"),
                    help="tujuan dataset (default: <repo>/3DCNN/dataset)")
    ap.add_argument("--staging_dir", default=str(_THIS.parent / "_raw_staging"),
                    help="folder ekstraksi sementara (di luar dataset)")
    ap.add_argument("--n_points", type=int, default=8192, help="n_points untuk FPS (R3)")
    ap.add_argument("--min_points", type=int, default=1000, help="filter frame sparse")
    ap.add_argument("--skip_align", action="store_true", help="lewati varian alignment (A1/A2/A4/A5)")
    ap.add_argument("--force", action="store_true", help="bangun ulang semua (timpa)")
    ap.add_argument("--keep_staging", action="store_true", help="jangan hapus folder staging setelah selesai")
    args = ap.parse_args()

    raw_dir  = Path(args.raw_dir)
    data_dir = Path(args.data_dir)
    staging  = Path(args.staging_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(raw_dir.glob("*.zip"))
    if not zips:
        sys.exit(f"Tidak ada ZIP di '{raw_dir}'. Pastikan 'Raw Depth Data/' berisi scan iPhone.")
    print(f"Raw ZIP   : {len(zips)} file di {raw_dir}")
    print(f"Dataset   : {data_dir}")
    t_all = time.time()
    timings = {}

    # ── Tahap 0: unzip ──────────────────────────────────────────────
    t0 = time.time()
    if staging.exists() and not args.force:
        print(f"\n[0/3] staging sudah ada ({staging}) — skip unzip (pakai --force utk ekstrak ulang).")
    else:
        print(f"\n[0/3] unzip {len(zips)} ZIP → {staging} ...")
        staging.mkdir(parents=True, exist_ok=True)
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(staging)
            print(f"  extracted {z.name}")
    # kumpulkan semua folder sesi <label>_<timestamp> (apa pun kedalaman staging)
    session_dirs = sorted({p.parent for p in staging.rglob("calibration.json")})
    if not session_dirs:
        sys.exit(f"Tidak ada folder sesi (berisi calibration.json) di {staging}")
    print(f"  {len(session_dirs)} folder sesi ditemukan")
    timings["unzip"] = time.time() - t0

    # ── Tahap 1: depth.bin → output.ply + geometry.json + cnn_input.npy ──
    t0 = time.time()
    print(f"\n[1/3] process_single_frames untuk {len(session_dirs)} sesi ...")
    common = ["--out_dir", data_dir, "--min_points", args.min_points]
    if args.force:
        common.append("--force")
    # tiap session_dir berisi calibration.json → process_single_frames menerima 1 sesi
    for i, sd in enumerate(session_dirs, 1):
        print(f"  ({i}/{len(session_dirs)}) {sd.name}")
        _run([sys.executable, "process_single_frames.py", "--data_dir", sd] + common, cwd=_THIS)
    timings["single_frames"] = time.time() - t0

    # ── Tahap 2: cnn_input.npy → cnn_input_fps.npy (R3) ──────────────
    t0 = time.time()
    print(f"\n[2/3] make_fps (R3) ...")
    fps_cmd = [sys.executable, "make_fps.py", "--data_dir", data_dir, "--n_points", args.n_points]
    if args.force:
        fps_cmd.append("--force")
    _run(fps_cmd, cwd=_THIS)
    timings["fps"] = time.time() - t0

    # ── Tahap 3: align variants (A1/A2/A4/A5) ───────────────────────
    if not args.skip_align:
        t0 = time.time()
        print(f"\n[3/3] make_align_variants (A1/A2/A4/A5) ...")
        al_cmd = [sys.executable, "make_align_variants.py", "--data_dir", data_dir]
        if args.force:
            al_cmd.append("--force")
        _run(al_cmd, cwd=_THIS)
        timings["align"] = time.time() - t0
    else:
        print("\n[3/3] align: dilewati (--skip_align).")

    # ── cleanup staging ─────────────────────────────────────────────
    if not args.keep_staging:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        print(f"\nstaging dihapus ({staging}).")

    # ── ringkasan ───────────────────────────────────────────────────
    print(f"\n{'='*60}\nSELESAI ({time.time()-t_all:.0f}s total)")
    for k, v in timings.items():
        print(f"  {k:14s}: {v:7.0f}s")
    print(f"{'-'*60}")
    n_ply = _count(data_dir, "output.ply")
    print(f"  output.ply         : {n_ply}")
    print(f"  geometry.json      : {_count(data_dir, 'geometry.json')}")
    print(f"  cnn_input.npy      : {_count(data_dir, 'cnn_input.npy')}")
    print(f"  cnn_input_fps.npy  : {_count(data_dir, 'cnn_input_fps.npy')}")
    if not args.skip_align:
        for m in ("align_center", "align_centerscale", "align_pca_robust", "align_anatomical"):
            print(f"  {m+'.npy':19s}: {_count(data_dir, m + '.npy')}")
    print(f"{'='*60}")
    if n_ply == 0:
        sys.exit("PERINGATAN: 0 output.ply — cek error di tahap 1.")


if __name__ == "__main__":
    main()
