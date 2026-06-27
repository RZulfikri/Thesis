"""
pack_dataset_release.py — kemas dataset jadi tarball untuk GitHub Release (v8+).

Strategi repo ramping: dataset TIDAK disimpan di git. Tiap versi data dikemas jadi
satu tarball (multi-part bila > batas asset Release GitHub) + MANIFEST sha256, lalu
di-upload sebagai aset Release (lihat release_assets.py). Colab mengunduh + extract
tarball ini, BUKAN meng-clone dataset dari git.

Isi tarball (per frame, relatif terhadap data_dir):
  <subject>/<session>/frame_XX/output.ply
  <subject>/<session>/frame_XX/geometry.json
  <subject>/<session>/frame_XX/cnn_input.npy        (A3 canonical / R2)
  <subject>/<session>/frame_XX/cnn_input_fps.npy    (R3)
  <subject>/<session>/frame_XX/align_*.npy          (A1/A2/A4/A5 — bila ada)

Kompresi: pakai `zstd -T0` bila tersedia (cepat, multi-thread), fallback `gzip`.
Bila tarball > --split-size (default 1900M, di bawah limit 2GB/asset GitHub) dipecah
jadi part .partNN; Colab menyatukan ulang sebelum extract.

Output (di --out_dir):
  dataset_<ver>.tar.zst                 (atau .tar.gz; atau .partNN bila displit)
  MANIFEST_<ver>.json                   (versi, kompresor, jumlah file, sha256 tiap part)

Usage:
  python pack_dataset_release.py --data_dir ../3DCNN/dataset --version v8
  python pack_dataset_release.py --data_dir ../3DCNN/dataset --version v8 --limit 3   # uji cepat
  python pack_dataset_release.py --data_dir ../3DCNN/dataset --version v8 --split-size 1900M
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# pola file artefak per frame yang dikemas (urut deterministik di dalam tar)
FRAME_GLOBS = [
    "output.ply",
    "geometry.json",
    "cnn_input.npy",
    "cnn_input_fps.npy",
    "align_center.npy",
    "align_centerscale.npy",
    "align_pca_robust.npy",
    "align_anatomical.npy",
]


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024


def _sha256(path: Path, buf: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = 1
    if s.endswith("G"):
        mult, s = 1024 ** 3, s[:-1]
    elif s.endswith("M"):
        mult, s = 1024 ** 2, s[:-1]
    elif s.endswith("K"):
        mult, s = 1024, s[:-1]
    return int(float(s) * mult)


def _pick_compressor(choice: str):
    """Return (ext, tar_compress_program). zstd bila ada, else gzip."""
    if choice in ("auto", "zstd") and shutil.which("zstd"):
        return "tar.zst", "zstd -T0"
    if choice == "zstd":
        sys.exit("Error: --compressor zstd diminta tapi `zstd` tidak ada di PATH.")
    return "tar.gz", "gzip"


def _collect_files(data_dir: Path, limit: int):
    """Kembalikan list path relatif (str) terhadap data_dir, urut deterministik."""
    frames = sorted(data_dir.glob("*/*/frame_*"))
    if limit:
        frames = frames[:limit]
    rels = []
    for fr in frames:
        for name in FRAME_GLOBS:
            p = fr / name
            if p.exists():
                rels.append(str(p.relative_to(data_dir)))
    return rels


def main():
    ap = argparse.ArgumentParser(description="Kemas dataset jadi tarball Release (v8+)")
    ap.add_argument("--data_dir", default="../3DCNN/dataset")
    ap.add_argument("--version", required=True, help="label versi data, mis. v8 → dataset_v8.tar.zst")
    ap.add_argument("--out_dir", default=".", help="direktori output tarball + manifest")
    ap.add_argument("--split-size", default="1900M", help="pecah bila tarball > ukuran ini (limit asset GitHub 2GB)")
    ap.add_argument("--compressor", default="auto", choices=["auto", "zstd", "gzip"])
    ap.add_argument("--limit", type=int, default=0, help="batasi N frame (0=semua) — uji cepat")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        sys.exit(f"Error: '{data_dir}' tidak ada")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ext, comp_prog = _pick_compressor(args.compressor)
    rels = _collect_files(data_dir, args.limit)
    if not rels:
        sys.exit(f"Tidak ada file frame di '{data_dir}'")
    n_frames = len({str(Path(r).parent) for r in rels})
    print(f"data_dir   : {data_dir}")
    print(f"file       : {len(rels)} ({n_frames} frame) | kompresor: {comp_prog}\n")

    tar_path = out_dir / f"dataset_{args.version}.{ext}"
    # daftar file → stdin `tar -T -` (aman utk path banyak/panjang)
    filelist = "\n".join(rels).encode()

    t0 = time.time()
    print(f"[1/3] tar+compress → {tar_path.name} ...")
    # -C data_dir agar path di dalam tar relatif (subject/session/frame/file)
    # daftar berisi file reguler eksplisit (bukan dir) → tak ada rekursi yg perlu dimatikan
    cmd = ["tar", "-C", str(data_dir), "--use-compress-program", comp_prog,
           "-c", "-f", str(tar_path), "-T", "-"]
    r = subprocess.run(cmd, input=filelist)
    if r.returncode != 0:
        sys.exit(f"tar gagal (rc={r.returncode})")
    tar_size = tar_path.stat().st_size
    print(f"      ukuran tarball: {_human(tar_size)} ({time.time()-t0:.1f}s)")

    # split bila perlu
    split_bytes = _parse_size(args.split_size)
    parts = []
    if tar_size > split_bytes:
        print(f"[2/3] split (> {args.split_size}) → {tar_path.name}.partNN ...")
        # split -b <size> -d -a 2 <file> <prefix>.part  → .part00, .part01, ...
        prefix = str(tar_path) + ".part"
        r = subprocess.run(["split", "-b", str(split_bytes), "-d", "-a", "2",
                            str(tar_path), prefix])
        if r.returncode != 0:
            sys.exit(f"split gagal (rc={r.returncode})")
        tar_path.unlink()  # hapus tar utuh; simpan part saja
        parts = sorted(Path(out_dir).glob(f"{tar_path.name}.part*"))
    else:
        print("[2/3] split: tidak perlu (di bawah batas).")
        parts = [tar_path]

    # manifest
    print(f"[3/3] sha256 + manifest ({len(parts)} part) ...")
    part_meta = []
    for p in parts:
        part_meta.append({"name": p.name, "size": p.stat().st_size, "sha256": _sha256(p)})
        print(f"      {p.name}  {_human(p.stat().st_size)}")
    manifest = {
        "version": args.version,
        "compressor": comp_prog,
        "archive_ext": ext,
        "tar_basename": tar_path.name,         # nama tar setelah part disatukan
        "split": len(parts) > 1,
        "file_count": len(rels),
        "frame_count": n_frames,
        "parts": part_meta,
    }
    man_path = out_dir / f"MANIFEST_{args.version}.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{'='*60}")
    print(f"SELESAI ({time.time()-t0:.1f}s)")
    print(f"  manifest : {man_path}")
    print(f"  parts    : {len(parts)} → upload semua + MANIFEST sebagai aset Release")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
