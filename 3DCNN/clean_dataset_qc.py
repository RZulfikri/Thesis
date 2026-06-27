"""
clean_dataset_qc.py — Pindahkan session dataset bermasalah ke quarantine.

Jalankan SEKALI sebelum training v0.3.0:
    python clean_dataset_qc.py

Session yang akan dipindahkan:
  - reysa/20260512_140514  (10 frame, semua incomplete: 6K-12K points)
  - nola/20260513_112517   (hanya 1 frame, incomplete session)

Hasil:
  - Folder asli di-rename dengan prefix `_QUARANTINE_`
  - Bisa di-restore kalau perlu
"""

from pathlib import Path
import shutil

DATA_DIR = Path("dataset")
QUARANTINE_LIST = [
    # (subjek, session, alasan)
    ("reysa", "20260512_140514", "Incomplete scan: avg 10,457 pts (normal ~16,500)"),
    ("nola", "20260513_112517", "Incomplete session: hanya 1 frame (harus 10)"),
]


def main():
    moved = 0
    for subject, session, reason in QUARANTINE_LIST:
        src = DATA_DIR / subject / session
        dst_name = f"_QUARANTINE_{session}"
        dst = DATA_DIR / subject / dst_name

        if not src.exists():
            print(f"  [SKIP] {src} tidak ditemukan (mungkin sudah dipindahkan)")
            continue

        if dst.exists():
            print(f"  [SKIP] {dst} sudah ada")
            continue

        shutil.move(str(src), str(dst))
        print(f"  [MOVED] {subject}/{session} → {dst_name}")
        print(f"          Alasan: {reason}")
        moved += 1

    print(f"\nTotal session dipindahkan: {moved}")
    print("Dataset siap untuk training v0.3.0")


if __name__ == "__main__":
    main()
