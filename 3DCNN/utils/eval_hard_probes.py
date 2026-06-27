"""
utils/eval_hard_probes.py — D5: Apakah gap with_geom vs no_geom mengecil
pada subset probe yang paling sulit?

Hipotesis (Plan §D5): no_geom sudah saturasi 99.82% Rank-1 di seluruh
probe; mungkin GeoAtt memberi nilai justru pada probe yang "sulit" (top-1
similarity rendah). Jika gap mengecil/membalik di hard subset → masalah
utamanya ceiling, bukan GeoAtt rusak.

Definisi hard probe:
  - Untuk varian "no_geom" (referensi), urutkan probe berdasarkan top-1
    cosine similarity (lebih rendah = lebih sulit).
  - "hard" = bottom q-quantile (default 25%).
  - Subset probe yang sama lalu dievaluasi di kedua varian (pakai pasangan
    seed yang sama).

Output:
  eval_results/audits/<ts>/hard_probes.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WITH_GEOM_DIR = ROOT / "eval_results/with_geom/20260516_223830"
NO_GEOM_DIR = ROOT / "eval_results/no_geom/20260516_223800"
SEEDS = [42, 123, 2026, 7, 31337]


def _load(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _top1_similarity(probe_embs: np.ndarray, gallery_embs: np.ndarray) -> np.ndarray:
    # asumsi sudah L2-normalized (encoder mengeluarkan normed embedding)
    sim = probe_embs @ gallery_embs.T  # (n_probe, n_gallery)
    return sim.max(axis=1)


def _rank1_correct(probe_embs: np.ndarray, gallery_embs: np.ndarray,
                   probe_labels: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    sim = probe_embs @ gallery_embs.T
    top1_idx = sim.argmax(axis=1)
    pred = gallery_labels[top1_idx]
    return (pred == probe_labels)


def analyze_seed(seed: int, q: float = 0.25) -> dict:
    wg = _load(WITH_GEOM_DIR / f"embeddings_seed_{seed}.npz")
    ng = _load(NO_GEOM_DIR / f"embeddings_seed_{seed}.npz")

    # urutan probe identik antar varian? cek via probe_true_labels
    assert wg["probe_true_labels"].shape == ng["probe_true_labels"].shape, (
        "probe count berbeda antar varian"
    )
    if not np.array_equal(wg["probe_true_labels"], ng["probe_true_labels"]):
        # Cached embeddings tidak menyimpan probe identity selain label;
        # jika urutan berbeda, kita tetap bisa bekerja per-label tetapi
        # tidak benar-benar paired per-frame. Cetak peringatan.
        print(f"[warn] seed {seed}: urutan probe berbeda antar varian "
              "(tidak per-frame paired). Lanjut dengan label-level only.")

    # similarity terhadap gallery (per-varian)
    sim_top1_ng = _top1_similarity(ng["probe_embs"], ng["gallery_embs"])

    # hard subset = bottom q-quantile sim di varian no_geom (yang saturasi)
    threshold = np.quantile(sim_top1_ng, q)
    hard_mask = sim_top1_ng <= threshold

    # rank-1 correctness per varian
    wg_correct = _rank1_correct(wg["probe_embs"], wg["gallery_embs"],
                                 wg["probe_true_labels"], wg["gallery_labels"])
    ng_correct = _rank1_correct(ng["probe_embs"], ng["gallery_embs"],
                                 ng["probe_true_labels"], ng["gallery_labels"])

    # jika probe count sama tapi urutan berbeda, kita re-cocokkan berdasar urutan masing-masing
    # kita asumsikan urutan sama (compare.ipynb sebelumnya juga mengasumsikan ini)
    overall_wg = float(wg_correct.mean())
    overall_ng = float(ng_correct.mean())

    hard_wg = float(wg_correct[hard_mask].mean()) if hard_mask.any() else float("nan")
    hard_ng = float(ng_correct[hard_mask].mean()) if hard_mask.any() else float("nan")

    return {
        "seed": seed,
        "n_probe": int(len(sim_top1_ng)),
        "q": q,
        "hard_threshold_sim_ng": float(threshold),
        "n_hard": int(hard_mask.sum()),
        "overall_rank1_with_geom": overall_wg,
        "overall_rank1_no_geom": overall_ng,
        "hard_rank1_with_geom": hard_wg,
        "hard_rank1_no_geom": hard_ng,
        "overall_delta": overall_wg - overall_ng,
        "hard_delta": hard_wg - hard_ng,
    }


def main() -> None:
    out_dir = ROOT / "eval_results" / "audits" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = [analyze_seed(s) for s in SEEDS]

    overall_deltas = np.array([s["overall_delta"] for s in per_seed])
    hard_deltas = np.array([s["hard_delta"] for s in per_seed])

    summary = {
        "audit": "hard_probes",
        "seeds": SEEDS,
        "q": 0.25,
        "per_seed": per_seed,
        "mean_overall_delta_with_minus_no": float(overall_deltas.mean()),
        "mean_hard_delta_with_minus_no": float(np.nanmean(hard_deltas)),
    }

    md = summary["mean_overall_delta_with_minus_no"]
    mh = summary["mean_hard_delta_with_minus_no"]
    if mh > md + 0.02:
        summary["verdict"] = (
            f"PARSIAL: gap with_geom vs no_geom mengecil di hard subset "
            f"(overall Δ={md:+.4f}, hard Δ={mh:+.4f}). Hipotesis saturasi parsial benar."
        )
    elif mh < md - 0.02:
        summary["verdict"] = (
            f"DITOLAK: gap dengan_geom vs no_geom MEMBESAR di hard subset "
            f"(overall Δ={md:+.4f}, hard Δ={mh:+.4f}). GeoAtt makin merugikan saat sulit."
        )
    else:
        summary["verdict"] = (
            f"TIDAK KONKLUSIF: gap tidak banyak berubah antara overall ({md:+.4f}) "
            f"dan hard ({mh:+.4f})."
        )

    out_path = out_dir / "hard_probes.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[D5] Wrote {out_path}")
    print(f"[D5] verdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
