"""
utils/audit_embedding_stats.py — D3: Audit perilaku dropout & GAM saat
train-mode vs eval-mode pada checkpoint v0.3.0-baseline.

Hipotesis (Plan §D3): Dropout(0.3) di proj head + ketiadaan dropout di
geom_encoder/GAM bisa membuat embedding direction berbeda antara train()
dan eval(). L2-normalize menjaga magnitudo tetap 1, tapi *arah* embedding
bisa berputar — yang berpengaruh pada cosine similarity yang dipakai ArcFace
dan evaluasi 1:1 maupun 1:N.

Pendekatan ringan (tanpa harus load dataset penuh):
  - Bangkitkan synthetic batch input (pts, geom) dari distribusi yang masuk akal
  - Load checkpoint with_geom dan no_geom
  - Hitung cosine(emb_train, emb_eval) untuk sampel yang sama
  - Hitung magnitudo pre-L2-norm (jika diakses)

Output:
  eval_results/audits/<ts>/embedding_stats.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.siamese import SiamesePalmNet  # noqa: E402
from utils.dataset import GEOMETRY_DIM  # noqa: E402


CHECKPOINTS = {
    "with_geom": ROOT / "runs/with_geom/20260516_210959/seed_42/best.pth",
    "no_geom":   ROOT / "runs/no_geom/20260516_211407/seed_42/best.pth",
}


def _strip_compile_prefix(sd: dict) -> dict:
    """Remove _orig_mod. prefix introduced by torch.compile."""
    out = {}
    for k, v in sd.items():
        out[k.removeprefix("_orig_mod.")] = v
    return out


def _load_model(ckpt_path: Path, use_geom: bool, num_classes: int = 11) -> SiamesePalmNet:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    sd = _strip_compile_prefix(sd)
    model = SiamesePalmNet(geom_dim=GEOMETRY_DIM, use_geom=use_geom, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[warn] {ckpt_path.name}: unexpected keys = {unexpected[:5]}...")
    if missing:
        print(f"[warn] {ckpt_path.name}: missing keys = {missing[:5]}...")
    return model


def _make_input(batch: int, n_points: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    # pts: xyz dalam unit ~0.05-0.2 m, normals unit vektor
    xyz = torch.randn(batch, n_points, 3, generator=g) * 0.05
    normals = torch.randn(batch, n_points, 3, generator=g)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pts = torch.cat([xyz, normals], dim=-1)
    # geom: z-score-like → N(0, 1)
    geom = torch.randn(batch, GEOMETRY_DIM, generator=g)
    return pts, geom


@torch.no_grad()
def _forward(model: SiamesePalmNet, pts: torch.Tensor, geom: torch.Tensor, train_mode: bool):
    model.train(train_mode)
    return model.encoder(pts, geom)


def audit_variant(name: str, ckpt_path: Path, use_geom: bool,
                  batch: int = 8, n_points: int = 2048, n_trials: int = 3) -> dict:
    if not ckpt_path.exists():
        return {"variant": name, "error": f"checkpoint not found: {ckpt_path}"}
    model = _load_model(ckpt_path, use_geom=use_geom)

    # use eval mode for BN stats; we only toggle train() for the dropout flag
    # We do this by manually setting just Dropout to training mode while BN stays in eval
    # → that's the realistic comparison (BN at inference uses running stats).
    cos_train_vs_eval = []
    cos_train_consistency = []
    for trial in range(n_trials):
        pts, geom = _make_input(batch, n_points, seed=2026 + trial)

        # eval-mode forward (reference)
        emb_eval = _forward(model, pts, geom, train_mode=False)

        # train-mode forward but BN frozen → only dropout differs
        # Approach: set entire model.train(False), then explicitly enable dropout
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train(True)
        with torch.no_grad():
            emb_dropout_a = model.encoder(pts, geom)
            emb_dropout_b = model.encoder(pts, geom)  # second sample → dropout mask different
        model.eval()

        cos_train_vs_eval.append(
            (emb_dropout_a * emb_eval).sum(dim=1).mean().item()
        )
        cos_train_consistency.append(
            (emb_dropout_a * emb_dropout_b).sum(dim=1).mean().item()
        )

    return {
        "variant": name,
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "use_geom": use_geom,
        "batch": batch,
        "n_points": n_points,
        "n_trials": n_trials,
        "cos_train_vs_eval_mean": float(np.mean(cos_train_vs_eval)),
        "cos_train_vs_eval_std": float(np.std(cos_train_vs_eval)),
        "cos_train_consistency_mean": float(np.mean(cos_train_consistency)),
        "cos_train_consistency_std": float(np.std(cos_train_consistency)),
    }


def main() -> None:
    out_dir = ROOT / "eval_results" / "audits" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in CHECKPOINTS.items():
        use_geom = name == "with_geom"
        print(f"[D3] auditing {name} ...")
        results[name] = audit_variant(name, path, use_geom=use_geom)

    # verdict berbasis perbandingan with_geom vs no_geom
    wg = results.get("with_geom", {})
    ng = results.get("no_geom", {})
    summary = {
        "audit": "embedding_stats_dropout",
        "results": results,
    }
    if "cos_train_vs_eval_mean" in wg and "cos_train_vs_eval_mean" in ng:
        gap_wg = 1.0 - wg["cos_train_vs_eval_mean"]
        gap_ng = 1.0 - ng["cos_train_vs_eval_mean"]
        ratio = gap_wg / gap_ng if gap_ng > 1e-9 else float("inf")
        summary["delta_cos_with_minus_no"] = wg["cos_train_vs_eval_mean"] - ng["cos_train_vs_eval_mean"]
        summary["gap_ratio_with_over_no"] = ratio
        if gap_wg > 0.05 and gap_wg > 2 * gap_ng:
            summary["verdict"] = (
                f"TERKONFIRMASI: dropout men-rotasi embedding lebih besar di with_geom "
                f"(gap cos = {gap_wg:.4f} vs no_geom {gap_ng:.4f}, ratio {ratio:.2f}x). "
                f"Mengurangi atau menyimetriskan dropout layak dicoba."
            )
        elif gap_wg > 0.05:
            summary["verdict"] = (
                f"TERKONFIRMASI ringan: dropout men-rotasi embedding di kedua varian "
                f"(gap with_geom={gap_wg:.4f}, no_geom={gap_ng:.4f})."
            )
        else:
            summary["verdict"] = (
                f"DITOLAK: gap kecil di kedua varian (with_geom={gap_wg:.4f}, no_geom={gap_ng:.4f})."
            )

    out_path = out_dir / "embedding_stats.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[D3] Wrote {out_path}")
    print(f"[D3] verdict: {summary.get('verdict', 'n/a')}")


if __name__ == "__main__":
    main()
