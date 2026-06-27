"""
utils/audit_init_parity.py — D1: Audit parity inisialisasi bobot antara
SiamesePalmNet(use_geom=True) vs SiamesePalmNet(use_geom=False).

Hipotesis (Plan §D1): membangun GeometryEncoder + GAM1 + GAM2 hanya pada
use_geom=True menggeser konsumsi RNG global, sehingga SA1/SA2/SA3 dan
ArcFace head di kedua varian punya bobot awal yang berbeda meskipun seed sama.
Implikasi: hasil ablasi tidak sepenuhnya "fair".

Output:
    eval_results/audits/<ts>/init_parity.json
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# allow running as `python -m utils.audit_init_parity` dari root 3DCNN
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.siamese import SiamesePalmNet  # noqa: E402
from utils.dataset import GEOMETRY_DIM  # noqa: E402


SHARED_LAYERS = (
    "sa1", "sa2", "sa3",
    "proj_with_geom", "proj_no_geom", "proj",  # `proj` untuk backward compat
    "geom_encoder", "gam1", "gam2",
)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build(use_geom: bool, seed: int, geom_dim: int, num_classes: int) -> SiamesePalmNet:
    _set_global_seed(seed)
    return SiamesePalmNet(geom_dim=geom_dim, use_geom=use_geom, num_classes=num_classes)


def _shared_params(model: SiamesePalmNet) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if any(name.startswith(f"encoder.{prefix}.") for prefix in SHARED_LAYERS):
            out[name] = p.detach().clone()
        elif name.startswith("arcface."):
            out[name] = p.detach().clone()
    return out


def compare_parity(seed: int, geom_dim: int = GEOMETRY_DIM, num_classes: int = 11) -> dict:
    model_geom = _build(use_geom=True, seed=seed, geom_dim=geom_dim, num_classes=num_classes)
    model_nog = _build(use_geom=False, seed=seed, geom_dim=geom_dim, num_classes=num_classes)

    params_g = _shared_params(model_geom)
    params_n = _shared_params(model_nog)

    common = sorted(set(params_g) & set(params_n))
    layer_report = []
    n_identical_layers = 0
    n_total_param_elems = 0
    n_identical_elems = 0
    max_abs_delta = 0.0

    for name in common:
        a, b = params_g[name], params_n[name]
        if a.shape != b.shape:
            # mis. proj[0].weight: input_dim 320 (with geom) vs 256 (no geom)
            layer_report.append({
                "name": name,
                "shape_with_geom": list(a.shape),
                "shape_no_geom": list(b.shape),
                "equal": False,
                "note": "shape_mismatch_structural",
            })
            continue
        equal = bool(torch.equal(a, b))
        delta = (a - b).abs()
        layer_report.append({
            "name": name,
            "shape": list(a.shape),
            "numel": int(a.numel()),
            "equal": equal,
            "max_abs_delta": float(delta.max().item()) if a.numel() > 0 else 0.0,
            "l2_delta": float(delta.norm().item()),
        })
        if equal:
            n_identical_layers += 1
            n_identical_elems += int(a.numel())
        n_total_param_elems += int(a.numel())
        max_abs_delta = max(max_abs_delta, float(delta.max().item()))

    return {
        "seed": seed,
        "n_shared_layers": len(common),
        "n_identical_layers": n_identical_layers,
        "n_total_param_elems": n_total_param_elems,
        "n_identical_param_elems": n_identical_elems,
        "fraction_identical_elems": (
            n_identical_elems / n_total_param_elems if n_total_param_elems > 0 else 0.0
        ),
        "max_abs_delta_any_layer": max_abs_delta,
        "layers": layer_report,
    }


def main() -> None:
    seeds = [42, 123, 2026, 7, 31337]
    out_dir = ROOT / "eval_results" / "audits" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = [compare_parity(s) for s in seeds]
    summary = {
        "audit": "init_parity",
        "hypothesis": "RNG consumption order differs between use_geom=True/False, "
                       "so SA layers + ArcFace head have different initial weights.",
        "seeds": seeds,
        "per_seed": per_seed,
        "verdict": _verdict(per_seed),
    }
    out_path = out_dir / "init_parity.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[D1] Wrote {out_path}")
    print(f"[D1] verdict: {summary['verdict']}")


def _verdict(per_seed: list[dict]) -> str:
    any_diff = any(s["n_identical_layers"] < s["n_shared_layers"] for s in per_seed)
    if any_diff:
        worst = max(s["max_abs_delta_any_layer"] for s in per_seed)
        return (
            "TERKONFIRMASI: bobot awal SA/ArcFace tidak identik antara use_geom=True/False "
            f"(max |Δ| antar seed = {worst:.4g}). Saran: tambahkan placeholder layer untuk "
            "geom_encoder/gam1/gam2 di use_geom=False agar konsumsi RNG identik."
        )
    return "DITOLAK: bobot awal SA/ArcFace identik di semua seed yang diuji."


if __name__ == "__main__":
    main()
