"""
utils/geometry_schema.py — Pure NumPy geometry feature schema.

Tidak bergantung pada PyTorch. Bisa di-import oleh skrip diagnostic
yang berjalan di environment tanpa torch.
"""

import numpy as np

GEOMETRY_KEYS = [
    "finger_lengths_mm",   # list[5]
    "palm_width_mm",        # float
    "palm_height_mm",       # float
    "palm_depth_std_mm",    # float
    "finger_widths_mm",     # list[5] → flatten[1:5] saat di _flatten_geometry
    "scan_distance_mm",     # float — context (scaling hint)
]
GEOMETRY_DIM = 13  # 5 + 1 + 1 + 1 + 4 + 1


def _flatten_geometry(geo_dict: dict) -> np.ndarray:
    """Flatten geometry.json dict menjadi array (13,) float32 (nilai mm absolut)."""
    values = []
    for key in GEOMETRY_KEYS:
        v = geo_dict.get(key)
        if isinstance(v, list):
            if key == "finger_widths_mm":
                # v5.0.0: skip thumb (index 0), ambil index 1–5
                values.extend([float(x) if x is not None else 0.0 for x in v[1:5]])
            else:
                values.extend([float(x) if x is not None else 0.0 for x in v])
        else:
            values.append(float(v) if v is not None else 0.0)
    arr = np.array(values, dtype=np.float32)
    assert arr.shape == (GEOMETRY_DIM,), (
        f"Diharapkan {GEOMETRY_DIM} nilai, dapat {arr.shape[0]}. "
        f"Pastikan geometry.json menggunakan extract_geometry.py terbaru ({GEOMETRY_DIM} fitur)."
    )
    return arr
