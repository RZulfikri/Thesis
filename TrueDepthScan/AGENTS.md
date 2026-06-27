# Multi-Agent Swarm — TrueDepthScan

> **Kontrak Agent Lokal** — Override spesifik untuk project `TrueDepthScan/`.  
> **Referensi global:** `~/Projects/Thesis/AGENTS.md`

## Project Context
- **Scope:** iOS TrueDepth Scanner (Swift, ARKit)
- **Status:** ✅ Stable — data acquisition ongoing
- **Key files:** `TrueDepthStreamer/`, `Configuration/`
- **Output:** `.ply` files → consumed by `3DCNN/dataset/`

## Agent Constraints
- Code Agent: Must verify build in Xcode before committing changes
- Code Agent: Never break backward compat with `3DCNN/dataset/` layout
- Analysis Agent: Scan quality metrics (coverage, noise, geometry stability)
- Planning Agent: Coordinate with 3DCNN Planning Agent when dataset expansion needed

## Integration Points
- Output `.ply` must be compatible with `3DCNN/` preprocessing pipeline
- `geometry.json` format must match `3DCNN/utils/extract_geometry.py` expectations

*Referensi global: ~/Projects/Thesis/AGENTS.md*
