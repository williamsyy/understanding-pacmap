"""Precompute PaCMAP over a hyperparameter grid for each toy dataset.

Writes public/hyperparameters_encoded.json in the format expected by the
existing hyperparameters_visualization front-end (with UMAP-shaped keys
`nNeighbors=X,minDist=Y` — we reuse those keys and let the front-end labels
be re-titled MN_ratio in the article prose)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pacmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _encoding import encode_embedding_with_ranges  # noqa: E402
from toy_generators import DEMOS  # noqa: E402

REPO = HERE.parent
OUT = REPO / "public" / "hyperparameters_encoded.json"

# Match the on-page slider values: 5 n_neighbors x 6 "min_dist"/MN_ratio slots.
N_NEIGHBORS = [5, 15, 30, 50, 100]
MN_RATIOS = [0, 0.01, 0.05, 0.1, 0.5, 1.0]


def _safe_pacmap_neighbors(n_points: int, requested: int) -> int:
    """PaCMAP will error out if n_neighbors ≥ n_points. Clamp defensively."""
    return max(2, min(requested, n_points - 1))


def main() -> None:
    output = []
    total = len(DEMOS) * len(N_NEIGHBORS) * len(MN_RATIOS)
    done = 0
    for demo_index, cfg in enumerate(DEMOS):
        rng = np.random.default_rng(42 + demo_index)
        points = cfg.generator(*cfg.args, rng=rng).astype(np.float64)
        entry = {}
        for n_nb in N_NEIGHBORS:
            eff = _safe_pacmap_neighbors(points.shape[0], n_nb)
            for mn in MN_RATIOS:
                key = f"nNeighbors={n_nb},minDist={mn}"
                # PaCMAP's MN_ratio replaces the UMAP-shaped "minDist" key.
                # A 0 MN_ratio disables mid-near pairs entirely.
                effective_mn = max(mn, 0.01)  # PaCMAP requires >0
                reducer = pacmap.PaCMAP(
                    n_components=2,
                    n_neighbors=eff,
                    MN_ratio=float(effective_mn),
                    FP_ratio=2.0,
                    random_state=42,
                    verbose=False,
                    apply_pca=False,
                )
                try:
                    emb = reducer.fit_transform(points, init="pca")
                except Exception as exc:  # pragma: no cover
                    print(f"  {cfg.name}: {key} FAILED ({exc}); using zeros")
                    emb = np.zeros((points.shape[0], 2))
                entry[key] = encode_embedding_with_ranges(emb)
                done += 1
        output.append(entry)
        print(f"[{demo_index + 1:2d}/{len(DEMOS)}] {cfg.name:35s}  {done}/{total} fits done")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f)
    print(f"Wrote {OUT.name} ({len(output)} demos, {done} fits total).")


if __name__ == "__main__":
    main()
