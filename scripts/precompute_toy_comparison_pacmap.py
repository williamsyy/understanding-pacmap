"""Precompute PaCMAP + t-SNE embeddings for the toy_comparison figure.

Writes public/toy_comparison_encoded.json in the shape:
    [ {umap: {nNeighbors=N: enc}, tsne: {perplexity=P: enc}}, ... ]
The front-end reads `umap` and `tsne` sub-objects; in this port `umap` holds
PaCMAP output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pacmap
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _encoding import encode_embedding_with_ranges  # noqa: E402
from toy_generators import DEMOS  # noqa: E402

REPO = HERE.parent
OUT = REPO / "public" / "toy_comparison_encoded.json"

PACMAP_N_NEIGHBORS = [3, 5, 15, 30, 50, 100]
TSNE_PERPLEXITIES = [3, 5, 15, 30, 50, 100]


def _safe(n_points: int, requested: int) -> int:
    return max(2, min(requested, n_points - 1))


def main() -> None:
    output = []
    for demo_index, cfg in enumerate(DEMOS):
        rng = np.random.default_rng(42 + demo_index)
        points = cfg.generator(*cfg.args, rng=rng).astype(np.float64)
        n_points = points.shape[0]
        umap_entry = {}
        tsne_entry = {}

        for n_nb in PACMAP_N_NEIGHBORS:
            eff = _safe(n_points, n_nb)
            reducer = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=eff,
                MN_ratio=0.5,
                FP_ratio=2.0,
                random_state=42,
                verbose=False,
                apply_pca=False,
            )
            try:
                emb = reducer.fit_transform(points, init="pca")
            except Exception as exc:
                print(f"  {cfg.name}: pacmap n={n_nb} FAILED ({exc})")
                emb = np.zeros((n_points, 2))
            umap_entry[f"nNeighbors={n_nb}"] = encode_embedding_with_ranges(emb)

        for perp in TSNE_PERPLEXITIES:
            eff = _safe(n_points, int(perp))
            # sklearn requires perplexity < n_samples.
            perp_eff = min(float(perp), max(2.0, n_points - 2.0))
            try:
                emb = TSNE(
                    n_components=2,
                    perplexity=perp_eff,
                    random_state=42,
                    init="pca",
                    learning_rate="auto",
                ).fit_transform(points)
            except Exception as exc:
                print(f"  {cfg.name}: tsne p={perp} FAILED ({exc})")
                emb = np.zeros((n_points, 2))
            tsne_entry[f"perplexity={perp}"] = encode_embedding_with_ranges(emb)

        output.append({"umap": umap_entry, "tsne": tsne_entry})
        print(f"[{demo_index + 1:2d}/{len(DEMOS)}] {cfg.name}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f)
    print(f"Wrote {OUT.name} for {len(output)} demos.")


if __name__ == "__main__":
    main()
