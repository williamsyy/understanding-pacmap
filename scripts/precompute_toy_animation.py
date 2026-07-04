"""Precompute animated PaCMAP trajectories for each toy demo.

PaCMAP's intermediate=True flag returns per-snapshot embeddings taken at fixed
iteration checkpoints. We ship those as animation frames so the browser can
play back the optimization without running PaCMAP live.

Grid: 3 x 3 x 3 = 27 (n_neighbors, MN_ratio, FP_ratio) triplets per demo.
"""
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
OUT = REPO / "public" / "toy_animation_encoded.json"

N_NEIGHBORS = [5, 15, 50]
MN_RATIOS = [0.1, 0.5, 1.0]
FP_RATIOS = [1.0, 2.0, 4.0]


def _safe(n_points: int, requested: int) -> int:
    return max(2, min(requested, n_points - 1))


def main() -> None:
    output = []
    total = len(DEMOS) * len(N_NEIGHBORS) * len(MN_RATIOS) * len(FP_RATIOS)
    done = 0
    for demo_index, cfg in enumerate(DEMOS):
        rng = np.random.default_rng(42 + demo_index)
        points = cfg.generator(*cfg.args, rng=rng).astype(np.float64)
        n_points = points.shape[0]
        triplets = {}
        for n_nb in N_NEIGHBORS:
            eff = _safe(n_points, n_nb)
            for mn in MN_RATIOS:
                for fp in FP_RATIOS:
                    key = f"n={n_nb},mn={mn},fp={fp}"
                    reducer = pacmap.PaCMAP(
                        n_components=2,
                        n_neighbors=eff,
                        MN_ratio=float(mn),
                        FP_ratio=float(fp),
                        random_state=42,
                        verbose=False,
                        apply_pca=False,
                        intermediate=True,
                    )
                    try:
                        # shape (n_snapshots, n_points, 2)
                        frames = reducer.fit_transform(points, init="pca")
                    except Exception as exc:
                        print(f"  {cfg.name} {key} FAILED: {exc}")
                        frames = np.zeros((1, n_points, 2))
                    # Pick 6 evenly-spaced frames from the 13 snapshots (or all if <=6).
                    n_frames = frames.shape[0]
                    if n_frames > 6:
                        picks = np.linspace(0, n_frames - 1, 6).astype(int)
                    else:
                        picks = np.arange(n_frames)
                    encoded_frames = [encode_embedding_with_ranges(frames[i]) for i in picks]
                    triplets[key] = encoded_frames
                    done += 1
        output.append(triplets)
        print(f"[{demo_index + 1:2d}/{len(DEMOS)}] {cfg.name:35s}  {done}/{total} fits")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f)
    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT.name} — {size_mb:.2f} MB, {len(output)} demos, {done} fits.")


if __name__ == "__main__":
    main()
