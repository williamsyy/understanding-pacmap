"""Precompute PaCMAP embeddings for the 3D mammoth over a 3-parameter grid
(n_neighbors x MN_ratio x FP_ratio).

Output matches the encoding produced by scripts/compress_mammoth.js so the
existing scatter-plot front-end can consume it unchanged: bit-packed 10-bit
integers → base64, keys of the form ``n=<n_neighbors>,mn=<MN_ratio>,fp=<FP_ratio>``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pacmap

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW = REPO / "raw_data" / "mammoth_umap.json"
OUT_ENCODED = REPO / "public" / "mammoth_10k_encoded.json"
OUT_3D = REPO / "public" / "mammoth_3d.json"

N_BITS = 10
MAX_VAL = (1 << N_BITS) - 1

# 3-dim grid — kept small to stay under a couple minutes total wall time.
N_NEIGHBORS = [5, 15, 50, 200]
MN_RATIOS = [0.25, 0.5, 1.0]
FP_RATIOS = [1.0, 2.0, 4.0]


def encode_bits(numbers: np.ndarray, n_bits: int) -> bytes:
    numbers = np.asarray(numbers, dtype=np.uint32)
    count = numbers.size
    total_bits = count * n_bits
    bits = np.zeros(total_bits, dtype=np.uint8)
    for j in range(n_bits):
        bits[j::n_bits] = (numbers >> j) & 1
    byte_count = (total_bits + 7) // 8
    padded_len = byte_count * 8
    if padded_len > total_bits:
        bits = np.concatenate([bits, np.zeros(padded_len - total_bits, dtype=np.uint8)])
    powers = (1 << np.arange(8, dtype=np.uint32)).astype(np.uint32)
    packed = (bits.reshape(-1, 8).astype(np.uint32) * powers).sum(axis=1).astype(np.uint8)
    return packed.tobytes()


def quantize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = np.where(maxs > mins, maxs - mins, 1.0)
    q = np.rint((points - mins) / ranges * MAX_VAL).astype(np.int64)
    q = np.clip(q, 0, MAX_VAL)
    return q.astype(np.uint32)


def main() -> None:
    with open(RAW, "r", encoding="utf-8") as f:
        raw = json.load(f)

    points3d = np.asarray(raw["3d"], dtype=np.float64)
    labels = np.asarray(raw["labels"], dtype=np.int64)
    print(f"Loaded {points3d.shape[0]} points; {labels.max() + 1} unique labels.")

    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    label_offsets = np.bincount(sorted_labels).tolist()

    p3d_sorted = points3d[order]
    p3d_reordered = np.stack([p3d_sorted[:, 1], p3d_sorted[:, 2], p3d_sorted[:, 0]], axis=1)
    with open(OUT_3D, "w", encoding="utf-8") as f:
        json.dump(p3d_reordered.tolist(), f)
    print(f"Wrote {OUT_3D.name}.")

    projections: dict[str, str] = {}
    total = len(N_NEIGHBORS) * len(MN_RATIOS) * len(FP_RATIOS)
    done = 0
    for n_nb in N_NEIGHBORS:
        for mn in MN_RATIOS:
            for fp in FP_RATIOS:
                key = f"n={n_nb},mn={mn},fp={fp}"
                reducer = pacmap.PaCMAP(
                    n_components=2,
                    n_neighbors=n_nb,
                    MN_ratio=float(mn),
                    FP_ratio=float(fp),
                    random_state=42,
                    verbose=False,
                )
                embedding = reducer.fit_transform(points3d, init="pca")
                embedding = embedding[order]

                q = quantize(embedding).reshape(-1)
                packed = encode_bits(q, N_BITS)
                projections[key] = base64.b64encode(packed).decode("ascii")
                done += 1
                print(f"  [{done:2d}/{total}] {key}")

    out = {"projections": projections, "labelOffsets": label_offsets}
    with open(OUT_ENCODED, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Wrote {OUT_ENCODED.name} with {len(projections)} projections.")


if __name__ == "__main__":
    main()
