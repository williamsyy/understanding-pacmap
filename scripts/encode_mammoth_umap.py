"""Encode the original PAIR UMAP mammoth projections (from raw_data/mammoth_umap.json)
into the same 10-bit format that mammoth_10k_encoded.json uses, so a UmapReal2d
component can load real UMAP data alongside PaCMAP + t-SNE."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAW = REPO / "raw_data" / "mammoth_umap.json"
OUT = REPO / "public" / "mammoth_umap_encoded.json"

N_BITS = 10
MAX_VAL = (1 << N_BITS) - 1


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


def main() -> None:
    with open(RAW, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = np.asarray(raw["labels"], dtype=np.int64)
    order = np.argsort(labels, kind="stable")
    label_offsets = np.bincount(labels[order]).tolist()

    projections = {}
    for key, proj in raw["projections"].items():
        p = np.asarray(proj, dtype=np.float64)
        p = p[order]
        # Values are already integer in [0, 1024]; clamp and pack.
        q = np.clip(np.rint(p).astype(np.int64), 0, MAX_VAL).astype(np.uint32).reshape(-1)
        projections[key] = base64.b64encode(encode_bits(q, N_BITS)).decode("ascii")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"projections": projections, "labelOffsets": label_offsets}, f)
    print(f"Wrote {OUT.name} with {len(projections)} projections.")


if __name__ == "__main__":
    main()
