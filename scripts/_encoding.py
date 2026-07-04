"""Shared 8-bit / 10-bit range-normalized encoder used by both the
hyperparameters and toy_comparison figures. Matches the format the front-end
expects: {data, length, nDimensions, ranges}."""
from __future__ import annotations

import base64
from typing import Dict, List

import numpy as np


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


def encode_embedding_with_ranges(embedding: np.ndarray, n_bits: int = 8) -> Dict:
    """Encode a 2D projection with per-dimension min/max ranges stored."""
    assert embedding.ndim == 2 and embedding.shape[1] == 2, embedding.shape
    scale = 1 << n_bits
    mins = embedding.min(axis=0)
    maxs = embedding.max(axis=0)
    ranges = np.where(maxs > mins, maxs - mins, 1.0)
    q = np.floor((embedding - mins) / ranges * scale).astype(np.int64)
    q = np.clip(q, 0, scale - 1).astype(np.uint32)
    interleaved = q.reshape(-1)
    packed = encode_bits(interleaved, n_bits)
    return {
        "data": base64.b64encode(packed).decode("ascii"),
        "length": int(embedding.shape[0]),
        "nDimensions": 2,
        "ranges": [
            {"min": float(mins[0]), "max": float(maxs[0])},
            {"min": float(mins[1]), "max": float(maxs[1])},
        ],
    }
