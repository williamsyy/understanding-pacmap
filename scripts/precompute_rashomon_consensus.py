"""Rerun ONLY the Rashomon section of extensions_encoded.json:
   3 PaCMAP embeddings (random init) + 1 consensus embedding via PCA-referenced
   Procrustes alignment and averaging.

   Loads the existing extensions_encoded.json, replaces the `rashomon` field
   and adds `rashomon_consensus`, keeps `methods` (LocalMAP, ParamRepulsor)
   untouched.
"""
from __future__ import annotations

import gzip
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pacmap
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _encoding import encode_embedding_with_ranges  # noqa: E402

REPO = HERE.parent
PUBLIC = REPO / "public"
OUT = PUBLIC / "extensions_encoded.json"

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _read_images(data):
    _, n, r, c = struct.unpack(">IIII", data[:16])
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, r * c)


def _read_labels(data):
    _, _n = struct.unpack(">II", data[:8])
    return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)


def load_mnist_full():
    def _fetch(url):
        print(f"  downloading {url}")
        with urllib.request.urlopen(url) as r:
            return r.read()
    train = _read_images(gzip.decompress(_fetch(MNIST_URLS["train_images"])))
    trainL = _read_labels(gzip.decompress(_fetch(MNIST_URLS["train_labels"])))
    test  = _read_images(gzip.decompress(_fetch(MNIST_URLS["test_images"])))
    testL = _read_labels(gzip.decompress(_fetch(MNIST_URLS["test_labels"])))
    x = np.concatenate([train, test], axis=0).astype(np.float64) / 255.0
    y = np.concatenate([trainL, testL], axis=0)
    return x, y


def fit_pacmap_random(x, seed):
    return pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
        random_state=seed, verbose=False,
    ).fit_transform(x, init="random")


def procrustes_align(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes + scale + translation, aligning `source` to `target`."""
    a = source - source.mean(0)
    b = target - target.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b, full_matrices=False)
    R = U @ Vt
    aligned = a @ R
    scale = (b * aligned).sum() / (aligned * aligned).sum()
    return aligned * scale + target.mean(0)


def main() -> None:
    with open(OUT, "r", encoding="utf-8") as f:
        payload = json.load(f)

    x, y = load_mnist_full()
    print(f"MNIST loaded: {x.shape}")

    print("PCA reference (2D of raw pixels) ...")
    pca_ref = PCA(n_components=2, random_state=0).fit_transform(x)

    embeddings = []
    for s in (11, 22, 33):
        print(f"Fitting PaCMAP seed={s} (random init) ...")
        embeddings.append(fit_pacmap_random(x, s))

    print("Procrustes-aligning to PCA reference and averaging ...")
    aligned = [procrustes_align(pca_ref, emb) for emb in embeddings]
    consensus = np.mean(np.stack(aligned, axis=0), axis=0)

    # Store the unaligned members (so the reader sees the raw Rashomon variability),
    # and the aggregated consensus.
    payload["rashomon"] = [encode_embedding_with_ranges(e) for e in embeddings]
    payload["rashomon_consensus"] = encode_embedding_with_ranges(consensus)
    payload["labels"] = y.tolist()  # ensure alignment

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Wrote {OUT.name} — 3 Rashomon members + 1 consensus.")


if __name__ == "__main__":
    main()
