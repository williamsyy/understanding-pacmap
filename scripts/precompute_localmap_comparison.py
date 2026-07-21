"""Precompute PaCMAP + LocalMAP MNIST embeddings side-by-side, plus a
sampled subset of neighbor-pair edges for each so the reader can see the
neighbor graph laid over the embedding.  Merges into extensions_encoded.json."""
from __future__ import annotations

import gzip
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pacmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _encoding import encode_embedding_with_ranges  # noqa: E402

REPO = HERE.parent
PUBLIC = REPO / "public"
OUT = PUBLIC / "extensions_encoded.json"

N_EDGES_SAMPLED = 3000  # how many neighbor edges to ship as gray overlay
SEED = 42

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


def fit_pacmap(x, seed=SEED):
    r = pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
        random_state=seed, verbose=False, save_tree=True,
    )
    emb = r.fit_transform(x, init="pca")
    return emb, np.asarray(r.pair_neighbors)


def fit_localmap(x, seed=SEED):
    r = pacmap.LocalMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
        random_state=seed, verbose=False, save_tree=True,
    )
    emb = r.fit_transform(x, init="pca")
    return emb, np.asarray(r.pair_neighbors)


def main() -> None:
    payload = {}
    if OUT.exists():
        with open(OUT, "r", encoding="utf-8") as f:
            payload = json.load(f)

    x, y = load_mnist_full()

    print("Fitting PaCMAP ...")
    p_emb, p_pairs = fit_pacmap(x)
    print("Fitting LocalMAP ...")
    l_emb, l_pairs = fit_localmap(x)

    rng = np.random.default_rng(0)
    def sample_edges(pairs):
        if len(pairs) <= N_EDGES_SAMPLED:
            return pairs
        idx = rng.choice(len(pairs), size=N_EDGES_SAMPLED, replace=False)
        return pairs[idx]

    p_edges = sample_edges(p_pairs).astype(np.int32)
    l_edges = sample_edges(l_pairs).astype(np.int32)

    # Store as flat int32 arrays serialized to lists (small — 3000 pairs = 6000 ints).
    payload["localmap_comparison"] = {
        "pacmap":       encode_embedding_with_ranges(p_emb),
        "localmap":     encode_embedding_with_ranges(l_emb),
        "pacmap_edges": p_edges.reshape(-1).tolist(),
        "localmap_edges": l_edges.reshape(-1).tolist(),
    }
    # Refresh labels + LocalMAP top-level entry to stay in sync.
    payload.setdefault("methods", {})["localmap"] = encode_embedding_with_ranges(l_emb)
    payload["labels"] = y.tolist()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT.name} — comparison section ({size_mb:.2f} MB total).")


if __name__ == "__main__":
    main()
