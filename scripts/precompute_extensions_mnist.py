"""Precompute MNIST embeddings for the three PaCMAP extensions shown in §9.

Methods:
  - localmap      : pacmap.LocalMAP  (LocalMAP paper — AAAI 2025)
  - parampacmap   : parampacmap.ParamPaCMAP  (ParamRepulsor / NeurIPS 2024)
  - rashomon      : 4 unaligned PaCMAP embeddings with different seeds
                    (RashomonDR / AISTATS 2026 — a member of the Rashomon set)

Outputs public/extensions_encoded.json in the same format as
stability_encoded.json — 2000 labels + per-method encoded embeddings.
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


def _read_idx_images(data: bytes) -> np.ndarray:
    _, n_img, rows, cols = struct.unpack(">IIII", data[:16])
    assert rows == cols == 28
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n_img, 28, 28)


def _read_idx_labels(data: bytes) -> np.ndarray:
    _, _n = struct.unpack(">II", data[:8])
    return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)


def load_mnist_full() -> tuple[np.ndarray, np.ndarray]:
    """Full 70k-image MNIST (60k train + 10k test), 784-dim vectors in [0,1]."""
    def _fetch(url):
        print(f"  downloading {url}")
        with urllib.request.urlopen(url) as r:
            return r.read()

    train_imgs = _read_idx_images(gzip.decompress(_fetch(MNIST_URLS["train_images"])))
    train_lbls = _read_idx_labels(gzip.decompress(_fetch(MNIST_URLS["train_labels"])))
    test_imgs  = _read_idx_images(gzip.decompress(_fetch(MNIST_URLS["test_images"])))
    test_lbls  = _read_idx_labels(gzip.decompress(_fetch(MNIST_URLS["test_labels"])))

    images = np.concatenate([train_imgs, test_imgs], axis=0)
    labels = np.concatenate([train_lbls, test_lbls], axis=0)
    pixels = images.reshape(len(images), -1).astype(np.float32) / 255.0
    print(f"Full MNIST loaded: {pixels.shape}, labels {labels.shape}")
    return pixels.astype(np.float64), labels


def fit_localmap(x, seed=42):
    return pacmap.LocalMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
        random_state=seed, verbose=False,
    ).fit_transform(x, init="pca")


def fit_pacmap(x, seed=42, init="pca"):
    return pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
        random_state=seed, verbose=False,
    ).fit_transform(x, init=init)


def fit_parampacmap(x, seed=42):
    try:
        import parampacmap  # type: ignore
    except Exception as exc:
        print(f"  parampacmap unavailable ({exc}); skipping ParamRepulsor.")
        return None
    # ParamPaCMAP uses `seed`, not `random_state`.
    reducer = parampacmap.ParamPaCMAP(n_components=2, seed=seed, verbose=False)
    return reducer.fit_transform(x)


def main() -> None:
    x, y = load_mnist_full()

    out = {"labels": y.tolist(), "methods": {}}

    print("Fitting LocalMAP ...")
    out["methods"]["localmap"] = encode_embedding_with_ranges(fit_localmap(x))

    print("Fitting ParamRepulsor (parampacmap.ParamPaCMAP) ...")
    pp = fit_parampacmap(x)
    if pp is not None:
        out["methods"]["parampacmap"] = encode_embedding_with_ranges(pp)

    print("Fitting 4 PaCMAP seeds (RashomonDR) with random init ...")
    # Random init (not PCA) — otherwise all four embeddings inherit the same
    # PCA-driven global structure and look nearly identical, which defeats
    # the purpose of showing the Rashomon set.
    out["rashomon"] = [
        encode_embedding_with_ranges(fit_pacmap(x, seed=s, init="random"))
        for s in (11, 22, 33, 44)
    ]

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
