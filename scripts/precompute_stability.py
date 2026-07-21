"""Precompute UMAP / t-SNE / PaCMAP embeddings for the seed-stability figure.

Three datasets:
  - mammoth: 2000-point subsample of the 3D woolly-mammoth skeleton.
  - fmnist:  the 2000-image Fashion-MNIST subset produced by
             precompute_fmnist_pacmap.py (raw pixels, 784 dim).
  - hierarchical: Duke DR Group's 5x5x5 nested Gaussian clusters (125 leaves,
                  ~2500 points), a stress test for global structure.

For each (dataset, method, seed) triple we fit the DR method with that seed
as random_state / random_seed and store the 2-D embedding. The front-end
picks a triple via sliders and shows the three methods side by side.
"""
from __future__ import annotations

import gzip
import json
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pacmap
import umap
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _encoding import encode_embedding_with_ranges  # noqa: E402

REPO = HERE.parent
PUBLIC = REPO / "public"
OUT = PUBLIC / "stability_encoded.json"

SEEDS = [1, 2, 3, 4]
METHODS = ["umap", "tsne", "pacmap"]

MAMMOTH_SAMPLE = 2000
MNIST_SAMPLE = None  # None → use the full 10k test set
HIER_LEAVES_PER = 5

MNIST_IMAGES_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
MNIST_LABELS_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
HIER_LEAVES_LEVELS = 3   # 5^3 = 125 leaves
HIER_POINTS_PER_LEAF = 20
HIER_DIM = 20
HIER_SEP = (4.0, 1.6, 0.6)  # macro / meso / micro Gaussian centres separation


# ---------- dataset loaders ---------------------------------------------

def load_mammoth() -> tuple[np.ndarray, np.ndarray]:
    with open(REPO / "raw_data" / "mammoth_umap.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = np.asarray(raw["3d"], dtype=np.float64)
    labels = np.asarray(raw["labels"], dtype=np.int64)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=MAMMOTH_SAMPLE, replace=False)
    idx.sort()
    return pts[idx], labels[idx]


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Fetch MNIST test set (10k digits). Uses full set when MNIST_SAMPLE is None."""
    def _fetch(url: str) -> bytes:
        print(f"  downloading {url}")
        with urllib.request.urlopen(url) as r:
            return r.read()

    img_data = gzip.decompress(_fetch(MNIST_IMAGES_URL))
    lbl_data = gzip.decompress(_fetch(MNIST_LABELS_URL))

    magic, n_img, rows, cols = struct.unpack(">IIII", img_data[:16])
    assert magic == 2051 and rows == cols == 28
    images = np.frombuffer(img_data[16:], dtype=np.uint8).reshape(n_img, 28, 28)
    magic, n_lbl = struct.unpack(">II", lbl_data[:8])
    assert magic == 2049
    labels = np.frombuffer(lbl_data[8:], dtype=np.uint8).astype(np.int64)

    if MNIST_SAMPLE is None:
        pixels = images.reshape(n_img, 28 * 28).astype(np.float64) / 255.0
        return pixels, labels
    rng = np.random.default_rng(0)
    idx = rng.choice(n_img, size=MNIST_SAMPLE, replace=False)
    idx.sort()
    pixels = images[idx].reshape(MNIST_SAMPLE, 28 * 28).astype(np.float64) / 255.0
    return pixels, labels[idx]


def make_hierarchical() -> tuple[np.ndarray, np.ndarray]:
    """Duke-style nested Gaussian clusters: 5 macro × 5 meso × 5 micro."""
    rng = np.random.default_rng(0)
    macro_sep, meso_sep, micro_sep = HIER_SEP
    centres_macro = rng.standard_normal((HIER_LEAVES_PER, HIER_DIM)) * macro_sep
    points: List[np.ndarray] = []
    labels: List[int] = []
    leaf_id = 0
    for i, m1 in enumerate(centres_macro):
        centres_meso = m1 + rng.standard_normal((HIER_LEAVES_PER, HIER_DIM)) * meso_sep
        for j, m2 in enumerate(centres_meso):
            centres_micro = m2 + rng.standard_normal((HIER_LEAVES_PER, HIER_DIM)) * micro_sep
            for k, m3 in enumerate(centres_micro):
                p = m3 + rng.standard_normal((HIER_POINTS_PER_LEAF, HIER_DIM)) * 0.15
                points.append(p)
                # Color by macro cluster so the visualization tests whether the 5
                # top-level groups survive projection.
                labels.extend([i] * HIER_POINTS_PER_LEAF)
                leaf_id += 1
    return np.vstack(points), np.asarray(labels, dtype=np.int64)


# ---------- fit wrappers ------------------------------------------------

def fit_umap(x: np.ndarray, seed: int) -> np.ndarray:
    # `init='random'` is important — the default spectral init is essentially
    # deterministic given the neighbor graph, so it hides the seed-dependence
    # this figure is meant to demonstrate.
    return umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        init="random",
        random_state=seed,
    ).fit_transform(x)


def fit_tsne(x: np.ndarray, seed: int) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=30,
        random_state=seed,
        init="random",   # different init per seed → variance is visible
        learning_rate="auto",
    ).fit_transform(x)


def fit_pacmap(x: np.ndarray, seed: int) -> np.ndarray:
    return pacmap.PaCMAP(
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=seed,
        verbose=False,
    ).fit_transform(x, init="pca")


FIT = {"umap": fit_umap, "tsne": fit_tsne, "pacmap": fit_pacmap}


# ---------- alignment ---------------------------------------------------

def procrustes_align(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Rotate/reflect/scale `source` to best match `target` in the L2 sense.
    Used only so seed 2..8 embeddings sit in the same frame as seed 1 —
    prevents distracting rotations that would obscure the real seed variance."""
    a = source - source.mean(0)
    b = target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
    r = u @ vt
    aligned = a @ r
    scale = (b * aligned).sum() / (aligned * aligned).sum()
    return aligned * scale + target.mean(0)


# ---------- main --------------------------------------------------------

def main() -> None:
    datasets = {
        "mammoth": load_mammoth(),
        "mnist": load_mnist(),
    }
    for name, (x, y) in datasets.items():
        print(f"[{name}] shape={x.shape} classes={len(set(y.tolist()))}")

    out: Dict = {"seeds": SEEDS, "datasets": {}}

    for ds_name, (x, y) in datasets.items():
        print(f"\n=== {ds_name} ===")
        ds_out = {"labels": y.tolist(), "methods": {}}
        for method in METHODS:
            print(f"  {method}: ", end="", flush=True)
            per_seed: List[dict] = []
            reference: np.ndarray | None = None
            for seed in SEEDS:
                try:
                    emb = FIT[method](x, seed)
                    if reference is None:
                        reference = emb.copy()
                    else:
                        emb = procrustes_align(reference, emb)
                    per_seed.append(encode_embedding_with_ranges(emb))
                    print(seed, end=" ", flush=True)
                except Exception as exc:
                    print(f"seed {seed} FAILED ({exc})", end=" ", flush=True)
                    per_seed.append(encode_embedding_with_ranges(np.zeros((x.shape[0], 2))))
            ds_out["methods"][method] = per_seed
            print()
        out["datasets"][ds_name] = ds_out

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nWrote {OUT.name} ({size_mb:.2f} MB).")


if __name__ == "__main__":
    main()
