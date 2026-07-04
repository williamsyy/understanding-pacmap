"""Refetch a subset of Fashion-MNIST, then produce Figure 2 assets from scratch.

The site's original ``fmnist_spritesheet.png`` and ``fmnist_tsne_vs_umap.json``
labels were generated from an unknown subset in an unknown order, so labels
and pixels no longer line up when we run our own PaCMAP on the sprite tiles.

This script fixes that by (1) downloading the official Fashion-MNIST test
set from Zalando's release, (2) taking a 2000-sample subset with a fixed
seed, (3) fitting PaCMAP AND t-SNE on the exact same 2000 images, and
(4) rewriting the shipped spritesheet, ``fmnist_tsne_vs_umap.json`` and
``fmnist.json`` so labels, projections and sprite hover images all agree.
"""
from __future__ import annotations

import gzip
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pacmap
from PIL import Image
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "public"

N_IMAGES = 2000
TILE = 28
GRID = 45  # match the original 45x45 sprite layout

FMNIST_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
IMAGES_URL = FMNIST_BASE + "t10k-images-idx3-ubyte.gz"
LABELS_URL = FMNIST_BASE + "t10k-labels-idx1-ubyte.gz"

LABEL_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _fetch(url: str) -> bytes:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r:
        return r.read()


def load_fmnist_test() -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels): images shape (10000, 28, 28) uint8, labels (10000,) int."""
    img_gz = _fetch(IMAGES_URL)
    lbl_gz = _fetch(LABELS_URL)

    img_data = gzip.decompress(img_gz)
    lbl_data = gzip.decompress(lbl_gz)

    magic, n_img, rows, cols = struct.unpack(">IIII", img_data[:16])
    assert magic == 2051 and rows == cols == TILE, (magic, rows, cols)
    images = np.frombuffer(img_data[16:], dtype=np.uint8).reshape(n_img, TILE, TILE)

    magic, n_lbl = struct.unpack(">II", lbl_data[:8])
    assert magic == 2049 and n_lbl == n_img
    labels = np.frombuffer(lbl_data[8:], dtype=np.uint8)
    return images, labels


def write_spritesheet(images: np.ndarray, out: Path) -> None:
    """Pack up to GRID*GRID tiles row-major into a spritesheet PNG."""
    sheet = np.full((GRID * TILE, GRID * TILE), 255, dtype=np.uint8)
    for i, tile in enumerate(images):
        r, c = divmod(i, GRID)
        if r >= GRID:
            break
        # Invert so black-on-white matches the original site's sprite convention.
        sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = 255 - tile
    Image.fromarray(sheet, mode="L").save(out)
    print(f"  wrote {out.name}")


def main() -> None:
    images, labels = load_fmnist_test()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(images), size=N_IMAGES, replace=False)
    idx.sort()  # keep a canonical order

    sub_images = images[idx]                    # (2000, 28, 28) uint8
    sub_labels = labels[idx].astype(int)        # (2000,)
    print(f"Selected {N_IMAGES} images; label counts: {np.bincount(sub_labels).tolist()}")

    write_spritesheet(sub_images, PUBLIC / "fmnist_spritesheet.png")

    pixels = sub_images.reshape(N_IMAGES, TILE * TILE).astype(np.float64) / 255.0

    print("Fitting PaCMAP (3D) ...")
    pacmap_embedding = pacmap.PaCMAP(
        n_components=3,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=42,
        verbose=False,
    ).fit_transform(pixels, init="pca")

    print("Fitting t-SNE (3D) ...")
    tsne_embedding = TSNE(
        n_components=3,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(pixels)

    def rescale(emb: np.ndarray, target: float = 25.0) -> np.ndarray:
        s = float(np.abs(emb).max())
        return emb * (target / s) if s > 0 else emb

    pacmap_embedding = rescale(pacmap_embedding)
    tsne_embedding = rescale(tsne_embedding)

    side_by_side = {
        "umapProjection": pacmap_embedding.tolist(),
        "tsneProjection": tsne_embedding.tolist(),
        "labels": sub_labels.tolist(),
        "labelNames": LABEL_NAMES,
    }
    with open(PUBLIC / "fmnist_tsne_vs_umap.json", "w", encoding="utf-8") as f:
        json.dump(side_by_side, f)
    print("  wrote fmnist_tsne_vs_umap.json")

    # fmnist.json is the flat (single-projection) file the classifier viz uses.
    single = {
        "labels": sub_labels.tolist(),
        "labelNames": LABEL_NAMES,
        "projection": pacmap_embedding.tolist(),
    }
    with open(PUBLIC / "fmnist.json", "w", encoding="utf-8") as f:
        json.dump(single, f)
    print("  wrote fmnist.json")


if __name__ == "__main__":
    main()
