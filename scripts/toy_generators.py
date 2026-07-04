"""Python port of understanding-umap's src/shared/js/generators.js and
toy-configs.js. Each generator function takes the same positional arguments
as its JavaScript counterpart and returns a NumPy array of shape (n, dim).

Random state is threaded through a per-call rng argument so precompute is
reproducible."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw


# ----- basic helpers -----------------------------------------------------

def _normal_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    return rng.standard_normal(dim)


# ----- generators (each returns np.ndarray of shape (N, D)) --------------

def grid_data(size: int, *, rng: np.random.Generator) -> np.ndarray:
    return np.array([[x, y] for x in range(size) for y in range(size)], dtype=np.float64)


def two_clusters_data(n: int, dim: int = 50, *, rng: np.random.Generator) -> np.ndarray:
    out = []
    for _ in range(n):
        out.append(_normal_vector(rng, dim))
        v = _normal_vector(rng, dim)
        v[0] += 10.0
        out.append(v)
    return np.asarray(out, dtype=np.float64)


def three_clusters_data(n: int, dim: int = 50, *, rng: np.random.Generator) -> np.ndarray:
    out = []
    for _ in range(n):
        out.append(_normal_vector(rng, dim))
        v2 = _normal_vector(rng, dim); v2[0] += 10.0
        out.append(v2)
        v3 = _normal_vector(rng, dim); v3[0] += 50.0
        out.append(v3)
    return np.asarray(out, dtype=np.float64)


def two_different_clusters_data(n: int, dim: int = 50, scale: float = 10.0, *, rng: np.random.Generator) -> np.ndarray:
    out = []
    for _ in range(n):
        out.append(_normal_vector(rng, dim))
        v = _normal_vector(rng, dim) / scale
        v[0] += 20.0
        out.append(v)
    return np.asarray(out, dtype=np.float64)


def long_cluster_data(n: int, *, rng: np.random.Generator) -> np.ndarray:
    s = 0.03 * n
    out = []
    for i in range(n):
        x1 = i + s * rng.standard_normal()
        y1 = i + s * rng.standard_normal()
        out.append([x1, y1])
        x2 = i + s * rng.standard_normal() + n / 5
        y2 = i + s * rng.standard_normal() - n / 5
        out.append([x2, y2])
    return np.asarray(out, dtype=np.float64)


def subset_clusters_data(n: int, dim: int = 2, *, rng: np.random.Generator) -> np.ndarray:
    out = []
    for _ in range(n):
        out.append(_normal_vector(rng, dim))
        out.append(_normal_vector(rng, dim) * 50.0)
    return np.asarray(out, dtype=np.float64)


def circle_data(num_points: int, *, rng: np.random.Generator) -> np.ndarray:
    t = 2 * np.pi * np.arange(num_points) / num_points
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def random_circle_data(num_points: int, *, rng: np.random.Generator) -> np.ndarray:
    t = 2 * np.pi * rng.random(num_points)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def gaussian_data(n: int, dim: int, *, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, dim))


def long_gaussian_data(n: int, dim: int, *, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal((n, dim))
    scales = 1.0 / (1.0 + np.arange(dim))
    return x * scales


def trefoil_data(n: int, *, rng: np.random.Generator) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.stack([x, y, z], axis=1)


def _rotate_yz(x: np.ndarray, y: np.ndarray, z: np.ndarray, angle: float = 0.4) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    u = x
    v = c * y + s * z
    w = -s * y + c * z
    return np.stack([u, v, w], axis=1)


def link_data(n: int, *, rng: np.random.Generator) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    sin, cos = np.sin(t), np.cos(t)
    ring1 = _rotate_yz(cos, sin, np.zeros_like(t))
    ring2 = _rotate_yz(1 + cos, np.zeros_like(t), sin)
    interleaved = np.empty((2 * n, 3), dtype=np.float64)
    interleaved[0::2] = ring1
    interleaved[1::2] = ring2
    return interleaved


def unlink_data(n: int, *, rng: np.random.Generator) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    sin, cos = np.sin(t), np.cos(t)
    ring1 = _rotate_yz(cos, sin, np.zeros_like(t))
    ring2 = _rotate_yz(3 + cos, np.zeros_like(t), sin)
    interleaved = np.empty((2 * n, 3), dtype=np.float64)
    interleaved[0::2] = ring1
    interleaved[1::2] = ring2
    return interleaved


def ortho_curve(n: int, *, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        out[i, :i] = 1.0
    return out


def random_walk(n: int, dim: int, *, rng: np.random.Generator) -> np.ndarray:
    steps = rng.standard_normal((n, dim))
    return np.cumsum(steps, axis=0)


def random_jump(n: int, dim: int, *, rng: np.random.Generator) -> np.ndarray:
    current = np.zeros(dim)
    out = []
    for _ in range(n):
        step = _normal_vector(rng, dim)
        current = current + step
        r = _normal_vector(rng, dim) * np.sqrt(dim)
        out.append(current + r)
    return np.asarray(out, dtype=np.float64)


def simplex_data(n: int, noise: float = 0.5, *, rng: np.random.Generator) -> np.ndarray:
    diag = 1.0 + noise * rng.standard_normal(n)
    return np.diag(diag)


def cube_data(n: int, dim: int, *, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, dim))


def star(n: int, n_arms: int, dim: int, *, rng: np.random.Generator) -> np.ndarray:
    per_arm = n // n_arms
    out = []
    for _ in range(n_arms):
        arm_vec = _normal_vector(rng, dim)
        for i in range(per_arm):
            pct = i / per_arm
            noise = rng.random(dim) * 0.01 - 0.005
            out.append(arm_vec * pct + noise)
    return np.asarray(out, dtype=np.float64)


def linked_clusters(n_clusters: int, per_cluster: int, per_link: int, dim: int, *, rng: np.random.Generator) -> np.ndarray:
    centroids: List[np.ndarray] = []
    out = []
    for i in range(n_clusters):
        centroid = _normal_vector(rng, dim)
        centroids.append(centroid)
        for _ in range(per_cluster):
            noise = rng.random(dim) * 0.2 - 0.1
            out.append(centroid + noise)
        if i > 0:
            last = centroids[i - 1]
            for j in range(per_link):
                pct = j / per_link
                p = centroid + (last - centroid) * pct
                noise = rng.random(dim) * 0.01 - 0.005
                out.append(p + noise)
    return np.asarray(out, dtype=np.float64)


def _draw_line_pixels(angle: float, n_pixels: int) -> np.ndarray:
    """Rasterize a single white line on a black square, return length-N*N grayscale."""
    center = n_pixels / 2.0
    line_distance = n_pixels * 2
    dY = np.sin(angle) * line_distance
    dX = np.cos(angle) * line_distance
    img = Image.new("L", (n_pixels, n_pixels), color=0)
    draw = ImageDraw.Draw(img)
    draw.line([(center - dX, center - dY), (center + dX, center + dY)], fill=255, width=2)
    return np.asarray(img, dtype=np.uint8).reshape(-1)


def continuous_line_images(n_lines: int, n_pixels: int = 28, *, rng: np.random.Generator) -> np.ndarray:
    return np.stack([
        _draw_line_pixels(np.pi * i / n_lines, n_pixels) for i in range(n_lines)
    ]).astype(np.float64)


def clustered_line_images(n_lines: int, n_clusters: int, noise_param: float = 25.0, n_pixels: int = 28, *, rng: np.random.Generator) -> np.ndarray:
    per_cluster = n_lines // n_clusters
    out = []
    for i in range(n_clusters):
        progress = i / n_clusters
        for _ in range(per_cluster):
            noise = rng.random() * (noise_param / 100.0) * np.pi
            angle = np.pi * progress + noise
            out.append(_draw_line_pixels(angle, n_pixels))
    return np.asarray(out, dtype=np.float64)


def sine_frequency(n_vectors: int, vector_size: int, *, rng: np.random.Generator) -> np.ndarray:
    min_freq = np.pi / (2 * vector_size)
    max_freq = np.pi / ((1 / 10) * vector_size)
    out = np.empty((n_vectors, vector_size), dtype=np.float64)
    xs = np.arange(vector_size)
    for i in range(n_vectors):
        progress = i / n_vectors
        freq = (max_freq - min_freq) * progress + min_freq
        out[i] = np.sin(freq * xs)
    return out


def sine_phase(n_vectors: int, vector_size: int, *, rng: np.random.Generator) -> np.ndarray:
    freq = (2 * np.pi) / vector_size
    phase = vector_size / n_vectors
    out = np.empty((n_vectors, vector_size), dtype=np.float64)
    xs = np.arange(vector_size)
    for i in range(n_vectors):
        phase_offset = phase * (i / n_vectors)
        out[i] = np.sin(freq * (xs + phase_offset))
    return out


# ----- dataset registry --------------------------------------------------
# Order matches allDemos = [...extendedDemos, ...demos] from toy-configs.js,
# with the option defaults overridden per originalOverrides/extendedOverrides
# in toy_comparison_visualization/js/demos.js and preprocess.js's
# "Dimensions = max/2" rule.

@dataclass
class DemoConfig:
    name: str
    generator: Callable
    args: Sequence

DEMOS: List[DemoConfig] = [
    # ---- extended (6) ----
    DemoConfig("Star", star, (300, 12, 25)),  # dim was max/2=25
    DemoConfig("Linked Clusters", linked_clusters, (6, 100, 50, 50)),  # dim was max/2=50
    DemoConfig("Rotated lines", continuous_line_images, (200, 28)),
    DemoConfig("Rotated lines, clustered", clustered_line_images, (200, 10, 8, 28)),
    DemoConfig("Sine frequency", sine_frequency, (200, 256)),
    DemoConfig("Sine phase", sine_phase, (200, 256)),
    # ---- original (18) ----
    DemoConfig("Grid", grid_data, (20,)),
    DemoConfig("Two Clusters", two_clusters_data, (100, 50)),
    DemoConfig("Three Clusters", three_clusters_data, (100, 50)),
    DemoConfig("Two Different-Sized Clusters", two_different_clusters_data, (100, 50, 5)),
    DemoConfig("Two Long Linear Clusters", long_cluster_data, (100,)),
    DemoConfig("Cluster In Cluster", subset_clusters_data, (100, 50)),
    DemoConfig("Circle (Evenly Spaced)", circle_data, (200,)),
    DemoConfig("Circle (Randomly Spaced)", random_circle_data, (200,)),
    DemoConfig("Gaussian Cloud", gaussian_data, (250, 50)),
    DemoConfig("Ellipsoidal Gaussian Cloud", long_gaussian_data, (250, 50)),
    DemoConfig("Trefoil Knot", trefoil_data, (200,)),
    DemoConfig("Linked Rings", link_data, (200,)),
    DemoConfig("Unlinked Rings", unlink_data, (200,)),
    DemoConfig("Orthogonal Steps", ortho_curve, (200,)),
    DemoConfig("Random Walk", random_walk, (200, 100)),
    DemoConfig("Random Jump", random_jump, (200, 100)),
    DemoConfig("Equally Spaced", simplex_data, (200,)),
    DemoConfig("Uniform Distribution", cube_data, (200, 5)),  # dim was max/2=5
]


def generate_all_demos(seed: int = 42) -> List[dict]:
    """Return list of {name, points} dicts, one per demo."""
    out = []
    for i, cfg in enumerate(DEMOS):
        rng = np.random.default_rng(seed + i)
        pts = cfg.generator(*cfg.args, rng=rng)
        out.append({"name": cfg.name, "points": pts})
    return out


if __name__ == "__main__":
    all_data = generate_all_demos()
    for d in all_data:
        print(f"{d['name']:35s} shape={d['points'].shape}")
