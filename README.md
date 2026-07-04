# Understanding PaCMAP

A local port of Google PAIR's *[Understanding UMAP](https://pair-code.github.io/understanding-umap/)*
article ([source](https://github.com/PAIR-code/understanding-umap), Apache 2.0)
rewired for [PaCMAP](https://github.com/YingfanWang/PaCMAP) — the article shell,
figure interactions, and rhetorical arc come from the original; the algorithm
content, precomputed embeddings, and the seed-stability comparison figure are
new.

**Live version:** https://williamsyy.github.io/understanding-pacmap/

**Article shell credit:** Andy Coenen and Adam Pearce, Google PAIR (2019), Apache 2.0.

**PaCMAP:** Wang, Huang, Sun, Rudin — *"Understanding How Dimensionality Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization"* (JMLR 2021).

**Empirical benchmarks:** Duke DR Group, https://sites.duke.edu/dimensionreduction/

## Running locally

```bash
yarn                                    # (or npm install)
yarn dev                                # rollup watch + sirv serve on port 5000
# — or just build + serve —
npx rollup -c --environment build:main
npx rollup -c --environment build:supplement
npx sirv-cli public --single --port 5174
```

## Deploying to GitHub Pages

The `scripts/deploy.js` script publishes `public/` to the `gh-pages` branch via
the `gh-pages` npm package. Set `URL_PREFIX` to the project-page subpath so the
built HTML uses correct absolute paths:

```bash
URL_PREFIX=/understanding-pacmap npx rollup -c --environment build:main
URL_PREFIX=/understanding-pacmap npx rollup -c --environment build:supplement
node scripts/deploy.js
```

Then in **Repo → Settings → Pages**, set *Source* to the `gh-pages` branch.

## Precompute pipeline

All interactive figures load pre-encoded JSON payloads from `public/`. The
Python scripts under `scripts/` regenerate them:

| Figure | Script |
|---|---|
| 1 (interactive toy datasets, play/pause animation) | `scripts/precompute_toy_animation.py` |
| 2 (FMNIST 3D scatter) | `scripts/precompute_fmnist_pacmap.py` |
| 4 (hyperparameter grid) | `scripts/precompute_hyperparameters_pacmap.py` |
| 5, 6 (mammoth, 3-parameter grid) | `scripts/precompute_mammoth_pacmap.py` |
| 7 (toy comparison PaCMAP vs t-SNE) | `scripts/precompute_toy_comparison_pacmap.py` |
| 8 (seed stability across UMAP / t-SNE / PaCMAP) | `scripts/precompute_stability.py` |

## License

Apache 2.0. See `LICENSE`.
