<script>
  /* Method comparison + seed stability figure.
     Loads stability_encoded.json and shows UMAP / t-SNE / PaCMAP side by side
     for a selectable (dataset, seed) pair.  As the user drags the seed slider
     PaCMAP stays put while UMAP and t-SNE flicker — the point of the figure. */

  import { onMount, tick } from "svelte";
  import * as d3 from "d3";

  import { decode, fromString } from "../../../shared/js/parse-binary";
  import { N_BITS_HYPERPARAMETERS } from "../../../shared/js/parameters";

  import Slider from "../../../shared/components/Slider.svelte";

  const METHOD_LABEL = { umap: "UMAP", tsne: "t-SNE", pacmap: "PaCMAP" };
  const METHODS = ["umap", "tsne", "pacmap"];

  const DATASETS = [
    { key: "mammoth", label: "Mammoth (3D → 2D)", colorScheme: "spectral", numeric: true },
    { key: "mnist",   label: "MNIST (784D → 2D)", colorScheme: "category" },
  ];

  let allData = null;
  let seedIndex = 0;
  let selectedDatasetIndex = 0;

  $: selectedDataset = DATASETS[selectedDatasetIndex];
  $: seeds = allData ? allData.seeds : [1];
  $: seed = seeds[seedIndex];

  let umapCanvas;
  let tsneCanvas;
  let pacmapCanvas;
  $: canvasRefs = { umap: umapCanvas, tsne: tsneCanvas, pacmap: pacmapCanvas };

  const decodeEntry = encoded => {
    const { data, length, nDimensions, ranges } = encoded;
    const byteArray = fromString(atob(data));
    const decoded = decode(byteArray, nDimensions * length, N_BITS_HYPERPARAMETERS);
    const [rangeX, rangeY] = ranges;
    const scale = 2 ** N_BITS_HYPERPARAMETERS;
    const out = [];
    for (let i = 0; i < decoded.length; i += 2) {
      const x = (decoded[i] / scale) * (rangeX.max - rangeX.min) + rangeX.min;
      const y = (decoded[i + 1] / scale) * (rangeY.max - rangeY.min) + rangeY.min;
      out.push([x, y]);
    }
    return out;
  };

  const buildColorFn = (labels, scheme) => {
    if (scheme === "spectral") {
      const min = d3.min(labels);
      const max = d3.max(labels);
      const s = d3.scaleSequential(d3.interpolateSpectral).domain([min, max]);
      return i => s(labels[i]);
    }
    const palette = d3.schemeCategory10.concat(d3.schemeSet3);
    return i => palette[labels[i] % palette.length];
  };

  const renderPanel = (method, coords, colorFn) => {
    const canvas = canvasRefs[method];
    if (!canvas || !coords || !coords.length) return;
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    const xs = coords.map(c => c[0]);
    const ys = coords.map(c => c[1]);
    const xMin = d3.min(xs), xMax = d3.max(xs);
    const yMin = d3.min(ys), yMax = d3.max(ys);
    const range = Math.max(xMax - xMin, yMax - yMin) || 1;
    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;
    const s = (Math.min(w, h) * 0.9) / range;

    const r = coords.length > 2000 ? 2.0 : coords.length > 800 ? 2.4 : 3.0;
    for (let i = 0; i < coords.length; i++) {
      const [x, y] = coords[i];
      const sx = (x - cx) * s + w / 2;
      const sy = -(y - cy) * s + h / 2;
      ctx.fillStyle = colorFn(i);
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };

  const rerender = async () => {
    await tick();
    if (!allData) return;
    const ds = allData.datasets[selectedDataset.key];
    if (!ds) return;
    const colorFn = buildColorFn(ds.labels, selectedDataset.colorScheme);
    for (const method of METHODS) {
      const encoded = ds.methods[method][seedIndex];
      const coords = decodeEntry(encoded);
      renderPanel(method, coords, colorFn);
    }
  };

  onMount(async () => {
    try {
      const res = await fetch("stability_encoded.json");
      allData = await res.json();
    } catch (err) {
      console.error("Failed to load stability_encoded.json", err);
      return;
    }
    await rerender();
  });

  $: if (allData) rerender(selectedDatasetIndex, seedIndex);
</script>

<style>
  .stability {
    font-family: "Open Sans", sans-serif;
    width: 100%;
    box-sizing: border-box;
  }
  .stability * { box-sizing: border-box; }

  .panels {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 20px;
  }
  @media (max-width: 720px) {
    .panels { grid-template-columns: 1fr; }
  }
  .panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  .panel-title {
    font-weight: 700;
    font-size: 14px;
    color: #222;
    letter-spacing: 0.02em;
  }
  .panel canvas {
    width: 100%;
    height: auto;
    aspect-ratio: 1 / 1;
    display: block;
    background: #ffffff;
    margin-bottom: 8px;
  }
  .panel-title {
    text-align: center;
    font-weight: 700;
    font-size: 14px;
    color: #222;
    margin-bottom: 16px;
    letter-spacing: 0.02em;
  }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 14px 16px;
    background: #f6f7f9;
    border-radius: 6px;
    font-size: 13px;
  }

  .row {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .row-label {
    font-weight: 700;
    color: #222;
    min-width: 90px;
  }

  .dataset-tabs {
    display: flex;
    flex-direction: row;
    gap: 6px;
    flex-wrap: wrap;
  }
  .dataset-tabs button {
    padding: 6px 12px;
    background: white;
    border: 1px solid #cfd6df;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    color: #333;
  }
  .dataset-tabs button.active {
    background: steelblue;
    border-color: steelblue;
    color: white;
    font-weight: 600;
  }

  .seed-row {
    display: grid;
    grid-template-columns: 90px 1fr 60px;
    gap: 12px;
    align-items: center;
  }
  .seed-value {
    font-family: "Roboto Mono", monospace;
    font-weight: 600;
    color: #1a4b7a;
    text-align: right;
  }

  .loading {
    padding: 12px;
    font-size: 13px;
    color: #666;
  }
</style>

<div class="stability">
  {#if !allData}
    <div class="loading">Loading stability comparison…</div>
  {/if}

  <div class="panels">
    <div class="panel">
      <div class="panel-title">UMAP</div>
      <canvas bind:this={umapCanvas} width="512" height="512" />
    </div>
    <div class="panel">
      <div class="panel-title">t-SNE</div>
      <canvas bind:this={tsneCanvas} width="512" height="512" />
    </div>
    <div class="panel">
      <div class="panel-title">PaCMAP</div>
      <canvas bind:this={pacmapCanvas} width="512" height="512" />
    </div>
  </div>

  <div class="controls">
    <div class="row">
      <div class="row-label">Dataset</div>
      <div class="dataset-tabs">
        {#each DATASETS as ds, i}
          <button
            class:active={i === selectedDatasetIndex}
            on:click={() => (selectedDatasetIndex = i)}
          >
            {ds.label}
          </button>
        {/each}
      </div>
    </div>

    <div class="row">
      <div class="row-label">Random seed</div>
      <div class="seed-row" style="flex:1;">
        <span>&larr; drag &rarr;</span>
        <Slider
          min={0}
          max={seeds.length - 1}
          step={1}
          bind:value={seedIndex}
        />
        <span class="seed-value">seed = {seed}</span>
      </div>
    </div>
  </div>
</div>
