<script>
  /* Single-panel MNIST scatter for one PaCMAP extension method.
     Loads extensions_encoded.json and renders the embedding for the given
     `method` prop (e.g. "localmap", "parampacmap"). */

  import { onMount, tick } from "svelte";
  import * as d3 from "d3";
  import { decode, fromString } from "../../../shared/js/parse-binary";
  import { N_BITS_HYPERPARAMETERS } from "../../../shared/js/parameters";

  export let method;

  let allData = null;
  let canvas;
  let missing = false;

  const decodeEntry = enc => {
    const { data, length, nDimensions, ranges } = enc;
    const byteArray = fromString(atob(data));
    const decoded = decode(byteArray, nDimensions * length, N_BITS_HYPERPARAMETERS);
    const [rx, ry] = ranges;
    const scale = 2 ** N_BITS_HYPERPARAMETERS;
    const out = [];
    for (let i = 0; i < decoded.length; i += 2) {
      const x = (decoded[i]     / scale) * (rx.max - rx.min) + rx.min;
      const y = (decoded[i + 1] / scale) * (ry.max - ry.min) + ry.min;
      out.push([x, y]);
    }
    return out;
  };

  const renderPanel = (coords, labels) => {
    if (!canvas || !coords || !coords.length) return;
    const w = canvas.width;
    const h = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = 512 * dpr;
    canvas.height = 512 * dpr;
    canvas.style.width  = "512px";
    canvas.style.height = "512px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, 512, 512);

    const xs = coords.map(c => c[0]);
    const ys = coords.map(c => c[1]);
    const xMin = d3.min(xs), xMax = d3.max(xs);
    const yMin = d3.min(ys), yMax = d3.max(ys);
    const range = Math.max(xMax - xMin, yMax - yMin) || 1;
    const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2;
    const s = (512 * 0.9) / range;

    const palette = d3.schemeCategory10;
    for (let i = 0; i < coords.length; i++) {
      const [x, y] = coords[i];
      const sx = (x - cx) * s + 256;
      const sy = -(y - cy) * s + 256;
      ctx.fillStyle = palette[labels[i] % palette.length];
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.arc(sx, sy, 1.2, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };

  onMount(async () => {
    try {
      const res = await fetch("extensions_encoded.json");
      allData = await res.json();
    } catch (err) {
      console.error("Failed to load extensions_encoded.json", err);
      return;
    }
    const entry = allData.methods && allData.methods[method];
    if (!entry) { missing = true; return; }
    const coords = decodeEntry(entry);
    await tick();
    renderPanel(coords, allData.labels);
  });
</script>

<style>
  .wrap {
    display: flex; flex-direction: column; align-items: center;
    font-family: "Open Sans", sans-serif;
  }
  canvas {
    width: 100%; max-width: 512px; height: auto;
    background: white;
    border: 1px solid #eaeaea;
    border-radius: 4px;
    display: block;
  }
  .missing { padding: 12px; color: #888; font-size: 13px; font-style: italic; }
  .loading { padding: 12px; color: #666; font-size: 13px; }
  .legend { font-size: 11px; color: #666; margin-top: 6px; }
</style>

<div class="wrap">
  {#if missing}
    <div class="missing">
      MNIST embedding for <code>{method}</code> not precomputed. Run <code>python scripts/precompute_extensions_mnist.py</code>.
    </div>
  {:else if !allData}
    <div class="loading">Loading extension embedding…</div>
  {/if}
  <canvas bind:this={canvas} width="512" height="512" />
  <div class="legend">Colored by MNIST digit label (0–9).</div>
</div>
