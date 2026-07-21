<script>
  /* Side-by-side PaCMAP vs LocalMAP on full MNIST, with a sample of the
     neighbor-pair graph drawn as thin gray edges over the scatter. Legend
     on the right maps digit labels to colours. */

  import { onMount, tick } from "svelte";
  import * as d3 from "d3";
  import { decode, fromString } from "../../../shared/js/parse-binary";
  import { N_BITS_HYPERPARAMETERS } from "../../../shared/js/parameters";

  let allData = null;
  let pacmapCanvas;
  let localmapCanvas;

  const DIGIT_PALETTE = [
    "#8b0033", "#c03038", "#e35d2c", "#f0942a", "#f2c93f",
    "#c9d64a", "#77b350", "#4d9d78", "#3a80b4", "#4b3b8d",
  ];

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

  const renderPanel = (canvas, coords, labels, edgesFlat, size) => {
    if (!canvas || !coords || !coords.length) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width  = size + "px";
    canvas.style.height = size + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, size, size);

    const xs = coords.map(c => c[0]);
    const ys = coords.map(c => c[1]);
    const xMin = d3.min(xs), xMax = d3.max(xs);
    const yMin = d3.min(ys), yMax = d3.max(ys);
    const range = Math.max(xMax - xMin, yMax - yMin) || 1;
    const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2;
    const s = (size * 0.9) / range;
    const project = ([x, y]) => [(x - cx) * s + size / 2, -(y - cy) * s + size / 2];

    // Edges first (behind points)
    if (edgesFlat && edgesFlat.length) {
      ctx.strokeStyle = "rgba(60,60,72,0.14)";
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      for (let k = 0; k < edgesFlat.length; k += 2) {
        const a = coords[edgesFlat[k]];
        const b = coords[edgesFlat[k + 1]];
        if (!a || !b) continue;
        const [ax, ay] = project(a);
        const [bx, by] = project(b);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
    }

    // Depth-independent point pass
    ctx.globalAlpha = 0.6;
    for (let i = 0; i < coords.length; i++) {
      const [sx, sy] = project(coords[i]);
      ctx.fillStyle = DIGIT_PALETTE[labels[i] % 10];
      ctx.beginPath();
      ctx.arc(sx, sy, 1.0, 0, 2 * Math.PI);
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
    const cmp = allData.localmap_comparison;
    if (!cmp) return;
    await tick();
    renderPanel(pacmapCanvas,   decodeEntry(cmp.pacmap),   allData.labels, cmp.pacmap_edges,   440);
    renderPanel(localmapCanvas, decodeEntry(cmp.localmap), allData.labels, cmp.localmap_edges, 440);
  });
</script>

<style>
  .wrap {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    gap: 16px;
    align-items: start;
    font-family: "Open Sans", sans-serif;
    width: 100%;
  }
  @media (max-width: 720px) {
    .wrap { grid-template-columns: 1fr; }
  }
  .panel {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
  }
  .panel-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--ink, #1f2937);
    letter-spacing: 0.02em;
  }
  .panel canvas {
    width: 100%;
    max-width: 440px;
    aspect-ratio: 1 / 1;
    background: white;
    border: 1px solid var(--figure-border, #eaecef);
    border-radius: 4px;
  }
  .legend {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 12px 14px;
    background: var(--bg-soft, #f7f8fa);
    border: 1px solid var(--border-soft, #e5e7eb);
    border-radius: 6px;
    align-self: center;
  }
  .legend-title {
    font-size: 11px;
    color: var(--ink-muted, #6b7280);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 4px;
  }
  .legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 13px;
    color: var(--ink, #1f2937);
  }
  .legend-dot {
    width: 12px; height: 12px; border-radius: 50%;
    display: inline-block;
  }
  .loading { padding: 12px; color: #666; font-size: 13px; grid-column: 1 / -1; }
</style>

<div class="wrap">
  {#if !allData}
    <div class="loading">Loading PaCMAP + LocalMAP comparison…</div>
  {/if}
  <div class="panel">
    <div class="panel-title">PaCMAP</div>
    <canvas bind:this={pacmapCanvas} width="440" height="440" />
  </div>
  <div class="panel">
    <div class="panel-title">LocalMAP</div>
    <canvas bind:this={localmapCanvas} width="440" height="440" />
  </div>
  <div class="legend">
    <div class="legend-title">Digit</div>
    {#each Array(10) as _, i}
      <div class="legend-row">
        <span class="legend-dot" style="background: {DIGIT_PALETTE[i]}"></span>
        {i}
      </div>
    {/each}
  </div>
</div>
