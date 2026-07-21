<script>
  /* Three unaligned PaCMAP embeddings of MNIST — three members of the
     Rashomon set (random-init seeds; not Procrustes-aligned).
     Each member on its own is a valid embedding; laid out side-by-side
     the visual differences are the "Rashomon variability" the paper studies. */

  import { onMount, tick } from "svelte";
  import * as d3 from "d3";
  import { decode, fromString } from "../../../shared/js/parse-binary";
  import { N_BITS_HYPERPARAMETERS } from "../../../shared/js/parameters";

  let allData = null;
  let member0Canvas;
  let member1Canvas;
  let member2Canvas;

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

  const renderPanel = (canvas, coords, labels, size) => {
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

    ctx.globalAlpha = 0.6;
    for (let i = 0; i < coords.length; i++) {
      const [x, y] = coords[i];
      const sx = (x - cx) * s + size / 2;
      const sy = -(y - cy) * s + size / 2;
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
    const set = allData.rashomon;
    if (!set) return;
    await tick();
    const targets = [member0Canvas, member1Canvas, member2Canvas];
    for (let i = 0; i < 3; i++) {
      if (targets[i] && set[i]) {
        renderPanel(targets[i], decodeEntry(set[i]), allData.labels, 320);
      }
    }
  });
</script>

<style>
  .wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    font-family: "Open Sans", sans-serif;
    width: 100%;
    box-sizing: border-box;
  }
  .members {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    width: 100%;
  }
  @media (max-width: 640px) {
    .members { grid-template-columns: 1fr; }
  }
  .panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }
  .panel-title {
    font-size: 12px;
    color: var(--ink-muted, #6b7280);
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .panel canvas {
    width: 100%;
    max-width: 100%;
    aspect-ratio: 1 / 1;
    background: white;
    border: 1px solid var(--figure-border, #eaecef);
    border-radius: 4px;
  }
  .legend {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: center;
    gap: 14px;
    padding: 10px 14px;
    background: var(--bg-soft, #f7f8fa);
    border: 1px solid var(--border-soft, #e5e7eb);
    border-radius: 6px;
  }
  .legend-title {
    font-size: 11px;
    color: var(--ink-muted, #6b7280);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
    align-self: center;
  }
  .legend-row {
    display: flex; align-items: center; gap: 6px;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 12px;
    color: var(--ink, #1f2937);
  }
  .legend-dot {
    width: 10px; height: 10px; border-radius: 50%;
    display: inline-block;
  }
  .loading { padding: 12px; color: #666; font-size: 13px; }
</style>

<div class="wrap">
  {#if !allData}
    <div class="loading">Loading Rashomon members…</div>
  {/if}
  <div class="members">
    <div class="panel">
      <div class="panel-title">Embedding 1</div>
      <canvas bind:this={member0Canvas} width="320" height="320" />
    </div>
    <div class="panel">
      <div class="panel-title">Embedding 2</div>
      <canvas bind:this={member1Canvas} width="320" height="320" />
    </div>
    <div class="panel">
      <div class="panel-title">Embedding 3</div>
      <canvas bind:this={member2Canvas} width="320" height="320" />
    </div>
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
