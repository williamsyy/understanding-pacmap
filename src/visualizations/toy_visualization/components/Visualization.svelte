<script>
  /* Interactive PaCMAP toy visualization.
     Plays back precomputed intermediate-frame animations produced by
     scripts/precompute_toy_animation.py. Three sliders (n_neighbors,
     MN_ratio, FP_ratio) snap to the closest precomputed triplet. */

  import { onDestroy, onMount, tick } from "svelte";
  // Use the option-override list from toy_comparison_visualization so
  // getPoints() generates the SAME number of points per dataset as our
  // Python precompute (which used the same overridden option.start values).
  import demos from "../../toy_comparison_visualization/js/demos";
  import { visualize, getPoints } from "../../../shared/js/visualize";

  import { decode, fromString } from "../../../shared/js/parse-binary";
  import { N_BITS_HYPERPARAMETERS } from "../../../shared/js/parameters";

  import Preview from "./Preview.svelte";
  import Parameter from "./Parameter.svelte";

  const N_NEIGHBORS_OPTIONS = [5, 15, 50];
  const MN_RATIO_OPTIONS = [0.1, 0.5, 1.0];
  const FP_RATIO_OPTIONS = [1.0, 2.0, 4.0];
  const FRAME_INTERVAL_MS = 350;

  let allData = null;
  let selectedDemoIndex = 0;
  let demo = demos[selectedDemoIndex];
  let canvas;

  let nNeighbors = 15;
  let mnRatio = 0.5;
  let fpRatio = 2.0;

  let currentFrames = [];
  let frameIndex = 0;
  let isPlaying = false;
  let playTimer = null;

  const nearest = (options, value) =>
    options.reduce((best, candidate) =>
      Math.abs(candidate - value) < Math.abs(best - value) ? candidate : best
    );

  const decodeFrame = encoded => {
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

  const loadCurrentFrames = () => {
    if (!allData) return;
    const entry = allData[selectedDemoIndex];
    if (!entry) return;
    const n = nearest(N_NEIGHBORS_OPTIONS, nNeighbors);
    const mn = nearest(MN_RATIO_OPTIONS, mnRatio);
    const fp = nearest(FP_RATIO_OPTIONS, fpRatio);
    // Precomputed keys use Python's repr for floats (e.g. "2.0", "0.1"), so
    // format with .toFixed(1) to preserve the trailing zero JavaScript would drop.
    const key = `n=${n},mn=${mn.toFixed(1)},fp=${fp.toFixed(1)}`;
    const rawFrames = entry[key];
    currentFrames = rawFrames ? rawFrames.map(decodeFrame) : [];
  };

  const renderFrame = idx => {
    if (!canvas || !currentFrames.length) return;
    const clamped = Math.max(0, Math.min(idx, currentFrames.length - 1));
    const coords = currentFrames[clamped];
    const demoPoints = getPoints(demo);
    const output = coords.map(([x, y], i) => ({
      coords: [x, y],
      color: demoPoints[i] ? demoPoints[i].color : "#039",
    }));
    visualize(output, canvas, null, null);
  };

  const stopTimer = () => {
    if (playTimer !== null) {
      clearInterval(playTimer);
      playTimer = null;
    }
  };

  const play = () => {
    if (!currentFrames.length) return;
    isPlaying = true;
    if (frameIndex >= currentFrames.length - 1) frameIndex = 0;
    stopTimer();
    playTimer = setInterval(() => {
      if (frameIndex >= currentFrames.length - 1) {
        stopTimer();
        isPlaying = false;
        frameIndex = currentFrames.length - 1;
        renderFrame(frameIndex);
        return;
      }
      frameIndex += 1;
      renderFrame(frameIndex);
    }, FRAME_INTERVAL_MS);
  };

  const pause = () => {
    stopTimer();
    isPlaying = false;
  };

  const restart = () => {
    stopTimer();
    frameIndex = 0;
    renderFrame(0);
    play();
  };

  const playPause = () => {
    if (isPlaying) pause();
    else play();
  };

  const handleSliderChange = () => {
    stopTimer();
    isPlaying = false;
    loadCurrentFrames();
    frameIndex = currentFrames.length ? currentFrames.length - 1 : 0;
    renderFrame(frameIndex);
  };

  const handlePreviewClick = index => async () => {
    stopTimer();
    isPlaying = false;
    selectedDemoIndex = index;
    demo = demos[selectedDemoIndex];
    loadCurrentFrames();
    frameIndex = 0;
    await tick();
    renderFrame(0);
    play();
  };

  onMount(async () => {
    try {
      const res = await fetch("toy_animation_encoded.json");
      allData = await res.json();
    } catch (err) {
      console.error("Failed to load toy_animation_encoded.json", err);
      return;
    }
    demo = demos[selectedDemoIndex];
    loadCurrentFrames();
    frameIndex = currentFrames.length ? currentFrames.length - 1 : 0;
    await tick();
    renderFrame(frameIndex);
  });

  onDestroy(() => {
    stopTimer();
  });
</script>

<style>
  .playground {
    font-family: "Open Sans", sans-serif;
    width: 100%;
    box-sizing: border-box;
  }
  .playground * { box-sizing: border-box; }

  .top-row {
    display: flex;
    flex-direction: row;
    gap: 24px;
    align-items: flex-start;
    margin-bottom: 24px;
  }
  @media (max-width: 800px) {
    .top-row { flex-direction: column; }
  }

  .canvas-wrap {
    flex: 1 1 55%;
    min-width: 0;
  }
  .canvas-wrap canvas {
    width: 100%;
    height: auto;
    display: block;
    background: #fafafa;
    border: 1px solid #eaeaea;
    border-radius: 4px;
  }
  .loading {
    padding: 12px;
    font-size: 13px;
    color: #666;
  }

  .controls-wrap {
    flex: 1 1 45%;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .play-controls {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 12px;
  }
  .play-controls button {
    cursor: pointer;
    outline: none;
    border-radius: 50%;
    background: steelblue;
    color: white;
    width: 44px;
    height: 44px;
    padding: 0;
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .play-controls button:disabled {
    background: lightgray;
    cursor: default;
  }
  .play-controls i {
    font-size: 24px;
    line-height: 1;
  }
  .steps-display {
    font-size: 13px;
    color: #444;
    line-height: 1.3;
  }
  .steps-display .step { font-weight: 600; }

  .section-label {
    font-weight: 700;
    font-size: 14px;
    color: #222;
    margin-bottom: 6px;
    letter-spacing: 0.02em;
  }

  .pacmap-options, .dataset-info {
    padding: 12px 14px;
    background: #f6f7f9;
    border-radius: 6px;
    font-size: 13px;
  }

  .dataset-description {
    margin-bottom: 10px;
    line-height: 1.4;
    color: #333;
  }
  .demo-parameter {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    font-family: "Roboto Mono", monospace;
    font-size: 12px;
  }
  .demo-parameter .name { color: #333; }
  .demo-parameter .value { font-weight: 600; color: #1a4b7a; }

  .picker-row {
    border-top: 1px solid #eaeaea;
    padding-top: 16px;
  }
  .picker-label {
    font-weight: 700;
    font-size: 14px;
    color: #222;
    margin-bottom: 8px;
  }
  .data-menu {
    overflow: hidden;   /* clearfix — Preview items float:left */
  }
</style>

<div class="playground">
  <div class="top-row">
    <div class="canvas-wrap">
      <canvas bind:this={canvas} class="output" width="600" height="600" />
      {#if !allData}
        <div class="loading">Loading precomputed PaCMAP animations…</div>
      {/if}
    </div>

    <div class="controls-wrap">
      <div class="play-controls">
        <button class="play-pause" on:click={playPause} disabled={!currentFrames.length} aria-label="play/pause">
          {#if isPlaying}
            <i class="material-icons">pause</i>
          {:else}
            <i class="material-icons">play_arrow</i>
          {/if}
        </button>
        <button class="restart" on:click={restart} disabled={!currentFrames.length} aria-label="restart">
          <i class="material-icons">refresh</i>
        </button>
        <div class="steps-display">
          Frame
          <span class="step">{currentFrames.length ? frameIndex + 1 : 0} / {currentFrames.length}</span>
        </div>
      </div>

      <div class="pacmap-options">
        <div class="section-label">PaCMAP Parameters</div>
        <Parameter
          options={{ name: 'n_neighbors (NN)', min: 5, max: 50, step: 1 }}
          bind:value={nNeighbors}
          onChange={handleSliderChange} />
        <Parameter
          options={{ name: 'MN_ratio (MN)', min: 0.1, max: 1.0, step: 0.01 }}
          bind:value={mnRatio}
          onChange={handleSliderChange} />
        <Parameter
          options={{ name: 'FP_ratio (FP)', min: 1.0, max: 4.0, step: 0.1 }}
          bind:value={fpRatio}
          onChange={handleSliderChange} />
      </div>

      <div class="dataset-info">
        <div class="section-label">Dataset: {demo.name}</div>
        <div class="dataset-description">{demo.description}</div>
        {#each demo.options as demoOption (demoOption.name)}
          <div class="demo-parameter">
            <span class="name">{demoOption.name}</span>
            <span class="value">{demoOption.start}</span>
          </div>
        {/each}
      </div>
    </div>
  </div>

  <div class="picker-row">
    <div class="picker-label">Choose a dataset</div>
    <div class="data-menu">
      {#each demos as demoItem, i}
        <Preview
          demo={demoItem}
          onClick={handlePreviewClick(i)}
          selected={i === selectedDemoIndex} />
      {/each}
    </div>
  </div>
</div>
