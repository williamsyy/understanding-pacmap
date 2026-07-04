<script>
  /* Adapted from the UMAP article's Umap2d component. Now drives PaCMAP
     projections keyed by three axes: n_neighbors × MN_ratio × FP_ratio. */

  import { afterUpdate, onMount } from "svelte";
  import Slider from "../../../shared/components/Slider.svelte";
  import Projection2d from "./Projection2d.svelte";

  let sliderWidth = 230;
  function matchResponsive(x) {
    if (x.matches) {
      sliderWidth = 100;
    } else {
      sliderWidth = 230;
    }
  }

  const mediaQuery = window.matchMedia("(max-width: 800px)");
  matchResponsive(mediaQuery);
  mediaQuery.addListener(matchResponsive);

  const neighbors = [5, 15, 50, 200];
  const mnRatios = ["0.25", "0.5", "1.0"];
  const fpRatios = ["1.0", "2.0", "4.0"];

  let canvas;
  let nNeighborsIndex = 1;   // 15
  let mnRatioIndex = 1;      // 0.5
  let fpRatioIndex = 1;      // 2.0
  $: nNeighbors = neighbors[nNeighborsIndex];
  $: mnRatio = mnRatios[mnRatioIndex];
  $: fpRatio = fpRatios[fpRatioIndex];

  export let colorIndices;
  export let projections;
  export let title = "";
  export let times = null;
  export let hoveredPointIndex = -1;

  $: key = `n=${nNeighbors},mn=${mnRatio},fp=${fpRatio}`;
  $: projection = projections ? projections[key] : null;
</script>

<style>
  .container {
    width: 50%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-sizing: border-box;
  }

  @media only screen and (max-width: 800px) {
    .container {
      width: 90%;
      margin-bottom: 0;
    }
  }

  .controls {
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    margin-top: 6px;
  }

  .title {
    text-align: center;
    width: 100%;
    font-weight: bold;
    margin-bottom: 20px;
  }

  label {
    margin-right: 10px;
    width: 150px;
  }

  .label-text {
    font-weight: 600;
  }
</style>

<div class="container">
  {#if title}
    <div class="title">{title}</div>
  {/if}
  <Projection2d on:hover {projection} {colorIndices} {hoveredPointIndex} />
  <div class="controls">
    <label class="label">
      <span class="label-text">n_neighbors:</span>
      {nNeighbors}
    </label>
    <div style="width: {sliderWidth}px">
      <Slider
        min={0}
        max={neighbors.length - 1}
        step={1}
        bind:value={nNeighborsIndex} />
    </div>
  </div>
  <div class="controls">
    <label class="label">
      <span class="label-text">MN_ratio:</span>
      {mnRatio}
    </label>
    <div style="width: {sliderWidth}px">
      <Slider min={0} max={mnRatios.length - 1} step={1} bind:value={mnRatioIndex} />
    </div>
  </div>
  <div class="controls">
    <label class="label">
      <span class="label-text">FP_ratio:</span>
      {fpRatio}
    </label>
    <div style="width: {sliderWidth}px">
      <Slider min={0} max={fpRatios.length - 1} step={1} bind:value={fpRatioIndex} />
    </div>
  </div>
  {#if times}
    <div class="controls">
      <label class="label">
        <span class="label-text">time:</span>
        {times[nNeighborsIndex].t}
      </label>
      <div style="width: {sliderWidth}px" />
    </div>
  {/if}
</div>
