<script>
  /* Real UMAP mammoth panel (loads the PAIR original UMAP projections
     encoded as public/mammoth_umap_encoded.json). Two sliders — the
     standard UMAP n_neighbors × min_dist grid. */

  import Slider from "../../../shared/components/Slider.svelte";
  import Projection2d from "./Projection2d.svelte";

  let sliderWidth = 230;
  const mediaQuery = window.matchMedia("(max-width: 800px)");
  const matchResponsive = x => { sliderWidth = x.matches ? 100 : 230; };
  matchResponsive(mediaQuery);
  mediaQuery.addListener(matchResponsive);

  const neighbors = [3, 5, 10, 15, 20, 50, 100, 200];
  const dists = ["0.0", "0.1", "0.25", "0.5", "0.8", "0.99"];

  let nNeighborsIndex = 3; // 15
  let distIndex = 2;       // 0.25
  $: nNeighbors = neighbors[nNeighborsIndex];
  $: dist = dists[distIndex];

  export let colorIndices;
  export let projections;
  export let title = "";
  export let hoveredPointIndex = -1;

  $: key = `n=${nNeighbors},d=${dist}`;
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
    .container { width: 90%; margin-bottom: 0; }
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
  label { margin-right: 10px; width: 150px; }
  .label-text { font-weight: 600; }
</style>

<div class="container">
  {#if title}
    <div class="title">{title}</div>
  {/if}
  <Projection2d on:hover {projection} {colorIndices} {hoveredPointIndex} />
  <div class="controls">
    <label class="label"><span class="label-text">n_neighbors:</span> {nNeighbors}</label>
    <div style="width: {sliderWidth}px">
      <Slider min={0} max={neighbors.length - 1} step={1} bind:value={nNeighborsIndex} />
    </div>
  </div>
  <div class="controls">
    <label class="label"><span class="label-text">min_dist:</span> {dist}</label>
    <div style="width: {sliderWidth}px">
      <Slider min={0} max={dists.length - 1} step={1} bind:value={distIndex} />
    </div>
  </div>
</div>
