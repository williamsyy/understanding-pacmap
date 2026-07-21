<script>
  /* Top-of-article 2x2 mammoth comparison:
       [ Original 3D ] [ t-SNE ]
       [ UMAP        ] [ PaCMAP ]
     Each panel keeps its own hyperparameter slider(s). */

  import { onMount } from "svelte";
  import { loadData, loadTSNE, loadUmapReal } from "../js/load-data";
  import { times } from "../js/times";
  import Projection3d from "./Projection3d.svelte";
  import Tsne2d from "./Tsne2d.svelte";
  import Umap2d from "./Umap2d.svelte";
  import UmapReal2d from "./UmapReal2d.svelte";

  let isLoaded = false;
  let colorIndices;
  let pacmapProjections;
  let tsneProjections;
  let umapProjections;
  let mammoth3d;
  let hoveredPointIndex = -1;

  onMount(async () => {
    const data = await loadData();
    tsneProjections = await loadTSNE();
    const umap = await loadUmapReal();
    umapProjections = umap.projections;
    colorIndices = data.colorIndices;
    pacmapProjections = data.projections;
    mammoth3d = data.mammoth3d;
    isLoaded = true;
  });
</script>

<style>
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px 32px;
    width: 100%;
  }
  @media only screen and (max-width: 800px) {
    .grid { grid-template-columns: 1fr; }
  }
  .cell {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    width: 100%;
  }
  .cell :global(.container) {
    width: 100% !important;
  }
  .loading { padding: 12px; color: #666; font-size: 13px; }
</style>

{#if isLoaded}
  <div class="grid">
    <div class="cell">
      <Projection3d
        {colorIndices}
        {mammoth3d}
        on:hover={e => (hoveredPointIndex = e.detail)}
        {hoveredPointIndex}
        title={'Original 3D data'} />
    </div>
    <div class="cell">
      <Tsne2d
        {colorIndices}
        on:hover={e => (hoveredPointIndex = e.detail)}
        {hoveredPointIndex}
        projections={tsneProjections}
        title={'t-SNE'}
        times={times.tsne} />
    </div>
    <div class="cell">
      <UmapReal2d
        {colorIndices}
        projections={umapProjections}
        on:hover={e => (hoveredPointIndex = e.detail)}
        {hoveredPointIndex}
        title={'UMAP'} />
    </div>
    <div class="cell">
      <Umap2d
        {colorIndices}
        projections={pacmapProjections}
        on:hover={e => (hoveredPointIndex = e.detail)}
        {hoveredPointIndex}
        title={'PaCMAP'} />
    </div>
  </div>
{:else}
  <div class="loading">Loading mammoth projections…</div>
{/if}
