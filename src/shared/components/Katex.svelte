<script>
  /* Render a LaTeX string via the globally-loaded KaTeX library. The formula
     is passed as a plain string prop so Svelte's template compiler never
     touches its braces or backslashes — bypasses the mdsvex-vs-Svelte parse
     issue that would otherwise fight $$…$$ and \[…\] delimiters. */
  import { onMount } from "svelte";
  export let formula = "";
  export let display = true;
  let el;

  const render = () => {
    if (!el || typeof window === "undefined" || !window.katex) return;
    try {
      window.katex.render(formula, el, {
        displayMode: display,
        throwOnError: false,
      });
    } catch (err) {
      el.textContent = formula;
      console.warn("KaTeX render failed", err);
    }
  };

  const whenReady = () => {
    if (window.katex) render();
    else setTimeout(whenReady, 50);
  };

  onMount(whenReady);
  $: if (el && formula) whenReady();
</script>

{#if display}
  <div class="katex-block" bind:this={el}></div>
{:else}
  <span class="katex-inline" bind:this={el}></span>
{/if}

<style>
  .katex-block { text-align: center; margin: 18px 0; overflow-x: auto; }
  .katex-inline { display: inline-block; }
</style>
