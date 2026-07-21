<script>
  /* Figure wrapper — centered content column that can bleed slightly wider
     than the prose column on desktop so interactive visualisations breathe. */
</script>

<style>
  figure {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    margin: 2.6em 0 1.4em;
    padding: 18px 18px 6px;
    background: var(--figure-bg, #ffffff);
    border: 1px solid var(--figure-border, #eaecef);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    box-sizing: border-box;
    width: 100%;
  }
  @media (min-width: 900px) {
    figure {
      width: 112%;
      margin-left: -6%;
      padding: 24px 24px 10px;
    }
  }
  .figure-container {
    width: 100%;
    max-width: 1100px;
  }
  caption {
    display: block;
    margin: 18px auto 6px;
    max-width: 780px;
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--ink-soft, #4b5563);
  }
</style>

<figure>
  <div class="figure-container">
    <slot />
  </div>
  <caption>
    <slot name="caption" />
  </caption>
</figure>
