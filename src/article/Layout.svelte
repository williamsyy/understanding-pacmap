<script>
  /* Layout for Understanding PaCMAP.
     Article-shell wrapper — provides the centered column and generous margins. */
</script>

<style>
  section {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .container {
    max-width: 780px;
    width: 100%;
    padding: 0 18px;
  }
  @media (min-width: 900px) {
    .container { padding: 0; }
  }
</style>

<section>
  <div class="container">
    <slot />
  </div>
</section>
