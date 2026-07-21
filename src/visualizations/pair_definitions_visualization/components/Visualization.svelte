<script>
  /* Interactive illustration of the three PaCMAP pair types on a 3D manifold
     (Swiss roll).  A rotating scatter view; click any point to pick the anchor;
     toggle to see which points are its neighbor / mid-near / further partners.
     Pair distances are computed in the ORIGINAL 3D coordinates so "neighbor"
     means neighbor in the manifold, not neighbor on screen. */

  import { onDestroy, onMount, tick } from "svelte";

  const N_POINTS = 300;
  const RNG_SEED = 42;

  const PAIR_TYPES = [
    { key: "neighbor", label: "Neighbor pairs (NN)",  color: "#0a7f3b",
      help: "The k nearest points in the original 3D manifold — attractive pairs that pull together in the embedding." },
    { key: "midnear",  label: "Mid-near pairs (MN)",  color: "#e88b0e",
      help: "For each of 6 random draws we keep the 2nd-closest — a moderate-distance partner (attractive, weaker). This is what preserves global structure." },
    { key: "further",  label: "Further pairs (FP)",   color: "#c02f2f",
      help: "Random distant points — repulsive pairs that push apart." },
  ];

  const N_NEIGHBORS = 6;
  const N_MID_NEAR  = 5;
  const N_FURTHER   = 12;

  let canvas;
  let width = 640;
  let height = 420;

  let points3d = [];        // Array<{x, y, z}> in the raw manifold
  let anchor = 0;
  let activePair = "neighbor";
  let partners = [];

  let rotY = 0.3;
  let rotX = 0.25;
  let autoRotate = true;
  let dragState = null;
  let rafHandle = null;

  const mulberry32 = seed => () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let rng = mulberry32(RNG_SEED);

  const makeSwissRoll = () => {
    // Standard Swiss roll: theta ∈ [1.5π, 4.5π], y ∈ [0, 20], with a little jitter.
    const r = mulberry32(RNG_SEED + 7);
    const pts = [];
    for (let i = 0; i < N_POINTS; i++) {
      const t = 1.5 * Math.PI * (1 + 2 * r());
      const y = 20 * r() - 10;
      const x =  t * Math.cos(t);
      const z =  t * Math.sin(t);
      pts.push({ x, y, z });
    }
    return pts;
  };

  const dist3 = (a, b) => {
    const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return Math.hypot(dx, dy, dz);
  };

  const neighborsOf = i => {
    return points3d
      .map((p, j) => ({ j, d: dist3(points3d[i], p) }))
      .filter(o => o.j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, N_NEIGHBORS)
      .map(o => o.j);
  };

  const midNearOf = i => {
    const out = new Set();
    let attempts = 0;
    while (out.size < N_MID_NEAR && attempts < 500) {
      attempts++;
      const sample = [];
      for (let s = 0; s < 6; s++) {
        let j;
        do { j = Math.floor(rng() * N_POINTS); } while (j === i);
        sample.push({ j, d: dist3(points3d[i], points3d[j]) });
      }
      sample.sort((a, b) => a.d - b.d);
      out.add(sample[1].j);
    }
    return [...out];
  };

  const furtherOf = i => {
    const nb = new Set(neighborsOf(i));
    const out = new Set();
    let attempts = 0;
    while (out.size < N_FURTHER && attempts < 1000) {
      attempts++;
      const j = Math.floor(rng() * N_POINTS);
      if (j === i || nb.has(j)) continue;
      out.add(j);
    }
    return [...out];
  };

  const recomputePartners = () => {
    if (activePair === "neighbor") partners = neighborsOf(anchor);
    else if (activePair === "midnear")  partners = midNearOf(anchor);
    else partners = furtherOf(anchor);
  };

  const project = (p) => {
    // Y-then-X rotation, then a mild perspective divide.
    const sy = Math.sin(rotY), cy = Math.cos(rotY);
    const sx = Math.sin(rotX), cx = Math.cos(rotX);
    const x1 = p.x * cy - p.z * sy;
    const z1 = p.x * sy + p.z * cy;
    const y2 = p.y * cx - z1 * sx;
    const z2 = p.y * sx + z1 * cx;
    const focal = 100;
    const scale = focal / (focal + z2 + 40);
    return { x: x1 * scale, y: y2 * scale, z: z2 };
  };

  const projectAll = () => {
    const cx = width / 2, cy = height / 2;
    // Auto-fit scale
    const raw = points3d.map(project);
    const maxAbs = raw.reduce((m, p) => Math.max(m, Math.abs(p.x), Math.abs(p.y)), 1);
    const s = (Math.min(width, height) * 0.42) / maxAbs;
    return raw.map(p => ({ x: p.x * s + cx, y: -p.y * s + cy, z: p.z }));
  };

  const render = () => {
    if (!canvas || !points3d.length) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = width  * dpr;
    canvas.height = height * dpr;
    canvas.style.width  = width  + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#fbfbfd";
    ctx.fillRect(0, 0, width, height);

    const projected = projectAll();

    // Lines from anchor to partners (drawn before points so they sit under)
    const meta = PAIR_TYPES.find(p => p.key === activePair);
    ctx.strokeStyle = meta.color;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.6;
    const a = projected[anchor];
    for (const j of partners) {
      const b = projected[j];
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Depth-sorted points (paint far-first so nearer points sit on top)
    const indexed = projected.map((p, i) => ({ i, p })).sort((u, v) => v.p.z - u.p.z);
    for (const { i, p } of indexed) {
      const isAnchor = i === anchor;
      const isPartner = partners.includes(i);
      // Color by manifold "unrolled" angle so the roll's spiral is visible.
      const raw = points3d[i];
      const theta = Math.atan2(raw.z, raw.x); // roughly maps to the spiral parameter
      const hue = ((theta / Math.PI) * 180 + 180) % 360;
      ctx.fillStyle = isAnchor
        ? "#111"
        : isPartner
          ? meta.color
          : `hsl(${hue}, 55%, 65%)`;
      const focal = 100;
      const zScale = focal / (focal + p.z + 60);
      const r = (isAnchor ? 6 : isPartner ? 4.5 : 3) * zScale;
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(1.5, r), 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  const step = () => {
    if (autoRotate && !dragState) {
      rotY += 0.005;
    }
    render();
    rafHandle = requestAnimationFrame(step);
  };

  const setActive = key => {
    activePair = key;
    rng = mulberry32(RNG_SEED + (key.charCodeAt(0) << 8) + anchor);
    recomputePartners();
  };

  const resample = () => {
    rng = mulberry32(RNG_SEED + Math.floor(Math.random() * 1e6));
    recomputePartners();
  };

  const findClickedPoint = (x, y) => {
    const projected = projectAll();
    let bestI = -1, bestD = Infinity;
    for (let i = 0; i < projected.length; i++) {
      const d = Math.hypot(projected[i].x - x, projected[i].y - y);
      if (d < bestD) { bestD = d; bestI = i; }
    }
    return bestD < 12 ? bestI : -1;
  };

  const onCanvasMouseDown = e => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    dragState = { x0: x, y0: y, rotX0: rotX, rotY0: rotY, moved: false };
  };
  const onCanvasMouseMove = e => {
    if (!dragState) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const dx = x - dragState.x0;
    const dy = y - dragState.y0;
    if (Math.abs(dx) + Math.abs(dy) > 3) dragState.moved = true;
    rotY = dragState.rotY0 + dx * 0.008;
    rotX = Math.max(-1.2, Math.min(1.2, dragState.rotX0 + dy * 0.008));
  };
  const onCanvasMouseUp = e => {
    if (dragState && !dragState.moved) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const i = findClickedPoint(x, y);
      if (i >= 0) { anchor = i; setActive(activePair); }
    }
    dragState = null;
  };

  onMount(async () => {
    points3d = makeSwissRoll();
    recomputePartners();
    await tick();
    step();
    window.addEventListener("mouseup", onCanvasMouseUp);
    window.addEventListener("mousemove", onCanvasMouseMove);
  });

  onDestroy(() => {
    if (rafHandle) cancelAnimationFrame(rafHandle);
    window.removeEventListener("mouseup", onCanvasMouseUp);
    window.removeEventListener("mousemove", onCanvasMouseMove);
  });
</script>

<style>
  .wrap {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 100%;
    box-sizing: border-box;
    font-family: "Open Sans", sans-serif;
  }
  .controls {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
  }
  .toggles {
    display: flex;
    flex-direction: row;
    gap: 6px;
    flex-wrap: wrap;
  }
  .toggles button, .aux button {
    padding: 6px 12px;
    background: white;
    border: 1px solid #cfd6df;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    color: #333;
  }
  .toggles button.active {
    color: white;
    font-weight: 600;
    border-color: transparent;
  }
  .toggles button.n.active { background: #0a7f3b; }
  .toggles button.m.active { background: #e88b0e; }
  .toggles button.f.active { background: #c02f2f; }
  .aux { display: flex; gap: 6px; }
  .help {
    font-size: 13px;
    color: #444;
    line-height: 1.4;
    background: #f6f7f9;
    padding: 8px 12px;
    border-radius: 4px;
  }
  canvas {
    border: 1px solid #eaeaea;
    background: #fbfbfd;
    border-radius: 4px;
    max-width: 100%;
    height: auto;
    cursor: grab;
  }
  canvas:active { cursor: grabbing; }
</style>

<div class="wrap">
  <div class="controls">
    <div class="toggles">
      {#each PAIR_TYPES as t}
        <button
          class="{t.key[0]}"
          class:active={activePair === t.key}
          on:click={() => setActive(t.key)}
        >{t.label}</button>
      {/each}
    </div>
    <div class="aux">
      <button on:click={() => (autoRotate = !autoRotate)}>
        {autoRotate ? "Pause rotation" : "Auto-rotate"}
      </button>
      <button on:click={resample}>Resample random draws</button>
    </div>
  </div>
  <div class="help">
    {PAIR_TYPES.find(t => t.key === activePair).help} Drag the view to rotate; click any point to move the anchor. Pair distances are measured in the 3D manifold, not on screen.
  </div>
  <canvas bind:this={canvas} on:mousedown={onCanvasMouseDown} />
</div>
