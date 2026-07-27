<script setup>
// Seeded RNG so cluster node scatter and backdrop are stable across reloads.
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(0x5c1f7);

// Three clusters: center + node count + label (placed below each cluster).
const clusters = [
  { cx: 175, cy: 200, n: 6, label: 'mRNA Splicing', labelY: 305 },
  { cx: 410, cy: 175, n: 7, label: 'Chromatin', labelY: 280 },
  { cx: 645, cy: 205, n: 6, label: 'nucleocytoplasmic Transport', labelY: 310 },
];

// Build tight cyan node clusters with seeded radial scatter.
const nodeSets = clusters.map((c) => {
  const nodes = [];
  for (let i = 0; i < c.n; i++) {
    const ang = rand() * Math.PI * 2;
    const r = 12 + rand() * 30;
    nodes.push({
      x: c.cx + Math.cos(ang) * r,
      y: c.cy + Math.sin(ang) * r * 0.85,
      rad: 5 + rand() * 2.5,
    });
  }
  // Intra-cluster edges between nearest few for a "neighborhood" feel.
  const edges = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      const d = Math.hypot(dx, dy);
      if (d < 34) edges.push({ x1: nodes[i].x, y1: nodes[i].y, x2: nodes[j].x, y2: nodes[j].y });
    }
  }
  return { nodes, edges, ...c };
});

// Faint purple backdrop map: scattered nodes + a few connecting edges.
const bgRand = mulberry32(0x9a3d1);
const bgNodes = [];
for (let i = 0; i < 26; i++) {
  bgNodes.push({
    x: 40 + bgRand() * 740,
    y: 55 + bgRand() * 300,
    rad: 2.5 + bgRand() * 2,
  });
}
const bgEdges = [];
for (let i = 0; i < bgNodes.length; i++) {
  for (let j = i + 1; j < bgNodes.length; j++) {
    const d = Math.hypot(bgNodes[i].x - bgNodes[j].x, bgNodes[i].y - bgNodes[j].y);
    if (d < 130 && bgRand() < 0.22) {
      bgEdges.push({ x1: bgNodes[i].x, y1: bgNodes[i].y, x2: bgNodes[j].x, y2: bgNodes[j].y });
    }
  }
}
</script>

<template>
  <div class="three-cluster-signature">
    <svg viewBox="0 0 820 400" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="tcs-halo" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#4ecadf" stop-opacity="0.32" />
          <stop offset="55%" stop-color="#4ecadf" stop-opacity="0.12" />
          <stop offset="100%" stop-color="#4ecadf" stop-opacity="0" />
        </radialGradient>
      </defs>

      <!-- Faint purple INDRA map backdrop -->
      <g opacity="0.5">
        <line v-for="(e, i) in bgEdges" :key="'be' + i"
              :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
              stroke="#6d4a8a" stroke-width="1" opacity="0.45" />
        <circle v-for="(n, i) in bgNodes" :key="'bn' + i"
                :cx="n.x" :cy="n.y" :r="n.rad"
                fill="#6d4a8a" opacity="0.5" />
      </g>

      <!-- Clusters settle onto the purple map: halo + edges + nodes + label
           arrive as one coherent mark, staggered ~100ms apart (echoes s5 layering). -->
      <g v-for="(c, ci) in nodeSets" :key="'c' + ci"
         class="tcs-cluster" :style="{ animationDelay: ci * 100 + 'ms' }">
        <!-- Cluster glow halo (behind nodes) -->
        <circle :cx="c.cx" :cy="c.cy" r="70" fill="url(#tcs-halo)" />
        <line v-for="(e, ei) in c.edges" :key="'e' + ci + '-' + ei"
              :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
              stroke="rgba(78,202,223,0.6)" stroke-width="1.2" />
        <circle v-for="(n, ni) in c.nodes" :key="'n' + ci + '-' + ni"
                :cx="n.x" :cy="n.y" :r="n.rad"
                fill="#4ecadf" stroke="#0a0f1a" stroke-width="1" />
        <!-- Cluster label beside / below cluster -->
        <text :x="c.cx" :y="c.labelY" text-anchor="middle"
              font-family="Inter, system-ui, sans-serif" font-size="15px"
              font-weight="600" fill="#4ecadf">{{ c.label }}</text>
      </g>

      <!-- Provenance note -->
      <text x="410" y="378" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="12px" fill="#94a3b8">
        discovery-derived themes &#183; fixed same-cohort panel
      </text>
    </svg>
  </div>
</template>

<style scoped>
.three-cluster-signature {
  display: flex;
  justify-content: center;
  width: 100%;
}
.three-cluster-signature svg {
  width: 100%;
  max-width: 820px;
  height: auto;
}

/* Arrival: each cluster settles onto the purple map once on mount —
   opacity fade + tiny upward settle. ease-out, ~300ms, runs ONCE (both). */
@keyframes tcs-settle {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
.tcs-cluster {
  transform-box: fill-box;
  transform-origin: center;
  animation: tcs-settle 300ms ease-out both;
}
@media (prefers-reduced-motion: reduce) {
  .tcs-cluster {
    animation: none;
  }
}
</style>
