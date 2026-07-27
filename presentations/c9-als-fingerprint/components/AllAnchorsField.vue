<script setup>
import primaryAnalysis from '../../../data/publication/c9_primary_analysis.json';

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(31171407);
const validGradientCount = primaryAnalysis.feature_accounting.valid_two_shell_gradients
  .toLocaleString('en-US');
const primaryRankingCount = primaryAnalysis.feature_accounting.primary_confirmatory_anchors
  .toLocaleString('en-US');

// Field bounds (the scatter area)
const fx0 = 40, fx1 = 560, fy0 = 70, fy1 = 360;

// Quiet background: many faint flat ticks / dots
const quiet = [];
for (let i = 0; i < 460; i++) {
  const x = fx0 + rng() * (fx1 - fx0);
  const y = fy0 + rng() * (fy1 - fy0);
  // mostly flat (near-horizontal) angle, tiny jitter around 0
  const ang = (rng() - 0.5) * 18; // degrees
  const len = 5 + rng() * 5;
  // bake alpha into rgba (floor 0.18) — defeats the UnoCSS :stroke-opacity crush
  const op = Math.max(0.18, 0.08 + rng() * 0.12);
  quiet.push({ x, y, ang, len, stroke: `rgba(51,65,85,${op.toFixed(3)})` });
}

// Steep cluster: ~30 glowing cyan down-angled ticks that PILE UP in a region
const steep = [];
const cx = 300, cy = 230; // cluster centroid
for (let i = 0; i < 30; i++) {
  // tight-ish gaussian-ish blob via summed uniforms
  const gx = (rng() + rng() + rng()) / 3 - 0.5;
  const gy = (rng() + rng() + rng()) / 3 - 0.5;
  const x = cx + gx * 150;
  const y = cy + gy * 150;
  const ang = 38 + rng() * 22; // steep DOWN angle (positive = down-right in SVG)
  const len = 13 + rng() * 6;
  steep.push({ x, y, ang, len });
}

function seg(p) {
  const rad = (p.ang * Math.PI) / 180;
  const hx = (Math.cos(rad) * p.len) / 2;
  const hy = (Math.sin(rad) * p.len) / 2;
  return { x1: p.x - hx, y1: p.y - hy, x2: p.x + hx, y2: p.y + hy };
}
</script>

<template>
  <div class="all-anchors-field">
    <svg viewBox="0 0 820 372" width="100%" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="aaf-pile" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="rgba(78,202,223,0.15)" />
          <stop offset="100%" stop-color="rgba(78,202,223,0)" />
        </radialGradient>
      </defs>

      <!-- subtle glow under the pile-up -->
      <ellipse cx="300" cy="230" rx="130" ry="120" fill="url(#aaf-pile)" />

      <!-- quiet background slope-glyphs -->
      <g stroke-width="1.2" stroke-linecap="round">
        <line v-for="(p, i) in quiet" :key="'q' + i"
          :x1="seg(p).x1" :y1="seg(p).y1" :x2="seg(p).x2" :y2="seg(p).y2"
          :stroke="p.stroke" />
      </g>

      <!-- the steep, glowing cluster (the only thing that pops) -->
      <g stroke="#4ecadf" stroke-width="2.2" stroke-linecap="round">
        <line v-for="(p, i) in steep" :key="'st' + i"
          :x1="seg(p).x1" :y1="seg(p).y1" :x2="seg(p).x2" :y2="seg(p).y2" />
      </g>

      <!-- pointer note to the cluster (end-anchored, kept clear of the right panel) -->
      <line x1="392" y1="176" x2="362" y2="206"
        stroke="#4ecadf" stroke-width="1" stroke-opacity="0.6" stroke-dasharray="3 3" />
      <text x="556" y="170" text-anchor="end"
        font-family="Inter, system-ui, sans-serif" font-size="14px" fill="#4ecadf"
        font-weight="600">the steep ones pile up</text>

      <!-- ===== Right caption block ===== -->
      <text x="600" y="160" font-family="'JetBrains Mono', monospace"
        font-size="52px" font-weight="800" fill="#f1f5f9">{{ validGradientCount }}</text>
      <text x="602" y="190" font-family="Inter, system-ui, sans-serif"
        font-size="14px" fill="#94a3b8">valid two-shell gradients</text>

      <text x="602" y="220" font-family="'JetBrains Mono', monospace"
        font-size="21px" font-weight="800" fill="#4ecadf">{{ primaryRankingCount }}</text>
      <text x="674" y="220" font-family="Inter, system-ui, sans-serif"
        font-size="12px" fill="#94a3b8">enter the robust ranking</text>

      <text x="602" y="250" font-family="Inter, system-ui, sans-serif"
        font-size="12px" fill="#94a3b8">most are flat &#8212; quiet neighborhoods</text>

      <!-- mini key, inline (not a legend, describes the two glyph kinds in place) -->
      <line x1="602" y1="280" x2="618" y2="284" stroke="#64748b" stroke-width="1.6" stroke-opacity="0.9" />
      <text x="628" y="288" font-family="Inter, system-ui, sans-serif"
        font-size="12px" fill="#94a3b8">flat tick = quiet</text>
      <line x1="602" y1="306" x2="615" y2="316" stroke="#4ecadf" stroke-width="2.2" />
      <text x="628" y="315" font-family="Inter, system-ui, sans-serif"
        font-size="12px" fill="#4ecadf">steep tick = concentrated</text>
    </svg>
  </div>
</template>

<style scoped>
.all-anchors-field {
  display: flex;
  justify-content: center;
  width: 100%;
}
.all-anchors-field svg {
  width: 100%;
  height: auto;
  max-width: 840px;
}
</style>
