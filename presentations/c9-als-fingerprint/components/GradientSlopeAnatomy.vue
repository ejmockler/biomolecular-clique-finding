<script setup>
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Plot geometry — slope figure is the sole strong center, breathing across the full window
const px0 = 130;   // x at hop 0 edge (y-axis)
const px1 = 360;   // x at hop 1
const px2 = 620;   // x at hop 2
const pxEnd = 700;
const pyBase = 330; // y axis baseline (loudness = 0)
const pyTop = 80;   // y at loudness ~2.0

// loudness → y. domain 0..2.1 maps pyBase..pyTop
function ly(v) {
  const max = 2.1;
  return pyBase - (v / max) * (pyBase - pyTop);
}

// Two real data points
const ptHop1 = { x: px1, y: ly(1.8) };
const ptHop2 = { x: px2, y: ly(1.05) };

// Faint scatter behind (seeded)
const rng = mulberry32(20250516);
const scatter = [];
for (let i = 0; i < 16; i++) {
  const hop = rng() < 0.5 ? 1 : 2;
  const baseX = hop === 1 ? px1 : px2;
  const baseV = hop === 1 ? 1.8 : 1.05;
  const jx = (rng() - 0.5) * 90;
  const jv = (rng() - 0.5) * 0.7;
  scatter.push({ x: baseX + jx, y: ly(Math.max(0.1, baseV + jv)) });
}
// shuffle so order is not sorted
for (let i = scatter.length - 1; i > 0; i--) {
  const j = Math.floor(rng() * (i + 1));
  [scatter[i], scatter[j]] = [scatter[j], scatter[i]];
}

// Fitted line through the two points, extended
function lineY(x) {
  const m = (ptHop2.y - ptHop1.y) / (ptHop2.x - ptHop1.x);
  return ptHop1.y + m * (x - ptHop1.x);
}
const fitX1 = px0 + 30;
const fitX2 = pxEnd;
const fitY1 = lineY(fitX1);
const fitY2 = lineY(fitX2);
</script>

<template>
  <div class="gradient-slope-anatomy">
    <svg viewBox="60 40 720 360" width="100%" xmlns="http://www.w3.org/2000/svg">
      <!-- ===== x/y scatter plot — sole strong center ===== -->
      <!-- axes -->
      <line :x1="px0" :y1="pyBase" :x2="pxEnd + 10" :y2="pyBase"
        stroke="#475569" stroke-width="1.5" />
      <line :x1="px0" :y1="pyBase" :x2="px0" :y2="pyTop - 10"
        stroke="#475569" stroke-width="1.5" />

      <!-- x ticks at hop 1 and hop 2 -->
      <line :x1="px1" :y1="pyBase" :x2="px1" :y2="pyBase + 6" stroke="#475569" stroke-width="1.5" />
      <line :x1="px2" :y1="pyBase" :x2="px2" :y2="pyBase + 6" stroke="#475569" stroke-width="1.5" />
      <text :x="px1" :y="pyBase + 22" text-anchor="middle"
        font-family="'JetBrains Mono', monospace" font-size="12px" fill="#94a3b8">ring 1</text>
      <text :x="px2" :y="pyBase + 22" text-anchor="middle"
        font-family="'JetBrains Mono', monospace" font-size="12px" fill="#94a3b8">ring 2</text>
      <text :x="(px1 + px2) / 2" :y="pyBase + 42" text-anchor="middle"
        font-family="Inter, system-ui, sans-serif" font-size="13px" fill="#94a3b8">undirected regulatory distance</text>

      <!-- y axis label -->
      <text :x="px0 - 14" :y="(pyTop + pyBase) / 2" text-anchor="middle"
        font-family="Inter, system-ui, sans-serif" font-size="13px" fill="#94a3b8"
        :transform="`rotate(-90 ${px0 - 14} ${(pyTop + pyBase) / 2})`">loudness (mean |t|)</text>

      <!-- faint scatter behind -->
      <g>
        <circle v-for="(s, i) in scatter" :key="'s' + i" :cx="s.x" :cy="s.y" r="3.5"
          fill="rgba(78,202,223,0.62)" stroke="#4ecadf" stroke-opacity="0.6" stroke-width="1" />
      </g>

      <!-- fitted line sloping DOWN left→right -->
      <line :x1="fitX1" :y1="fitY1" :x2="fitX2" :y2="fitY2"
        stroke="#4ecadf" stroke-width="2.5" stroke-linecap="round" />

      <!-- the two real points (pop) -->
      <circle :cx="ptHop1.x" :cy="ptHop1.y" r="7" fill="#4ecadf" />
      <circle :cx="ptHop2.x" :cy="ptHop2.y" r="7" fill="#4ecadf" />
      <text :x="ptHop1.x + 12" :y="ptHop1.y - 6"
        font-family="'JetBrains Mono', monospace" font-size="12px" fill="#f1f5f9">1.8</text>
      <text :x="ptHop2.x + 12" :y="ptHop2.y - 6"
        font-family="'JetBrains Mono', monospace" font-size="12px" fill="#f1f5f9">1.05</text>

      <!-- conclusion -->
      <text :x="px0 + 6" :y="pyTop - 16"
        font-family="Inter, system-ui, sans-serif" font-size="14px" fill="#4ecadf"
        font-weight="600">direct partners higher than two-hop partners</text>
    </svg>
  </div>
</template>

<style scoped>
.gradient-slope-anatomy {
  display: flex;
  justify-content: center;
  width: 100%;
}
</style>
