<script setup>
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Tidy grid of dots representing the measured proteome.
const cols = 30;
const rows = 5;
const gridX = 40;
const gridY = 56;
const dx = 14;
const dy = 18;
const rand = mulberry32(3264);

const dots = [];
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    // baked rgba (floor 0.55) for the anonymous mass — avoids the bound :fill-opacity crush
    const o = 0.55 + rand() * 0.3;
    dots.push({
      x: gridX + c * dx,
      y: gridY + r * dy,
      fill: `rgba(78,202,223,${o.toFixed(3)})`,
      rad: 2.6,
    });
  }
}

// A few named exemplars: pick stable grid cells and enlarge/label them.
const named = [
  { idx: 0 * cols + 4, name: 'TARDBP' },
  { idx: 1 * cols + 12, name: 'SOD1' },
  { idx: 2 * cols + 21, name: 'FUS' },
];
const namedSet = new Map(named.map((n) => [n.idx, n.name]));
dots.forEach((d, i) => {
  if (namedSet.has(i)) {
    d.named = true;
    d.label = namedSet.get(i);
    d.rad = 4.2;
    d.fill = '#4ecadf';
  }
});
</script>

<template>
  <div class="protein-readout">
    <svg viewBox="0 0 820 300" width="100%" preserveAspectRatio="xMidYMid meet">
      <!-- dot field -->
      <g>
        <circle
          v-for="(d, i) in dots"
          :key="i"
          :cx="d.x"
          :cy="d.y"
          :r="d.rad"
          :fill="d.fill"
        />
      </g>

      <!-- exemplar labels, sitting next to their dot -->
      <g font-family="'JetBrains Mono', monospace" font-size="10px" fill="#94a3b8">
        <text
          v-for="(d, i) in dots.filter((x) => x.named)"
          :key="'lbl' + i"
          :x="d.x + 7"
          :y="d.y + 3.5"
        >{{ d.label }}</text>
      </g>

      <!-- hero number -->
      <text
        x="40"
        y="225"
        font-family="'JetBrains Mono', monospace"
        font-size="64px"
        font-weight="800"
        fill="#4ecadf"
      >3,264</text>
      <text
        x="40"
        y="258"
        font-family="Inter, system-ui, sans-serif"
        font-size="14px"
        fill="#94a3b8"
      >feature rows · one level per person</text>

      <!-- supporting line connecting field to number, subtle -->
      <text
        x="320"
        y="225"
        font-family="Inter, system-ui, sans-serif"
        font-size="13px"
        fill="#94a3b8"
      >3,263 human UniProt rows + 1 iRT standard</text>
    </svg>
  </div>
</template>

<style scoped>
.protein-readout {
  display: flex;
  justify-content: center;
  width: 100%;
}
.protein-readout svg {
  width: 100%;
  height: auto;
  max-width: 720px;
}
</style>
