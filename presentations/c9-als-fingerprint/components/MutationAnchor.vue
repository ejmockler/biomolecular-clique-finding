<script setup>
// A horizontal double-helix-ish strand with rungs; one segment highlighted as the repeat.
const x0 = 40;
const x1 = 780;
const midY = 120;
const amp = 22;
const period = 56;

// Two sine paths in antiphase form the helix backbone.
function helixPath(phase) {
  const pts = [];
  for (let x = x0; x <= x1; x += 4) {
    const y = midY + amp * Math.sin(((x - x0) / period) * Math.PI * 2 + phase);
    pts.push(`${x},${y.toFixed(2)}`);
  }
  return 'M ' + pts.join(' L ');
}

// Rungs connecting the two strands at intervals.
const rungs = [];
for (let x = x0 + 14; x <= x1 - 14; x += 14) {
  const y1 = midY + amp * Math.sin(((x - x0) / period) * Math.PI * 2);
  const y2 = midY + amp * Math.sin(((x - x0) / period) * Math.PI * 2 + Math.PI);
  rungs.push({ x, y1, y2 });
}

// Highlighted repeat-expansion segment.
const segX = 300;
const segW = 280;
</script>

<template>
  <div class="mutation-anchor">
    <svg viewBox="0 0 820 280" width="100%" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="ma-glow" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#cbb994" stop-opacity="0" />
          <stop offset="0.5" stop-color="#cbb994" stop-opacity="0.2" />
          <stop offset="1" stop-color="#cbb994" stop-opacity="0" />
        </linearGradient>
      </defs>

      <!-- highlight band behind the repeat region -->
      <rect :x="segX" :y="midY - amp - 24" :width="segW" :height="amp * 2 + 48" rx="10" fill="url(#ma-glow)" />
      <rect
        :x="segX"
        :y="midY - amp - 24"
        :width="segW"
        :height="amp * 2 + 48"
        rx="10"
        fill="none"
        stroke="#cbb994"
        stroke-width="1.2"
        stroke-dasharray="4 4"
        opacity="0.7"
      />

      <!-- rungs -->
      <g stroke="#475569" stroke-width="1.5">
        <line
          v-for="(r, i) in rungs"
          :key="'r' + i"
          :x1="r.x"
          :y1="r.y1"
          :x2="r.x"
          :y2="r.y2"
          :stroke="r.x >= segX && r.x <= segX + segW ? 'rgba(203,185,148,0.9)' : 'rgba(71,85,105,0.75)'"
        />
      </g>

      <!-- backbones: gray overall; the repeat is marked by the patient-label highlight band -->
      <path :d="helixPath(0)" fill="none" stroke="#64748b" stroke-width="2.5" />
      <path :d="helixPath(Math.PI)" fill="none" stroke="#64748b" stroke-width="2.5" />

      <!-- the repeated motif in mono, under the highlighted segment -->
      <text
        :x="segX + segW / 2"
        :y="midY + amp + 34"
        text-anchor="middle"
        font-family="'JetBrains Mono', monospace"
        font-size="15px"
        font-weight="800"
        fill="#cbb994"
      >GGGGCC · GGGGCC · GGGGCC …</text>

      <!-- bottom caption -->
      <text
        x="410"
        y="248"
        text-anchor="middle"
        font-family="Inter, system-ui, sans-serif"
        font-size="13px"
        fill="#94a3b8"
      >the genetic cause that defines a C9 carrier — not a place on the protein map</text>
    </svg>
  </div>
</template>

<style scoped>
.mutation-anchor {
  display: flex;
  justify-content: center;
  width: 100%;
}
</style>
