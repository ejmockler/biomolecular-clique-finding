<script setup>
// Descriptive database-term row counts at FDR < 0.05, summed across four DBs.
const bars = [
  { label: 'C9 vs Sporadic', value: 284, color: '#4ecadf', cyan: true },
  { label: 'C9 vs Control', value: 260, color: '#4ecadf', cyan: true },
  { label: 'Sporadic vs Control', value: 0, color: '#7dd629', cyan: false },
];

const MAX = 284;

// Plot geometry.
const baseY = 312;          // bar baseline
const maxBarH = 210;        // pixel height of the tallest bar
const barW = 120;
const slots = [180, 410, 640]; // x-centers of the three bars

const drawn = bars.map((b, i) => {
  const h = Math.max((b.value / MAX) * maxBarH, 4);
  return {
    ...b,
    x: slots[i] - barW / 2,
    w: barW,
    h,
    top: baseY - h,
    cx: slots[i],
  };
});
</script>

<template>
  <div class="triangulation-drop">
    <svg viewBox="0 0 820 380" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="td-cyan" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#4ecadf" stop-opacity="0.95" />
          <stop offset="100%" stop-color="#2a7f8c" stop-opacity="0.85" />
        </linearGradient>
      </defs>

      <!-- Title -->
      <text x="410" y="30" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="15px" fill="#94a3b8">
        FDR &lt; 0.05 database-term rows · four databases summed
      </text>

      <!-- Baseline -->
      <line x1="80" :y1="baseY" x2="740" :y2="baseY" stroke="#475569" stroke-width="1.5" />

      <!-- Bars -->
      <g v-for="(b, i) in drawn" :key="i">
        <rect :x="b.x" :y="b.top" :width="b.w" :height="b.h" rx="4"
              :fill="b.cyan ? 'url(#td-cyan)' : b.color"
              :stroke="b.cyan ? 'none' : b.color" :stroke-width="b.cyan ? 0 : 1.5" />
        <!-- Count on top (mono) -->
        <text :x="b.cx" :y="b.top - 12" text-anchor="middle"
              font-family="'JetBrains Mono', monospace" font-size="30px"
              font-weight="800" :fill="b.cyan ? '#4ecadf' : b.color">{{ b.value }}</text>
        <!-- Bar label below baseline -->
        <text :x="b.cx" :y="baseY + 22" text-anchor="middle"
              font-family="Inter, system-ui, sans-serif" font-size="13px"
              font-weight="600" :fill="b.cyan ? '#94a3b8' : '#94a3b8'">{{ b.label }}</text>
      </g>

      <!-- Same-cohort comparison label under the zero stub. -->
      <text :x="drawn[2].cx" :y="baseY + 40" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="11px"
            font-style="italic" fill="#94a3b8">same-cohort contrast</text>

      <!-- The within-cohort zero is the visual state change. -->
      <g>
        <!-- bracket from top of tall bars down to stub level -->
        <path d="M 575 110 q 18 0 18 18 L 593 250 q 0 18 18 18"
              fill="none" stroke="#f59e0b" stroke-width="1.5" opacity="0.65" />
        <text x="552" y="190" text-anchor="end"
              font-family="Inter, system-ui, sans-serif" font-size="26px"
              font-weight="700" fill="#7dd629">zero rows</text>
      </g>

      <!-- Tertiary caption -->
      <text x="410" y="362" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="12px"
            font-style="italic" fill="#94a3b8">
        descriptive rows, not unique pathways · overlapping biology may repeat across libraries
      </text>
    </svg>
  </div>
</template>

<style scoped>
.triangulation-drop {
  display: flex;
  justify-content: center;
  width: 100%;
}
.triangulation-drop svg {
  width: 100%;
  height: auto;
}
</style>
