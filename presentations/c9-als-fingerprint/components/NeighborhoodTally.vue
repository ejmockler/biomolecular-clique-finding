<script setup>
// Each bar: summed slope strength per job. Three pop, rest are quiet.
const baseY = 290;   // bar baseline
const maxH = 210;    // max bar height
const bars = [
  { label: 'splicing',   v: 1.00, hot: true },
  { label: 'chromatin',  v: 0.86, hot: true },
  { label: 'transport',  v: 0.71, hot: true },
  { label: 'metabolism', v: 0.20, hot: false },
  { label: 'signaling',  v: 0.16, hot: false },
  { label: 'adhesion',   v: 0.12, hot: false },
  { label: 'immune',     v: 0.14, hot: false },
  { label: 'other',      v: 0.09, hot: false },
];

const x0 = 90;          // first bar left
const slot = 86;        // spacing between bar centers
const barW = 48;

function bx(i) { return x0 + i * slot; }
function bh(v) { return v * maxH; }
</script>

<template>
  <div class="neighborhood-tally">
    <svg viewBox="0 0 820 360" width="100%" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="nt-hot" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#4ecadf" />
          <stop offset="100%" stop-color="#2a7f8c" />
        </linearGradient>
      </defs>

      <!-- baseline -->
      <line :x1="x0 - 30" :y1="baseY" x2="790" :y2="baseY"
        stroke="#334155" stroke-width="1.5" />

      <!-- y-axis caption -->
      <text x="48" y="75" font-family="Inter, system-ui, sans-serif"
        font-size="13px" fill="#94a3b8"
        transform="rotate(-90 48 170)" text-anchor="middle">illustrative enrichment</text>

      <!-- bars -->
      <g v-for="(b, i) in bars" :key="b.label">
        <rect :x="bx(i)" :y="baseY - bh(b.v)" :width="barW" :height="bh(b.v)"
          :fill="b.hot ? 'url(#nt-hot)' : '#64748b'" rx="3" />
        <!-- value on hot bars only (the ones that matter) -->
        <text v-if="b.hot" :x="bx(i) + barW / 2" :y="baseY - bh(b.v) - 9"
          text-anchor="middle" font-family="'JetBrains Mono', monospace"
          font-size="13px" font-weight="800" fill="#4ecadf">{{ b.v.toFixed(2) }}</text>
        <!-- job label -->
        <text :x="bx(i) + barW / 2" :y="baseY + 20" text-anchor="middle"
          font-family="Inter, system-ui, sans-serif" font-size="12px"
          :fill="b.hot ? '#f1f5f9' : '#94a3b8'"
          :font-weight="b.hot ? '600' : '400'">{{ b.label }}</text>
      </g>

      <!-- conclusion note above the hot trio -->
      <text :x="bx(1) + barW / 2" y="42" text-anchor="middle"
        font-family="Inter, system-ui, sans-serif" font-size="14px"
        font-weight="600" fill="#4ecadf">the fixed panel spans three themes</text>
      <path :d="`M ${bx(0) - 4} 52 Q ${bx(1) + barW / 2} 34, ${bx(2) + barW + 4} 52`"
        fill="none" stroke="#4ecadf" stroke-width="1.2" stroke-opacity="0.4" />

      <!-- corner provenance label -->
      <text x="790" y="345" text-anchor="end"
        font-family="'JetBrains Mono', monospace" font-size="11px" fill="#94a3b8">
        preranked enrichment &#183; 4 libraries</text>
    </svg>
  </div>
</template>

<style scoped>
.neighborhood-tally {
  display: flex;
  justify-content: center;
  width: 100%;
}
</style>
