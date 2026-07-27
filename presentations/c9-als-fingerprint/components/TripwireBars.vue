<script setup>
defineProps({
  mode: { type: String, default: 'setup' }
})
</script>

<template>
  <div class="tripwire-bars">
    <!-- ============ MODE: setup (slide 29) ============ -->
    <svg v-if="mode === 'setup'" viewBox="0 0 820 360" xmlns="http://www.w3.org/2000/svg">
      <text x="410" y="34" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="18px" font-weight="700" fill="#f1f5f9">stability check: shrink Sporadic to match C9</text>

      <!-- Sporadic 294: long gray bar -->
      <text x="60" y="120" font-family="Inter, system-ui, sans-serif" font-size="13px"
            font-weight="700" fill="#94a3b8">Sporadic</text>
      <rect x="60" y="132" width="520" height="40" rx="3" fill="#64748b" stroke="#64748b" stroke-width="1.5" />
      <text x="320" y="158" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="20px" font-weight="800" fill="#f1f5f9">294</text>

      <!-- arrow to 25 -->
      <line x1="592" y1="152" x2="648" y2="152" stroke="#94a3b8" stroke-width="2" />
      <polyline points="640,146 650,152 640,158" fill="none" stroke="#94a3b8" stroke-width="2" />

      <!-- shrunk bar -> 25, matching C9 (cyan) -->
      <rect x="660" y="132" width="44" height="40" rx="3" fill="#4ecadf" />
      <text x="682" y="158" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="15px" font-weight="800" fill="#0a0f1a">25</text>
      <text x="724" y="158" font-family="Inter, system-ui, sans-serif" font-size="12px"
            fill="#4ecadf">= C9</text>

      <!-- green target gauge: preregistered selector-overlap requirement -->
      <line x1="60" y1="252" x2="760" y2="252" stroke="#7dd629" stroke-width="2" stroke-dasharray="6 4" />
      <text x="60" y="244" font-family="Inter, system-ui, sans-serif" font-size="13px"
            font-weight="700" fill="#7dd629">require Jaccard &#8805; 0.70</text>
      <!-- a tick mark / gauge node on the target line -->
      <circle cx="760" cy="252" r="5" fill="#7dd629" />
      <text x="752" y="274" text-anchor="end" font-family="'JetBrains Mono', monospace"
            font-size="12px" font-weight="800" fill="#7dd629">0.70</text>

      <text x="410" y="332" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">does this selector return the same pairs after SPOR 294→25?</text>
    </svg>

    <!-- ============ MODE: result (slide 30) ============ -->
    <svg v-else viewBox="0 0 820 360" xmlns="http://www.w3.org/2000/svg">
      <text x="410" y="34" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="20px" font-weight="800" fill="#f1f5f9">the tripwire fired — production halted by design</text>

      <!-- vertical comparison: required line vs achieved amber bar -->
      <!-- scale: 0..1.0 maps x = 120 (0%) -> 720 (100%); bar height region y 110..250 -->
      <!-- baseline (0%) -->
      <line x1="120" y1="260" x2="720" y2="260" stroke="#475569" stroke-width="1.5" />
      <text x="120" y="280" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="10px" fill="#94a3b8">0</text>

      <!-- amber bar reaching 28.5% : 120 + 600*0.285 = 291 -->
      <!-- arrival: bar settles ONCE on mount toward 0.285, falling short of 0.70 (~300ms ease-out) -->
      <g class="settle-bar">
        <rect x="120" y="120" width="171" height="140" rx="3" fill="#f59e0b" />
      </g>
      <text x="205" y="108" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="20px" font-weight="800" fill="#f59e0b" class="settle-number">0.285</text>
      <text x="205" y="245" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" font-weight="600" fill="#0a0f1a">reproduced</text>

      <!-- green required line at 70% : 120 + 600*0.70 = 540 -->
      <line x1="540" y1="92" x2="540" y2="272" stroke="#7dd629" stroke-width="2.5" stroke-dasharray="6 4" />
      <text x="540" y="84" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="16px" font-weight="800" fill="#7dd629">0.70</text>
      <text x="540" y="290" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" font-weight="600" fill="#7dd629">required overlap</text>

      <!-- gap indicator: amber falls short of green (arrives last, after the bar settles short) -->
      <g class="settle-gap">
        <line x1="291" y1="190" x2="538" y2="190" stroke="#64748b" stroke-width="1" stroke-dasharray="3 3" />
        <polyline points="298,184 290,190 298,196" fill="none" stroke="#64748b" stroke-width="1.5" />
        <text x="414" y="182" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
              font-size="11px" fill="#94a3b8">falls short</text>
      </g>

      <text x="410" y="334" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">raw-p selector/WASC pipeline unstable under SPOR 294→25</text>
    </svg>
  </div>
</template>

<style scoped>
.tripwire-bars {
  display: flex;
  justify-content: center;
  width: 100%;
}
.tripwire-bars svg {
  width: 100%;
  height: auto;
  max-width: 820px;
}

/* ---- result/fired mode: the deck's ONE dread-motion ---- */
/* The amber bar settles ONCE on mount toward 0.285, coming up short of the */
/* static 0.70 line. Vertical scale anchored at the baseline (bottom) so the */
/* gut feels it rise and fall short. ease-out only, ~300ms, no loop. */
@keyframes settle-short {
  from { transform: scaleY(0.55); }
  to   { transform: scaleY(1); }
}
.settle-bar {
  transform-box: fill-box;
  transform-origin: bottom;
  animation: settle-short 300ms ease-out both;
}

/* The number arrives just after the bar settles (opacity fade — bulletproof on SVG). */
@keyframes arrive {
  from { opacity: 0; }
  to   { opacity: 1; }
}
.settle-number {
  animation: arrive 280ms ease-out both;
  animation-delay: 300ms;
}
.settle-gap {
  animation: arrive 280ms ease-out both;
  animation-delay: 420ms;
}
</style>
