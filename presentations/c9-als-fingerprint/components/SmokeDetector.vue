<script setup>
// Seeded RNG for stable shuffle of which ticks are red.
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TOTAL = 200;
const REJECTED = 113;

// Build an array of 200, mark the observed self-contained rejections, then shuffle.
const ticks = Array.from({ length: TOTAL }, (_, i) => i < REJECTED);
const rand = mulberry32(20250609);
for (let i = ticks.length - 1; i > 0; i--) {
  const j = Math.floor(rand() * (i + 1));
  const tmp = ticks[i]; ticks[i] = ticks[j]; ticks[j] = tmp;
}

// Tick strip geometry. Gapless barcode: 1.65px ticks washed out under
// anti-aliasing, so tile the cells full-width at full opacity instead.
const stripX = 110;
const stripW = 600;
const tickY = 264;
const tickH = 38;
const gap = stripW / TOTAL;
const tickW = gap; // gapless — adjacent same-color cells merge into legible bands

const tickData = ticks.map((fired, i) => ({
  x: stripX + i * gap,
  rejected: fired,
}));
</script>

<template>
  <div class="smoke-detector">
    <svg viewBox="0 0 820 340" xmlns="http://www.w3.org/2000/svg">
      <!-- Hero number — the one number on this slide. Arrives once on mount:
           opacity fade + tiny translateY rise so the gut registers it landing. -->
      <g class="hero-arrive">
        <text x="410" y="118" text-anchor="middle"
              font-family="'JetBrains Mono', monospace" font-size="84px" font-weight="800"
              fill="#f59e0b">56.5%</text>
      </g>

      <!-- Primary explanation -->
      <text x="410" y="158" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="16px" fill="#94a3b8">
        113 of 200 random observed-data gene lists rejected (&#945; = 0.05)
      </text>

      <!-- Italic caption -->
      <text x="410" y="190" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="13px" font-style="italic"
            fill="#94a3b8">
        observed-set rejection fraction · not a null-calibration estimate
      </text>

      <!-- Tick strip: 200 cells, 113 red (fired) / 87 gray (quiet), seeded shuffle -->
      <!-- framed track so the barcode reads as a delineated object on the dark field -->
      <rect :x="stripX - 5" :y="tickY - 5" :width="stripW + 10" :height="tickH + 10"
            rx="5" fill="#121a2d" stroke="#334155" stroke-width="1" />
      <g>
        <rect v-for="(t, i) in tickData" :key="i"
              :x="t.x" :y="tickY" :width="tickW" :height="tickH"
              :fill="t.rejected ? '#f59e0b' : '#64748b'" />
      </g>

      <!-- Strip end labels -->
      <text :x="stripX" :y="tickY - 10"
            font-family="Inter, system-ui, sans-serif" font-size="11px" fill="#f59e0b">
        rejected
      </text>
      <text :x="stripX + stripW" :y="tickY - 10" text-anchor="end"
            font-family="Inter, system-ui, sans-serif" font-size="11px" fill="#94a3b8">
        quiet
      </text>
      <text x="410" :y="tickY + tickH + 22" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="11px" fill="#94a3b8">
        each tick = one random set sampled from the observed proteome
      </text>
    </svg>
  </div>
</template>

<style scoped>
.smoke-detector {
  display: flex;
  justify-content: center;
  width: 100%;
}
.smoke-detector svg {
  width: 100%;
  height: auto;
}

/* RATIONED arrival: the one number (56.5%) settles in once on mount —
   opacity fade + a tiny translateY rise over ~320ms ease-out. Nothing else
   animates. Final frame identical: ends at full opacity, zero translate. */
.hero-arrive {
  transform-box: fill-box;
  animation: hero-arrive 320ms ease-out both;
}
@keyframes hero-arrive {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
