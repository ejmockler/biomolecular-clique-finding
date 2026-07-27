<script setup>
// Two protein chips, each a vertical stack of 6 volume segments filled from the bottom.
const SEGMENTS = 6;

// Segment geometry (shared).
const segH = 22;
const segGap = 6;
const segW = 64;
const stackH = SEGMENTS * segH + (SEGMENTS - 1) * segGap;

// Baseline (bottom of the lowest segment).
const baseY = 250;

function buildChip(cx, filled) {
  const segs = [];
  for (let s = 0; s < SEGMENTS; s++) {
    // s = 0 is bottom segment.
    const y = baseY - segH - s * (segH + segGap);
    const isFilled = s < filled;
    segs.push({
      x: cx - segW / 2,
      y,
      w: segW,
      h: segH,
      filled: isFilled,
    });
  }
  return segs;
}

const leftCx = 285;
const rightCx = 535;

const quietSegs = buildChip(leftCx, 2);   // barely moved
const loudSegs = buildChip(rightCx, 6);   // moved a lot
</script>

<template>
  <div class="volume-knob">
    <svg viewBox="0 0 820 320" xmlns="http://www.w3.org/2000/svg">
      <!-- LEFT chip: quiet -->
      <g>
        <rect v-for="(seg, i) in quietSegs" :key="'q' + i"
              :x="seg.x" :y="seg.y" :width="seg.w" :height="seg.h" rx="3"
              :fill="seg.filled ? '#4ecadf' : '#1a2540'"
              :stroke="seg.filled ? 'none' : '#64748b'"
              :stroke-width="seg.filled ? 0 : 1" />
        <!-- chip name -->
        <text :x="leftCx" :y="baseY + 28" text-anchor="middle"
              font-family="Inter, system-ui, sans-serif" font-size="14px"
              font-weight="600" fill="#94a3b8">barely moved</text>
        <!-- tiny |t| tag near the bars -->
        <text :x="leftCx + segW / 2 + 12" :y="baseY - stackH + 12" text-anchor="start"
              font-family="'JetBrains Mono', monospace" font-size="11px"
              fill="#94a3b8">|t|</text>
      </g>

      <!-- RIGHT chip: loud -->
      <g>
        <rect v-for="(seg, i) in loudSegs" :key="'l' + i"
              :x="seg.x" :y="seg.y" :width="seg.w" :height="seg.h" rx="3"
              :fill="seg.filled ? '#4ecadf' : '#1a2540'"
              :stroke="seg.filled ? 'none' : '#64748b'"
              :stroke-width="seg.filled ? 0 : 1" />
        <!-- chip name -->
        <text :x="rightCx" :y="baseY + 28" text-anchor="middle"
              font-family="Inter, system-ui, sans-serif" font-size="14px"
              font-weight="600" fill="#94a3b8">moved a lot</text>
        <!-- tiny |t| tag near the bars -->
        <text :x="rightCx + segW / 2 + 12" :y="baseY - stackH + 12" text-anchor="start"
              font-family="'JetBrains Mono', monospace" font-size="11px"
              fill="#94a3b8">|t|</text>
      </g>

      <!-- Caption beneath -->
      <text x="410" y="308" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="13px" fill="#94a3b8">
        loudness = size of the change, turned down when we're unsure it's real
      </text>
    </svg>
  </div>
</template>

<style scoped>
.volume-knob {
  display: flex;
  justify-content: center;
  width: 100%;
}
.volume-knob svg {
  width: 100%;
  height: auto;
}
</style>
