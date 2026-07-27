<script setup>
// Three cohorts; bar length scales linearly with count so 25 reads as a stub.
const maxCount = 294;
const maxLen = 560;
const barX = 168;
const barH = 34;

const groups = [
  {
    label: 'C9 carriers',
    count: 25,
    color: '#4ecadf',
    y: 50,
    note: 'rare mutation — only 25',
  },
  {
    label: 'Sporadic ALS',
    count: 294,
    color: '#64748b',
    countColor: '#94a3b8',
    y: 120,
    note: null,
  },
  {
    label: 'Healthy control',
    count: 91,
    color: '#7dd629',
    y: 190,
    note: null,
  },
];

function len(c) {
  return (c / maxCount) * maxLen;
}
</script>

<template>
  <div class="cohort-bars">
    <svg viewBox="0 0 820 260" width="100%" preserveAspectRatio="xMidYMid meet">
      <g v-for="(g, i) in groups" :key="i">
        <!-- left label -->
        <text
          :x="barX - 14"
          :y="g.y + barH / 2 + 4"
          text-anchor="end"
          font-family="Inter, system-ui, sans-serif"
          font-size="14px"
          :fill="g.color === '#64748b' ? '#94a3b8' : g.color"
        >{{ g.label }}</text>

        <!-- bar -->
        <rect
          :x="barX"
          :y="g.y"
          :width="len(g.count)"
          :height="barH"
          :fill="g.color"
          :stroke="g.color === '#64748b' ? '#64748b' : 'none'"
          :stroke-width="g.color === '#64748b' ? 1.5 : 0"
          rx="3"
        />

        <!-- mono count at right end -->
        <text
          :x="barX + len(g.count) + 12"
          :y="g.y + barH / 2 + 6"
          font-family="'JetBrains Mono', monospace"
          font-size="22px"
          font-weight="800"
          :fill="g.countColor || g.color"
        >{{ g.count }}</text>

        <!-- note under C9 -->
        <text
          v-if="g.note"
          :x="barX"
          :y="g.y + barH + 18"
          font-family="Inter, system-ui, sans-serif"
          font-size="12px"
          fill="#f59e0b"
        >{{ g.note }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.cohort-bars {
  display: flex;
  justify-content: center;
  width: 100%;
}
</style>
