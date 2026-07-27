<script setup>
// Triangle vertices for the three-cohort comparison design.
const V = {
  c9:      { x: 410, y: 88,  color: '#4ecadf', name: 'C9 carriers', count: '25' },
  spor:    { x: 150, y: 322, color: '#64748b', name: 'Sporadic',    count: '294' },
  control: { x: 670, y: 322, color: '#7dd629', name: 'Control',     count: '91' },
};

// Midpoint of an edge for placing the comparison-name tag.
function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}
const mTopLeft = mid(V.c9, V.spor);       // C9 vs Sporadic
const mTopRight = mid(V.c9, V.control);   // C9 vs Control
const mBottom = mid(V.spor, V.control);   // Sporadic vs Control (specificity leg)
</script>

<template>
  <div class="contrast-triangle">
    <svg viewBox="0 0 820 388" xmlns="http://www.w3.org/2000/svg">
      <!-- Title -->
      <text x="410" y="34" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="15px" fill="#94a3b8">
        same pipeline, three comparisons
      </text>

      <!-- Edges (drawn first, behind vertices) -->
      <line :x1="V.c9.x" :y1="V.c9.y" :x2="V.spor.x" :y2="V.spor.y"
            stroke="#475569" stroke-width="2" />
      <line :x1="V.c9.x" :y1="V.c9.y" :x2="V.control.x" :y2="V.control.y"
            stroke="#475569" stroke-width="2" />
      <line :x1="V.spor.x" :y1="V.spor.y" :x2="V.control.x" :y2="V.control.y"
            stroke="#cbd5e1" stroke-width="2" stroke-dasharray="6 5" />

      <!-- Edge tags: comparison names beside each edge -->
      <!-- C9 vs Sporadic (top-left edge) -->
      <text :x="mTopLeft.x - 56" :y="mTopLeft.y - 4"
            font-family="Inter, system-ui, sans-serif" font-size="13px"
            font-weight="600" fill="#94a3b8" transform="rotate(-42 218 205)">
        C9 vs Sporadic
      </text>

      <!-- C9 vs Control (top-right edge) -->
      <text :x="mTopRight.x + 4" :y="mTopRight.y - 4"
            font-family="Inter, system-ui, sans-serif" font-size="13px"
            font-weight="600" fill="#94a3b8" transform="rotate(42 602 205)">
        C9 vs Control
      </text>

      <!-- Sporadic vs Control (bottom edge) — within-cohort specificity leg -->
      <text :x="mBottom.x" :y="mBottom.y + 22" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="13px"
            font-weight="600" fill="#94a3b8">
        Sporadic vs Control
      </text>
      <text :x="mBottom.x" :y="mBottom.y + 40" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="11px"
            font-style="italic" fill="#94a3b8">
        within-cohort specificity leg
      </text>
      <text :x="mBottom.x" :y="mBottom.y + 56" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="11px"
            font-style="italic" fill="#94a3b8">
        quiet supports — but does not prove — a C9-associated reading
      </text>

      <!-- Vertices -->
      <g v-for="(v, k) in V" :key="k">
        <circle :cx="v.x" :cy="v.y" r="13" :fill="v.color" stroke="#0a0f1a" stroke-width="2" />
      </g>

      <!-- C9 vertex label (above) -->
      <text :x="V.c9.x" :y="V.c9.y - 22" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="15px"
            font-weight="600" :fill="V.c9.color">
        C9 carriers
        <tspan font-family="'JetBrains Mono', monospace" font-weight="800"
               fill="#f1f5f9"> ({{ V.c9.count }})</tspan>
      </text>

      <!-- Sporadic vertex label (below-left) -->
      <text :x="V.spor.x" :y="V.spor.y + 36" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="15px"
            font-weight="600" fill="#94a3b8">
        Sporadic
        <tspan font-family="'JetBrains Mono', monospace" font-weight="800"
               fill="#f1f5f9"> ({{ V.spor.count }})</tspan>
      </text>

      <!-- Control vertex label (below-right) -->
      <text :x="V.control.x" :y="V.control.y + 36" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="15px"
            font-weight="600" :fill="V.control.color">
        Control
        <tspan font-family="'JetBrains Mono', monospace" font-weight="800"
               fill="#f1f5f9"> ({{ V.control.count }})</tspan>
      </text>
    </svg>
  </div>
</template>

<style scoped>
.contrast-triangle {
  display: flex;
  justify-content: center;
  width: 100%;
}
.contrast-triangle svg {
  width: 100%;
  height: auto;
  max-width: 1080px;
}
</style>
