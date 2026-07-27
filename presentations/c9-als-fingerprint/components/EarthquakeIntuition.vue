<script setup>
const anchor = { x: 410, y: 176 };
const ring1 = [
  { x: 410, y: 86 },
  { x: 496, y: 148 },
  { x: 465, y: 248 },
  { x: 355, y: 248 },
  { x: 324, y: 148 },
];
const ring2 = [
  { x: 410, y: 24, via: 0 },
  { x: 550, y: 88, via: 1 },
  { x: 555, y: 260, via: 2 },
  { x: 410, y: 328, via: 3 },
  { x: 265, y: 260, via: 3 },
  { x: 270, y: 88, via: 4 },
];
</script>

<template>
  <div class="neighborhood-intuition">
    <svg viewBox="0 0 820 380" xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="An anchor with a ring of direct regulatory partners and a ring of two-hop partners. All connections are undirected for the distance calculation.">
      <circle :cx="anchor.x" :cy="anchor.y" r="92" fill="rgba(78,202,223,0.05)"
              stroke="#4ecadf" stroke-width="1.5" stroke-dasharray="5 5" />
      <circle :cx="anchor.x" :cy="anchor.y" r="154" fill="none"
              stroke="#64748b" stroke-width="1.2" stroke-dasharray="5 6" />

      <line v-for="(node, i) in ring1" :key="`a-${i}`"
            :x1="anchor.x" :y1="anchor.y" :x2="node.x" :y2="node.y"
            stroke="#bf6ff7" stroke-width="1.8" />
      <line v-for="(node, i) in ring2" :key="`b-${i}`"
            :x1="ring1[node.via].x" :y1="ring1[node.via].y" :x2="node.x" :y2="node.y"
            stroke="#64748b" stroke-width="1.3" />

      <circle v-for="(node, i) in ring2" :key="`r2-${i}`"
              :cx="node.x" :cy="node.y" r="8" fill="#182238" stroke="#94a3b8" stroke-width="1.4" />
      <circle v-for="(node, i) in ring1" :key="`r1-${i}`"
              :cx="node.x" :cy="node.y" r="11" fill="#bf6ff7" />
      <circle :cx="anchor.x" :cy="anchor.y" r="20" fill="#4ecadf" />
      <text :x="anchor.x" :y="anchor.y + 4" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="10px" font-weight="800" fill="#0a0f1a">anchor</text>

      <line x1="496" y1="148" x2="650" y2="126" stroke="#4ecadf" stroke-width="1" />
      <text x="660" y="122" font-family="Inter, system-ui, sans-serif" font-size="14px" font-weight="700" fill="#4ecadf">ring 1</text>
      <text x="660" y="142" font-family="Inter, system-ui, sans-serif" font-size="12px" fill="#cbd5e1">direct partners</text>
      <text x="660" y="160" font-family="Inter, system-ui, sans-serif" font-size="11px" fill="#94a3b8">regulators + regulatees</text>

      <line x1="555" y1="260" x2="650" y2="250" stroke="#64748b" stroke-width="1" />
      <text x="660" y="246" font-family="Inter, system-ui, sans-serif" font-size="14px" font-weight="700" fill="#94a3b8">ring 2</text>
      <text x="660" y="266" font-family="Inter, system-ui, sans-serif" font-size="12px" fill="#cbd5e1">two-hop partners</text>
      <text x="660" y="284" font-family="Inter, system-ui, sans-serif" font-size="11px" fill="#94a3b8">two undirected edges away</text>

      <text x="410" y="366" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">compare ring means · topological concentration only · no propagation direction</text>
    </svg>
  </div>
</template>

<style scoped>
.neighborhood-intuition {
  display: flex;
  justify-content: center;
  width: 100%;
}
.neighborhood-intuition svg {
  width: 100%;
  height: auto;
  max-width: 820px;
}
</style>
