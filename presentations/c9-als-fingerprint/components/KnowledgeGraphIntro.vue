<script setup>
// Left: stacked "paper" cards. Right: a small directed network (knowledge layer).
const papers = [
  { x: 40, y: 70 },
  { x: 52, y: 96 },
  { x: 64, y: 122 },
  { x: 76, y: 148 },
];
const paperW = 96;
const paperH = 60;

// Network nodes (purple knowledge). Hand-placed for a clean directed look.
const nodes = [
  { id: 'A', x: 430, y: 80, label: 'A' },
  { id: 'B', x: 560, y: 60, label: 'B' },
  { id: 'C', x: 690, y: 110, label: 'C' },
  { id: 'D', x: 470, y: 180 },
  { id: 'E', x: 600, y: 165 },
  { id: 'F', x: 720, y: 215 },
  { id: 'G', x: 410, y: 270 },
  { id: 'H', x: 560, y: 285 },
  { id: 'I', x: 690, y: 300 },
];
const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

// Cyan data settling onto purple knowledge: small data dots that ARRIVE on
// existing node positions once on mount. The literal layering gesture.
const dataDots = ['A', 'E', 'I'];

const edges = [
  ['A', 'B'],
  ['B', 'C'],
  ['A', 'D'],
  ['D', 'E'],
  ['B', 'E'],
  ['E', 'C'],
  ['E', 'F'],
  ['D', 'G'],
  ['G', 'H'],
  ['H', 'E'],
  ['H', 'I'],
  ['F', 'I'],
];

const nodeR = 16;

// Shorten each edge so the arrowhead lands at the node border, not its center.
function edgeGeom([a, b]) {
  const s = byId[a];
  const t = byId[b];
  const ang = Math.atan2(t.y - s.y, t.x - s.x);
  return {
    x1: s.x + Math.cos(ang) * nodeR,
    y1: s.y + Math.sin(ang) * nodeR,
    x2: t.x - Math.cos(ang) * (nodeR + 5),
    y2: t.y - Math.sin(ang) * (nodeR + 5),
  };
}
</script>

<template>
  <div class="knowledge-graph-intro">
    <svg viewBox="0 0 820 360" width="100%" preserveAspectRatio="xMidYMid meet">
      <defs>
        <marker
          id="kgi-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#bf6ff7" />
        </marker>
      </defs>

      <!-- paper stack -->
      <g>
        <rect
          v-for="(p, i) in papers"
          :key="'p' + i"
          :x="p.x"
          :y="p.y"
          :width="paperW"
          :height="paperH"
          rx="4"
          fill="#243052"
          stroke="#6d4a8a"
          stroke-width="1"
        />
        <!-- text lines on the topmost paper -->
        <g stroke="#7d5ba0" stroke-width="2">
          <line :x1="76 + 12" :y1="148 + 16" :x2="76 + paperW - 12" :y2="148 + 16" />
          <line :x1="76 + 12" :y1="148 + 28" :x2="76 + paperW - 20" :y2="148 + 28" />
          <line :x1="76 + 12" :y1="148 + 40" :x2="76 + paperW - 30" :y2="148 + 40" />
        </g>
      </g>

      <!-- caption near papers -->
      <text
        x="40"
        y="240"
        font-family="Inter, system-ui, sans-serif"
        font-size="13px"
        fill="#94a3b8"
      >every "A acts on B"</text>
      <text
        x="40"
        y="258"
        font-family="Inter, system-ui, sans-serif"
        font-size="13px"
        fill="#94a3b8"
      >in the literature →</text>
      <text
        x="40"
        y="276"
        font-family="Inter, system-ui, sans-serif"
        font-size="13px"
        fill="#bf6ff7"
      >one arrow</text>

      <!-- emanating faint arrows from papers to the network -->
      <g stroke="#9b7bc0" stroke-width="1.4" stroke-opacity="0.78" marker-end="url(#kgi-arrow)">
        <line x1="180" y1="110" x2="400" y2="90" />
        <line x1="184" y1="150" x2="392" y2="175" />
        <line x1="180" y1="190" x2="388" y2="255" />
      </g>

      <!-- network edges -->
      <g stroke="#bf6ff7" stroke-width="1.5" opacity="0.85">
        <line
          v-for="(e, i) in edges"
          :key="'e' + i"
          :x1="edgeGeom(e).x1"
          :y1="edgeGeom(e).y1"
          :x2="edgeGeom(e).x2"
          :y2="edgeGeom(e).y2"
          marker-end="url(#kgi-arrow)"
        />
      </g>

      <!-- network nodes -->
      <g>
        <circle
          v-for="n in nodes"
          :key="n.id"
          :cx="n.x"
          :cy="n.y"
          :r="nodeR"
          fill="#1a2540"
          stroke="#bf6ff7"
          stroke-width="2"
        />
        <text
          v-for="n in nodes.filter((x) => x.label)"
          :key="'t' + n.id"
          :x="n.x"
          :y="n.y + 5"
          text-anchor="middle"
          font-family="'JetBrains Mono', monospace"
          font-size="14px"
          font-weight="800"
          fill="#bf6ff7"
        >{{ n.label }}</text>
        <!-- one explicit "protein" tag next to node A -->
        <text
          :x="byId['A'].x - nodeR - 6"
          :y="byId['A'].y - nodeR - 4"
          text-anchor="start"
          font-family="Inter, system-ui, sans-serif"
          font-size="11px"
          fill="#c9a9ee"
        >protein</text>
      </g>

      <!-- cyan data settling onto purple knowledge: dots arrive once on mount -->
      <g class="kgi-data-layer">
        <g
          v-for="(id, i) in dataDots"
          :key="'dd' + id"
          :class="'kgi-data-dot kgi-data-dot--' + i"
        >
          <circle :cx="byId[id].x" :cy="byId[id].y" :r="5" fill="#4ecadf" />
        </g>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.knowledge-graph-intro {
  display: flex;
  justify-content: center;
  width: 100%;
}

/* Cyan data settling onto purple knowledge — arrives ONCE on mount.
   Opacity fade + a tiny downward settle onto the node. ease-out, no loop. */
@keyframes kgi-data-arrive {
  from {
    opacity: 0;
    transform: translateY(-7px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.kgi-data-dot {
  transform-box: fill-box;
  transform-origin: center;
  animation: kgi-data-arrive 300ms ease-out both;
}

/* short stagger — ~80ms apart — for the layering reveal */
.kgi-data-dot--0 {
  animation-delay: 0ms;
}
.kgi-data-dot--1 {
  animation-delay: 80ms;
}
.kgi-data-dot--2 {
  animation-delay: 160ms;
}

@media (prefers-reduced-motion: reduce) {
  .kgi-data-dot {
    animation: none;
  }
}
</style>
