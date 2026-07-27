<script setup>
// Seeded RNG so the swarm layout is stable across reloads.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(20250610);

const N = 80;
const centerX = 470;   // null cloud center (x)
const baseY = 150;     // swarm center line (y)
const yJitter = 58;    // vertical spread
const xSpread = 150;   // horizontal spread of the null cloud

// Gaussian-ish via two uniforms averaged -> tighter central clustering.
function bell() {
  return (rand() + rand() + rand() - 1.5) / 1.5; // ~[-1,1], peaked at 0
}

const dots = Array.from({ length: N }, () => ({
  cx: centerX + bell() * xSpread,
  cy: baseY + bell() * yJitter,
  r: 3 + rand() * 1.2,
}));

// Shuffle so any residual order doesn't read as a gradient.
for (let i = dots.length - 1; i > 0; i--) {
  const j = Math.floor(rand() * (i + 1));
  [dots[i], dots[j]] = [dots[j], dots[i]];
}
</script>

<template>
  <div class="permutation-null">
    <svg viewBox="0 0 820 300" xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="A cloud of degree-matched shuffled slopes and one observed anchor slope. This is an anchor-level diagnostic, not the pathway enrichment null.">
      <defs>
        <marker id="pn-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0,0 L8,3.2 L0,6.4 Z" fill="#4ecadf" />
        </marker>
      </defs>

      <!-- faint reference line where the null cloud centers -->
      <line x1="470" y1="56" x2="470" y2="244" stroke="#475569" stroke-width="1" stroke-dasharray="4 5" />
      <text x="470" y="262" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" fill="#94a3b8">center of the random cloud</text>

      <!-- shuffled-label swarm -->
      <g>
        <circle v-for="(d, i) in dots" :key="i" :cx="d.cx" :cy="d.cy" :r="d.r"
                fill="rgba(148,163,184,0.7)" />
      </g>

      <!-- swarm label -->
      <text x="470" y="36" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="600" fill="#94a3b8">degree-matched shuffles for one anchor</text>

      <!-- arrow pointing to the real answer -->
      <line x1="690" y1="100" x2="730" y2="136" stroke="#4ecadf" stroke-width="1.6"
            marker-end="url(#pn-arrow)" />
      <text x="690" y="92" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="14px" font-weight="700" fill="#4ecadf">observed slope</text>

      <!-- the real result: one bright cyan dot standing clearly to the side -->
      <circle cx="744" cy="150" r="11" fill="rgba(78,202,223,0.15)" />
      <circle cx="744" cy="150" r="6.5" fill="#4ecadf" />

      <!-- caption -->
      <text x="410" y="290" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">diagnostic p-value for this anchor only · not consumed by pathway GSEA</text>
    </svg>
  </div>
</template>

<style scoped>
.permutation-null {
  display: flex;
  justify-content: center;
  width: 100%;
}
.permutation-null svg {
  width: 100%;
  height: auto;
  max-width: 820px;
}
</style>
