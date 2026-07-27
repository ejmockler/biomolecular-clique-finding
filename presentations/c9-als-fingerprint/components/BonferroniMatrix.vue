<script setup>
import primaryAnalysis from '../../../data/publication/c9_primary_analysis.json';

defineProps({
  highlightCore: { type: Boolean, default: false },
});

// Production log2(x+1), bounded measured-only h<=2 same-cohort consistency matrix.
// Numeric state comes directly from the tracked publication manifest.
const confirmatory = primaryAnalysis.confirmatory;
const terms = confirmatory.term_order;
const cols = confirmatory.contrast_order;
const pass = terms.map(term =>
  cols.map(contrast => confirmatory.bounded[contrast].terms[term].pass),
);
const tallies = cols.map(contrast => `${confirmatory.bounded[contrast].pass_count} / 8`);

// Geometry — compact so the slide caption still fits under it.
const labelX = 18;
const gridX = 360;
const colW = 150;
const rowH = 30;
const gridTop = 58;
const coreRows = confirmatory.six_term_core.length;

function colCenter(c) { return gridX + c * colW + colW / 2; }
function cellY(r) { return gridTop + r * rowH; }
function rowMid(r) { return gridTop + r * rowH + rowH / 2; }
function checkPoints(c, r) {
  const cx = colCenter(c), cy = rowMid(r);
  return `${cx - 8},${cy + 1} ${cx - 2},${cy + 7} ${cx + 9},${cy - 8}`;
}
const annX = gridX + 3 * colW + 10;
</script>

<template>
  <div class="bonferroni-matrix" :class="highlightCore ? 'mode-core' : 'mode-headline'">
    <svg viewBox="0 0 820 360" xmlns="http://www.w3.org/2000/svg">
      <!-- Column headers -->
      <text v-for="(c, ci) in cols" :key="'h' + ci"
            :x="colCenter(ci)" :y="gridTop - 16" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="12px" font-weight="700"
            :fill="ci === 2 ? '#94a3b8' : '#4ecadf'">{{ c }}</text>

      <!-- Core highlight band (behind cells) — alpha baked into rgba() to survive UnoCSS -->
      <rect v-if="highlightCore" class="core-band"
            :x="labelX - 6" :y="gridTop - 4"
            :width="gridX + 2 * colW - labelX + 6 + colW * 0" :height="coreRows * rowH + 8"
            rx="8" fill="rgba(78,202,223,0.08)"
            stroke="rgba(78,202,223,0.6)" stroke-width="1.5" />

      <!-- Rows -->
      <g v-for="(t, r) in terms" :key="'r' + r">
        <!-- Row label -->
        <text :x="labelX" :y="rowMid(r) + 4"
              font-family="Inter, system-ui, sans-serif" font-size="12.5px"
              :fill="r >= coreRows ? '#94a3b8' : '#e2e8f0'">{{ t }}</text>

        <!-- Cells. When the core is highlighted, dim non-core rows (alpha baked
             into rgba) so the six-term core is isolated and visibly distinct. -->
        <g v-for="(c, ci) in cols" :key="'c' + r + '-' + ci"
           :class="{ 'dim-noncore': highlightCore && r >= coreRows }">
          <rect :x="gridX + ci * colW + 5" :y="cellY(r) + 3"
                :width="colW - 10" :height="rowH - 6" rx="5"
                :fill="pass[r][ci]
                  ? (highlightCore && r >= coreRows ? 'rgba(78,202,223,0.09)' : 'rgba(78,202,223,0.28)')
                  : (ci === 2 ? 'rgba(100,116,139,0.08)' : 'rgba(148,163,184,0.04)')"
                :stroke="pass[r][ci]
                  ? (highlightCore && r >= coreRows ? 'rgba(78,202,223,0.3)' : 'rgba(78,202,223,0.9)')
                  : '#334155'"
                stroke-width="1.2" />
          <polyline v-if="pass[r][ci]" :points="checkPoints(ci, r)"
                    fill="none"
                    :stroke="highlightCore && r >= coreRows ? 'rgba(78,202,223,0.35)' : '#4ecadf'"
                    stroke-width="2.6"
                    stroke-linecap="round" stroke-linejoin="round" />
          <!-- empty marker in the quiet column -->
          <line v-if="ci === 2" :x1="colCenter(2) - 6" :y1="rowMid(r)"
                :x2="colCenter(2) + 6" :y2="rowMid(r)"
                stroke="#64748b" stroke-width="2" stroke-linecap="round" />
        </g>
      </g>

      <!-- Core annotations -->
      <g v-if="highlightCore">
        <text :x="annX" :y="rowMid(2) - 2" font-family="Inter, system-ui, sans-serif"
              font-size="11px" font-weight="600" fill="#4ecadf">core</text>
        <text :x="annX" :y="rowMid(2) + 13" font-family="Inter, system-ui, sans-serif"
              font-size="10px" fill="#94a3b8">passes both C9</text>
        <text :x="annX" :y="rowMid(6) + 4" font-family="Inter, system-ui, sans-serif"
              font-size="10px" fill="#94a3b8">C9-vs-Sporadic only</text>
        <text :x="annX" :y="rowMid(7) + 4" font-family="Inter, system-ui, sans-serif"
              font-size="10px" fill="#94a3b8">C9-vs-Sporadic only</text>
      </g>

      <!-- Column tallies. Cyan C9 tallies arrive first; the control 0/8 settles
           in LAST so the empty column is the final beat the eye lands on.
           0/8 tinted green = correct-silence. -->
      <text v-for="(tot, ti) in tallies" :key="'t' + ti"
            :class="ti === 2 ? 'tally-control' : 'tally-c9'"
            :x="colCenter(ti)" :y="gridTop + 8 * rowH + 30" text-anchor="middle"
            font-family="'JetBrains Mono', monospace" font-size="22px" font-weight="800"
            :fill="ti === 2 ? '#5fd6a0' : '#4ecadf'">{{ tot }}</text>
      <text class="tally-control" :x="colCenter(2)" :y="gridTop + 8 * rowH + 48" text-anchor="middle"
            font-family="Inter, system-ui, sans-serif" font-size="10.5px"
            font-style="italic" fill="#7fcfb0">quiet within-cohort specificity leg</text>
    </svg>
  </div>
</template>

<style scoped>
.bonferroni-matrix { display: flex; justify-content: center; width: 100%; }
.bonferroni-matrix svg { width: 100%; height: auto; max-width: 820px; }

/* Rationed arrival motion: one state-change per slide, fires ONCE on mount.
   Opacity-only fades (bulletproof on SVG). ease-out, no loop, no bounce. */
@keyframes arrive {
  from { opacity: 0; }
  to   { opacity: 1; }
}

/* HEADLINE (s45): the cyan C9 tallies arrive first, then the control 0/8
   column settles in LAST — the empty control column is the final beat. */
.mode-headline .tally-c9 {
  animation: arrive 300ms ease-out both;
}
.mode-headline .tally-control {
  animation: arrive 320ms ease-out both;
  animation-delay: 320ms;
}

/* CORE (s47): the core-band highlight layers in once — the single focal
   state-change that distinguishes s47 from s45. */
.mode-core .core-band {
  animation: arrive 320ms ease-out both;
}

@media (prefers-reduced-motion: reduce) {
  .mode-headline .tally-c9,
  .mode-headline .tally-control,
  .mode-core .core-band {
    animation: none;
  }
}
</style>
