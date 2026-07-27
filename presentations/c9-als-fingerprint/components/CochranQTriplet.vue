<script setup>
// stage: 'one'  → single tilt (what "coupling" means)
//        'three'→ three tilts overlaid (same or fanned?)
//        'referee' → three tilts + the Cochran-Q judge
defineProps({ stage: { type: String, default: 'referee' } })
</script>

<template>
  <div class="cochran-q-triplet">
    <svg viewBox="0 0 820 400" xmlns="http://www.w3.org/2000/svg">
      <!-- ===== TOP INSET: an edge between two wired proteins ===== -->
      <line x1="330" y1="52" x2="490" y2="52" stroke="#bf6ff7" stroke-width="2" />
      <circle cx="330" cy="52" r="13" fill="#1a2540" stroke="#bf6ff7" stroke-width="2" />
      <circle cx="490" cy="52" r="13" fill="#1a2540" stroke="#bf6ff7" stroke-width="2" />
      <text x="330" y="56" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" font-weight="700" fill="#bf6ff7">A</text>
      <text x="490" y="56" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" font-weight="700" fill="#bf6ff7">B</text>
      <text x="410" y="30" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="12px" fill="#94a3b8">
        <template v-if="stage === 'one'">two wired-together proteins</template>
        <template v-else>one of <tspan font-weight="800" fill="#bf6ff7">944</tspan> linked pairs inside a neighborhood</template>
      </text>

      <!-- ===== PLOT: shared x/y axes ===== -->
      <line x1="150" y1="120" x2="150" y2="320" stroke="#334155" stroke-width="1.5" />
      <line x1="150" y1="320" x2="560" y2="320" stroke="#334155" stroke-width="1.5" />
      <text x="138" y="220" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11px"
            fill="#94a3b8" transform="rotate(-90 138 220)">protein B level</text>
      <text x="355" y="344" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11px"
            fill="#94a3b8">protein A (anchor) level</text>

      <!-- C9 cyan scatter + line (shown in every stage; "one group" in stage one) -->
      <g>
        <circle cx="190" cy="300" r="2.8" fill="#4ecadf" opacity="0.6" />
        <circle cx="255" cy="278" r="2.8" fill="#4ecadf" opacity="0.6" />
        <circle cx="330" cy="245" r="2.8" fill="#4ecadf" opacity="0.6" />
        <circle cx="420" cy="205" r="2.8" fill="#4ecadf" opacity="0.6" />
        <circle cx="510" cy="160" r="2.8" fill="#4ecadf" opacity="0.6" />
        <line x1="170" y1="305" x2="540" y2="155" stroke="#4ecadf" stroke-width="2.5" />
      </g>

      <!-- Sporadic + Control: only once we compare groups -->
      <g v-if="stage !== 'one'">
        <circle cx="200" cy="288" r="2.8" fill="#64748b" />
        <circle cx="285" cy="265" r="2.8" fill="#64748b" />
        <circle cx="360" cy="248" r="2.8" fill="#64748b" />
        <circle cx="445" cy="222" r="2.8" fill="#64748b" />
        <circle cx="520" cy="200" r="2.8" fill="#64748b" />
        <line x1="170" y1="298" x2="540" y2="198" stroke="#64748b" stroke-width="2.5" />

        <circle cx="210" cy="272" r="2.8" fill="#7dd629" opacity="0.6" />
        <circle cx="300" cy="258" r="2.8" fill="#7dd629" opacity="0.6" />
        <circle cx="385" cy="240" r="2.8" fill="#7dd629" opacity="0.6" />
        <circle cx="465" cy="226" r="2.8" fill="#7dd629" opacity="0.6" />
        <circle cx="530" cy="212" r="2.8" fill="#7dd629" opacity="0.6" />
        <line x1="170" y1="288" x2="540" y2="214" stroke="#7dd629" stroke-width="2.5" />
      </g>

      <!-- Group labels at the right ends of the lines -->
      <template v-if="stage === 'one'">
        <text x="548" y="156" font-family="Inter, system-ui, sans-serif" font-size="11px"
              font-weight="700" fill="#4ecadf">one group</text>
      </template>
      <template v-else>
        <text x="548" y="156" font-family="Inter, system-ui, sans-serif" font-size="11px"
              font-weight="700" fill="#4ecadf">C9</text>
        <text x="548" y="200" font-family="Inter, system-ui, sans-serif" font-size="11px"
              font-weight="700" fill="#94a3b8">Sporadic</text>
        <text x="548" y="226" font-family="Inter, system-ui, sans-serif" font-size="11px"
              font-weight="700" fill="#7dd629">Control</text>
      </template>

      <!-- stage 'one': name the tilt -->
      <text v-if="stage === 'one'" x="250" y="158" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="600" fill="#4ecadf">the tilt = how tightly they move together</text>

      <!-- stage 'three': the comparison question -->
      <text v-if="stage === 'three'" x="630" y="276" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="600" fill="#cbd5e1">same tilt,</text>
      <text v-if="stage === 'three'" x="630" y="296" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="600" fill="#cbd5e1">or fanned</text>
      <text v-if="stage === 'three'" x="630" y="316" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="600" fill="#cbd5e1">apart?</text>

      <!-- stage 'referee': the Cochran-Q judge -->
      <template v-if="stage === 'referee'">
        <text x="640" y="270" font-family="'JetBrains Mono', monospace" font-size="18px"
              font-weight="800" fill="#f1f5f9">Cochran-Q</text>
        <text x="640" y="294" font-family="Inter, system-ui, sans-serif" font-size="12px"
              fill="#94a3b8">scores how much</text>
        <text x="640" y="311" font-family="Inter, system-ui, sans-serif" font-size="12px"
              fill="#94a3b8">the tilts disagree</text>
      </template>

      <!-- bottom hint varies by stage -->
      <text x="410" y="388" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">
        <template v-if="stage === 'one'">plot one protein against its partner — the tilt is the link</template>
        <template v-else-if="stage === 'three'">measure that tilt separately in C9, Sporadic, and Control</template>
        <template v-else>small disagreement = concordant (the link is the same)</template>
      </text>
    </svg>
  </div>
</template>

<style scoped>
.cochran-q-triplet {
  display: flex;
  justify-content: center;
  width: 100%;
}
.cochran-q-triplet svg {
  width: 100%;
  height: auto;
  max-width: 840px;
}
</style>
