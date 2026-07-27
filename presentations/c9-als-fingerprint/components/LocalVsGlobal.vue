<script setup>
import primaryAnalysis from '../../../data/publication/c9_primary_analysis.json';

const bounded = primaryAnalysis.confirmatory.bounded_pass_pattern;
const unbounded = primaryAnalysis.confirmatory.unbounded_pass_pattern;
const passScale = 22.5;
const baselineY = 290;
const barHeight = value => Math.max(value * passScale, 2);
const barTop = value => baselineY - barHeight(value);
const pattern = values => values.join(' / ');
const total = values => values.reduce((sum, value) => sum + value, 0);
</script>

<template>
  <div class="local-vs-global">
    <svg viewBox="0 0 820 380" xmlns="http://www.w3.org/2000/svg">
      <!-- big sans takeaway -->
      <text x="410" y="34" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="22px" font-weight="800" fill="#f1f5f9">bounded {{ pattern(bounded) }} · unbounded {{ pattern(unbounded) }}</text>

      <!-- baseline for bars -->
      <line x1="60" y1="290" x2="760" y2="290" stroke="#334155" stroke-width="1.5" />

      <!-- y-axis hint: pass count ticks -->
      <text x="52" y="294" text-anchor="end" font-family="'JetBrains Mono', monospace" font-size="10px" fill="#94a3b8">0</text>
      <line x1="56" y1="200" x2="60" y2="200" stroke="#334155" stroke-width="1" />
      <text x="52" y="204" text-anchor="end" font-family="'JetBrains Mono', monospace" font-size="10px" fill="#94a3b8">4</text>
      <line x1="56" y1="110" x2="60" y2="110" stroke="#334155" stroke-width="1" />
      <text x="52" y="114" text-anchor="end" font-family="'JetBrains Mono', monospace" font-size="10px" fill="#94a3b8">8</text>
      <text x="40" y="180" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11px"
            fill="#94a3b8" transform="rotate(-90 40 180)">passes</text>

      <!-- ============ GROUP 1: within 2 hops (local) ============ -->
      <!-- bar height scale: 1 pass = 22.5px (8 passes spans 290->110) -->
      <!-- C9-vs-Spor 8 -->
      <rect x="130" :y="barTop(bounded[0])" width="48" :height="barHeight(bounded[0])" rx="2" fill="#4ecadf" />
      <text x="154" :y="barTop(bounded[0]) - 9" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="15px" font-weight="800" fill="#4ecadf">{{ bounded[0] }}</text>
      <text x="154" y="306" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="9px" fill="#94a3b8">C9–Spor</text>
      <!-- C9-vs-Ctrl 6 -->
      <rect x="186" :y="barTop(bounded[1])" width="48" :height="barHeight(bounded[1])" rx="2" fill="#4ecadf" />
      <text x="210" :y="barTop(bounded[1]) - 9" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="15px" font-weight="800" fill="#4ecadf">{{ bounded[1] }}</text>
      <text x="210" y="306" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="9px" fill="#94a3b8">C9–Ctrl</text>
      <!-- Spor-vs-Ctrl 0 -->
      <rect x="242" :y="barTop(bounded[2])" width="48" :height="barHeight(bounded[2])" rx="1" fill="#475569" />
      <text x="266" :y="barTop(bounded[2]) - 8" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="15px" font-weight="800" fill="#94a3b8">{{ bounded[2] }}</text>
      <text x="266" y="306" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="9px" fill="#94a3b8">Spor–Ctrl</text>

      <text x="210" y="334" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="700" fill="#4ecadf">within 2 hops (local)</text>
      <text x="210" y="352" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="12px" font-weight="800" fill="#f1f5f9">total {{ total(bounded) }}</text>

      <!-- ============ GROUP 2: unbounded (whole graph) ============ -->
      <!-- C9-vs-Spor 6 -->
      <rect x="500" :y="barTop(unbounded[0])" width="48" :height="barHeight(unbounded[0])" rx="2" fill="#4ecadf" />
      <text x="524" :y="barTop(unbounded[0]) - 9" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="15px" font-weight="800" fill="#4ecadf">{{ unbounded[0] }}</text>
      <text x="524" y="306" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="9px" fill="#94a3b8">C9–Spor</text>
      <!-- C9-vs-Ctrl 0 -->
      <rect x="556" :y="barTop(unbounded[1])" width="48" :height="barHeight(unbounded[1])" rx="1" fill="#475569" />
      <text x="580" :y="barTop(unbounded[1]) - 8" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="15px" font-weight="800" fill="#94a3b8">{{ unbounded[1] }}</text>
      <text x="580" y="306" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="9px" fill="#94a3b8">C9–Ctrl</text>
      <!-- Spor-vs-Ctrl 0 -->
      <rect x="612" :y="barTop(unbounded[2])" width="48" :height="barHeight(unbounded[2])" rx="1" fill="#475569" />
      <text x="636" :y="barTop(unbounded[2]) - 8" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="15px" font-weight="800" fill="#94a3b8">{{ unbounded[2] }}</text>
      <text x="636" y="306" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="9px" fill="#94a3b8">Spor–Ctrl</text>

      <text x="580" y="334" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="700" fill="#94a3b8">unbounded (whole graph)</text>
      <text x="580" y="352" text-anchor="middle" font-family="'JetBrains Mono', monospace"
            font-size="12px" font-weight="800" fill="#94a3b8">total {{ total(unbounded) }}</text>

      <!-- divider between groups -->
      <line x1="395" y1="80" x2="395" y2="290" stroke="#334155" stroke-width="1" />

      <!-- tertiary caption -->
      <text x="410" y="374" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" fill="#94a3b8">observed attenuation under the unbounded depth sensitivity</text>
    </svg>
  </div>
</template>

<style scoped>
.local-vs-global {
  display: flex;
  justify-content: center;
  width: 100%;
}
.local-vs-global svg {
  width: 100%;
  height: auto;
  max-width: 820px;
}
</style>
