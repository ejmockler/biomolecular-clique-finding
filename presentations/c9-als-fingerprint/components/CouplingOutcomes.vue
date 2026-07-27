<template>
  <div class="coupling-outcomes">
    <svg viewBox="0 0 900 300" xmlns="http://www.w3.org/2000/svg">
      <!-- divider -->
      <line x1="450" y1="34" x2="450" y2="250" stroke="#334155" stroke-width="1" />

      <!-- ===== LEFT: the three tilts line up ===== -->
      <text x="240" y="30" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="15px" font-weight="700" fill="#cbd5e1">the three tilts line up</text>

      <!-- minimal axes -->
      <line x1="80" y1="70" x2="80" y2="225" stroke="#475569" stroke-width="1.5" />
      <line x1="80" y1="225" x2="410" y2="225" stroke="#475569" stroke-width="1.5" />

      <!-- three near-coincident lines (same tilt, tiny offset so all read) -->
      <line x1="92" y1="218" x2="398" y2="96" stroke="#4ecadf" stroke-width="2.5" />
      <line x1="92" y1="221" x2="398" y2="103" stroke="#64748b" stroke-width="2.5" />
      <line x1="92" y1="224" x2="398" y2="110" stroke="#7dd629" stroke-width="2.5" />

      <text x="240" y="270" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="15px" font-weight="600" fill="#7dd629">→ the link is the same</text>

      <!-- ===== RIGHT: the three tilts fan apart ===== -->
      <text x="685" y="30" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="15px" font-weight="700" fill="#cbd5e1">the three tilts fan apart</text>

      <line x1="500" y1="70" x2="500" y2="225" stroke="#475569" stroke-width="1.5" />
      <line x1="500" y1="225" x2="830" y2="225" stroke="#475569" stroke-width="1.5" />

      <!-- three lines fanning from a common origin -->
      <line x1="512" y1="220" x2="812" y2="88" stroke="#4ecadf" stroke-width="2.5" />
      <line x1="512" y1="220" x2="812" y2="140" stroke="#64748b" stroke-width="2.5" />
      <line x1="512" y1="220" x2="812" y2="184" stroke="#7dd629" stroke-width="2.5" />
      <!-- group labels at the divergent ends -->
      <text x="820" y="91" font-family="Inter, system-ui, sans-serif" font-size="11px" font-weight="700" fill="#4ecadf">C9</text>
      <text x="820" y="143" font-family="Inter, system-ui, sans-serif" font-size="11px" font-weight="700" fill="#94a3b8">Spor</text>
      <text x="820" y="187" font-family="Inter, system-ui, sans-serif" font-size="11px" font-weight="700" fill="#7dd629">Ctrl</text>

      <text x="685" y="270" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="15px" font-weight="600" fill="#64748b">→ the link is different</text>
    </svg>
  </div>
</template>

<style scoped>
.coupling-outcomes {
  display: flex;
  justify-content: center;
  width: 100%;
}
.coupling-outcomes svg {
  width: 100%;
  height: auto;
  max-width: 1040px;
}
</style>
