<template>
  <div class="wrong-yardstick">
    <svg viewBox="0 0 900 300" xmlns="http://www.w3.org/2000/svg">
      <!-- ===== divider ===== -->
      <line x1="450" y1="18" x2="450" y2="272" stroke="#475569" stroke-width="1" />

      <!-- ===================== LEFT: self-contained question ===================== -->
      <text x="232" y="24" text-anchor="middle" fill="#f59e0b"
            font-family="Inter, system-ui, sans-serif" font-size="11px" font-weight="800" letter-spacing="1">SELF-CONTAINED QUESTION</text>
      <text x="232" y="44" text-anchor="middle" fill="#cbd5e1"
            font-family="Inter, system-ui, sans-serif" font-size="13px">vs no change at all — "did it move?"</text>

      <!-- floor = healthy / no change -->
      <line x1="70" y1="250" x2="400" y2="250" stroke="#64748b" stroke-width="1.5" />
      <text x="72" y="266" fill="#94a3b8" font-family="Inter, system-ui, sans-serif" font-size="11px">floor = no change at all</text>

      <!-- "significant" threshold, just above the floor -->
      <line x1="70" y1="208" x2="400" y2="208" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.85" />
      <text x="400" y="204" text-anchor="end" fill="#f59e0b" font-family="Inter, system-ui, sans-serif" font-size="11px">rejection threshold</text>

      <!-- the risen water: the whole proteome lifted off the floor -->
      <rect x="70" y="120" width="330" height="130" fill="rgba(78,202,223,0.18)" />
      <line x1="70" y1="120" x2="400" y2="120" stroke="#4ecadf" stroke-width="1.5" opacity="0.55" />
      <text x="235" y="172" text-anchor="middle" fill="#94a3b8" font-family="Inter, system-ui, sans-serif"
            font-size="11px" font-style="italic">the tide rose — nearly the whole proteome moved</text>

      <!-- every group floats ON the waterline: random (gray) + our candidate (cyan) -->
      <circle cx="120" cy="123" r="5" fill="#64748b" />
      <circle cx="178" cy="117" r="5" fill="#64748b" />
      <circle cx="250" cy="124" r="5" fill="#64748b" />
      <circle cx="358" cy="118" r="5" fill="#64748b" />
      <circle cx="302" cy="116" r="6.5" fill="#4ecadf" />
      <text x="302" y="103" text-anchor="middle" fill="#4ecadf"
            font-family="Inter, system-ui, sans-serif" font-size="11px" font-weight="700">ours</text>

      <text x="232" y="290" text-anchor="middle" fill="#f59e0b"
            font-family="Inter, system-ui, sans-serif" font-size="12px" font-weight="600">many observed-data sets reject → poor competitive specificity</text>

      <!-- ===================== RIGHT: RIGHT yardstick (vs the crowd) ===================== -->
      <text x="668" y="24" text-anchor="middle" fill="#7dd629"
            font-family="Inter, system-ui, sans-serif" font-size="11px" font-weight="800" letter-spacing="1">COMPETITIVE QUESTION</text>
      <text x="668" y="44" text-anchor="middle" fill="#cbd5e1"
            font-family="Inter, system-ui, sans-serif" font-size="13px">vs random groups from the same patients</text>

      <!-- the crowd it actually lives in -->
      <ellipse cx="664" cy="150" rx="112" ry="44" fill="none" stroke="#475569" stroke-width="1" stroke-dasharray="4,3" />
      <g fill="#64748b">
        <circle cx="584" cy="140" r="5" /><circle cx="614" cy="128" r="5" /><circle cx="646" cy="142" r="5" />
        <circle cx="676" cy="132" r="5" /><circle cx="706" cy="144" r="5" /><circle cx="732" cy="136" r="5" />
        <circle cx="598" cy="162" r="5" /><circle cx="630" cy="170" r="5" /><circle cx="688" cy="168" r="5" />
        <circle cx="718" cy="160" r="5" /><circle cx="610" cy="150" r="5" /><circle cx="700" cy="120" r="5" />
        <circle cx="650" cy="120" r="5" />
      </g>

      <!-- our candidate sits in the middle of the crowd -->
      <circle cx="660" cy="150" r="6.5" fill="#4ecadf" />
      <line x1="760" y1="150" x2="672" y2="150" stroke="#4ecadf" stroke-width="1" opacity="0.6" />
      <text x="764" y="154" fill="#4ecadf" font-family="Inter, system-ui, sans-serif" font-size="11px" font-weight="700">ours</text>

      <text x="664" y="222" text-anchor="middle" fill="#94a3b8"
            font-family="Inter, system-ui, sans-serif" font-size="11px">the crowd: random groups, same sick people</text>
      <text x="668" y="290" text-anchor="middle" fill="#7dd629"
            font-family="Inter, system-ui, sans-serif" font-size="12px" font-weight="600">ours sits mid-crowd → ordinary, not special</text>
    </svg>
  </div>
</template>

<style scoped>
.wrong-yardstick { display: flex; justify-content: center; width: 100%; }
.wrong-yardstick svg { width: 100%; height: auto; max-width: 720px; }
</style>
