<template>
  <div class="axis-contrast">
    <svg viewBox="0 0 900 350" xmlns="http://www.w3.org/2000/svg">
      <!-- divider -->
      <line x1="450" y1="42" x2="450" y2="290" stroke="#334155" stroke-width="1" />

      <!-- ===== LEFT: Act 2, the gradient — one protein on its own ===== -->
      <text x="245" y="30" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="14px" font-weight="700" fill="#cbd5e1">what we measured before</text>

      <!-- axes -->
      <line x1="95" y1="62" x2="95" y2="255" stroke="#334155" stroke-width="1.5" />
      <line x1="95" y1="255" x2="410" y2="255" stroke="#334155" stroke-width="1.5" />
      <text x="78" y="158" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11px"
            fill="#94a3b8" transform="rotate(-90 78 158)">loudness</text>
      <text x="252" y="278" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11px"
            fill="#94a3b8">undirected ring: direct → two-hop</text>

      <!-- down-tilt line: direct partners have larger mean |t| than two-hop partners -->
      <line x1="112" y1="84" x2="400" y2="238" stroke="#4ecadf" stroke-width="2.5" />
      <circle cx="122" cy="90" r="3.2" fill="#4ecadf" opacity="0.65" />
      <circle cx="185" cy="116" r="3.2" fill="#4ecadf" opacity="0.65" />
      <circle cx="250" cy="150" r="3.2" fill="#4ecadf" opacity="0.65" />
      <circle cx="320" cy="190" r="3.2" fill="#4ecadf" opacity="0.65" />
      <circle cx="392" cy="232" r="3.2" fill="#4ecadf" opacity="0.65" />

      <text x="252" y="306" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="600" fill="#4ecadf">one protein, on its own</text>

      <!-- ===== RIGHT: Now — two proteins, one dot per person ===== -->
      <text x="685" y="30" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="14px" font-weight="700" fill="#cbd5e1">Now — two proteins, per person</text>

      <line x1="505" y1="62" x2="505" y2="255" stroke="#334155" stroke-width="1.5" />
      <line x1="505" y1="255" x2="830" y2="255" stroke="#334155" stroke-width="1.5" />
      <text x="488" y="158" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11px"
            fill="#94a3b8" transform="rotate(-90 488 158)">protein B level</text>
      <text x="672" y="278" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11px"
            fill="#94a3b8">protein A (anchor) level</text>

      <!-- up-tilt: A high, B high -->
      <line x1="520" y1="238" x2="815" y2="84" stroke="#4ecadf" stroke-width="2.5" />
      <circle cx="528" cy="232" r="3.2" fill="#4ecadf" opacity="0.65" />
      <circle cx="595" cy="196" r="3.2" fill="#4ecadf" opacity="0.65" />
      <circle cx="665" cy="160" r="3.2" fill="#4ecadf" opacity="0.65" />
      <circle cx="735" cy="120" r="3.2" fill="#4ecadf" opacity="0.65" />
      <circle cx="808" cy="88" r="3.2" fill="#4ecadf" opacity="0.65" />

      <text x="672" y="306" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="600" fill="#4ecadf">do the two rise and fall together?</text>

      <!-- center one-liner: the level-vs-relationship distinction, in plain words -->
      <text x="450" y="338" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">different question — undirected neighborhood contrast, then protein B vs protein A</text>
    </svg>
  </div>
</template>

<style scoped>
.axis-contrast {
  display: flex;
  justify-content: center;
  width: 100%;
}
.axis-contrast svg {
  width: 100%;
  height: auto;
  max-width: 1040px;
}
</style>
