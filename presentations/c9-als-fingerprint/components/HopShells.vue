<template>
  <div class="hop-shells">
    <svg viewBox="0 0 820 400" xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="An anchor surrounded by direct regulatory partners and two-hop partners. Lines are undirected because regulatory distance pools incoming and outgoing INDRA edges.">
      <defs>
        <radialGradient id="hs-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="rgba(78,202,223,0.18)" />
          <stop offset="100%" stop-color="rgba(78,202,223,0)" />
        </radialGradient>
      </defs>

      <!-- soft glow behind anchor -->
      <circle cx="410" cy="200" r="150" fill="url(#hs-glow)" />

      <!-- faint shell guide rings -->
      <circle cx="410" cy="200" r="92" fill="none" stroke="#3b455c" stroke-width="1" stroke-dasharray="3 5" />
      <circle cx="410" cy="200" r="168" fill="none" stroke="#334155" stroke-width="1" stroke-dasharray="3 5" />

      <!-- ===== hop-2 edges (drawn first, behind) ===== -->
      <!-- from h1[0] (top) -->
      <line x1="410" y1="108" x2="356" y2="46"  stroke="#9b7bc0" stroke-width="1.2" />
      <line x1="410" y1="108" x2="470" y2="44"  stroke="#9b7bc0" stroke-width="1.2" />
      <!-- from h1[1] (upper right) -->
      <line x1="497" y1="172" x2="565" y2="118" stroke="#9b7bc0" stroke-width="1.2" />
      <!-- from h1[2] (lower right) -->
      <line x1="475" y1="270" x2="558" y2="288" stroke="#9b7bc0" stroke-width="1.2" />
      <line x1="475" y1="270" x2="466" y2="356" stroke="#9b7bc0" stroke-width="1.2" />
      <!-- from h1[3] (lower left) -->
      <line x1="345" y1="270" x2="262" y2="290" stroke="#9b7bc0" stroke-width="1.2" />
      <line x1="345" y1="270" x2="350" y2="356" stroke="#9b7bc0" stroke-width="1.2" />
      <!-- from h1[4] (upper left) -->
      <line x1="323" y1="172" x2="255" y2="118" stroke="#9b7bc0" stroke-width="1.2" />

      <!-- ===== hop-1 edges (anchor → shell) ===== -->
      <line x1="410" y1="200" x2="410" y2="116" stroke="#bf6ff7" stroke-width="1.6" />
      <line x1="410" y1="200" x2="489" y2="174" stroke="#bf6ff7" stroke-width="1.6" />
      <line x1="410" y1="200" x2="468" y2="262" stroke="#bf6ff7" stroke-width="1.6" />
      <line x1="410" y1="200" x2="352" y2="262" stroke="#bf6ff7" stroke-width="1.6" />
      <line x1="410" y1="200" x2="331" y2="174" stroke="#bf6ff7" stroke-width="1.6" />

      <!-- ===== hop-2 nodes (fainter) ===== -->
      <g>
        <circle cx="350" cy="40"  r="9" fill="#1a2540" stroke="#9b7bc0" stroke-width="1.5" />
        <circle cx="476" cy="38"  r="9" fill="#1a2540" stroke="#9b7bc0" stroke-width="1.5" />
        <circle cx="571" cy="113" r="9" fill="#1a2540" stroke="#9b7bc0" stroke-width="1.5" />
        <circle cx="565" cy="293" r="9" fill="#1a2540" stroke="#9b7bc0" stroke-width="1.5" />
        <circle cx="466" cy="363" r="9" fill="#1a2540" stroke="#9b7bc0" stroke-width="1.5" />
        <circle cx="350" cy="363" r="9" fill="#1a2540" stroke="#9b7bc0" stroke-width="1.5" />
        <circle cx="256" cy="295" r="9" fill="#1a2540" stroke="#9b7bc0" stroke-width="1.5" />
        <circle cx="249" cy="113" r="9" fill="#1a2540" stroke="#9b7bc0" stroke-width="1.5" />
      </g>

      <!-- ===== hop-1 nodes ===== -->
      <g>
        <circle cx="410" cy="108" r="13" fill="#bf6ff7" />
        <circle cx="497" cy="172" r="13" fill="#bf6ff7" />
        <circle cx="475" cy="270" r="13" fill="#bf6ff7" />
        <circle cx="345" cy="270" r="13" fill="#bf6ff7" />
        <circle cx="323" cy="172" r="13" fill="#bf6ff7" />
      </g>

      <!-- ===== anchor ===== -->
      <circle cx="410" cy="200" r="22" fill="#4ecadf" />
      <text x="410" y="205" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" font-weight="700" fill="#0a0f1a">anchor</text>

      <!-- ===== shell labels ===== -->
      <text x="612" y="178" font-family="Inter, system-ui, sans-serif" font-size="14px"
            font-weight="600" fill="#bf6ff7">1 hop</text>
      <text x="612" y="196" font-family="Inter, system-ui, sans-serif" font-size="12px"
            fill="#94a3b8">direct partner · either direction</text>

      <text x="612" y="244" font-family="Inter, system-ui, sans-serif" font-size="14px"
            font-weight="600" fill="#a98bcf">2 hops</text>
      <text x="612" y="262" font-family="Inter, system-ui, sans-serif" font-size="12px"
            fill="#94a3b8">two undirected edges away</text>

      <!-- connector ticks from labels to rings -->
      <line x1="528" y1="173" x2="604" y2="173" stroke="#bf6ff7" stroke-width="1" opacity="0.5" />
      <line x1="582" y1="239" x2="604" y2="239" stroke="#a98bcf" stroke-width="1" opacity="0.6" />

      <!-- stop note -->
      <text x="612" y="320" font-family="Inter, system-ui, sans-serif" font-size="12px"
            fill="#94a3b8">distance pools</text>
      <text x="612" y="338" font-family="Inter, system-ui, sans-serif" font-size="12px"
            fill="#f1f5f9" font-weight="600">regulators + regulatees</text>
      <text x="612" y="356" font-family="Inter, system-ui, sans-serif" font-size="11px"
            fill="#94a3b8">— no propagation direction</text>
    </svg>
  </div>
</template>

<style scoped>
.hop-shells {
  display: flex;
  justify-content: center;
  width: 100%;
}
.hop-shells svg {
  width: 100%;
  height: auto;
  max-width: 820px;
}
</style>
