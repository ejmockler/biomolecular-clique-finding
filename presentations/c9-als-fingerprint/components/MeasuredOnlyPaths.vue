<template>
  <div class="measured-only-paths">
    <svg viewBox="0 0 820 360" xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Undirected regulatory distance may connect measured proteins only through measured intermediates. A path through an unmeasured node is excluded.">

      <!-- ============ FORBIDDEN direct route (upper) ============ -->
      <!-- anchor -> ghost (faded dashed) -->
      <line x1="150" y1="118" x2="383" y2="86" stroke="#64748b" stroke-width="1.5"
            stroke-dasharray="6 5" />
      <!-- ghost -> target (faded dashed) -->
      <line x1="437" y1="86" x2="654" y2="142" stroke="#64748b" stroke-width="1.5"
            stroke-dasharray="6 5" />

      <!-- ghost node: dashed gray outline -->
      <circle cx="410" cy="86" r="24" fill="#0a0f1a" stroke="#64748b" stroke-width="1.6"
              stroke-dasharray="4 4" />
      <!-- struck-through × over ghost -->
      <line x1="394" y1="70" x2="426" y2="102" stroke="#64748b" stroke-width="2.4" />
      <line x1="426" y1="70" x2="394" y2="102" stroke="#64748b" stroke-width="2.4" />

      <!-- ghost labels -->
      <text x="410" y="36" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="13px" font-weight="600" fill="#94a3b8">unmeasured "ghost"</text>
      <text x="410" y="53" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">can't step here</text>

      <!-- forbidden route badge -->
      <text x="250" y="78" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" font-weight="600" fill="#64748b">path through an unmeasured node — excluded</text>

      <!-- ============ LEGAL detour (lower) ============ -->
      <!-- anchor -> measured -->
      <line x1="150" y1="156" x2="376" y2="244" stroke="#4ecadf" stroke-width="2.4"
            />
      <!-- measured -> target -->
      <line x1="444" y1="244" x2="630" y2="170" stroke="#4ecadf" stroke-width="2.4"
            />

      <!-- legal route badge -->
      <text x="300" y="300" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" font-weight="600" fill="#4ecadf">valid undirected path — measured proteins only</text>

      <!-- measured stepping node -->
      <circle cx="410" cy="244" r="22" fill="#4ecadf" />
      <text x="410" y="248" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="10px" font-weight="700" fill="#0a0f1a">measured</text>
      <text x="410" y="290" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#4ecadf">measured — allowed</text>

      <!-- ============ anchor + two-hop partner ============ -->
      <!-- anchor -->
      <circle cx="128" cy="138" r="26" fill="#4ecadf" />
      <text x="128" y="142" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" font-weight="700" fill="#0a0f1a">anchor</text>
      <text x="128" y="184" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">distance reference</text>

      <!-- two-hop partner -->
      <circle cx="676" cy="156" r="26" fill="#1a2540" stroke="#4ecadf" stroke-width="2" />
      <text x="676" y="160" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="11px" font-weight="700" fill="#4ecadf">partner</text>
      <text x="676" y="202" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
            font-size="12px" fill="#94a3b8">two undirected edges away</text>
    </svg>
  </div>
</template>

<style scoped>
.measured-only-paths {
  display: flex;
  justify-content: center;
  width: 100%;
}
.measured-only-paths svg {
  width: 100%;
  height: auto;
  max-width: 820px;
}
</style>
