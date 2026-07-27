<template>
  <div class="first-test">
    <svg viewBox="0 0 800 170" xmlns="http://www.w3.org/2000/svg">
      <!-- a group of proteins -->
      <g fill="#4ecadf">
        <circle cx="95" cy="62" r="6" /><circle cx="126" cy="58" r="6" />
        <circle cx="110" cy="84" r="6" /><circle cx="140" cy="82" r="6" />
        <circle cx="98" cy="106" r="6" /><circle cx="129" cy="104" r="6" />
      </g>
      <text x="118" y="148" text-anchor="middle" fill="#94a3b8"
            font-family="Inter, system-ui, sans-serif" font-size="13px">a group of proteins</text>

      <!-- arrow into the test -->
      <line x1="182" y1="84" x2="246" y2="84" stroke="#64748b" stroke-width="2" />
      <polyline points="238,78 248,84 238,90" fill="none" stroke="#64748b" stroke-width="2" />

      <!-- the test box -->
      <rect x="258" y="50" width="270" height="68" rx="8" fill="#121a2d" stroke="#475569" stroke-width="1.5" />
      <text x="393" y="42" text-anchor="middle" fill="#94a3b8"
            font-family="Inter, system-ui, sans-serif" font-size="11px" letter-spacing="1.5">ROAST — A GENE-SET TEST</text>
      <text x="393" y="80" text-anchor="middle" fill="#cbd5e1"
            font-family="Inter, system-ui, sans-serif" font-size="13px">is this group</text>
      <text x="393" y="99" text-anchor="middle" fill="#cbd5e1"
            font-family="Inter, system-ui, sans-serif" font-size="13px">disturbed in C9?</text>

      <!-- arrow out -->
      <line x1="540" y1="84" x2="604" y2="84" stroke="#64748b" stroke-width="2" />
      <polyline points="596,78 606,84 596,90" fill="none" stroke="#64748b" stroke-width="2" />

      <!-- the loud yes -->
      <text x="680" y="96" fill="#f1f5f9"
            font-family="Inter, system-ui, sans-serif" font-size="32px" font-weight="800">YES</text>
      <text x="694" y="126" text-anchor="middle" fill="#94a3b8"
            font-family="Inter, system-ui, sans-serif" font-size="13px">"significant"</text>
    </svg>
  </div>
</template>

<style scoped>
.first-test { display: flex; justify-content: center; width: 100%; }
.first-test svg { width: 100%; height: auto; max-width: 820px; }
</style>
