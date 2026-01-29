<script setup lang="ts">
/**
 * Cover layout for INDRA CoGEx presentations
 *
 * Perceptual Engineering Principles Applied:
 *
 * 1. TEMPORAL CAUSALITY
 *    - Node pulse → particle emission → edge trace → destination pulse
 *    - Unified 6s cycle with staggered phases maintaining <100ms causality
 *    - Data (0s) → Knowledge (2s) → Insight (4s) → cycle
 *
 * 2. SPATIAL MAPPING
 *    - Layer labels positioned as column headers above each semantic layer
 *    - Direct spatial correspondence eliminates cognitive translation
 *
 * 3. VISUAL CHANNEL SEPARATION
 *    - Title: Primary focal (left, horizontal)
 *    - Subtitle: Secondary peripheral (left edge, vertical)
 *    - Network: Ambient awareness (right, animated)
 */

defineProps<{
  color?: 'gyori' | 'cogex' | 'bio' | 'dark'
}>()
</script>

<template>
  <div class="slidev-layout cover" :class="[`cover-${color || 'dark'}`]">
    <!-- DEPTH FIELD - Intricate layered space
         The knowledge graph extends infinitely; this slide is a window into it
         Multiple subtle layers create true perceptual depth -->
    <div class="depth-field" aria-hidden="true">
      <!-- Layer 1: Distant scattered nodes - the graph continues beyond view -->
      <svg class="distant-nodes" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid slice">
        <defs>
          <radialGradient id="distant-glow">
            <stop offset="0%" stop-color="currentColor" stop-opacity="0.6"/>
            <stop offset="100%" stop-color="currentColor" stop-opacity="0"/>
          </radialGradient>
        </defs>
        <!-- Scattered latent nodes - varying sizes suggest depth -->
        <circle cx="8%" cy="15%" r="2" fill="url(#distant-glow)" class="distant-data"/>
        <circle cx="3%" cy="45%" r="1.5" fill="url(#distant-glow)" class="distant-data"/>
        <circle cx="12%" cy="72%" r="2.5" fill="url(#distant-glow)" class="distant-data"/>
        <circle cx="6%" cy="88%" r="1" fill="url(#distant-glow)" class="distant-data"/>
        <circle cx="18%" cy="25%" r="1.5" fill="url(#distant-glow)" class="distant-knowledge"/>
        <circle cx="25%" cy="65%" r="2" fill="url(#distant-glow)" class="distant-knowledge"/>
        <circle cx="22%" cy="92%" r="1" fill="url(#distant-glow)" class="distant-knowledge"/>
        <circle cx="35%" cy="8%" r="1.5" fill="url(#distant-glow)" class="distant-insight"/>
        <circle cx="30%" cy="38%" r="2" fill="url(#distant-glow)" class="distant-insight"/>
        <circle cx="40%" cy="78%" r="1" fill="url(#distant-glow)" class="distant-insight"/>
        <circle cx="15%" cy="55%" r="1" fill="url(#distant-glow)" class="distant-data"/>
        <circle cx="28%" cy="18%" r="1.5" fill="url(#distant-glow)" class="distant-knowledge"/>
      </svg>

      <!-- Layer 2: Faint edge traces - connections extending to infinity -->
      <svg class="distant-edges" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid slice">
        <path d="M -50 200 Q 200 180 400 300" class="trace trace-data"/>
        <path d="M -30 500 Q 150 480 350 420" class="trace trace-data"/>
        <path d="M -20 800 Q 180 750 400 680" class="trace trace-knowledge"/>
        <path d="M 100 -20 Q 200 150 380 280" class="trace trace-knowledge"/>
        <path d="M 300 1100 Q 350 900 450 750" class="trace trace-insight"/>
        <path d="M 200 1120 Q 280 950 400 820" class="trace trace-data"/>
      </svg>

      <!-- Layer 3: Semantic color field - horizontal flow -->
      <div class="semantic-field"></div>

      <!-- Layer 4: Depth vignette - focal attention -->
      <div class="depth-vignette"></div>
    </div>

    <!-- Main content: title only -->
    <div class="cover-content">
      <slot name="title">
        <slot />
      </slot>
    </div>

    <!-- Vertical subtitle along left edge -->
    <div class="cover-subtitle">
      <slot name="subtitle" />
    </div>

    <!-- Semantic Network Visualization -->
    <div class="network-container" aria-hidden="true">
      <!-- Layer column headers - spatially aligned -->
      <div class="layer-headers">
        <span class="layer-header header-data">Data</span>
        <span class="layer-header header-knowledge">Knowledge</span>
        <span class="layer-header header-insights">Insights</span>
      </div>

      <svg class="network-svg" viewBox="0 0 360 340" preserveAspectRatio="xMidYMid meet">
        <defs>
          <!-- Atmospheric depth gradients - subtle semantic halos -->
          <radialGradient id="depth-data" cx="55" cy="170" r="120" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stop-color="#17a2b8" stop-opacity="0.08"/>
            <stop offset="100%" stop-color="#17a2b8" stop-opacity="0"/>
          </radialGradient>
          <radialGradient id="depth-knowledge" cx="180" cy="150" r="140" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stop-color="#9d4edd" stop-opacity="0.06"/>
            <stop offset="100%" stop-color="#9d4edd" stop-opacity="0"/>
          </radialGradient>
          <radialGradient id="depth-insight" cx="305" cy="180" r="120" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stop-color="#5bb900" stop-opacity="0.08"/>
            <stop offset="100%" stop-color="#5bb900" stop-opacity="0"/>
          </radialGradient>

          <!-- Glow filters for node illumination -->
          <filter id="glow-data" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feFlood flood-color="#17a2b8" flood-opacity="0.7"/>
            <feComposite in2="blur" operator="in"/>
            <feMerge>
              <feMergeNode/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          <filter id="glow-knowledge" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feFlood flood-color="#9d4edd" flood-opacity="0.7"/>
            <feComposite in2="blur" operator="in"/>
            <feMerge>
              <feMergeNode/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          <filter id="glow-insight" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feFlood flood-color="#5bb900" flood-opacity="0.7"/>
            <feComposite in2="blur" operator="in"/>
            <feMerge>
              <feMergeNode/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        <!-- LOCAL DEPTH REINFORCEMENT - supplements the full-slide atmosphere
             Provides additional focal depth around the network nodes
             Reduced opacity since primary depth is now at slide level -->
        <g class="depth-layer" opacity="0.5">
          <ellipse cx="55" cy="170" rx="50" ry="140" fill="url(#depth-data)"/>
          <ellipse cx="180" cy="170" rx="70" ry="160" fill="url(#depth-knowledge)"/>
          <ellipse cx="305" cy="180" rx="50" ry="140" fill="url(#depth-insight)"/>
        </g>

        <!-- LAYER 1: Data Sources (x: 40-70)
             Data nodes illuminate when particles ARRIVE (t=4s+) -->
        <g class="layer layer-data">
          <!-- Gene with ripple -->
          <circle cx="55" cy="60" r="9" class="ripple ripple-data" style="animation-delay: 4s"/>
          <circle cx="55" cy="60" r="9" class="node node-data node-data-1" filter="url(#glow-data)"/>
          <text x="55" y="82" class="node-label">Gene</text>

          <!-- Protein with ripple -->
          <circle cx="50" cy="130" r="8" class="ripple ripple-data" style="animation-delay: 4.1s"/>
          <circle cx="50" cy="130" r="8" class="node node-data node-data-2" filter="url(#glow-data)"/>
          <text x="50" y="152" class="node-label">Protein</text>

          <!-- Drug with ripple -->
          <circle cx="60" cy="200" r="7" class="ripple ripple-data" style="animation-delay: 4.2s"/>
          <circle cx="60" cy="200" r="7" class="node node-data node-data-3" filter="url(#glow-data)"/>
          <text x="60" y="222" class="node-label">Drug</text>

          <!-- Disease with ripple -->
          <circle cx="52" cy="270" r="8" class="ripple ripple-data" style="animation-delay: 4.3s"/>
          <circle cx="52" cy="270" r="8" class="node node-data node-data-4" filter="url(#glow-data)"/>
          <text x="52" y="292" class="node-label">Disease</text>
        </g>

        <!-- LAYER 2: Knowledge Processing (x: 160-200) -->
        <g class="layer layer-knowledge">
          <!-- Agent node with ripple -->
          <circle cx="175" cy="100" r="14" class="ripple ripple-knowledge ripple-agent"/>
          <circle cx="175" cy="100" r="14" class="node node-knowledge node-agent" filter="url(#glow-knowledge)"/>
          <text x="175" y="78" class="node-label-lg">Agent</text>

          <!-- CoGEx node with ripple -->
          <circle cx="185" cy="200" r="16" class="ripple ripple-knowledge ripple-cogex"/>
          <circle cx="185" cy="200" r="16" class="node node-knowledge node-cogex" filter="url(#glow-knowledge)"/>
          <text x="185" y="232" class="node-label-lg">CoGEx</text>

          <!-- Supporting nodes -->
          <circle cx="150" cy="150" r="4" class="node node-knowledge-small"/>
          <circle cx="210" cy="140" r="3" class="node node-knowledge-small"/>
          <circle cx="155" cy="250" r="3" class="node node-knowledge-small"/>
          <circle cx="215" cy="230" r="4" class="node node-knowledge-small"/>
        </g>

        <!-- LAYER 3: Insights (x: 290-320) -->
        <g class="layer layer-insights">
          <!-- Pathway with ripple -->
          <circle cx="305" cy="70" r="8" class="ripple ripple-insight ripple-insight-1"/>
          <circle cx="305" cy="70" r="8" class="node node-insight node-insight-1" filter="url(#glow-insight)"/>
          <text x="305" y="52" class="node-label">Pathway</text>

          <!-- Target with ripple -->
          <circle cx="310" cy="145" r="9" class="ripple ripple-insight ripple-insight-2"/>
          <circle cx="310" cy="145" r="9" class="node node-insight node-insight-2" filter="url(#glow-insight)"/>
          <text x="310" y="170" class="node-label">Target</text>

          <!-- Hypothesis with ripple -->
          <circle cx="300" cy="220" r="10" class="ripple ripple-insight ripple-insight-3"/>
          <circle cx="300" cy="220" r="10" class="node node-insight node-insight-3" filter="url(#glow-insight)"/>
          <text x="300" y="248" class="node-label">Hypothesis</text>

          <!-- Action with ripple -->
          <circle cx="308" cy="290" r="7" class="ripple ripple-insight ripple-insight-4"/>
          <circle cx="308" cy="290" r="7" class="node node-insight node-insight-4" filter="url(#glow-insight)"/>
          <text x="308" y="312" class="node-label">Action</text>
        </g>

        <!-- EDGES: Reflecting true data flow
             The Agent orchestrates: Agent ↔ CoGEx ↔ Data, then Agent → Insights

             Flow: Query received → Agent → CoGEx → Data (fan out)
                   Data → CoGEx → Agent (retrieval) → Insights (synthesis)
        -->
        <g class="edges">
          <!-- Agent ↔ CoGEx: Primary orchestration link -->
          <path id="edge-agent-cogex" d="M175 114 Q 180 155 185 184" class="edge edge-internal"/>
          <path id="edge-cogex-agent" d="M185 184 Q 180 155 175 114" class="edge edge-internal"/>

          <!-- CoGEx → Data: Query fan-out (CoGEx queries the data sources) -->
          <path id="edge-cogex-d1" d="M169 200 Q 115 140 64 60" class="edge edge-query"/>
          <path id="edge-cogex-d2" d="M169 200 Q 110 170 58 130" class="edge edge-query"/>
          <path id="edge-cogex-d3" d="M169 200 Q 115 200 67 200" class="edge edge-query"/>
          <path id="edge-cogex-d4" d="M169 205 Q 115 240 60 270" class="edge edge-query"/>

          <!-- Data → CoGEx: Results return -->
          <path id="edge-d1-cogex" d="M64 60 Q 115 140 169 200" class="edge edge-return"/>
          <path id="edge-d2-cogex" d="M58 130 Q 110 170 169 200" class="edge edge-return"/>
          <path id="edge-d3-cogex" d="M67 200 Q 115 200 169 200" class="edge edge-return"/>
          <path id="edge-d4-cogex" d="M60 270 Q 115 240 169 205" class="edge edge-return"/>

          <!-- Agent → Insights: Synthesis output -->
          <path id="edge-agent-i1" d="M189 100 Q 250 75 297 70" class="edge edge-output"/>
          <path id="edge-agent-i2" d="M189 100 Q 255 120 301 145" class="edge edge-output"/>
          <path id="edge-agent-i3" d="M189 105 Q 250 165 290 220" class="edge edge-output"/>
          <path id="edge-agent-i4" d="M185 114 Q 250 220 301 290" class="edge edge-output"/>

          <!-- Supporting knowledge mesh -->
          <line x1="175" y1="100" x2="210" y2="140" class="edge edge-internal-thin"/>
          <line x1="175" y1="100" x2="150" y2="150" class="edge edge-internal-thin"/>
          <line x1="185" y1="216" x2="155" y2="250" class="edge edge-internal-thin"/>
          <line x1="185" y1="216" x2="215" y2="230" class="edge edge-internal-thin"/>
        </g>

        <!-- PARTICLES: Query-Retrieval-Synthesis cycle (8s total)

             CRITICAL: All animations start at t=0 with NO begin offsets.
             Timing is controlled ONLY through keyTimes/keyPoints/values.
             This ensures SMIL and CSS animations stay synchronized across cycles.

             Timeline (all times as % of 8s):
             - 0.125 (1s):    Agent → CoGEx departs
             - 0.3125 (2.5s): CoGEx receives, CoGEx → Data departs
             - 0.5 (4s):      Data receives, Data → CoGEx departs
             - 0.6875 (5.5s): CoGEx receives data
             - 0.8125 (6.5s): Agent receives aggregated, Agent → Insights departs
             - 1.0 (8s):      Insights receive (= 0 of next cycle)
        -->
        <g class="particles">
          <!-- Phase 2: Agent → CoGEx (depart 1s/0.125, arrive 2.5s/0.3125) -->
          <circle r="3" fill="#9d4edd" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.125;0.14;0.30;0.3125;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.125;0.3125;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-agent-cogex"/>
            </animateMotion>
          </circle>

          <!-- Phase 3: CoGEx → Data (depart 2.5s/0.3125, arrive 4s/0.5) -->
          <circle r="2.5" fill="#9d4edd" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.3125;0.33;0.48;0.5;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.3125;0.5;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-cogex-d1"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#9d4edd" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.325;0.34;0.49;0.5125;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.325;0.5125;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-cogex-d2"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#9d4edd" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.3375;0.35;0.50;0.525;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.3375;0.525;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-cogex-d3"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#9d4edd" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.35;0.36;0.51;0.5375;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.35;0.5375;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-cogex-d4"/>
            </animateMotion>
          </circle>

          <!-- Phase 4: Data → CoGEx (depart 4s/0.5, arrive 5.5s/0.6875) -->
          <circle r="2.5" fill="#17a2b8" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.5;0.52;0.67;0.6875;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.5;0.6875;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-d1-cogex"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#17a2b8" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.5125;0.53;0.68;0.7;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.5125;0.7;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-d2-cogex"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#17a2b8" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.525;0.54;0.69;0.7125;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.525;0.7125;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-d3-cogex"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#17a2b8" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0;0" keyTimes="0;0.5375;0.55;0.70;0.725;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.5375;0.725;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-d4-cogex"/>
            </animateMotion>
          </circle>

          <!-- Phase 5: CoGEx → Agent (depart 5.5s/0.6875, arrive 6.5s/0.8125)
               Larger, brighter particle showing combined knowledge -->
          <circle r="4" fill="#bf6ff7" opacity="0" filter="url(#glow-knowledge)">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;1;1;0;0" keyTimes="0;0.6875;0.70;0.80;0.8125;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.6875;0.8125;1" keyPoints="0;0;1;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1">
              <mpath href="#edge-cogex-agent"/>
            </animateMotion>
          </circle>

          <!-- Phase 6: Agent → Insights (depart 6.5s/0.8125, arrive 8s/1.0) -->
          <circle r="2.5" fill="#5bb900" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0" keyTimes="0;0.8125;0.83;0.98;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.8125;1" keyPoints="0;0;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1">
              <mpath href="#edge-agent-i1"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#5bb900" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0" keyTimes="0;0.825;0.84;0.99;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.825;1" keyPoints="0;0;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1">
              <mpath href="#edge-agent-i2"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#5bb900" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0" keyTimes="0;0.8375;0.85;0.995;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.8375;1" keyPoints="0;0;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1">
              <mpath href="#edge-agent-i3"/>
            </animateMotion>
          </circle>
          <circle r="2" fill="#5bb900" opacity="0">
            <animate attributeName="opacity" dur="8s" repeatCount="indefinite"
              values="0;0;0.95;0.95;0" keyTimes="0;0.85;0.86;0.998;1"/>
            <animateMotion dur="8s" repeatCount="indefinite"
              keyTimes="0;0.85;1" keyPoints="0;0;1" calcMode="spline"
              keySplines="0 0 1 1;0.4 0 0.2 1">
              <mpath href="#edge-agent-i4"/>
            </animateMotion>
          </circle>
        </g>
      </svg>
    </div>

    <!-- Author/meta info -->
    <div class="cover-meta">
      <slot name="meta" />
    </div>
  </div>
</template>

<style scoped>
/* ==========================================================================
   LAYOUT STRUCTURE
   ========================================================================== */

.cover {
  position: relative;
  display: grid;
  grid-template-columns: 1fr auto;
  grid-template-rows: 1fr auto;
  min-height: 100%;
  padding: 3rem;
  overflow: hidden;
}

/* Dark theme backgrounds */
.cover-dark {
  background: linear-gradient(145deg, #080c14 0%, #0f1624 40%, #0a1520 100%);
  color: white;
}

/* ==========================================================================
   DEPTH FIELD - Intricate layered perceptual space

   Perceptual Engineering:
   - The knowledge graph extends infinitely beyond this window
   - Distant nodes and edges suggest continuation of the network
   - Semantic color flow spans entire slide (data→knowledge→insight)
   - Multiple layers create atmospheric perspective (distant=faint, near=vivid)
   - Intricate detail felt subliminally, not consciously noticed
   ========================================================================== */

.depth-field {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
}

/* Layer 1: Distant scattered nodes - graph continues beyond view */
.distant-nodes {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0.4;
}

.distant-data { color: #17a2b8; }
.distant-knowledge { color: #9d4edd; }
.distant-insight { color: #5bb900; }

/* Subtle twinkle animation for distant nodes */
.distant-nodes circle {
  animation: distant-pulse 8s ease-in-out infinite;
}

.distant-nodes circle:nth-child(odd) { animation-delay: -2s; }
.distant-nodes circle:nth-child(3n) { animation-delay: -4s; }
.distant-nodes circle:nth-child(5n) { animation-delay: -6s; }

@keyframes distant-pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.7; }
}

/* Layer 2: Faint edge traces extending to infinity */
.distant-edges {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0.15;
}

.trace {
  fill: none;
  stroke-width: 1;
  stroke-linecap: round;
}

.trace-data { stroke: #17a2b8; }
.trace-knowledge { stroke: #9d4edd; }
.trace-insight { stroke: #5bb900; }

/* Layer 3: Semantic color field - full-width horizontal flow */
.semantic-field {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    100deg,
    rgba(23, 162, 184, 0.035) 0%,
    rgba(23, 162, 184, 0.02) 20%,
    rgba(120, 80, 180, 0.025) 40%,
    rgba(157, 78, 221, 0.03) 55%,
    rgba(130, 120, 100, 0.015) 70%,
    rgba(91, 185, 0, 0.025) 85%,
    rgba(91, 185, 0, 0.035) 100%
  );
}

/* Layer 4: Depth vignette - darker edges create focal depth */
.depth-vignette {
  position: absolute;
  inset: 0;
  background:
    /* Bottom depth - ground plane recedes */
    linear-gradient(
      to top,
      rgba(5, 10, 20, 0.25) 0%,
      transparent 40%
    ),
    /* Corner vignette - peripheral darkening */
    radial-gradient(
      ellipse 120% 120% at 50% 50%,
      transparent 50%,
      rgba(5, 10, 20, 0.2) 100%
    );
}

.cover-gyori {
  background: linear-gradient(145deg, #050a0c 0%, #0a2830 50%, #0c3540 100%);
  color: white;
}

.cover-cogex {
  background: linear-gradient(145deg, #120a14 0%, #1f1228 50%, #2a1838 100%);
  color: white;
}

.cover-bio {
  background: linear-gradient(145deg, #060a04 0%, #12200a 50%, #1a3010 100%);
  color: white;
}

/* ==========================================================================
   TITLE CONTENT (Primary focal)
   ========================================================================== */

.cover-content {
  grid-column: 1;
  grid-row: 1;
  align-self: center;
  max-width: 480px;
  z-index: 10;
}

.cover-content :deep(h1) {
  font-size: 2.75rem;
  font-weight: 700;
  line-height: 1.1;
  margin: 0;
  color: white;
  letter-spacing: -0.025em;
}

/* Hide h2 in main content - it goes to subtitle slot */
.cover-content :deep(h2) {
  display: none;
}

.cover-content :deep(p) {
  font-size: 0.95rem;
  color: rgba(255, 255, 255, 0.7);
  margin-top: 1.5rem;
  line-height: 1.6;
}

/* ==========================================================================
   VERTICAL SUBTITLE (Secondary peripheral)
   ========================================================================== */

.cover-subtitle {
  position: absolute;
  left: 1rem;
  top: 35%;
  transform: translateY(-50%) rotate(-90deg);
  transform-origin: center center;
  white-space: nowrap;
  z-index: 10;
}

.cover-subtitle :deep(h2),
.cover-subtitle :deep(p) {
  font-size: 0.75rem;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.45);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin: 0;
}

/* ==========================================================================
   NETWORK VISUALIZATION (Ambient awareness)
   ========================================================================== */

.network-container {
  grid-column: 2;
  grid-row: 1 / -1;
  position: relative;
  width: 420px;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

/* Layer headers - spatially aligned as column headers */
.layer-headers {
  display: flex;
  justify-content: space-between;
  width: 360px;
  padding: 0 20px;
  margin-bottom: 0.5rem;
}

.layer-header {
  font-size: 0.6rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  opacity: 0.7;
}

.header-data {
  color: #17a2b8;
  width: 70px;
  text-align: center;
}

.header-knowledge {
  color: #9d4edd;
  width: 100px;
  text-align: center;
}

.header-insights {
  color: #5bb900;
  width: 70px;
  text-align: center;
}

.network-svg {
  width: 360px;
  height: 340px;
}

/* ==========================================================================
   NODE STYLES - Query-Retrieval-Synthesis Cycle (8s total)

   The Agent orchestrates the flow:
   - Phase 1 (0-1s):     Agent activates (query received from user)
   - Phase 2 (1-2.5s):   Agent → CoGEx (dispatch query)
   - Phase 3 (2.5-4s):   CoGEx → Data (fan out to entities)
   - Phase 4 (4-5.5s):   Data → CoGEx (results return)
   - Phase 5 (5.5-6.5s): CoGEx → Agent (aggregated knowledge)
   - Phase 6 (6.5-8s):   Agent → Insights (synthesis output)

   Key principle: Nodes illuminate when they RECEIVE information,
   showing the causal flow of knowledge through the system.
   ========================================================================== */

.node {
  /* No transitions that could cause jitter */
}

/* Agent: Hub orchestrator - activates first (t=0) and when results return (t=5.5s) */
.node-agent {
  fill: #9d4edd;
  opacity: 0.5;
  animation: pulse-agent 8s ease-in-out infinite;
}

/* CoGEx: Knowledge graph - receives query (t=1s) and results (t=4s) */
.node-cogex {
  fill: #9d4edd;
  opacity: 0.5;
  animation: pulse-cogex 8s ease-in-out infinite;
}

.node-knowledge {
  fill: #9d4edd;
  opacity: 0.5;
}

.node-knowledge-small {
  fill: #bf40bf;
  opacity: 0.25;
  animation: pulse-knowledge-subtle 8s ease-in-out infinite;
  animation-delay: 1.5s;
}

/* Data nodes: illuminate when particles ARRIVE (t=4s) */
.node-data {
  fill: #17a2b8;
  opacity: 0.5;
  animation: pulse-data 8s ease-in-out infinite;
}

.node-data-1 { animation-delay: 4s; }
.node-data-2 { animation-delay: 4.1s; }
.node-data-3 { animation-delay: 4.2s; }
.node-data-4 { animation-delay: 4.3s; }

/* Insight nodes: illuminate when particles ARRIVE (t=8s = cycle boundary) */
.node-insight {
  fill: #5bb900;
  opacity: 0.5;
  animation: pulse-insight 8s ease-in-out infinite;
}

.node-insight-1 { animation-delay: 0s; }
.node-insight-2 { animation-delay: 0.1s; }
.node-insight-3 { animation-delay: 0.2s; }
.node-insight-4 { animation-delay: 0.3s; }

/* ==========================================================================
   RIPPLE RINGS - Expanding energy absorption visualization

   Perceptual principle: Ripples emanate outward showing "energy received"
   without moving the node itself. Like a water droplet hitting a surface.
   ========================================================================== */

.ripple {
  fill: none;
  stroke-width: 2;
  opacity: 0;
  pointer-events: none;
  /* Critical: Scale from element center, not SVG origin */
  transform-box: fill-box;
  transform-origin: center;
}

.ripple-knowledge {
  stroke: #9d4edd;
  animation: ripple-expand 8s ease-out infinite;
}

/* Agent ripples at t=0 (query received) and t=6.5s (results received) */
.ripple-agent {
  animation-delay: 0s;
}

/* CoGEx ripples when particles ARRIVE: t=2.5s (query) and t=5.5s (data) */
.ripple-cogex {
  animation-delay: 2.5s;
}

.ripple-insight {
  stroke: #5bb900;
  animation: ripple-expand 8s ease-out infinite;
}

/* Insights ripple when particles ARRIVE (t=8s = 0s next cycle) */
.ripple-insight-1 { animation-delay: 0s; }
.ripple-insight-2 { animation-delay: 0.1s; }
.ripple-insight-3 { animation-delay: 0.2s; }
.ripple-insight-4 { animation-delay: 0.3s; }

/* Data nodes ripple when particles ARRIVE (t=4s) */
.ripple-data {
  stroke: #17a2b8;
  animation: ripple-expand 8s ease-out infinite;
}

/* Node labels */
.node-label {
  font-size: 7px;
  fill: rgba(255, 255, 255, 0.45);
  text-anchor: middle;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.node-label-lg {
  font-size: 9px;
  fill: rgba(255, 255, 255, 0.6);
  text-anchor: middle;
  font-weight: 600;
  letter-spacing: 0.05em;
}

/* ==========================================================================
   EDGE STYLES - Reflecting true data flow directions
   ========================================================================== */

.edge {
  fill: none;
  stroke-linecap: round;
}

/* Agent ↔ CoGEx: Primary orchestration channel */
.edge-internal {
  stroke: rgba(157, 78, 221, 0.35);
  stroke-width: 2;
}

.edge-internal-thin {
  stroke: rgba(157, 78, 221, 0.15);
  stroke-width: 1;
}

/* CoGEx → Data: Query paths (purple, outbound) */
.edge-query {
  stroke: rgba(157, 78, 221, 0.2);
  stroke-width: 1.5;
  stroke-dasharray: 4 2;
}

/* Data → CoGEx: Return paths (cyan, inbound results) */
.edge-return {
  stroke: rgba(23, 162, 184, 0.25);
  stroke-width: 1.5;
}

/* Agent → Insights: Synthesis output (green) */
.edge-output {
  stroke: rgba(91, 185, 0, 0.25);
  stroke-width: 1.5;
}

/* ==========================================================================
   PARTICLE STYLES - Query-Retrieval-Synthesis cycle (8s)
   ========================================================================== */

.particle {
  opacity: 0;
}

/* Query particles (purple): Agent → CoGEx → Data */
.particle-knowledge {
  fill: #9d4edd;
  animation: particle-fade-8s ease-in-out infinite;
  animation-duration: 8s;
}

/* Return particles (cyan): Data → CoGEx */
.particle-data {
  fill: #17a2b8;
  animation: particle-fade-8s ease-in-out infinite;
  animation-duration: 8s;
}

/* Aggregated knowledge particle (larger, brighter purple) */
.particle-aggregated {
  fill: #bf6ff7;
  filter: url(#glow-knowledge);
}

/* Synthesis particles (green): Agent → Insights */
.particle-insight {
  fill: #5bb900;
  animation: particle-fade-8s ease-in-out infinite;
  animation-duration: 8s;
}

/* ==========================================================================
   META INFO
   ========================================================================== */

.cover-meta {
  grid-column: 1;
  grid-row: 2;
  align-self: end;
  z-index: 10;
}

.cover-meta :deep(p),
.cover-meta :deep(div) {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.6);
  margin: 0;
}

.cover-meta :deep(strong) {
  color: rgba(255, 255, 255, 0.85);
  font-weight: 600;
}

.cover-meta :deep(em) {
  font-style: normal;
  color: rgba(255, 255, 255, 0.5);
}

/* ==========================================================================
   KEYFRAME ANIMATIONS - Query-Retrieval-Synthesis cycle (8s)

   Perceptual Engineering: NO SCALE TRANSFORMS
   - Nodes stay spatially fixed (no jumping)
   - Illumination through opacity + filter only
   - Nodes illuminate when particles ARRIVE (not depart)

   Timeline (particle ARRIVAL times):
   - t=0:    Agent receives query (external trigger)
   - t=2.5s: CoGEx receives query (Agent→CoGEx particle arrives)
   - t=4s:   Data nodes receive queries (CoGEx→Data particles arrive)
   - t=5.5s: CoGEx receives results (Data→CoGEx particles arrive)
   - t=6.5s: Agent receives aggregated knowledge (CoGEx→Agent arrives)
   - t=8s:   Insights receive synthesis (Agent→Insights arrive = cycle boundary)
   ========================================================================== */

/* Agent: Dual activation - receives query (t=0) and results (t=6.5s/81.25%) */
@keyframes pulse-agent {
  0%, 2% {
    opacity: 1;
    filter: url(#glow-knowledge);
  }
  10%, 79% {
    opacity: 0.5;
    filter: none;
  }
  81.25%, 85% {
    opacity: 1;
    filter: url(#glow-knowledge);
  }
  93%, 100% {
    opacity: 0.5;
    filter: none;
  }
}

/* CoGEx: Dual activation - receives query (t=2.5s/31.25%) and data results (t=5.5s/68.75%) */
@keyframes pulse-cogex {
  0%, 29% {
    opacity: 0.5;
    filter: none;
  }
  31.25%, 35% {
    opacity: 1;
    filter: url(#glow-knowledge);
  }
  43%, 66% {
    opacity: 0.5;
    filter: none;
  }
  68.75%, 73% {
    opacity: 1;
    filter: url(#glow-knowledge);
  }
  81%, 100% {
    opacity: 0.5;
    filter: none;
  }
}

/* Data nodes: Illuminate when queried (with animation-delay offset) */
@keyframes pulse-data {
  0%, 2% {
    opacity: 1;
    filter: url(#glow-data);
  }
  15%, 100% {
    opacity: 0.5;
    filter: none;
  }
}

/* Insight nodes: Illuminate when Agent synthesizes (with animation-delay offset) */
@keyframes pulse-insight {
  0%, 2% {
    opacity: 1;
    filter: url(#glow-insight);
  }
  15%, 100% {
    opacity: 0.5;
    filter: none;
  }
}

/* Subtle ambient pulse for supporting knowledge nodes */
@keyframes pulse-knowledge-subtle {
  0%, 50% {
    opacity: 0.2;
  }
  12.5%, 50% {
    opacity: 0.4;
  }
}

/* Particle visibility for 8s cycle */
@keyframes particle-fade-8s {
  0% { opacity: 0; }
  5% { opacity: 0.95; }
  18% { opacity: 0.95; }
  22% { opacity: 0; }
  100% { opacity: 0; }
}

/* Ripple expand: energy absorption visualization (for 8s cycle)
   - Starts at node size, expands outward
   - Fades as it expands (energy dissipation)
   - Shorter relative duration since cycle is longer */
@keyframes ripple-expand {
  0% {
    opacity: 0.8;
    stroke-width: 2;
    transform: scale(1);
  }
  3% {
    opacity: 0.6;
    stroke-width: 2;
    transform: scale(1.3);
  }
  8% {
    opacity: 0.2;
    stroke-width: 1.5;
    transform: scale(2);
  }
  12% {
    opacity: 0;
    stroke-width: 1;
    transform: scale(2.5);
  }
  13%, 100% {
    opacity: 0;
    transform: scale(1);
  }
}

/* ==========================================================================
   REDUCED MOTION
   ========================================================================== */

@media (prefers-reduced-motion: reduce) {
  .node, .particle {
    animation: none !important;
  }
  .node-data { opacity: 0.6; }
  .node-knowledge { opacity: 0.6; }
  .node-insight { opacity: 0.6; }
  .particle { opacity: 0; }
}
</style>
