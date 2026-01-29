<script setup>
/**
 * MCP Architecture Visualization
 *
 * Redesigned with responsive CSS Grid/Flexbox layout.
 * Abandoned SVG for proper text scaling and layout control.
 */
</script>

<template>
  <div class="mcp-arch">
    <!-- Query Input -->
    <div class="layer layer-query">
      <div class="query-box">"What drugs target ROCK1?"</div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- Intelligent Gateway - The Key Innovation -->
    <div class="layer layer-gateway">
      <div class="gateway-header">INTELLIGENT GATEWAY</div>
      <div class="gateway-tools">
        <div class="tool">
          <div class="tool-name">ground_entity</div>
          <div class="tool-desc">semantic grounding</div>
        </div>
        <div class="tool">
          <div class="tool-name">suggest_endpoints</div>
          <div class="tool-desc">navigation hints</div>
        </div>
        <div class="tool">
          <div class="tool-name">call_endpoint</div>
          <div class="tool-desc">unified interface</div>
        </div>
        <div class="tool">
          <div class="tool-name">get_nav_schema</div>
          <div class="tool-desc">graph topology</div>
        </div>
      </div>
      <div class="compression">↓ 190+ autoclient functions ↓</div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- Processing Pipeline -->
    <div class="layer layer-processing">
      <div class="pipeline">
        <span class="step step-cyan">Validation</span>
        <span class="arrow">→</span>
        <span class="step step-purple">Cypher Exec</span>
        <span class="arrow">→</span>
        <span class="step step-purple">Pagination</span>
        <span class="arrow">→</span>
        <span class="step step-green">Enrichment</span>
      </div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- Knowledge Graph -->
    <div class="layer layer-knowledge">
      <div class="kg-title">CoGEx Knowledge Graph</div>
      <div class="kg-stats">30K genes • 15K drugs • 4K diseases • 25K pathways</div>
    </div>

    <!-- Key Callouts -->
    <div class="callouts">
      <div class="callout callout-tools">
        <span class="highlight">4 tools</span>
        <span class="sep">→</span>
        <span class="detail">190+ functions</span>
      </div>
      <div class="callout callout-disclosure">
        <span class="label">Progressive disclosure:</span>
        <span class="levels">minimal → standard → detailed → exploratory</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mcp-arch {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.4rem;
  width: 100%;
  max-width: 720px;
  margin: 0 auto;
  font-family: var(--font-sans, 'Inter', system-ui, sans-serif);
}

.layer {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

/* Query */
.query-box {
  background: #1a2540;
  border: 2px solid #4ecadf;
  border-radius: 8px;
  padding: 0.6rem 1.5rem;
  color: #4ecadf;
  font-size: 0.95rem;
  font-weight: 600;
  box-shadow: 0 0 15px rgba(78, 202, 223, 0.25);
}

/* Flow */
.flow-arrow {
  color: #64748b;
  font-size: 1rem;
  line-height: 1;
}

/* Gateway */
.layer-gateway {
  background: linear-gradient(180deg, rgba(191, 111, 247, 0.15) 0%, rgba(92, 45, 145, 0.08) 100%);
  border: 2px solid #bf6ff7;
  border-radius: 10px;
  padding: 0.6rem 0.8rem;
  width: 100%;
  box-shadow: 0 0 20px rgba(191, 111, 247, 0.15);
}

.gateway-header {
  color: #bf6ff7;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-align: center;
  margin-bottom: 0.5rem;
}

.gateway-tools {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.4rem;
}

.tool {
  background: #243052;
  border: 1px solid rgba(191, 111, 247, 0.6);
  border-radius: 5px;
  padding: 0.4rem 0.3rem;
  text-align: center;
  min-width: 0; /* Allow shrinking */
}

.tool-name {
  color: #e2e8f0;
  font-size: 0.6rem;
  font-weight: 600;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.tool-desc {
  color: #94a3b8;
  font-size: 0.5rem;
  margin-top: 0.1rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.compression {
  color: #bf6ff7;
  font-size: 0.6rem;
  font-weight: 600;
  text-align: center;
  margin-top: 0.4rem;
}

/* Processing */
.layer-processing {
  background: #121a2d;
  border: 1px solid #475569;
  border-radius: 6px;
  padding: 0.4rem 0.8rem;
}

.pipeline {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.35rem;
  flex-wrap: nowrap;
}

.step {
  background: #1a2540;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-size: 0.55rem;
  font-weight: 500;
  white-space: nowrap;
}

.step-cyan { color: #4ecadf; }
.step-purple { color: #bf6ff7; }
.step-green { color: #7dd629; }

.pipeline .arrow {
  color: #64748b;
  font-size: 0.6rem;
}

/* Knowledge */
.layer-knowledge {
  background: rgba(14, 98, 110, 0.15);
  border: 1.5px solid #4ecadf;
  border-radius: 6px;
  padding: 0.4rem 1rem;
  text-align: center;
}

.kg-title {
  color: #4ecadf;
  font-size: 0.75rem;
  font-weight: 600;
}

.kg-stats {
  color: #94a3b8;
  font-size: 0.55rem;
  margin-top: 0.15rem;
}

/* Callouts */
.callouts {
  display: flex;
  justify-content: center;
  gap: 0.75rem;
  margin-top: 0.4rem;
  flex-wrap: wrap;
}

.callout {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.3rem 0.6rem;
  border-radius: 5px;
  font-size: 0.6rem;
}

.callout-tools {
  background: rgba(191, 111, 247, 0.12);
  border: 1px solid rgba(191, 111, 247, 0.35);
}

.callout-tools .highlight {
  color: #bf6ff7;
  font-weight: 600;
}

.callout-tools .sep { color: #94a3b8; }
.callout-tools .detail { color: #e2e8f0; }

.callout-disclosure {
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(100, 116, 139, 0.35);
}

.callout-disclosure .label { color: #e2e8f0; }
.callout-disclosure .levels { color: #94a3b8; }
</style>
