<script setup>
/**
 * MCP Explainer - Perceptual Engineering
 *
 * Core insight: MCP is a UNIVERSAL ADAPTER that eliminates the need
 * to learn different query languages for different tools.
 *
 * Design:
 * - Top: The human question (what you SAY)
 * - Middle: MCP as the universal port (USB visual metaphor)
 * - Bottom: The many tools it connects to
 * - The translation is the hero
 */
</script>

<template>
  <div class="mcp-explainer">
    <!-- The Human Side -->
    <div class="human-side">
      <div class="human-input">
        <span class="input-label">You say:</span>
        <span class="input-text">"What genes interact with LRRK2?"</span>
      </div>
    </div>

    <!-- The Universal Adapter -->
    <div class="adapter-section">
      <div class="adapter-visual">
        <div class="port-label">MCP</div>
        <div class="port">
          <div class="port-icon">⚡</div>
        </div>
        <div class="port-subtitle">Universal Adapter</div>
      </div>

      <div class="translation">
        <div class="translation-arrow">↓</div>
        <div class="translation-detail">
          <span class="from">Natural language</span>
          <span class="to-icon">→</span>
          <span class="to">Structured queries</span>
        </div>
      </div>
    </div>

    <!-- The Tool Connections -->
    <div class="tools-section">
      <div class="connection-tree">
        <div class="trunk"></div>
        <div class="branches">
          <div class="branch branch-1"></div>
          <div class="branch branch-2"></div>
          <div class="branch branch-3"></div>
          <div class="branch branch-4"></div>
        </div>
      </div>

      <div class="tools-row">
        <div class="tool">
          <span class="tool-icon">🧬</span>
          <span class="tool-name">CoGEx</span>
        </div>
        <div class="tool">
          <span class="tool-icon">💊</span>
          <span class="tool-name">DrugBank</span>
        </div>
        <div class="tool">
          <span class="tool-icon">📄</span>
          <span class="tool-name">PubMed</span>
        </div>
        <div class="tool">
          <span class="tool-icon">🔬</span>
          <span class="tool-name">UniProt</span>
        </div>
      </div>
    </div>

    <!-- The Payoff -->
    <div class="payoff">
      <div class="payoff-before">
        <span class="code-sample">MATCH (g:Gene)-[:interacts_with]->(g2:Gene) WHERE g.name = 'LRRK2'...</span>
      </div>
      <div class="payoff-message">
        <span class="no-more">You never write this.</span>
        <span class="just-ask">Just ask in English.</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mcp-explainer {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  max-width: 700px;
  margin: 0 auto;
}

/* ========== HUMAN SIDE ========== */
.human-side {
  width: 100%;
}

.human-input {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  padding: 0.5rem 1.2rem;
  background: rgba(78, 202, 223, 0.1);
  border: 1px solid rgba(78, 202, 223, 0.3);
  border-radius: 8px;
}

.input-label {
  font-size: 0.7rem;
  color: var(--text-tertiary, #64748b);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.input-text {
  font-size: 0.95rem;
  font-weight: 600;
  color: #4ecadf;
}

/* ========== ADAPTER SECTION ========== */
.adapter-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
}

.adapter-visual {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.15rem;
}

.port-label {
  font-size: 0.65rem;
  font-weight: 700;
  color: #bf6ff7;
  text-transform: uppercase;
  letter-spacing: 0.15em;
}

.port {
  width: 60px;
  height: 36px;
  background: linear-gradient(180deg, #bf6ff7 0%, #9333ea 100%);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 20px rgba(191, 111, 247, 0.4);
}

.port-icon {
  font-size: 1.3rem;
}

.port-subtitle {
  font-size: 0.6rem;
  color: var(--text-tertiary, #64748b);
  font-style: italic;
}

.translation {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.1rem;
}

.translation-arrow {
  font-size: 1.2rem;
  color: #bf6ff7;
  line-height: 1;
}

.translation-detail {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.25rem 0.6rem;
  background: rgba(191, 111, 247, 0.1);
  border-radius: 4px;
}

.from {
  font-size: 0.65rem;
  color: #4ecadf;
}

.to-icon {
  font-size: 0.6rem;
  color: var(--text-tertiary, #64748b);
}

.to {
  font-size: 0.65rem;
  color: #bf6ff7;
}

/* ========== TOOLS SECTION ========== */
.tools-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0;
  width: 100%;
}

.connection-tree {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 380px;
}

.trunk {
  width: 2px;
  height: 10px;
  background: linear-gradient(180deg, #bf6ff7, rgba(191, 111, 247, 0.6));
}

.branches {
  display: flex;
  justify-content: space-between;
  width: 100%;
  height: 16px;
  position: relative;
  padding: 0 38px;
  box-sizing: border-box;
}

.branches::before {
  content: '';
  position: absolute;
  top: 0;
  left: 38px;
  right: 38px;
  height: 2px;
  background: rgba(191, 111, 247, 0.5);
}

.branch {
  width: 2px;
  height: 100%;
  background: linear-gradient(180deg, rgba(191, 111, 247, 0.6), rgba(191, 111, 247, 0.3));
}

.tools-row {
  display: flex;
  justify-content: space-between;
  width: 380px;
}

.tool {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.2rem;
  padding: 0.4rem 0.6rem;
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(100, 116, 139, 0.3);
  border-radius: 6px;
  min-width: 65px;
}

.tool-icon {
  font-size: 1.1rem;
}

.tool-name {
  font-size: 0.6rem;
  color: var(--text-secondary, #94a3b8);
  font-weight: 500;
}

/* ========== PAYOFF ========== */
.payoff {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding-top: 0.8rem;
  border-top: 1px solid rgba(100, 116, 139, 0.2);
  width: 100%;
}

.payoff-before {
  padding: 0.3rem 0.6rem;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  border-left: 2px solid #ef4444;
}

.code-sample {
  font-size: 0.7rem;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
  color: #64748b;
  text-decoration: line-through;
  opacity: 0.8;
}

.payoff-message {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.no-more {
  font-size: 0.75rem;
  color: #ef4444;
  font-weight: 500;
}

.just-ask {
  font-size: 0.9rem;
  color: #22c55e;
  font-weight: 700;
}
</style>
