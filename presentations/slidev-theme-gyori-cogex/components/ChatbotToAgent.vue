<script setup>
/**
 * From Chatbot to Agent - Perceptual Engineering
 *
 * Core insight: Adding tools transforms a guesser into a researcher.
 * Same question, radically different answers.
 *
 * Design:
 * - Show the SAME question to both
 * - Chatbot: memory-only answer (uncertain, maybe wrong)
 * - Plus sign: the transformation (+ TOOLS)
 * - Agent: grounded answer (specific, current)
 * - The concrete example is the hero
 */
</script>

<template>
  <div class="evolution-viz">
    <!-- The Question (same for both) -->
    <div class="question-bar">
      <span class="question-label">You ask:</span>
      <span class="question-text">"What drugs target ROCK1?"</span>
    </div>

    <!-- The Comparison -->
    <div class="comparison">
      <!-- Chatbot Side -->
      <div class="side chatbot-side">
        <div class="side-header">
          <span class="side-icon">💬</span>
          <span class="side-title">Chatbot</span>
        </div>
        <div class="side-subtitle">LLM alone</div>

        <div class="response-box chatbot-response">
          <div class="response-text">
            "ROCK1 is targeted by fasudil and Y-27632, which are commonly used in research..."
          </div>
          <div class="response-warning">
            <span class="warning-icon">⚠️</span>
            <span class="warning-text">From 2021 training data</span>
          </div>
        </div>

        <div class="verdict chatbot-verdict">
          <span class="verdict-label">Static knowledge</span>
          <span class="verdict-detail">May be outdated or incomplete</span>
        </div>
      </div>

      <!-- The Transformation -->
      <div class="transform">
        <div class="transform-plus">+</div>
        <div class="transform-label">TOOLS</div>
      </div>

      <!-- Agent Side -->
      <div class="side agent-side">
        <div class="side-header">
          <span class="side-icon">🤖</span>
          <span class="side-title">Agent</span>
        </div>
        <div class="side-subtitle">LLM + knowledge graph</div>

        <div class="response-box agent-response">
          <div class="response-action">
            <span class="action-icon">🔍</span>
            <span class="action-text">query_database("ROCK1")</span>
          </div>
          <div class="response-text">
            "ROCK1 has <strong>38 known modulators</strong> including fasudil, ripasudil, netarsudil..."
          </div>
        </div>

        <div class="verdict agent-verdict">
          <span class="verdict-label">Dynamic knowledge</span>
          <span class="verdict-detail">Current, verifiable, complete</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.evolution-viz {
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
}

/* ========== QUESTION BAR ========== */
.question-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  padding: 0.6rem 1.2rem;
  background: rgba(78, 202, 223, 0.1);
  border: 1px solid rgba(78, 202, 223, 0.3);
  border-radius: 8px;
}

.question-label {
  font-size: 0.7rem;
  color: var(--text-tertiary, #64748b);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.question-text {
  font-size: 1rem;
  font-weight: 600;
  color: #4ecadf;
}

/* ========== COMPARISON ========== */
.comparison {
  display: flex;
  align-items: stretch;
  gap: 0;
}

.side {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  padding: 1rem;
  border-radius: 10px;
}

.chatbot-side {
  background: linear-gradient(180deg, rgba(100, 116, 139, 0.15) 0%, rgba(100, 116, 139, 0.05) 100%);
  border: 1px solid rgba(100, 116, 139, 0.3);
  border-right: none;
  border-radius: 10px 0 0 10px;
}

.agent-side {
  background: linear-gradient(180deg, rgba(191, 111, 247, 0.12) 0%, rgba(191, 111, 247, 0.04) 100%);
  border: 2px solid rgba(191, 111, 247, 0.5);
  border-radius: 0 10px 10px 0;
}

/* Side Header */
.side-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.side-icon {
  font-size: 1.3rem;
}

.side-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-primary, #e2e8f0);
}

.side-subtitle {
  font-size: 0.7rem;
  color: var(--text-tertiary, #64748b);
  margin-top: -0.3rem;
}

/* Response Box */
.response-box {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.7rem;
  border-radius: 6px;
  min-height: 90px;
}

.chatbot-response {
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(100, 116, 139, 0.3);
}

.agent-response {
  background: rgba(191, 111, 247, 0.1);
  border: 1px solid rgba(191, 111, 247, 0.3);
}

.response-text {
  font-size: 0.75rem;
  color: var(--text-secondary, #94a3b8);
  line-height: 1.5;
  font-style: italic;
}

.response-text strong {
  color: #22c55e;
  font-style: normal;
  font-weight: 600;
}

.response-warning {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.25rem 0.5rem;
  background: rgba(245, 158, 11, 0.15);
  border-radius: 4px;
  width: fit-content;
}

.warning-icon {
  font-size: 0.7rem;
}

.warning-text {
  font-size: 0.6rem;
  color: #fbbf24;
}

.response-action {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.5rem;
  background: rgba(191, 111, 247, 0.2);
  border-radius: 4px;
  width: fit-content;
}

.action-icon {
  font-size: 0.7rem;
}

.action-text {
  font-size: 0.65rem;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
  color: #bf6ff7;
}

/* Verdict */
.verdict {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
  padding-top: 0.5rem;
  border-top: 1px solid rgba(100, 116, 139, 0.2);
  margin-top: auto;
}

.verdict-label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.chatbot-verdict .verdict-label {
  color: #94a3b8;
}

.agent-verdict .verdict-label {
  color: #22c55e;
}

.verdict-detail {
  font-size: 0.65rem;
  color: var(--text-tertiary, #64748b);
}

/* ========== TRANSFORM ========== */
.transform {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 0 0.8rem;
  gap: 0.2rem;
}

.transform-plus {
  font-size: 2rem;
  font-weight: 300;
  color: #bf6ff7;
  line-height: 1;
}

.transform-label {
  font-size: 0.6rem;
  font-weight: 700;
  color: #bf6ff7;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
</style>
