<script setup>
/**
 * LLM Capabilities - Perceptual Engineering
 *
 * Core insight: LLMs can REASON but can't ACCESS.
 * This tension sets up why we need agents with tools.
 *
 * Design:
 * - Simple metaphor at top: reasoning engine visual
 * - Clear Can/Cannot split with color coding
 * - The "Cannot" is the hook for the next slide
 */
</script>

<template>
  <div class="llm-capabilities">
    <!-- The Core Insight -->
    <div class="core-insight">
      <div class="insight-visual">
        <span class="brain">🧠</span>
        <span class="plus">+</span>
        <span class="text-icon">📚</span>
        <span class="equals">=</span>
        <span class="pattern">Pattern Matcher</span>
      </div>
      <div class="insight-text">
        Trained on scientific text, learns <em>relationships</em> between concepts
      </div>
    </div>

    <!-- Can / Cannot Split -->
    <div class="capabilities-split">
      <!-- CAN -->
      <div class="capability can">
        <div class="cap-header">
          <span class="cap-icon">✓</span>
          <span class="cap-label">Can do</span>
        </div>
        <div class="cap-items">
          <div class="cap-item">Interpret natural language</div>
          <div class="cap-item">Reason about biology</div>
          <div class="cap-item">Explain relationships</div>
        </div>
      </div>

      <!-- Divider -->
      <div class="cap-divider"></div>

      <!-- CANNOT -->
      <div class="capability cannot">
        <div class="cap-header">
          <span class="cap-icon">✗</span>
          <span class="cap-label">Cannot do</span>
        </div>
        <div class="cap-items">
          <div class="cap-item highlight">Access real-time data</div>
          <div class="cap-item">Guarantee accuracy</div>
          <div class="cap-item">Replace domain expertise</div>
        </div>
      </div>
    </div>

    <!-- The Hook -->
    <div class="hook">
      <span class="hook-text">So how do we ground LLM reasoning in real data?</span>
    </div>
  </div>
</template>

<style scoped>
.llm-capabilities {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  width: 100%;
  max-width: 560px;
}

/* ========== CORE INSIGHT ========== */
.core-insight {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.6rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(100, 116, 139, 0.2);
}

.insight-visual {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.1rem;
}

.brain, .text-icon {
  font-size: 1.4rem;
}

.plus, .equals {
  color: var(--text-tertiary, #64748b);
  font-weight: 300;
}

.pattern {
  font-size: 1rem;
  font-weight: 600;
  color: #bf6ff7;
  padding: 0.2rem 0.6rem;
  background: rgba(191, 111, 247, 0.15);
  border-radius: 4px;
}

.insight-text {
  font-size: 0.85rem;
  color: var(--text-secondary, #94a3b8);
  line-height: 1.5;
}

.insight-text em {
  color: var(--text-primary, #e2e8f0);
  font-style: normal;
  font-weight: 500;
}

/* ========== CAPABILITIES SPLIT ========== */
.capabilities-split {
  display: flex;
  gap: 1.2rem;
  align-items: flex-start;
}

.capability {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.cap-header {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.cap-icon {
  font-size: 1rem;
  font-weight: 700;
}

.can .cap-icon { color: #22c55e; }
.cannot .cap-icon { color: #ef4444; }

.cap-label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.can .cap-label { color: #22c55e; }
.cannot .cap-label { color: #ef4444; }

.cap-items {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.cap-item {
  font-size: 0.8rem;
  color: var(--text-secondary, #94a3b8);
  padding: 0.3rem 0.5rem;
  border-radius: 4px;
}

.can .cap-item {
  background: rgba(34, 197, 94, 0.08);
  border-left: 2px solid rgba(34, 197, 94, 0.4);
}

.cannot .cap-item {
  background: rgba(239, 68, 68, 0.08);
  border-left: 2px solid rgba(239, 68, 68, 0.4);
}

.cannot .cap-item.highlight {
  background: rgba(239, 68, 68, 0.15);
  border-left: 2px solid #ef4444;
  color: #fca5a5;
  font-weight: 500;
}

.cap-divider {
  width: 1px;
  min-height: 100px;
  background: linear-gradient(180deg, transparent, rgba(100, 116, 139, 0.3), transparent);
}

/* ========== HOOK ========== */
.hook {
  padding-top: 0.8rem;
  border-top: 1px solid rgba(78, 202, 223, 0.2);
}

.hook-text {
  font-size: 0.9rem;
  color: #4ecadf;
  font-weight: 500;
  font-style: italic;
}
</style>
