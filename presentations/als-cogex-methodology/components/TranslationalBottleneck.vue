<script setup>
/**
 * The Translational Bottleneck - Perceptual Engineering v2
 *
 * Core insight: We produce data easily, but hit a WALL at interpretation.
 *
 * Design:
 * - Left: What we produce (compact, green, "easy")
 * - Center: THE WALL (the bottleneck - a barrier, not a funnel)
 * - Right: Questions we can't answer (prominent, red, the pain)
 * - Bottom: The sad result
 *
 * The THREE QUESTIONS are the hero - everyone recognizes this pain.
 */
</script>

<template>
  <div class="bottleneck-viz">
    <div class="main-content">
      <!-- What We Produce (Easy, Compact) -->
      <div class="produce-section">
        <div class="section-header green">We produce</div>
        <div class="produce-stack">
          <div class="produce-item">
            <span class="item-icon">📊</span>
            <span class="item-text">Terabytes of omics data</span>
          </div>
          <div class="produce-item">
            <span class="item-icon">📈</span>
            <span class="item-text">Differential expression</span>
          </div>
          <div class="produce-item">
            <span class="item-icon">📋</span>
            <span class="item-text">500 dysregulated genes</span>
          </div>
        </div>
        <div class="produce-verdict">
          <span class="check">✓</span> This part is easy
        </div>
      </div>

      <!-- THE WALL -->
      <div class="wall-section">
        <div class="wall"></div>
        <div class="wall-label">THE<br/>GAP</div>
      </div>

      <!-- What We Need to Know (The Pain Point) -->
      <div class="need-section">
        <div class="section-header red">We need to know</div>
        <div class="question-stack">
          <div class="question">
            <span class="q-num">1</span>
            <span class="q-text">What do these proteins <em>do</em> together?</span>
          </div>
          <div class="question">
            <span class="q-num">2</span>
            <span class="q-text">Which findings are <em>actionable</em>?</span>
          </div>
          <div class="question">
            <span class="q-num">3</span>
            <span class="q-text">What drugs <em>already target</em> these?</span>
          </div>
        </div>
        <div class="need-verdict">
          <span class="x">✗</span> This is where we fail
        </div>
      </div>
    </div>

    <!-- The Result -->
    <div class="result-section">
      <div class="result-arrow">↓</div>
      <div class="result-box">
        <span class="result-text">Supplementary Table S7: 847 rows that nobody will ever read</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bottleneck-viz {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  width: 100%;
  max-width: 920px;
  margin: 0 auto;
}

.main-content {
  display: flex;
  align-items: stretch;
  gap: 0;
  width: 100%;
}

/* ========== SECTION HEADERS ========== */
.section-header {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  text-align: center;
  margin-bottom: 0.7rem;
}

.section-header.green { color: #22c55e; }
.section-header.red { color: #ef4444; }

/* ========== PRODUCE SECTION (Left) ========== */
.produce-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background: linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.02) 100%);
  border: 1px solid rgba(34, 197, 94, 0.25);
  border-radius: 12px 0 0 12px;
  border-right: none;
}

.produce-stack {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.produce-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.6rem;
  background: rgba(34, 197, 94, 0.1);
  border-radius: 6px;
}

.item-icon {
  font-size: 0.9rem;
}

.item-text {
  font-size: 0.7rem;
  color: var(--text-secondary, #94a3b8);
}

.produce-verdict {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  margin-top: auto;
  padding-top: 0.7rem;
  font-size: 0.65rem;
  color: #22c55e;
  font-weight: 500;
}

.check {
  font-size: 0.9rem;
}

/* ========== THE WALL (Center) ========== */
.wall-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 0 0.3rem;
  position: relative;
}

.wall {
  width: 8px;
  height: 100%;
  min-height: 160px;
  background: linear-gradient(180deg, #ef4444 0%, #dc2626 50%, #ef4444 100%);
  border-radius: 4px;
  box-shadow:
    0 0 20px rgba(239, 68, 68, 0.4),
    0 0 40px rgba(239, 68, 68, 0.2);
}

.wall-label {
  position: absolute;
  font-size: 0.55rem;
  font-weight: 800;
  color: #ef4444;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  text-align: center;
  line-height: 1.3;
  background: var(--surface-0, #0a0f1a);
  padding: 0.3rem 0.4rem;
  border-radius: 4px;
}

/* ========== NEED SECTION (Right - Hero) ========== */
.need-section {
  flex: 1.3;
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.03) 100%);
  border: 1.5px solid rgba(239, 68, 68, 0.4);
  border-radius: 0 12px 12px 0;
  border-left: none;
}

.question-stack {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.question {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.5rem 0.7rem;
  background: rgba(239, 68, 68, 0.12);
  border-radius: 8px;
  border-left: 3px solid #ef4444;
}

.q-num {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 1.4rem;
  height: 1.4rem;
  background: #ef4444;
  color: white;
  font-size: 0.7rem;
  font-weight: 700;
  border-radius: 50%;
  flex-shrink: 0;
}

.q-text {
  font-size: 0.8rem;
  color: var(--text-primary, #f1f5f9);
  line-height: 1.4;
}

.q-text em {
  color: #f97316;
  font-style: normal;
  font-weight: 600;
}

.need-verdict {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  margin-top: auto;
  padding-top: 0.7rem;
  font-size: 0.65rem;
  color: #ef4444;
  font-weight: 500;
}

.x {
  font-size: 0.9rem;
}

/* ========== RESULT SECTION ========== */
.result-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
}

.result-arrow {
  font-size: 1.2rem;
  color: var(--text-tertiary, #64748b);
}

.result-box {
  padding: 0.6rem 1.2rem;
  background: rgba(100, 116, 139, 0.1);
  border: 1px dashed rgba(100, 116, 139, 0.4);
  border-radius: 8px;
}

.result-text {
  font-size: 0.75rem;
  color: var(--text-tertiary, #64748b);
  font-style: italic;
}
</style>
