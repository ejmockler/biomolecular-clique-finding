<script setup>
/**
 * sALS Proteomics Setup - Perceptual Engineering
 *
 * Core insight: We test MODULES (cliques), not individual proteins.
 * This captures coordinated biology that single-protein tests miss.
 *
 * Design:
 * - Top: Data summary (cohorts compact)
 * - Middle: The visual transformation (proteins → cliques)
 * - Bottom: Why ROAST matters + scale
 */
</script>

<template>
  <div class="sals-setup">
    <!-- The Data -->
    <div class="data-section">
      <div class="data-label">CSF Proteomics · Sporadic ALS vs. Controls</div>
      <div class="cohorts">
        <div class="cohort female">
          <span class="cohort-icon">♀</span>
          <span class="cohort-name">Female</span>
          <span class="cohort-stats">67 cases · 33 controls</span>
        </div>
        <div class="cohort male">
          <span class="cohort-icon">♂</span>
          <span class="cohort-name">Male</span>
          <span class="cohort-stats">173 cases · 28 controls</span>
        </div>
      </div>
    </div>

    <!-- The Innovation -->
    <div class="innovation-section">
      <div class="innovation-label">The Approach</div>

      <div class="transformation">
        <!-- Individual Proteins -->
        <div class="stage proteins">
          <div class="stage-visual">
            <div class="scattered-dots">
              <span class="dot d1"></span>
              <span class="dot d2"></span>
              <span class="dot d3"></span>
              <span class="dot d4"></span>
              <span class="dot d5"></span>
              <span class="dot d6"></span>
              <span class="dot d7"></span>
              <span class="dot d8"></span>
            </div>
          </div>
          <div class="stage-label">Individual proteins</div>
          <div class="stage-problem">Multiple testing problem</div>
        </div>

        <div class="transform-arrow">→</div>

        <!-- Cliques -->
        <div class="stage cliques">
          <div class="stage-visual">
            <div class="grouped-dots">
              <div class="clique c1">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
              </div>
              <div class="clique c2">
                <span class="dot"></span>
                <span class="dot"></span>
              </div>
              <div class="clique c3">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
              </div>
            </div>
          </div>
          <div class="stage-label">Protein cliques</div>
          <div class="stage-benefit">Co-regulated modules</div>
        </div>

        <div class="transform-arrow">→</div>

        <!-- ROAST Test -->
        <div class="stage roast">
          <div class="stage-visual">
            <div class="roast-badge">ROAST</div>
          </div>
          <div class="stage-label">Rotation testing</div>
          <div class="stage-benefit">Preserves correlations</div>
        </div>
      </div>
    </div>

    <!-- The Payoff -->
    <div class="payoff-section">
      <div class="scale-stat">
        <span class="scale-number">1,777</span>
        <span class="scale-label">protein cliques tested</span>
      </div>
      <div class="payoff-insight">
        <span class="insight-key">Key:</span>
        <span class="insight-text">Detects coordinated dysregulation that single-protein tests miss</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sals-setup {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
  max-width: 850px;
  margin: 0 auto;
}

/* ========== DATA SECTION ========== */
.data-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.data-label {
  font-size: 0.7rem;
  color: var(--text-tertiary, #64748b);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  text-align: center;
}

.cohorts {
  display: flex;
  justify-content: center;
  gap: 1.5rem;
}

.cohort {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(30, 41, 59, 0.5);
  border-radius: 8px;
  border: 1px solid rgba(100, 116, 139, 0.2);
}

.cohort.female {
  border-left: 3px solid #f472b6;
}

.cohort.male {
  border-left: 3px solid #60a5fa;
}

.cohort-icon {
  font-size: 1.2rem;
}

.female .cohort-icon { color: #f472b6; }
.male .cohort-icon { color: #60a5fa; }

.cohort-name {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-primary, #e2e8f0);
}

.cohort-stats {
  font-size: 0.75rem;
  color: var(--text-secondary, #94a3b8);
}

/* ========== INNOVATION SECTION ========== */
.innovation-section {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  padding: 0.8rem;
  background: rgba(191, 111, 247, 0.05);
  border: 1px solid rgba(191, 111, 247, 0.2);
  border-radius: 10px;
}

.innovation-label {
  font-size: 0.7rem;
  color: #bf6ff7;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  text-align: center;
  font-weight: 600;
}

.transformation {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
}

.stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
  min-width: 140px;
}

.stage-visual {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stage-label {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-primary, #e2e8f0);
}

.stage-problem {
  font-size: 0.65rem;
  color: #ef4444;
  font-style: italic;
}

.stage-benefit {
  font-size: 0.65rem;
  color: #22c55e;
  font-style: italic;
}

/* Scattered dots */
.scattered-dots {
  position: relative;
  width: 50px;
  height: 50px;
}

.scattered-dots .dot {
  position: absolute;
  width: 8px;
  height: 8px;
  background: #94a3b8;
  border-radius: 50%;
}

.scattered-dots .d1 { top: 5px; left: 10px; }
.scattered-dots .d2 { top: 15px; left: 35px; }
.scattered-dots .d3 { top: 25px; left: 5px; }
.scattered-dots .d4 { top: 30px; left: 25px; }
.scattered-dots .d5 { top: 8px; left: 28px; }
.scattered-dots .d6 { top: 40px; left: 12px; }
.scattered-dots .d7 { top: 38px; left: 38px; }
.scattered-dots .d8 { top: 20px; left: 18px; }

/* Grouped dots (cliques) */
.grouped-dots {
  display: flex;
  gap: 0.5rem;
}

.clique {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 6px;
  background: rgba(34, 197, 94, 0.15);
  border: 1px solid rgba(34, 197, 94, 0.4);
  border-radius: 6px;
}

.clique .dot {
  width: 8px;
  height: 8px;
  background: #22c55e;
  border-radius: 50%;
}

/* ROAST badge */
.roast-badge {
  padding: 0.6rem 1rem;
  background: linear-gradient(135deg, #bf6ff7 0%, #9333ea 100%);
  color: white;
  font-size: 0.9rem;
  font-weight: 700;
  border-radius: 8px;
  letter-spacing: 0.05em;
  box-shadow: 0 0 20px rgba(191, 111, 247, 0.3);
}

.transform-arrow {
  font-size: 1.5rem;
  color: #bf6ff7;
  font-weight: 300;
}

/* ========== PAYOFF SECTION ========== */
.payoff-section {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.6rem 1rem;
  background: rgba(78, 202, 223, 0.08);
  border: 1px solid rgba(78, 202, 223, 0.2);
  border-radius: 8px;
}

.scale-stat {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
}

.scale-number {
  font-size: 1.6rem;
  font-weight: 700;
  color: #4ecadf;
}

.scale-label {
  font-size: 0.75rem;
  color: var(--text-secondary, #94a3b8);
}

.payoff-insight {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.insight-key {
  font-size: 0.7rem;
  font-weight: 600;
  color: #22c55e;
  text-transform: uppercase;
}

.insight-text {
  font-size: 0.8rem;
  color: var(--text-secondary, #94a3b8);
}
</style>
