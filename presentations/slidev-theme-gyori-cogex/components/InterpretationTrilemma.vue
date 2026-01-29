<script setup>
/**
 * The Interpretation Trilemma - Perceptual Engineering v3
 *
 * PERCEPTUAL LINKING: Group details into tight visual chunks.
 *
 * Each approach card:
 * - What you do (recognition - people nod)
 * - FATAL FLAW (hero - large, red)
 * - Time → What you get (the tradeoff)
 *
 * The flaws stay dominant. Context clusters around them.
 * Use the space - cards can breathe without scattering.
 */
</script>

<template>
  <div class="trilemma-viz">
    <!-- The Three Approaches -->
    <div class="approaches-row">
      <!-- Manual -->
      <div class="approach-card">
        <div class="approach-header">
          <span class="approach-name">Manual Review</span>
        </div>
        <div class="approach-how">
          <span>Google each gene</span>
          <span>Read abstracts</span>
        </div>
        <div class="approach-flaw">Doesn't scale</div>
        <div class="approach-tradeoff">
          <span class="time">Weeks</span>
          <span class="arrow">→</span>
          <span class="outcome good">~20 genes deep</span>
        </div>
      </div>

      <!-- Enrichment -->
      <div class="approach-card">
        <div class="approach-header">
          <span class="approach-name">Enrichment Analysis</span>
        </div>
        <div class="approach-how">
          <span>Run GSEA / DAVID</span>
          <span>Get GO categories</span>
        </div>
        <div class="approach-flaw">No mechanism</div>
        <div class="approach-tradeoff">
          <span class="time">Hours</span>
          <span class="arrow">→</span>
          <span class="outcome good">Genome-wide</span>
        </div>
      </div>

      <!-- Expert -->
      <div class="approach-card">
        <div class="approach-header">
          <span class="approach-name">Expert Intuition</span>
        </div>
        <div class="approach-how">
          <span>Focus on known hits</span>
          <span>Cite familiar work</span>
        </div>
        <div class="approach-flaw">Confirmation bias</div>
        <div class="approach-tradeoff">
          <span class="time">Fast</span>
          <span class="arrow">→</span>
          <span class="outcome good">Strong narrative</span>
        </div>
      </div>
    </div>

    <!-- The Breakthrough -->
    <div class="breakthrough">
      <div class="breakthrough-question">
        What if you could query <span class="emphasis">all of biomedical knowledge</span><br>
        as naturally as asking a colleague?
      </div>
    </div>
  </div>
</template>

<style scoped>
.trilemma-viz {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2rem;
  width: 100%;
  max-width: 920px;
  margin: 0 auto;
}

/* ========== APPROACHES ROW ========== */
.approaches-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  width: 100%;
}

.approach-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem 0.8rem;
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(30, 41, 59, 0.3) 100%);
  border: 1px solid rgba(100, 116, 139, 0.25);
  border-radius: 10px;
}

/* Header */
.approach-header {
  width: 100%;
  text-align: center;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(100, 116, 139, 0.2);
}

.approach-name {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-primary, #e2e8f0);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* How */
.approach-how {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.15rem;
  padding: 0.3rem 0;
}

.approach-how span {
  font-size: 0.65rem;
  color: var(--text-tertiary, #64748b);
}

/* Fatal Flaw - THE HERO */
.approach-flaw {
  font-size: 1.4rem;
  font-weight: 700;
  color: #ef4444;
  text-align: center;
  line-height: 1.2;
  padding: 0.3rem 0;
  text-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
}

/* Tradeoff */
.approach-tradeoff {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.6rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  margin-top: 0.2rem;
}

.time {
  font-size: 0.7rem;
  font-weight: 600;
  color: #fbbf24;
}

.arrow {
  font-size: 0.6rem;
  color: var(--text-tertiary, #64748b);
}

.outcome {
  font-size: 0.65rem;
  color: var(--text-secondary, #94a3b8);
}

.outcome.good {
  color: #86efac;
}

/* ========== BREAKTHROUGH ========== */
.breakthrough {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 1.2rem;
  border-top: 1px solid rgba(78, 202, 223, 0.2);
  width: 85%;
}

.breakthrough-question {
  text-align: center;
  font-size: 1.2rem;
  line-height: 1.6;
  color: var(--text-secondary, #94a3b8);
}

.breakthrough-question .emphasis {
  color: #4ecadf;
  font-weight: 600;
}
</style>
