<script setup>
/**
 * CoGEx Knowledge Graph Overview - For Gyori Lab Audience
 *
 * This audience BUILT the system. Don't oversimplify.
 * Communicate:
 * - True scale (92M entities, 573M relationships)
 * - Evidence provenance layer
 * - Mechanistic vocabulary depth
 * - Research ecosystem integration
 */
</script>

<template>
  <div class="cogex-overview">
    <!-- Scale Header -->
    <div class="scale-banner">
      <span class="scale-number">92M</span>
      <span class="scale-label">entities</span>
      <span class="scale-sep">/</span>
      <span class="scale-number">573M</span>
      <span class="scale-label">relationships</span>
    </div>

    <!-- Three Column Layout -->
    <div class="content-grid">
      <!-- Column 1: Core Biology -->
      <div class="column">
        <div class="column-header">Biological Entities</div>
        <div class="stat-list">
          <div class="stat-item">
            <span class="stat-val">49K</span>
            <span class="stat-name">genes</span>
            <span class="stat-src">HGNC</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">230K</span>
            <span class="stat-name">chemicals</span>
            <span class="stat-src">ChEBI/DrugBank</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">16.7M</span>
            <span class="stat-name highlight-purple">indra_rel</span>
            <span class="stat-src">machine-read</span>
          </div>
        </div>
      </div>

      <!-- Column 2: Evidence Layer -->
      <div class="column">
        <div class="column-header">Evidence Provenance</div>
        <div class="stat-list">
          <div class="stat-item">
            <span class="stat-val">45M</span>
            <span class="stat-name">evidence records</span>
            <span class="stat-src">with embeddings</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">39M</span>
            <span class="stat-name">publications</span>
            <span class="stat-src">PubMed linked</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">31</span>
            <span class="stat-name">statement types</span>
            <span class="stat-src">mechanistic</span>
          </div>
        </div>
      </div>

      <!-- Column 3: Research Ecosystem -->
      <div class="column">
        <div class="column-header">Research Ecosystem</div>
        <div class="stat-list">
          <div class="stat-item">
            <span class="stat-val">549K</span>
            <span class="stat-name">clinical trials</span>
            <span class="stat-src">ClinicalTrials.gov</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">NIH</span>
            <span class="stat-name">research projects</span>
            <span class="stat-src">RePORTER linked</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">GO</span>
            <span class="stat-name">+ pathways</span>
            <span class="stat-src">REACTOME/WP</span>
          </div>
        </div>
      </div>
    </div>

    <!-- INDRA Statement Types - The Mechanistic Vocabulary -->
    <div class="stmt-types-section">
      <div class="section-label">Mechanistic Vocabulary</div>
      <div class="stmt-pills">
        <span class="pill pill-mod">Phosphorylation</span>
        <span class="pill pill-act">Activation</span>
        <span class="pill pill-inh">Inhibition</span>
        <span class="pill pill-cplx">Complex</span>
        <span class="pill pill-amt">IncreaseAmount</span>
        <span class="pill pill-mod">Ubiquitination</span>
        <span class="pill pill-more">+25 more</span>
      </div>
    </div>

    <!-- Key Differentiator -->
    <div class="differentiator">
      <span class="diff-label">Each indra_rel carries:</span>
      <span class="diff-item">belief score</span>
      <span class="diff-sep">·</span>
      <span class="diff-item">evidence_count</span>
      <span class="diff-sep">·</span>
      <span class="diff-item">source_counts</span>
      <span class="diff-sep">·</span>
      <span class="diff-item">stmt_json</span>
    </div>
  </div>
</template>

<style scoped>
.cogex-overview {
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
  width: 100%;
  max-width: 880px;
  margin: 0 auto;
}

/* ========== SCALE BANNER ========== */
.scale-banner {
  display: flex;
  align-items: baseline;
  justify-content: center;
  gap: 0.4rem;
  padding: 0.5rem 1rem;
  background: linear-gradient(90deg, rgba(78, 202, 223, 0.1) 0%, rgba(191, 111, 247, 0.1) 50%, rgba(125, 214, 41, 0.1) 100%);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.scale-number {
  font-size: 1.6rem;
  font-weight: 700;
  color: #4ecadf;
  letter-spacing: -0.02em;
}

.scale-label {
  font-size: 0.75rem;
  color: var(--text-secondary, #94a3b8);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.scale-sep {
  color: var(--text-tertiary, #64748b);
  font-size: 1.2rem;
  margin: 0 0.3rem;
}

/* ========== THREE COLUMN GRID ========== */
.content-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.column {
  background: var(--surface-1, #121a2d);
  border: 1px solid var(--border-default, rgba(255, 255, 255, 0.1));
  border-radius: 8px;
  padding: 0.7rem;
}

.column-header {
  font-size: 0.6rem;
  font-weight: 600;
  color: var(--text-tertiary, #64748b);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.5rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.06));
}

.stat-list {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.stat-item {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: baseline;
  gap: 0.4rem;
}

.stat-val {
  font-size: 0.95rem;
  font-weight: 700;
  color: #4ecadf;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
  min-width: 3rem;
}

.stat-name {
  font-size: 0.7rem;
  color: var(--text-primary, #f1f5f9);
  white-space: nowrap;
}

.stat-name.highlight-purple {
  color: #bf6ff7;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
}

.stat-src {
  font-size: 0.55rem;
  color: var(--text-tertiary, #64748b);
  text-align: right;
}

/* ========== STATEMENT TYPES ========== */
.stmt-types-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.4rem;
}

.section-label {
  font-size: 0.55rem;
  font-weight: 600;
  color: var(--text-tertiary, #64748b);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.stmt-pills {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.35rem;
}

.pill {
  padding: 0.2rem 0.5rem;
  border-radius: 12px;
  font-size: 0.55rem;
  font-weight: 500;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
  background: var(--surface-2, #1a2540);
  border: 1px solid;
}

.pill-mod {
  border-color: rgba(191, 111, 247, 0.5);
  color: #bf6ff7;
}

.pill-act {
  border-color: rgba(34, 197, 94, 0.5);
  color: #22c55e;
}

.pill-inh {
  border-color: rgba(239, 68, 68, 0.5);
  color: #ef4444;
}

.pill-cplx {
  border-color: rgba(78, 202, 223, 0.5);
  color: #4ecadf;
}

.pill-amt {
  border-color: rgba(249, 115, 22, 0.5);
  color: #f97316;
}

.pill-more {
  border-color: rgba(100, 116, 139, 0.5);
  color: #64748b;
  font-style: italic;
}

/* ========== DIFFERENTIATOR ========== */
.differentiator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  padding: 0.4rem 0.8rem;
  background: rgba(191, 111, 247, 0.08);
  border: 1px solid rgba(191, 111, 247, 0.25);
  border-radius: 6px;
  flex-wrap: wrap;
}

.diff-label {
  font-size: 0.6rem;
  color: var(--text-secondary, #94a3b8);
}

.diff-item {
  font-size: 0.6rem;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
  color: #bf6ff7;
}

.diff-sep {
  color: var(--text-tertiary, #64748b);
}
</style>
