<script setup>
/**
 * The Scale of Biomedical Knowledge - Perceptual Engineering
 *
 * Core insight: The scale is cognitively impossible.
 * Even ONE protein overwhelms working memory.
 *
 * Design:
 * - Hero numbers dominate peripheral vision
 * - Visual "zoom" from ocean to one drop
 * - The drop is STILL overwhelming - that's the reveal
 * - Punchline lands with cognitive weight
 */
</script>

<template>
  <div class="scale-viz">
    <!-- The Ocean: Global Scale -->
    <div class="ocean-section">
      <div class="ocean-label">The ocean of biomedical knowledge</div>
      <div class="ocean-stats">
        <div class="mega-stat">
          <span class="mega-number">35M</span>
          <span class="mega-unit">papers</span>
        </div>
        <div class="mega-stat">
          <span class="mega-number">60M</span>
          <span class="mega-unit">interactions</span>
        </div>
        <div class="mega-stat">
          <span class="mega-number">15K</span>
          <span class="mega-unit">drugs</span>
        </div>
      </div>
    </div>

    <!-- The Zoom Arrow -->
    <div class="zoom-indicator">
      <span class="zoom-text">zoom to ONE protein</span>
      <span class="zoom-arrow">↓</span>
    </div>

    <!-- The Drop: Single Protein - Still Overwhelming -->
    <div class="drop-section">
      <div class="protein-badge">
        <span class="protein-name">CTNNB1</span>
        <span class="protein-alt">β-catenin</span>
      </div>
      <div class="drop-stats">
        <div class="drop-stat">
          <span class="drop-number">500+</span>
          <span class="drop-label">pathways</span>
        </div>
        <div class="drop-stat">
          <span class="drop-number">1,000+</span>
          <span class="drop-label">interactors</span>
        </div>
        <div class="drop-stat">
          <span class="drop-number">12,000+</span>
          <span class="drop-label">citations</span>
        </div>
      </div>
      <div class="still-text">still too much for one protein</div>
    </div>

    <!-- The Punchline -->
    <div class="punchline">
      <div class="punch-icon">🧠</div>
      <div class="punch-text">
        <span class="punch-main">Working memory: 4±1 items</span>
        <span class="punch-sub">You're looking at thousands.</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.scale-viz {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.8rem;
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
}

/* ========== THE OCEAN ========== */
.ocean-section {
  width: 100%;
  text-align: center;
}

.ocean-label {
  font-size: 0.65rem;
  font-weight: 500;
  color: var(--text-tertiary, #64748b);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin-bottom: 0.6rem;
}

.ocean-stats {
  display: flex;
  justify-content: center;
  gap: 2.5rem;
}

.mega-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.mega-number {
  font-size: 3.2rem;
  font-weight: 800;
  color: #4ecadf;
  line-height: 1;
  letter-spacing: -0.03em;
  text-shadow: 0 0 40px rgba(78, 202, 223, 0.4);
}

.mega-unit {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--text-secondary, #94a3b8);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.2rem;
}

/* ========== ZOOM INDICATOR ========== */
.zoom-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.1rem;
  padding: 0.3rem 0;
}

.zoom-text {
  font-size: 0.6rem;
  font-weight: 600;
  color: #bf6ff7;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.zoom-arrow {
  font-size: 1.2rem;
  color: #bf6ff7;
  animation: pulse-down 1.5s ease-in-out infinite;
}

@keyframes pulse-down {
  0%, 100% { opacity: 0.5; transform: translateY(0); }
  50% { opacity: 1; transform: translateY(3px); }
}

/* ========== THE DROP (Single Protein) ========== */
.drop-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding: 0.8rem 1.5rem;
  background: linear-gradient(180deg, rgba(191, 111, 247, 0.08) 0%, rgba(191, 111, 247, 0.02) 100%);
  border: 1.5px solid rgba(191, 111, 247, 0.4);
  border-radius: 12px;
  position: relative;
}

.protein-badge {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
}

.protein-name {
  font-size: 1.4rem;
  font-weight: 700;
  color: #bf6ff7;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
}

.protein-alt {
  font-size: 0.75rem;
  color: var(--text-tertiary, #64748b);
  font-style: italic;
}

.drop-stats {
  display: flex;
  gap: 1.5rem;
}

.drop-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.drop-number {
  font-size: 1.5rem;
  font-weight: 700;
  color: #f97316;
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
}

.drop-label {
  font-size: 0.6rem;
  color: var(--text-secondary, #94a3b8);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.still-text {
  font-size: 0.55rem;
  font-weight: 500;
  color: #f97316;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 0.2rem;
}

/* ========== THE PUNCHLINE ========== */
.punchline {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  padding: 0.6rem 1.2rem;
  background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
  border: 1.5px solid rgba(239, 68, 68, 0.5);
  border-radius: 8px;
  margin-top: 0.3rem;
}

.punch-icon {
  font-size: 1.4rem;
}

.punch-text {
  display: flex;
  flex-direction: column;
}

.punch-main {
  font-size: 0.9rem;
  font-weight: 700;
  color: #ef4444;
}

.punch-sub {
  font-size: 0.7rem;
  color: var(--text-secondary, #94a3b8);
}
</style>
