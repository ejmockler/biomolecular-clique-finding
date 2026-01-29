<script setup>
/**
 * Query Examples - Perceptual Engineering
 *
 * Core insight: These are QUESTIONS - they should feel conversational,
 * not like code. The user should immediately think "I could ask that!"
 *
 * Design:
 * - 2x2 grid of query categories
 * - Each category has icon + label + questions
 * - Questions styled as chat-like prompts, not code
 * - Subtle color coding per category
 */
</script>

<template>
  <div class="query-examples">
    <div class="categories-grid">
      <!-- Drug & Target -->
      <div class="category drug">
        <div class="category-header">
          <span class="category-icon">💊</span>
          <span class="category-label">Drug & Target</span>
        </div>
        <div class="questions">
          <div class="question">"What drugs target ROCK1?"</div>
          <div class="question">"Does fasudil have ALS trials?"</div>
        </div>
      </div>

      <!-- Disease -->
      <div class="category disease">
        <div class="category-header">
          <span class="category-icon">🏥</span>
          <span class="category-label">Disease</span>
        </div>
        <div class="questions">
          <div class="question">"What genes associate with ALS?"</div>
          <div class="question">"What ALS clinical trials exist?"</div>
        </div>
      </div>

      <!-- Pathway -->
      <div class="category pathway">
        <div class="category-header">
          <span class="category-icon">🔀</span>
          <span class="category-label">Pathway</span>
        </div>
        <div class="questions">
          <div class="question">"What pathways involve CTNNB1?"</div>
          <div class="question">"Do BAX and CYCS share pathways?"</div>
        </div>
      </div>

      <!-- Gene Function -->
      <div class="category gene">
        <div class="category-header">
          <span class="category-icon">🧬</span>
          <span class="category-label">Gene Function</span>
        </div>
        <div class="questions">
          <div class="question">"What GO terms describe DNMT1?"</div>
          <div class="question">"What tissues express NEFH?"</div>
        </div>
      </div>
    </div>

    <!-- The Flow -->
    <div class="flow-note">
      <span class="flow-item you">You ask</span>
      <span class="flow-arrow">→</span>
      <span class="flow-item agent">Agent queries</span>
      <span class="flow-arrow">→</span>
      <span class="flow-item result">You get answers</span>
    </div>
  </div>
</template>

<style scoped>
.query-examples {
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
}

/* ========== CATEGORIES GRID ========== */
.categories-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.category {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.8rem 1rem;
  border-radius: 10px;
  background: rgba(30, 41, 59, 0.5);
  border: 1px solid rgba(100, 116, 139, 0.2);
}

/* Category-specific accent colors */
.category.drug {
  border-left: 3px solid #f472b6;
}
.category.disease {
  border-left: 3px solid #fb923c;
}
.category.pathway {
  border-left: 3px solid #a78bfa;
}
.category.gene {
  border-left: 3px solid #34d399;
}

/* Category Header */
.category-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.category-icon {
  font-size: 1.1rem;
}

.category-label {
  font-size: 0.85rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.drug .category-label { color: #f472b6; }
.disease .category-label { color: #fb923c; }
.pathway .category-label { color: #a78bfa; }
.gene .category-label { color: #34d399; }

/* Questions */
.questions {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.question {
  font-size: 0.85rem;
  color: var(--text-secondary, #94a3b8);
  padding: 0.4rem 0.6rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
  font-style: italic;
}

.question::before {
  content: '→ ';
  color: #4ecadf;
  font-style: normal;
}

/* ========== FLOW NOTE ========== */
.flow-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  padding: 0.6rem 1rem;
  background: rgba(78, 202, 223, 0.08);
  border: 1px solid rgba(78, 202, 223, 0.2);
  border-radius: 8px;
}

.flow-item {
  font-size: 0.8rem;
  font-weight: 500;
  padding: 0.25rem 0.6rem;
  border-radius: 4px;
}

.flow-item.you {
  background: rgba(78, 202, 223, 0.2);
  color: #4ecadf;
}

.flow-item.agent {
  background: rgba(191, 111, 247, 0.2);
  color: #bf6ff7;
}

.flow-item.result {
  background: rgba(34, 197, 94, 0.2);
  color: #22c55e;
}

.flow-arrow {
  color: var(--text-tertiary, #64748b);
  font-size: 0.75rem;
}
</style>
