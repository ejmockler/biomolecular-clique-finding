<script setup lang="ts">
/**
 * Results highlight component
 *
 * Prominent display of key findings with visual emphasis.
 */

defineProps<{
  title: string
  finding: string
  type?: 'discovery' | 'validation' | 'insight'
  pvalue?: number
  effect?: string
}>()

const typeConfig = {
  discovery: { icon: '🔬', color: 'gyori', label: 'Discovery' },
  validation: { icon: '✓', color: 'bio', label: 'Validated' },
  insight: { icon: '💡', color: 'cogex', label: 'Insight' },
}
</script>

<template>
  <div class="results-highlight" :class="[`type-${type || 'discovery'}`]">
    <div class="highlight-header">
      <span class="highlight-icon">{{ typeConfig[type || 'discovery'].icon }}</span>
      <span class="highlight-label">{{ typeConfig[type || 'discovery'].label }}</span>
    </div>
    <h3 class="highlight-title">{{ title }}</h3>
    <p class="highlight-finding">{{ finding }}</p>
    <div class="highlight-stats" v-if="pvalue || effect">
      <span v-if="pvalue" class="stat-pvalue">
        p = {{ pvalue < 0.001 ? pvalue.toExponential(2) : pvalue.toFixed(4) }}
      </span>
      <span v-if="effect" class="stat-effect">{{ effect }}</span>
    </div>
  </div>
</template>

<style scoped>
.results-highlight {
  background: white;
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 2px solid;
}

html.dark .results-highlight {
  background: #1e293b;
}

.type-discovery {
  border-color: var(--gyori-primary);
}

.type-validation {
  border-color: var(--bio-green);
}

.type-insight {
  border-color: var(--cogex-purple);
}

.highlight-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.highlight-icon {
  font-size: 1.25rem;
}

.highlight-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.type-discovery .highlight-label { color: var(--gyori-primary); }
.type-validation .highlight-label { color: var(--bio-green); }
.type-insight .highlight-label { color: var(--cogex-purple); }

.highlight-title {
  font-size: 1.25rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: #1e293b;
}

html.dark .highlight-title {
  color: #f1f5f9;
}

.highlight-finding {
  font-size: 1rem;
  color: #475569;
  line-height: 1.6;
  margin-bottom: 0.75rem;
}

html.dark .highlight-finding {
  color: #94a3b8;
}

.highlight-stats {
  display: flex;
  gap: 1rem;
  padding-top: 0.75rem;
  border-top: 1px solid #e5e7eb;
}

html.dark .highlight-stats {
  border-top-color: #374151;
}

.stat-pvalue {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: var(--bio-green);
  font-weight: 600;
}

.stat-effect {
  font-size: 0.85rem;
  color: #64748b;
}
</style>
