<script setup lang="ts">
/**
 * Clique visualization card
 *
 * Displays a protein clique with direction and significance.
 */

defineProps<{
  id: string | number
  name?: string
  size: number
  direction: 'up' | 'down' | 'bidirectional'
  pvalue: number
  genes?: string[]
  showGenes?: boolean
}>()

const formatPValue = (p: number) => {
  if (p < 0.001) return p.toExponential(2)
  return p.toFixed(4)
}
</script>

<template>
  <div class="clique-card" :class="[`clique-${direction}`]">
    <div class="clique-header">
      <span class="clique-id">Clique {{ id }}</span>
      <span class="clique-direction">
        <span v-if="direction === 'up'" class="direction-icon">↑</span>
        <span v-else-if="direction === 'down'" class="direction-icon">↓</span>
        <span v-else class="direction-icon">↕</span>
        {{ direction === 'up' ? 'Up' : direction === 'down' ? 'Down' : 'Mixed' }}
      </span>
    </div>

    <div class="clique-name" v-if="name">{{ name }}</div>

    <div class="clique-stats">
      <div class="stat">
        <span class="stat-value">{{ size }}</span>
        <span class="stat-label">proteins</span>
      </div>
      <div class="stat">
        <span class="stat-value pvalue" :class="{ significant: pvalue < 0.05 }">
          {{ formatPValue(pvalue) }}
        </span>
        <span class="stat-label">p-value</span>
      </div>
    </div>

    <div class="clique-genes" v-if="showGenes && genes?.length">
      <span v-for="gene in genes.slice(0, 5)" :key="gene" class="gene-tag">
        {{ gene }}
      </span>
      <span v-if="genes.length > 5" class="gene-more">+{{ genes.length - 5 }} more</span>
    </div>
  </div>
</template>

<style scoped>
.clique-card {
  background: white;
  border-radius: 12px;
  padding: 1rem;
  border-top: 4px solid;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

html.dark .clique-card {
  background: #1e293b;
}

.clique-up { border-top-color: #0d9488; }
.clique-down { border-top-color: #f97316; }
.clique-bidirectional { border-top-color: #7c3aed; }

.clique-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.clique-id {
  font-weight: 700;
  font-size: 0.9rem;
  color: #374151;
}

html.dark .clique-id {
  color: #e5e7eb;
}

.clique-direction {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.8rem;
  font-weight: 600;
}

.clique-up .clique-direction { color: #0d9488; }
.clique-down .clique-direction { color: #f97316; }
.clique-bidirectional .clique-direction { color: #7c3aed; }

.direction-icon {
  font-size: 1rem;
}

.clique-name {
  font-size: 0.85rem;
  color: #6b7280;
  margin-bottom: 0.75rem;
}

html.dark .clique-name {
  color: #9ca3af;
}

.clique-stats {
  display: flex;
  gap: 1.5rem;
}

.stat {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--gyori-primary);
}

.stat-value.pvalue {
  font-family: var(--font-mono);
  font-size: 1rem;
  color: #6b7280;
}

.stat-value.pvalue.significant {
  color: var(--bio-green);
}

.stat-label {
  font-size: 0.7rem;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.clique-genes {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid #e5e7eb;
}

html.dark .clique-genes {
  border-top-color: #374151;
}

.gene-tag {
  font-size: 0.7rem;
  font-style: italic;
  background: rgba(23, 162, 184, 0.1);
  color: var(--gyori-primary);
  padding: 2px 6px;
  border-radius: 4px;
}

.gene-more {
  font-size: 0.7rem;
  color: #9ca3af;
}
</style>
