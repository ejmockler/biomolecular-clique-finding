<script setup lang="ts">
/**
 * Evidence list component
 *
 * Displays literature evidence supporting statements.
 */

interface Evidence {
  pmid?: string
  text: string
  source?: string
  year?: number
}

defineProps<{
  items: Evidence[]
  maxItems?: number
}>()
</script>

<template>
  <div class="evidence-list">
    <div
      v-for="(item, i) in items.slice(0, maxItems || 5)"
      :key="i"
      class="evidence-item"
    >
      <div class="evidence-source">
        <a
          v-if="item.pmid"
          :href="`https://pubmed.ncbi.nlm.nih.gov/${item.pmid}`"
          target="_blank"
          rel="noopener"
          class="pmid-link"
        >
          PMID:{{ item.pmid }}
        </a>
        <span v-if="item.source" class="source-badge">{{ item.source }}</span>
        <span v-if="item.year" class="year">({{ item.year }})</span>
      </div>
      <div class="evidence-text">{{ item.text }}</div>
    </div>
    <div v-if="items.length > (maxItems || 5)" class="evidence-more">
      +{{ items.length - (maxItems || 5) }} more evidence items
    </div>
  </div>
</template>

<style scoped>
.evidence-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.evidence-item {
  background: #f8fafc;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  border-left: 3px solid var(--gyori-primary);
}

html.dark .evidence-item {
  background: #1e293b;
}

.evidence-source {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.pmid-link {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--gyori-primary);
  text-decoration: none;
  font-weight: 600;
}

.pmid-link:hover {
  text-decoration: underline;
}

.source-badge {
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  background: var(--cogex-purple);
  color: white;
  padding: 1px 6px;
  border-radius: 4px;
}

.year {
  font-size: 0.75rem;
  color: #6b7280;
}

.evidence-text {
  font-size: 0.85rem;
  color: #374151;
  line-height: 1.5;
}

html.dark .evidence-text {
  color: #d1d5db;
}

.evidence-more {
  font-size: 0.8rem;
  color: #6b7280;
  text-align: center;
  padding: 0.5rem;
}
</style>
