<script setup lang="ts">
/**
 * Inline gene symbol component
 *
 * Renders gene symbols with proper biological convention (italic).
 */

defineProps<{
  symbol: string
  linked?: boolean
  ncbiId?: string
}>()
</script>

<template>
  <a
    v-if="linked && ncbiId"
    :href="`https://www.ncbi.nlm.nih.gov/gene/${ncbiId}`"
    target="_blank"
    rel="noopener"
    class="gene linked"
  >
    {{ symbol }}
  </a>
  <span v-else class="gene">{{ symbol }}</span>
</template>

<style scoped>
.gene {
  font-style: italic;
  color: var(--gyori-primary);
  font-weight: 500;
}

html.dark .gene {
  color: var(--gyori-primary-light);
}

.gene.linked {
  text-decoration: none;
  border-bottom: 1px dotted var(--gyori-primary);
  cursor: pointer;
  transition: border-color 0.2s;
}

.gene.linked:hover {
  border-bottom-style: solid;
}
</style>
