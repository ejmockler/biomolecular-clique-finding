<script setup lang="ts">
/**
 * Default content layout for INDRA CoGEx presentations
 *
 * Clean, readable layout for standard content slides
 * with consistent spacing and typography.
 */
</script>

<template>
  <div class="slidev-layout default">
    <slot />
  </div>
</template>

<style scoped>
.default {
  padding: 2.5rem 3.5rem;
  min-height: 100%;
  display: flex;
  flex-direction: column;
}

.default :deep(h1) {
  font-size: 2.25rem;
  font-weight: 700;
  color: var(--gyori-primary);
  margin-bottom: 1.5rem;
  line-height: 1.2;
}

.default :deep(h2) {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--slidev-theme-primary, #17a2b8);
  margin-bottom: 1rem;
  margin-top: 1.5rem;
}

.default :deep(h3) {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
  margin-top: 1rem;
}

.default :deep(p) {
  margin-bottom: 1rem;
  line-height: 1.7;
}

.default :deep(ul),
.default :deep(ol) {
  margin-left: 1.5rem;
  margin-bottom: 1rem;
}

.default :deep(li) {
  margin-bottom: 0.5rem;
  line-height: 1.6;
}

.default :deep(li::marker) {
  color: var(--gyori-primary);
}

.default :deep(code:not(pre code)) {
  background: rgba(78, 202, 223, 0.15);
  color: var(--gyori-primary);
  padding: 0.15em 0.4em;
  border-radius: 4px;
  font-size: 0.9em;
}

.default :deep(a) {
  color: var(--gyori-primary);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.2s;
}

.default :deep(a:hover) {
  border-bottom-color: var(--gyori-primary);
}

.default :deep(blockquote) {
  border-left: 4px solid var(--gyori-primary);
  padding-left: 1.5rem;
  margin: 1.5rem 0;
  color: var(--text-secondary);
  font-style: italic;
}

.default :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.9rem;
}

.default :deep(th) {
  background: var(--gyori-primary);
  color: white;
  font-weight: 600;
  text-align: left;
  padding: 0.75rem 1rem;
}

.default :deep(td) {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--border-default);
}

.default :deep(tr:hover td) {
  background: rgba(23, 162, 184, 0.05);
}
</style>
