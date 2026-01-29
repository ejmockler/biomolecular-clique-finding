<script setup lang="ts">
/**
 * Two-column layout for INDRA CoGEx presentations
 *
 * Side-by-side content for comparisons, figure + explanation,
 * or any dual-panel content arrangement.
 */

defineProps<{
  gap?: 'sm' | 'md' | 'lg'
  leftWidth?: string
}>()
</script>

<template>
  <div class="slidev-layout two-cols" :class="[`gap-${gap || 'md'}`]" :style="leftWidth ? `--left-width: ${leftWidth}` : ''">
    <div class="col-header" v-if="$slots.header">
      <slot name="header" />
    </div>
    <div class="col-container">
      <div class="col-left">
        <slot name="left" />
        <slot />
      </div>
      <div class="col-right">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.two-cols {
  --left-width: 1fr;
  padding: 2.5rem 3.5rem;
  min-height: 100%;
  display: flex;
  flex-direction: column;
}

.col-header {
  margin-bottom: 1.5rem;
}

.col-header :deep(h1) {
  font-size: 2rem;
  font-weight: 700;
  color: var(--gyori-primary);
  margin: 0;
}

.col-container {
  display: grid;
  grid-template-columns: var(--left-width) 1fr;
  flex: 1;
  min-height: 0;
}

.gap-sm .col-container { gap: 1rem; }
.gap-md .col-container { gap: 2rem; }
.gap-lg .col-container { gap: 3rem; }

.col-left,
.col-right {
  display: flex;
  flex-direction: column;
  overflow: auto;
}

.col-left :deep(h2),
.col-right :deep(h2) {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--gyori-primary);
  margin-bottom: 1rem;
}

.col-left :deep(img),
.col-right :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
}

.col-left :deep(p),
.col-right :deep(p) {
  margin-bottom: 0.75rem;
  line-height: 1.6;
}

.col-left :deep(ul),
.col-right :deep(ul) {
  margin-left: 1.25rem;
  margin-bottom: 0.75rem;
}

.col-left :deep(li),
.col-right :deep(li) {
  margin-bottom: 0.35rem;
  line-height: 1.5;
}
</style>
