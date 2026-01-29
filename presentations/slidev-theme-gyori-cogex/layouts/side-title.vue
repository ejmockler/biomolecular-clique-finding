<script setup lang="ts">
/**
 * Side-title layout for INDRA CoGEx presentations
 *
 * A prominent title panel on the left with main content on the right.
 * Great for introducing new topics or emphasizing section themes.
 */

defineProps<{
  color?: 'gyori' | 'cogex' | 'bio'
  titleWidth?: 'narrow' | 'normal' | 'wide'
}>()
</script>

<template>
  <div class="slidev-layout side-title" :class="[`side-${color || 'gyori'}`, `title-${titleWidth || 'normal'}`]">
    <div class="title-panel">
      <div class="title-content">
        <slot name="title" />
      </div>
    </div>
    <div class="main-content">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.side-title {
  display: grid;
  min-height: 100%;
  padding: 0;
}

.title-narrow { grid-template-columns: 200px 1fr; }
.title-normal { grid-template-columns: 280px 1fr; }
.title-wide { grid-template-columns: 360px 1fr; }

.title-panel {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 2.5rem 1.5rem;
  color: white;
}

.side-gyori .title-panel {
  background: linear-gradient(180deg, #17a2b8 0%, #0e626e 100%);
}

.side-cogex .title-panel {
  background: linear-gradient(180deg, #BF40BF 0%, #732673 100%);
}

.side-bio .title-panel {
  background: linear-gradient(180deg, #5bb900 0%, #376f00 100%);
}

.title-content :deep(h1) {
  font-size: 1.75rem;
  font-weight: 700;
  color: white;
  line-height: 1.25;
  margin-bottom: 0.75rem;
}

.title-content :deep(h2) {
  font-size: 1rem;
  font-weight: 400;
  color: rgba(255, 255, 255, 0.85);
  line-height: 1.4;
}

.title-content :deep(p) {
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.8);
  margin-top: 1rem;
}

.main-content {
  padding: 2.5rem;
  overflow: auto;
}

.main-content :deep(h2) {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.main-content :deep(h3) {
  font-size: 1.15rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
  margin-top: 1.25rem;
}

.main-content :deep(p) {
  margin-bottom: 0.75rem;
  line-height: 1.6;
}

.main-content :deep(ul) {
  margin-left: 1.25rem;
  margin-bottom: 0.75rem;
}

.main-content :deep(li) {
  margin-bottom: 0.35rem;
}

.main-content :deep(img) {
  max-width: 100%;
  border-radius: 8px;
}
</style>
