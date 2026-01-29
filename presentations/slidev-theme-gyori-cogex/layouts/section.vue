<script setup lang="ts">
/**
 * Section divider layout for INDRA CoGEx presentations
 *
 * Creates visual breaks between major sections with
 * sophisticated dark-first gradients that reinforce
 * the semantic color coding:
 * - gyori (cyan): Data sources
 * - cogex (purple): Knowledge processing
 * - bio (green): Insights/outcomes
 * - gradient: Full pipeline visualization
 */

defineProps<{
  color?: 'gyori' | 'cogex' | 'bio' | 'gradient'
}>()
</script>

<template>
  <div class="slidev-layout section" :class="[`section-${color || 'gradient'}`]">
    <div class="section-decoration" aria-hidden="true"></div>
    <div class="section-content">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.section {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  min-height: 100%;
  padding: 3rem 4rem;
  overflow: hidden;
}

/* Gradient backgrounds - dark-first design */
.section-gradient {
  background: linear-gradient(
    145deg,
    #0a0f1a 0%,
    #1a1530 30%,
    #1a2540 70%,
    #0a1a1a 100%
  );
  color: white;
}

.section-gyori {
  background: linear-gradient(
    145deg,
    #071518 0%,
    #0e3a42 50%,
    #0d4a54 100%
  );
  color: white;
}

.section-cogex {
  background: linear-gradient(
    145deg,
    #1a0f1a 0%,
    #2d1a35 50%,
    #3d1f4d 100%
  );
  color: white;
}

.section-bio {
  background: linear-gradient(
    145deg,
    #0a1508 0%,
    #1a3010 50%,
    #254518 100%
  );
  color: white;
}

/* Decorative background element */
.section-decoration {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 600px;
  height: 600px;
  border-radius: 50%;
  opacity: 0.06;
  pointer-events: none;
}

.section-gradient .section-decoration {
  background: radial-gradient(circle, #9d4edd 0%, transparent 70%);
}

.section-gyori .section-decoration {
  background: radial-gradient(circle, #17a2b8 0%, transparent 70%);
}

.section-cogex .section-decoration {
  background: radial-gradient(circle, #9d4edd 0%, transparent 70%);
}

.section-bio .section-decoration {
  background: radial-gradient(circle, #5bb900 0%, transparent 70%);
}

.section-content {
  position: relative;
  z-index: 10;
  max-width: 800px;
}

.section-content :deep(h1) {
  font-size: 2.75rem;
  font-weight: 700;
  color: white;
  margin-bottom: 0.75rem;
  line-height: 1.15;
  letter-spacing: -0.02em;
}

.section-content :deep(h2) {
  font-size: 1.35rem;
  font-weight: 400;
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.4;
}

.section-content :deep(hr) {
  width: 80px;
  height: 3px;
  border: none;
  border-radius: 2px;
  margin: 1.25rem auto;
  opacity: 0.8;
}

.section-gradient :deep(hr) {
  background: linear-gradient(90deg, #17a2b8 0%, #9d4edd 50%, #5bb900 100%);
}

.section-gyori :deep(hr) {
  background: linear-gradient(90deg, rgba(255,255,255,0.3) 0%, #4ecadf 50%, rgba(255,255,255,0.3) 100%);
}

.section-cogex :deep(hr) {
  background: linear-gradient(90deg, rgba(255,255,255,0.3) 0%, #bf6ff7 50%, rgba(255,255,255,0.3) 100%);
}

.section-bio :deep(hr) {
  background: linear-gradient(90deg, rgba(255,255,255,0.3) 0%, #7dd629 50%, rgba(255,255,255,0.3) 100%);
}

.section-content :deep(p) {
  font-size: 1.15rem;
  color: rgba(255, 255, 255, 0.75);
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}
</style>
