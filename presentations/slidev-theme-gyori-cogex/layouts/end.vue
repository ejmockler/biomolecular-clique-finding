<script setup lang="ts">
/**
 * End/Thank You layout for INDRA CoGEx presentations
 *
 * Closing slide with contact information and acknowledgments.
 */
</script>

<template>
  <div class="slidev-layout end">
    <div class="end-content">
      <slot />
    </div>
    <div class="dna-decoration" aria-hidden="true">
      <svg viewBox="0 0 200 400" preserveAspectRatio="xMidYMid slice">
        <path d="M100,0 Q150,50 100,100 Q50,150 100,200 Q150,250 100,300 Q50,350 100,400"
              fill="none" stroke="rgba(23,162,184,0.3)" stroke-width="2" class="helix-1"/>
        <path d="M100,0 Q50,50 100,100 Q150,150 100,200 Q50,250 100,300 Q150,350 100,400"
              fill="none" stroke="rgba(191,64,191,0.3)" stroke-width="2" class="helix-2"/>
        <g class="base-pairs">
          <line x1="60" y1="50" x2="140" y2="50" stroke="rgba(91,185,0,0.4)" stroke-width="1.5"/>
          <line x1="65" y1="100" x2="135" y2="100" stroke="rgba(91,185,0,0.4)" stroke-width="1.5"/>
          <line x1="60" y1="150" x2="140" y2="150" stroke="rgba(91,185,0,0.4)" stroke-width="1.5"/>
          <line x1="65" y1="200" x2="135" y2="200" stroke="rgba(91,185,0,0.4)" stroke-width="1.5"/>
          <line x1="60" y1="250" x2="140" y2="250" stroke="rgba(91,185,0,0.4)" stroke-width="1.5"/>
          <line x1="65" y1="300" x2="135" y2="300" stroke="rgba(91,185,0,0.4)" stroke-width="1.5"/>
          <line x1="60" y1="350" x2="140" y2="350" stroke="rgba(91,185,0,0.4)" stroke-width="1.5"/>
        </g>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.end {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  min-height: 100%;
  padding: 3rem 4rem;
  background: linear-gradient(135deg, #0a1628 0%, #1e3a5f 50%, #0e626e 100%);
  color: white;
  overflow: hidden;
}

.end-content {
  position: relative;
  z-index: 10;
  max-width: 700px;
}

.end-content :deep(h1) {
  font-size: 3rem;
  font-weight: 800;
  color: white;
  margin-bottom: 1.5rem;
}

.end-content :deep(h2) {
  font-size: 1.5rem;
  font-weight: 400;
  color: rgba(255, 255, 255, 0.9);
  margin-bottom: 2rem;
}

.end-content :deep(p) {
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 0.5rem;
}

.end-content :deep(a) {
  color: var(--gyori-primary-light);
  text-decoration: none;
  transition: color 0.2s;
}

.end-content :deep(a:hover) {
  color: white;
}

.end-content :deep(.contact-info) {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 2rem;
  margin-top: 2rem;
}

.end-content :deep(.contact-item) {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: rgba(255, 255, 255, 0.8);
  font-size: 1rem;
}

.end-content :deep(.acknowledgments) {
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.2);
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.6);
}

.dna-decoration {
  position: absolute;
  right: -50px;
  top: 0;
  width: 200px;
  height: 100%;
  opacity: 0.5;
  pointer-events: none;
}

.dna-decoration svg {
  width: 100%;
  height: 100%;
}

.helix-1 {
  animation: helix-rotate 8s linear infinite;
}

.helix-2 {
  animation: helix-rotate 8s linear infinite reverse;
}

@keyframes helix-rotate {
  0% { stroke-dashoffset: 0; }
  100% { stroke-dashoffset: 100; }
}
</style>
