<script setup>
defineProps({ text: { type: String, required: true } })
</script>

<template>
  <div class="method-strip">
    <span class="ms-tag">METHOD</span>
    <span class="ms-text" v-html="text" />
  </div>
</template>

<style scoped>
.method-strip {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding: 7px 40px 9px;
  border-top: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(10, 15, 26, 0.92);
}
.ms-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 2px;
  color: #4ecadf;
  flex-shrink: 0;
  position: relative;
  top: -1px;
}
.ms-text {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  line-height: 1.35;
  color: #94a3b8;
}
.ms-text :deep(b),
.ms-text :deep(strong) {
  color: #cbd5e1;
  font-weight: 700;
}
</style>
