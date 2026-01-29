<script setup lang="ts">
/**
 * Statistics display box
 *
 * Prominent display of key metrics with optional icons and trends.
 */

defineProps<{
  value: string | number
  label: string
  icon?: string
  color?: 'gyori' | 'cogex' | 'bio' | 'neutral'
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
}>()
</script>

<template>
  <div class="stat-box" :class="[`color-${color || 'gyori'}`]">
    <div class="stat-icon" v-if="icon">
      <component :is="icon" />
    </div>
    <div class="stat-content">
      <div class="stat-value">{{ value }}</div>
      <div class="stat-label">{{ label }}</div>
      <div class="stat-trend" v-if="trend && trendValue" :class="[`trend-${trend}`]">
        <span v-if="trend === 'up'">↑</span>
        <span v-else-if="trend === 'down'">↓</span>
        <span v-else>→</span>
        {{ trendValue }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.stat-box {
  background: white;
  border-radius: 12px;
  padding: 1.25rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  border: 1px solid #e5e7eb;
}

html.dark .stat-box {
  background: #1e293b;
  border-color: #374151;
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
}

.color-gyori .stat-icon {
  background: rgba(23, 162, 184, 0.15);
  color: var(--gyori-primary);
}

.color-cogex .stat-icon {
  background: rgba(191, 64, 191, 0.15);
  color: var(--cogex-purple);
}

.color-bio .stat-icon {
  background: rgba(91, 185, 0, 0.15);
  color: var(--bio-green);
}

.color-neutral .stat-icon {
  background: #f1f5f9;
  color: #64748b;
}

html.dark .color-neutral .stat-icon {
  background: #334155;
  color: #94a3b8;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 1.75rem;
  font-weight: 800;
  line-height: 1;
}

.color-gyori .stat-value { color: var(--gyori-primary); }
.color-cogex .stat-value { color: var(--cogex-purple); }
.color-bio .stat-value { color: var(--bio-green); }
.color-neutral .stat-value { color: #374151; }

html.dark .color-neutral .stat-value { color: #e5e7eb; }

.stat-label {
  font-size: 0.85rem;
  color: #64748b;
  margin-top: 0.25rem;
}

html.dark .stat-label {
  color: #94a3b8;
}

.stat-trend {
  font-size: 0.75rem;
  font-weight: 600;
  margin-top: 0.35rem;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.trend-up {
  color: var(--bio-green);
}

.trend-down {
  color: #dc3545;
}

.trend-neutral {
  color: #6b7280;
}
</style>
