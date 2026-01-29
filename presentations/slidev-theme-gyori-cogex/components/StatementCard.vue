<script setup lang="ts">
/**
 * INDRA Statement visualization card
 *
 * Displays causal relationships with proper biological notation:
 * - Subject → Object for activation
 * - Subject ⊣ Object for inhibition
 */

defineProps<{
  subject: string
  object: string
  type: 'activation' | 'inhibition' | 'complex' | 'modification'
  evidence?: number
  sources?: string[]
}>()

const typeConfig = {
  activation: { arrow: '→', color: 'bio', label: 'Activation' },
  inhibition: { arrow: '⊣', color: 'red', label: 'Inhibition' },
  complex: { arrow: '—', color: 'gyori', label: 'Complex' },
  modification: { arrow: '→P', color: 'cogex', label: 'Modification' },
}
</script>

<template>
  <div class="statement-card" :class="[`stmt-${type}`]">
    <div class="stmt-badge">{{ typeConfig[type].label }}</div>
    <div class="stmt-content">
      <span class="entity subject">{{ subject }}</span>
      <span class="arrow">{{ typeConfig[type].arrow }}</span>
      <span class="entity object">{{ object }}</span>
    </div>
    <div class="stmt-meta" v-if="evidence || sources?.length">
      <span class="evidence" v-if="evidence">
        <carbon-document /> {{ evidence }} evidence{{ evidence > 1 ? 's' : '' }}
      </span>
      <span class="sources" v-if="sources?.length">
        {{ sources.join(', ') }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.statement-card {
  background: white;
  border-radius: 12px;
  padding: 1rem 1.25rem;
  border-left: 4px solid;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s, box-shadow 0.2s;
}

html.dark .statement-card {
  background: #1e293b;
}

.statement-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.stmt-activation { border-left-color: var(--bio-green); }
.stmt-inhibition { border-left-color: #dc3545; }
.stmt-complex { border-left-color: var(--gyori-primary); }
.stmt-modification { border-left-color: var(--cogex-purple); }

.stmt-badge {
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 2px 8px;
  border-radius: 4px;
  margin-bottom: 0.5rem;
}

.stmt-activation .stmt-badge {
  background: rgba(91, 185, 0, 0.15);
  color: var(--bio-green);
}

.stmt-inhibition .stmt-badge {
  background: rgba(220, 53, 69, 0.15);
  color: #dc3545;
}

.stmt-complex .stmt-badge {
  background: rgba(23, 162, 184, 0.15);
  color: var(--gyori-primary);
}

.stmt-modification .stmt-badge {
  background: rgba(191, 64, 191, 0.15);
  color: var(--cogex-purple);
}

.stmt-content {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.1rem;
}

.entity {
  font-weight: 600;
}

.subject {
  color: var(--gyori-primary);
  font-style: italic;
}

.object {
  color: var(--cogex-purple);
  font-style: italic;
}

.arrow {
  font-size: 1.25rem;
  font-weight: bold;
}

.stmt-activation .arrow { color: var(--bio-green); }
.stmt-inhibition .arrow { color: #dc3545; }
.stmt-complex .arrow { color: var(--gyori-primary); }
.stmt-modification .arrow { color: var(--cogex-purple); }

.stmt-meta {
  display: flex;
  gap: 1rem;
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #64748b;
}

html.dark .stmt-meta {
  color: #94a3b8;
}

.evidence {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}
</style>
