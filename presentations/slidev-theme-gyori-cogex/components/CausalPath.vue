<script setup lang="ts">
/**
 * Causal pathway visualization
 *
 * Linear display of causal relationships with arrows.
 */

interface PathNode {
  name: string
  type?: 'gene' | 'protein' | 'process'
}

interface PathEdge {
  type: 'activation' | 'inhibition'
}

defineProps<{
  nodes: PathNode[]
  edges?: PathEdge[]
}>()
</script>

<template>
  <div class="causal-path">
    <template v-for="(node, i) in nodes" :key="i">
      <div class="path-node" :class="[`node-${node.type || 'gene'}`]">
        {{ node.name }}
      </div>
      <div
        v-if="i < nodes.length - 1"
        class="path-edge"
        :class="[edges?.[i]?.type === 'inhibition' ? 'edge-inhibition' : 'edge-activation']"
      >
        <span v-if="edges?.[i]?.type === 'inhibition'">⊣</span>
        <span v-else>→</span>
      </div>
    </template>
  </div>
</template>

<style scoped>
.causal-path {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 12px;
  overflow-x: auto;
}

html.dark .causal-path {
  background: #0f172a;
}

.path-node {
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
  white-space: nowrap;
}

.node-gene {
  background: rgba(23, 162, 184, 0.15);
  color: var(--gyori-primary);
  font-style: italic;
  border: 2px solid var(--gyori-primary);
}

.node-protein {
  background: rgba(191, 64, 191, 0.15);
  color: var(--cogex-purple);
  font-family: var(--font-mono);
  border: 2px solid var(--cogex-purple);
}

.node-process {
  background: rgba(91, 185, 0, 0.15);
  color: var(--bio-green);
  border: 2px solid var(--bio-green);
  border-radius: 20px;
}

.path-edge {
  font-size: 1.5rem;
  font-weight: bold;
  display: flex;
  align-items: center;
}

.edge-activation {
  color: var(--bio-green);
}

.edge-inhibition {
  color: #dc3545;
}
</style>
