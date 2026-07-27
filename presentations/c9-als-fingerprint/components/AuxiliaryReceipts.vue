<script setup>
import primaryAnalysis from '../../../data/publication/c9_primary_analysis.json';
import degreeNull from '../../../data/publication/c9_degree_stratified_null.json';

const contrasts = primaryAnalysis.confirmatory.contrast_order;
const sizePattern = primaryAnalysis.auxiliary_evidence.size_matched_gene_set_null
  .eightfold_threshold_pattern;
const degreeSummary = degreeNull.summary.fixed_canonical_scope_full_degree_mean;
const degreePattern = contrasts.map(contrast => degreeSummary[contrast].threshold_count);
const pattern = values => values.join(' / ');
</script>

<template>
  <div class="auxiliary-receipts">
    <section class="receipt size-receipt">
      <div class="kicker">graph-independent size control</div>
      <div class="question">Does each term's mean moderated t² exceed uniform same-size HGNC sets?</div>
      <div class="pattern">{{ pattern(sizePattern) }}</div>
      <div class="contrast-order">C9–Sporadic · C9–Control · Sporadic–Control</div>
      <div class="method">10,000 sets per term · production log₂ EB model · eightfold reporting threshold</div>
    </section>

    <section class="receipt degree-receipt">
      <div class="kicker">graph-conditional degree control</div>
      <div class="question">Do robust-scope term anchors exceed full-INDRA-degree-matched nonmembers in mean −slope?</div>
      <div class="pattern">{{ pattern(degreePattern) }}</div>
      <div class="contrast-order">Vpr-mediated import is the sole C9 non-pass</div>
      <div class="method">9,999 matched references · not per-anchor slope p · not GSEA</div>
    </section>

    <div class="boundary">
      Both reuse the same cohort and discovery-derived fixed terms. They are auxiliary consistency checks—not independent confirmation, selective inference, or external replication.
    </div>
  </div>
</template>

<style scoped>
.auxiliary-receipts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  width: 100%;
  margin-top: 0.65rem;
}
.receipt {
  min-height: 245px;
  padding: 1.05rem 1.15rem;
  border-radius: 0.6rem;
  background: rgba(100, 116, 139, 0.05);
}
.size-receipt { border: 1px solid #4ecadf; }
.degree-receipt { border: 1px solid #bf6ff7; }
.kicker {
  color: #94a3b8;
  font: 700 0.68rem/1.2 'JetBrains Mono', monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.question {
  color: #cbd5e1;
  font-size: 0.92rem;
  line-height: 1.4;
  margin-top: 0.7rem;
  min-height: 3.9rem;
}
.pattern {
  color: #4ecadf;
  font: 800 2.2rem/1 'JetBrains Mono', monospace;
  margin-top: 0.65rem;
}
.degree-receipt .pattern { color: #bf6ff7; }
.contrast-order {
  color: #f1f5f9;
  font-size: 0.72rem;
  margin-top: 0.45rem;
}
.method {
  color: #94a3b8;
  font-size: 0.68rem;
  line-height: 1.35;
  margin-top: 0.8rem;
}
.boundary {
  grid-column: 1 / -1;
  color: #94a3b8;
  border-left: 3px solid #f59e0b;
  padding: 0.55rem 0.8rem;
  font-size: 0.76rem;
  line-height: 1.4;
}
</style>
