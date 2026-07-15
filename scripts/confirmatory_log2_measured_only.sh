#!/usr/bin/env bash
# Measured-only log2-vs-raw confirmatory (gated GO from wf_58c4bf7e-746).
# G1 gate: faithfulness on the measured-only graph MUST pass before trusting log2.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PY="${PYTHON:-.venv/bin/python}"
BK=output/_remote_backup_measured_only
echo "=================== STEP 1-2: FAITHFULNESS GATES (measured-only graph) ==================="
echo "--- gate C9-vs-SPORADIC ---"
"${PY}" scripts/log2_sensitivity.py --landscape-dir "${BK}/landscape_proteome_measured_only" --c1 C9ORF72 --c2 SPORADIC --nperm 0 --gate-only
echo "--- gate C9-vs-CONTROL ---"
"${PY}" scripts/log2_sensitivity.py --landscape-dir "${BK}/landscape_c9_vs_control_measured_only" --c1 C9ORF72 --c2 CONTROL --nperm 0 --gate-only
echo "GATES PASSED (script would have exited nonzero otherwise)"
echo "=================== STEP 3-4: EMIT log2 result.json (measured-only) ==================="
"${PY}" scripts/log2_sensitivity.py --landscape-dir "${BK}/landscape_proteome_measured_only" --c1 C9ORF72 --c2 SPORADIC --nperm 0 \
   --emit-log2-dir output/landscape_proteome_measured_only_log2 >/tmp/emit_mo_spor.log 2>&1; tail -3 /tmp/emit_mo_spor.log
"${PY}" scripts/log2_sensitivity.py --landscape-dir "${BK}/landscape_c9_vs_control_measured_only" --c1 C9ORF72 --c2 CONTROL --nperm 0 \
   --emit-log2-dir output/landscape_c9_vs_control_measured_only_log2 >/tmp/emit_mo_ctrl.log 2>&1; tail -3 /tmp/emit_mo_ctrl.log
echo "=================== STEP 5-6: GSEA on log2 measured-only (perm=1000, both scopes) ==================="
"${PY}" scripts/run_landscape_gsea.py --result-dir output/landscape_proteome_measured_only_log2 \
   --out-dir output/landscape_gsea_log2_c9spor_measured_only --permutation-num 1000 --scope-set both >/tmp/gsea_mo_spor.log 2>&1; tail -3 /tmp/gsea_mo_spor.log
"${PY}" scripts/run_landscape_gsea.py --result-dir output/landscape_c9_vs_control_measured_only_log2 \
   --out-dir output/landscape_gsea_log2_c9ctrl_measured_only --permutation-num 1000 --scope-set both >/tmp/gsea_mo_ctrl.log 2>&1; tail -3 /tmp/gsea_mo_ctrl.log
echo "=================== DONE ==================="
