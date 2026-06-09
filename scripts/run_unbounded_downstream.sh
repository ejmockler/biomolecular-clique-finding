#!/usr/bin/env bash
# Wave 24l unbounded downstream orchestrator.
# Runs after all 3 unbounded landscape result.json files exist.
# Sequence: convergence + GSEA in parallel → confirmatory → comparison.
#
# Idempotent: re-running is safe (each script writes to its own out-dir
# and overwrites cleanly).

set -euo pipefail

cd ~/biomolecular-clique-finding

VENV=.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)

CONTRASTS=(
  "proteome:c9spor"
  "c9_vs_control:c9ctrl"
  "sporadic_vs_control:spctrl"
)

verify_inputs() {
  local missing=0
  for entry in "${CONTRASTS[@]}"; do
    local indir=${entry%:*}
    local rj=output/landscape_${indir}_measured_only_unbounded/result.json
    if [[ ! -f $rj ]]; then
      echo "MISSING: $rj"
      missing=1
    fi
  done
  if (( missing )); then
    echo "FATAL: one or more unbounded result.json files missing." >&2
    exit 1
  fi
  echo "All 3 unbounded result.json files present."
}

run_convergence() {
  echo "=== Convergence diagnostic ==="
  for entry in "${CONTRASTS[@]}"; do
    local indir=${entry%:*} tag=${entry#*:}
    $VENV scripts/analyze_convergence.py \
      --result-dir output/landscape_${indir}_measured_only_unbounded \
      --out-dir output/landscape_convergence_${tag}_unbounded \
      2>&1 | tee output/logs/convergence_${tag}_${TS}.log
  done
}

launch_gsea() {
  echo "=== Launching 3 GSEA discoveries in parallel tmux ==="
  for entry in "${CONTRASTS[@]}"; do
    local indir=${entry%:*} tag=${entry#*:}
    local sname="gu-${tag}"
    # If a stale tmux session exists, kill it.
    tmux kill-session -t "$sname" 2>/dev/null || true
    tmux new-session -d -s "$sname" \
      "cd ~/biomolecular-clique-finding && \
       $VENV -u scripts/run_landscape_gsea.py \
         --result-dir output/landscape_${indir}_measured_only_unbounded \
         --out-dir output/landscape_gsea_${tag}_measured_only_unbounded \
       2>&1 | tee output/logs/gsea_unbounded_${tag}_${TS}.log"
    echo "  launched tmux $sname"
  done
}

wait_gsea() {
  echo "=== Waiting for GSEA completion ==="
  local timeout=900
  local elapsed=0
  while (( elapsed < timeout )); do
    local active=0
    for entry in "${CONTRASTS[@]}"; do
      local tag=${entry#*:}
      if tmux has-session -t "gu-${tag}" 2>/dev/null; then
        active=$((active+1))
      fi
    done
    if (( active == 0 )); then
      echo "All GSEA sessions exited."
      break
    fi
    sleep 15
    elapsed=$((elapsed+15))
    echo "  ${active}/3 GSEA sessions still active (${elapsed}s elapsed)"
  done
  # Sanity: summary.csv per contrast.
  for entry in "${CONTRASTS[@]}"; do
    local tag=${entry#*:}
    local sc=output/landscape_gsea_${tag}_measured_only_unbounded/summary.csv
    if [[ ! -f $sc ]]; then
      echo "WARNING: $sc missing — GSEA may have failed" >&2
    fi
  done
}

run_confirmatory() {
  echo "=== Bonferroni-8 confirmatory per contrast ==="
  for entry in "${CONTRASTS[@]}"; do
    local tag=${entry#*:}
    $VENV scripts/analyze_landscape_confirmatory.py \
      --gsea-dir output/landscape_gsea_${tag}_measured_only_unbounded \
      --out-dir output/landscape_confirmatory_${tag}_measured_only_unbounded \
      --scope robust 2>&1 | tee output/logs/confirmatory_unbounded_${tag}_${TS}.log
  done
}

run_comparison() {
  echo "=== Comparison: bounded h=2 vs unbounded ==="
  $VENV scripts/compare_bounded_unbounded.py \
    --bounded-dir "output/landscape_confirmatory_{c9spor,c9ctrl,spctrl}_measured_only" \
    --unbounded-dir "output/landscape_confirmatory_{c9spor,c9ctrl,spctrl}_measured_only_unbounded" \
    --out output/wave_24l_bounded_vs_unbounded.md \
    2>&1 | tee output/logs/comparison_${TS}.log
  echo "Wrote output/wave_24l_bounded_vs_unbounded.md"
}

verify_inputs
run_convergence
launch_gsea
wait_gsea
run_confirmatory
run_comparison

echo
echo "=== WAVE_24L_UNBOUNDED_DOWNSTREAM_DONE ==="
echo "Artifacts:"
echo "  output/landscape_convergence_*_unbounded/"
echo "  output/landscape_gsea_*_measured_only_unbounded/"
echo "  output/landscape_confirmatory_*_measured_only_unbounded/"
echo "  output/wave_24l_bounded_vs_unbounded.md"
