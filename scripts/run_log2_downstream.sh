#!/usr/bin/env bash
# Bounded h=2 log2 downstream: GSEA + Bonferroni-8 confirmatory on the
# log2 published-primary re-run.  Mirror of run_unbounded_downstream.sh but
# for the *_measured_only_log2 dirs produced by run_landscape_log2.py.
#
# Runs after the 3 log2 landscape result.json files exist.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p output/logs

# tag : landscape-dir-stem
CONTRASTS=(
  "c9spor:landscape_proteome"
  "c9ctrl:landscape_c9_vs_control"
  "spctrl:landscape_sporadic_vs_control"
)

echo "=== verify 3 log2 landscape result.json present ==="
missing=0
for entry in "${CONTRASTS[@]}"; do
  stem=${entry#*:}
  rj="output/${stem}_measured_only_log2/result.json"
  if [[ ! -f $rj ]]; then echo "MISSING: $rj"; missing=1; else echo "OK: $rj"; fi
done
(( missing )) && { echo "FATAL: log2 landscapes incomplete." >&2; exit 1; }

echo "=== GSEA per contrast (sequential; each logs its consumed transform) ==="
for entry in "${CONTRASTS[@]}"; do
  tag=${entry%:*}; stem=${entry#*:}
  $VENV -u scripts/run_landscape_gsea.py \
    --result-dir "output/${stem}_measured_only_log2" \
    --out-dir    "output/landscape_gsea_${tag}_measured_only_log2" \
    --permutation-num 1000 --scope-set both \
    2>&1 | tee "output/logs/gsea_log2_${tag}_${TS}.log"
done

echo "=== Bonferroni-8 confirmatory per contrast (robust scope) ==="
for entry in "${CONTRASTS[@]}"; do
  tag=${entry%:*}
  $VENV scripts/analyze_landscape_confirmatory.py \
    --gsea-dir "output/landscape_gsea_${tag}_measured_only_log2" \
    --out-dir  "output/landscape_confirmatory_${tag}_measured_only_log2" \
    --scope robust \
    2>&1 | tee "output/logs/confirmatory_log2_${tag}_${TS}.log"
done

echo
echo "=== Bonferroni-8 pass counts (compare vs raw canonical 7/6/0) ==="
for entry in "${CONTRASTS[@]}"; do
  tag=${entry%:*}
  cdir="output/landscape_confirmatory_${tag}_measured_only_log2"
  $VENV - "$cdir" "$tag" <<'PY'
import sys, os, pandas as pd
cdir, tag = sys.argv[1], sys.argv[2]
f = os.path.join(cdir, "confirmatory_8terms_robust.csv")
if not os.path.exists(f):
    print(f"  {tag}: (missing {f})"); sys.exit(0)
df = pd.read_csv(f)
def truthy(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])
n_pass = int(truthy(df["bonferroni_pass"]).sum()) if "bonferroni_pass" in df else None
print(f"  {tag}: {n_pass}/{len(df)} Bonferroni-8 pass")
PY
done
echo "=== LOG2_DOWNSTREAM_DONE ==="
