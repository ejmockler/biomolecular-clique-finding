#!/usr/bin/env bash
# Historical transform-control: run the archived raw, measured-only C9
# landscapes against the current enrichment corpus.  This isolates annotation
# corpus drift from the raw-vs-log2 intensity change; it is not a production
# analysis driver.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
BACKUP_DIR=${BACKUP_DIR:-"$ROOT/output/_remote_backup_measured_only"}

cd "$ROOT"

echo "=== RAW measured-only GSEA against the current corpus ==="

"$PYTHON" scripts/run_landscape_gsea.py \
  --result-dir "$BACKUP_DIR/landscape_proteome_measured_only" \
  --out-dir output/landscape_gsea_raw_TODAY_c9spor_measured_only \
  --permutation-num 1000 \
  --scope-set both

"$PYTHON" scripts/run_landscape_gsea.py \
  --result-dir "$BACKUP_DIR/landscape_c9_vs_control_measured_only" \
  --out-dir output/landscape_gsea_raw_TODAY_c9ctrl_measured_only \
  --permutation-num 1000 \
  --scope-set both

echo "=== RAW CORPUS CONTROL DONE ==="
