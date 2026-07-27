#!/usr/bin/env bash
# Mirror the analytical-workflow narratives into the iCloud Drive project folder.
#
# These two documents live under output/, which is gitignored as a whole, so the
# iCloud copy is a second durable location for them. The copy is one-way
# (repo -> iCloud): the repo is the source of truth. Only differing files are
# copied, so re-running is cheap and idempotent.
set -euo pipefail

SRC_DIR="/Users/noot/Documents/biomolecular-clique-finding/output"
DEST_DIR="$HOME/Library/Mobile Documents/com~apple~CloudDocs/biomolecular-clique-finding"

DOCS=(
  analytical_workflow_methods.md
  analytical_workflow_breakdown.md
)

mkdir -p "$DEST_DIR"

for doc in "${DOCS[@]}"; do
  src="$SRC_DIR/$doc"
  dest="$DEST_DIR/$doc"
  [[ -f "$src" ]] || continue
  if ! cmp -s "$src" "$dest"; then
    cp -p "$src" "$dest"
    [[ "${1:-}" == "--verbose" ]] && echo "synced: $doc"
  fi
done

if [[ "${1:-}" == "--verbose" ]]; then
  echo "dest: $DEST_DIR"
  ls -la "$DEST_DIR"
fi
exit 0
