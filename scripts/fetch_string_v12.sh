#!/usr/bin/env bash
# Fetch STRING v12.0 human physical-PPI links for a future audited rerun of the
# withdrawn Wave 24k T43 INDRA-vs-STRING comparison.
#
# The downloaded archive is ~8.5 MB and is gitignored under data/string/.
# No legacy STRING numbers remain in the publication figure source. This file
# is only a reference substrate; reuse requires a same-orientation canonical
# log2, measured-only derivation with frozen mapping and machine-readable output.
#
# Source: https://stringdb-downloads.org/
# License: STRING data is freely available for academic use under CC BY 4.0.
#
# Usage:
#   bash scripts/fetch_string_v12.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/data/string"
URL="https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz"
EXPECTED_FILE="${DEST_DIR}/9606.protein.physical.links.v12.0.txt.gz"

mkdir -p "${DEST_DIR}"

if [[ -f "${EXPECTED_FILE}" ]]; then
  echo "STRING v12 file already present: ${EXPECTED_FILE}"
  echo "  size: $(wc -c < "${EXPECTED_FILE}") bytes"
  echo "  sha256: $(shasum -a 256 "${EXPECTED_FILE}" | awk '{print $1}')"
  exit 0
fi

echo "Downloading STRING v12 human physical-PPI links to ${EXPECTED_FILE}..."
curl -fL "${URL}" -o "${EXPECTED_FILE}"

echo
echo "Downloaded:"
echo "  path: ${EXPECTED_FILE}"
echo "  size: $(wc -c < "${EXPECTED_FILE}") bytes"
echo "  sha256: $(shasum -a 256 "${EXPECTED_FILE}" | awk '{print $1}')"
