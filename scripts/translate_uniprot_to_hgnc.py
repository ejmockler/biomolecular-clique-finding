#!/usr/bin/env python3
"""Translate UniProt protein IDs in the proteomics matrix to HGNC gene symbols
using the INDRA CoGEx knowledge graph (via its REST API)."""

import json
import re
import urllib.request
import urllib.error
import time
import sys

INPUT_FILE = "Data_AnswerALS-436-P_proteomics-protein-matrix_correctedImputed.txt"
OUTPUT_TSV = "uniprot_to_hgnc_mapping.tsv"

COGEX_URL = "https://discovery.indra.bio/cogex/api/cypher"

QUERY = """
MATCH (u:BioEntity)
WHERE u.id IN $ids
RETURN u.id AS uniprot_id, u.name AS name
"""

BATCH_SIZE = 300


def extract_uniprot_ids(filepath: str) -> list[str]:
    """Extract UniProt accessions from the Protein column."""
    ids = []
    with open(filepath) as f:
        header = f.readline()  # skip header
        for line in f:
            protein = line.split("\t")[0]
            # format: 1/sp|ACCESSION|NAME_HUMAN
            match = re.search(r"\|([A-Z0-9]+)\|", protein)
            if match:
                ids.append(match.group(1))
    return ids


def query_cogex_batch(uniprot_ids: list[str]) -> dict[str, str]:
    """Query CoGEx for a batch of UniProt IDs, return {accession: gene_name}."""
    prefixed = [f"uniprot:{uid}" for uid in uniprot_ids]
    payload = {
        "query": QUERY.strip(),
        "parameters": {"ids": prefixed},
        "max_results": len(prefixed) + 50,
        "timeout_ms": 60000,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        COGEX_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  ERROR querying CoGEx: {e}", file=sys.stderr)
        return {}

    mapping = {}
    for row in result.get("results", []):
        uid = row["uniprot_id"].replace("uniprot:", "")
        name = row.get("name") or ""
        if name:
            mapping[uid] = name
    return mapping


def main():
    print(f"Reading protein IDs from {INPUT_FILE}...")
    ids = extract_uniprot_ids(INPUT_FILE)
    print(f"  Found {len(ids)} UniProt accessions")

    all_mappings: dict[str, str] = {}
    n_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Querying batch {batch_num}/{n_batches} ({len(batch)} IDs)...")
        mappings = query_cogex_batch(batch)
        all_mappings.update(mappings)
        if batch_num < n_batches:
            time.sleep(0.5)  # be polite

    # Write output
    mapped = 0
    unmapped = 0
    with open(OUTPUT_TSV, "w") as f:
        f.write("uniprot_accession\thgnc_symbol\n")
        for uid in ids:
            symbol = all_mappings.get(uid, "")
            f.write(f"{uid}\t{symbol}\n")
            if symbol:
                mapped += 1
            else:
                unmapped += 1

    print(f"\nResults written to {OUTPUT_TSV}")
    print(f"  Mapped:   {mapped}/{len(ids)}")
    print(f"  Unmapped: {unmapped}/{len(ids)}")

    if unmapped > 0:
        print("\nUnmapped IDs:")
        for uid in ids:
            if uid not in all_mappings:
                print(f"  {uid}")


if __name__ == "__main__":
    main()
