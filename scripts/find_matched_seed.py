"""Find a degree-matched candidate seed for the matched-seed null test.

Reviewer's hypothesis: the C9orf72 gradient may be a hub artifact — any gene
of similar degree might produce a similar decay pattern. To test, we need a
gene with INDRA degree close to C9orf72's, that:

1. Is measured in the proteomics data (so |t| stats exist for the gradient)
2. Is NOT in known ALS / neurodegeneration pathways (avoid double-counting biology)
3. Has reasonable proteomics signal (non-trivial |t| values)

Output: ranked list of candidates.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd

# Local imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cliquefinder.knowledge.cogex import CoGExClient
from cliquefinder.stats.network_proximity import query_gene_degrees_batched
from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols

# Genes to exclude — ALS/FTD/neurodegeneration adjacent
ALS_EXCLUDE = {
    "C9orf72", "TARDBP", "TDP43", "FUS", "SOD1", "ATXN2", "OPTN", "UBQLN2",
    "VCP", "TBK1", "ANG", "PFN1", "MATR3", "CHCHD10", "TIA1", "HNRNPA1",
    "HNRNPA2B1", "SQSTM1", "CCNF", "KIF5A", "DCTN1", "SETX", "ALS2",
    "NEFL", "NEFM", "NEFH", "PRPH",
    # Pathway-adjacent
    "MAPT", "GRN", "APP", "PSEN1", "PSEN2", "SNCA", "PARK2", "LRRK2",
    "HTT", "ATXN1", "ATXN3", "ATXN7",
}


def main():
    # 1. Load measured genes from the proteomics metadata
    data_path = ROOT / "output/proteomics/all_als.data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Proteomics data not found at {data_path}")
    print(f"Loading measured genes from {data_path}")
    data = pd.read_csv(data_path)
    # First column is feature_id (UniProt); rest are samples
    feature_ids = data[data.columns[0]].dropna().astype(str).tolist()
    print(f"  {len(feature_ids)} measured features (UniProt accessions)")
    print("Mapping UniProt → HGNC symbols...")
    sym_to_feat = map_feature_ids_to_symbols(feature_ids, verbose=False)
    measured_genes = sorted(sym_to_feat.keys())
    print(f"  {len(measured_genes)} unique HGNC symbols mapped")

    # 2. Connect to INDRA and query degrees
    env_file = ROOT / ".env"
    print(f"Loading INDRA credentials from {env_file}")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v.strip("\"'")

    print("Connecting to INDRA CoGEx...")
    client = CoGExClient()

    # 3. Query C9orf72's degree first
    target_seed = "C9orf72"
    print(f"\nQuerying {target_seed}'s degree...")
    target_deg = query_gene_degrees_batched(
        client, [target_seed]
    ).get(target_seed, 0)
    print(f"  {target_seed} degree = {target_deg}")

    if target_deg == 0:
        print("ERROR: C9orf72 has zero degree — check INDRA connectivity")
        return

    # 4. Query degrees for all measured genes (batched)
    print(f"\nQuerying degrees for {len(measured_genes)} measured genes...")
    all_degs = query_gene_degrees_batched(
        client, measured_genes, batch_size=500
    )
    print(f"  Got degrees for {len(all_degs)} genes")

    # 5. Filter candidates: degree within ±20% of C9orf72, not in ALS exclude
    lower = int(0.8 * target_deg)
    upper = int(1.2 * target_deg)
    print(f"\nLooking for genes with degree in [{lower}, {upper}]...")
    candidates = []
    for gene, deg in all_degs.items():
        if gene in ALS_EXCLUDE:
            continue
        if lower <= deg <= upper:
            candidates.append((gene, deg))

    candidates.sort(key=lambda x: abs(x[1] - target_deg))
    print(f"  {len(candidates)} candidates")
    print(f"\nTop 25 closest by degree:")
    print(f"{'Gene':<15} {'Degree':>8}  {'|Δ from C9|':>12}")
    print("-" * 40)
    for gene, deg in candidates[:25]:
        delta = abs(deg - target_deg)
        print(f"{gene:<15} {deg:>8}  {delta:>12}")

    # 6. Save full list to file
    out = ROOT / "output/validation/gradient_specificity/matched_seed_candidates.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# C9orf72 degree = {target_deg}\n")
        f.write(f"# Filter: ±20% degree, not in ALS_EXCLUDE\n")
        f.write("gene\tdegree\tdelta_from_C9orf72\n")
        for gene, deg in candidates:
            f.write(f"{gene}\t{deg}\t{deg - target_deg}\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
