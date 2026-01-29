"""
Example: Using Knowledge Graph Queries for ALS Research.

This script demonstrates how to use the graph query functions to:
1. Query neighbors for ALS-associated genes
2. Split results by relationship type (activation, inhibition)
3. Convert to feature sets for permutation testing
4. Run comparative analysis across multiple genes

Requirements:
    - INDRA CoGEx credentials in .env file
    - Proteomics data matrix with gene/protein IDs
    - Sample metadata for experimental design

Usage:
    python examples/als/example_graph_queries.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import knowledge graph query functions
from examples.als.graph_queries import (
    get_gene_neighbor_sets,
    get_c9orf72_neighbor_sets,
    get_als_gene_neighbor_sets,
    get_gene_neighbors_custom_categories,
)

# Import core library components
from cliquefinder.knowledge import INDRAKnowledgeSource
from cliquefinder.stats.permutation_framework import (
    PermutationTestEngine,
    MedianPolishSummarizer,
    MixedModelStatistic,
)
from examples.als.genetic_contrasts import create_c9_vs_sporadic_design


def example_single_gene_query():
    """Example 1: Query neighbors for a single gene."""
    print("=" * 80)
    print("Example 1: Single Gene Query")
    print("=" * 80)

    # Initialize knowledge source
    source = INDRAKnowledgeSource(env_file=".env")

    # Simulate a gene universe (in real use, extract from your data matrix)
    gene_universe = {
        "TP53", "AKT1", "MAPK1", "EGFR", "STAT3", "JUN", "FOS",
        "CASP3", "BCL2", "BAX", "MYC", "CDKN1A", "MDM2",
    }

    # Query C9orf72 neighbors
    print("\nQuerying C9orf72 neighbors...")
    neighbors = get_gene_neighbor_sets(
        gene_name="C9orf72",
        source=source,
        universe=gene_universe,
        min_evidence=2,
    )

    # Print results
    print(f"\nC9orf72 Activated Neighbors (n={len(neighbors['activated'])})")
    print(f"  Genes: {neighbors['activated'].entities}")

    print(f"\nC9orf72 Inhibited Neighbors (n={len(neighbors['inhibited'])})")
    print(f"  Genes: {neighbors['inhibited'].entities}")

    print(f"\nAll C9orf72 Neighbors (n={len(neighbors['all'])})")
    print(f"  Genes: {neighbors['all'].entities}")

    # Show relationship breakdown
    print("\nRelationship Breakdown:")
    by_rel = neighbors["all"].by_relationship()
    for rel_name, rel_result in by_rel.items():
        print(f"  {rel_name}: {len(rel_result)} genes")

    return neighbors


def example_multiple_als_genes():
    """Example 2: Query neighbors for all major ALS genes."""
    print("\n" + "=" * 80)
    print("Example 2: Multiple ALS Genes")
    print("=" * 80)

    # Initialize knowledge source
    source = INDRAKnowledgeSource(env_file=".env")

    # Simulate a gene universe
    gene_universe = {
        "TP53", "AKT1", "MAPK1", "EGFR", "STAT3", "JUN", "FOS",
        "CASP3", "BCL2", "BAX", "MYC", "CDKN1A", "MDM2",
    }

    # Query all ALS genes at once
    print("\nQuerying neighbors for C9orf72, SOD1, TARDBP, and FUS...")
    als_neighbors = get_als_gene_neighbor_sets(
        source=source,
        universe=gene_universe,
        min_evidence=2,
    )

    # Compare neighbor counts across genes
    print("\nNeighbor Counts by Gene:")
    print(f"{'Gene':<10} {'Activated':<12} {'Inhibited':<12} {'Total':<12}")
    print("-" * 50)
    for gene, neighbors in als_neighbors.items():
        n_activated = len(neighbors["activated"])
        n_inhibited = len(neighbors["inhibited"])
        n_total = len(neighbors["all"])
        print(f"{gene:<10} {n_activated:<12} {n_inhibited:<12} {n_total:<12}")

    # Find common neighbors across all ALS genes
    print("\nFinding common neighbors across all ALS genes...")
    all_gene_neighbors = [n["all"].entities for n in als_neighbors.values()]
    common_neighbors = set.intersection(*all_gene_neighbors) if all_gene_neighbors else set()

    print(f"Common neighbors (n={len(common_neighbors)}): {common_neighbors}")

    return als_neighbors


def example_custom_relationship_categories():
    """Example 3: Custom relationship categories."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Relationship Categories")
    print("=" * 80)

    # Initialize knowledge source
    source = INDRAKnowledgeSource(env_file=".env")

    # Define custom relationship categories
    custom_categories = {
        "transcriptional": {"IncreaseAmount", "DecreaseAmount"},
        "post_translational": {"Phosphorylation", "Dephosphorylation"},
        "protein_interaction": {"Complex", "BindsTo"},
    }

    # Simulate a gene universe
    gene_universe = {
        "TP53", "AKT1", "MAPK1", "EGFR", "STAT3", "JUN", "FOS",
        "CASP3", "BCL2", "BAX", "MYC", "CDKN1A", "MDM2",
    }

    # Query TP53 with custom categories
    print("\nQuerying TP53 with custom relationship categories...")
    neighbors = get_gene_neighbors_custom_categories(
        gene_name="TP53",
        source=source,
        relationship_categories=custom_categories,
        universe=gene_universe,
        min_evidence=1,
    )

    # Print results by category
    print("\nTP53 Neighbors by Category:")
    for category, result in neighbors.items():
        if category != "all":
            print(f"\n{category.upper()} (n={len(result)})")
            print(f"  Genes: {result.entities}")


def example_integration_with_permutation_testing():
    """Example 4: Integration with permutation testing framework."""
    print("\n" + "=" * 80)
    print("Example 4: Integration with Permutation Testing")
    print("=" * 80)

    # This example shows the full workflow from query to statistical test
    # Note: This requires actual proteomics data, so we'll just show the pattern

    print("\nWorkflow:")
    print("1. Load proteomics data and metadata")
    print("2. Query knowledge graph for gene neighbors")
    print("3. Convert to feature sets")
    print("4. Run permutation test")
    print("\nCode pattern:")

    code_example = '''
    # 1. Load data
    matrix = np.load("proteomics_matrix.npy")  # features × samples
    feature_ids = pd.read_csv("feature_ids.csv")["gene_symbol"].tolist()
    metadata = pd.read_csv("sample_metadata.csv")

    # 2. Initialize knowledge source
    source = INDRAKnowledgeSource(env_file=".env")

    # 3. Query neighbors for C9orf72
    gene_universe = set(feature_ids)
    neighbors = get_gene_neighbor_sets(
        "C9orf72", source, gene_universe, min_evidence=2
    )

    # 4. Convert to feature sets
    activated_fs = neighbors["activated"].to_feature_set("C9_activated")
    inhibited_fs = neighbors["inhibited"].to_feature_set("C9_inhibited")

    # 5. Setup permutation test
    design = create_c9_vs_sporadic_design(blocking_column="subject_id")
    engine = PermutationTestEngine(
        data=matrix,
        feature_ids=feature_ids,
        metadata=metadata,
        summarizer=MedianPolishSummarizer(),
        test=MixedModelStatistic(),
    )

    # 6. Run competitive test
    results = engine.run_competitive_test(
        feature_sets=[activated_fs, inhibited_fs],
        design=design,
        n_permutations=1000,
    )

    # 7. Analyze results
    for result in results:
        print(f"{result.feature_set_id}: p={result.p_value:.4f}")
    '''

    print(code_example)


def example_comparative_analysis():
    """Example 5: Comparative analysis across genes."""
    print("\n" + "=" * 80)
    print("Example 5: Comparative Analysis Pattern")
    print("=" * 80)

    print("\nPattern: Compare regulation across multiple genes")
    print("\nThis pattern is useful for questions like:")
    print("  - Do different ALS genes regulate similar pathways?")
    print("  - Are there gene-specific vs. shared regulatory targets?")
    print("  - Which genes have the most extensive regulatory networks?")

    code_example = '''
    # Get neighbors for multiple genes
    genes_of_interest = ["C9orf72", "SOD1", "TARDBP", "TP53", "EGFR"]
    all_neighbors = {}

    for gene in genes_of_interest:
        neighbors = get_gene_neighbor_sets(
            gene, source, gene_universe, min_evidence=2
        )
        all_neighbors[gene] = neighbors

    # Create feature sets for competitive testing
    feature_sets = []
    for gene, neighbors in all_neighbors.items():
        # Add activated targets as separate feature sets
        activated_fs = neighbors["activated"].to_feature_set(f"{gene}_activated")
        feature_sets.append(activated_fs)

        # Add inhibited targets as separate feature sets
        inhibited_fs = neighbors["inhibited"].to_feature_set(f"{gene}_inhibited")
        feature_sets.append(inhibited_fs)

    # Run competitive test to see which gene's targets are most enriched
    results = engine.run_competitive_test(
        feature_sets=feature_sets,
        design=design,
        n_permutations=1000,
    )

    # Analyze which gene's regulatory targets are most significantly altered
    significant_results = [r for r in results if r.p_value < 0.05]
    for result in sorted(significant_results, key=lambda r: r.p_value):
        print(f"{result.feature_set_id}: ES={result.enrichment_score:.3f}, p={result.p_value:.4f}")
    '''

    print(code_example)


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  Knowledge Graph Query Examples for ALS Research".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\nNote: These examples require INDRA CoGEx credentials in .env file")
    print("      Some examples are code patterns only (no actual execution)")
    print()

    # Run examples that don't require real data
    try:
        # Example 1: Single gene query (requires INDRA credentials)
        # example_single_gene_query()

        # Example 2: Multiple ALS genes (requires INDRA credentials)
        # example_multiple_als_genes()

        # Example 3: Custom categories (requires INDRA credentials)
        # example_custom_relationship_categories()

        # Example 4: Integration pattern (code only)
        example_integration_with_permutation_testing()

        # Example 5: Comparative analysis pattern (code only)
        example_comparative_analysis()

        print("\n" + "=" * 80)
        print("Examples completed!")
        print("=" * 80)
        print("\nTo run the actual queries, uncomment the example functions and ensure:")
        print("  1. INDRA CoGEx credentials are in .env file")
        print("  2. Gene universe is extracted from your actual data")
        print("  3. Network connection is available")
        print()

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nThis is expected if INDRA credentials are not configured.")
        print("See README for setup instructions.")


if __name__ == "__main__":
    main()
