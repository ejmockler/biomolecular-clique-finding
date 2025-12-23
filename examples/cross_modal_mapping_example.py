"""
Example: Cross-Modal ID Mapping for Multi-Omics Integration

This script demonstrates how to use CrossModalIDMapper to unify gene identifiers
between proteomics (gene symbols) and RNA-seq data (Ensembl IDs or symbols).

Typical Use Cases:
    1. Validate proteomics cliques with RNA-seq expression data
    2. Filter discovered regulators to transcriptionally active genes
    3. Cross-reference multi-omics datasets

Examples run through three scenarios:
    - Scenario 1: Both datasets use gene symbols (fast path)
    - Scenario 2: RNA uses Ensembl IDs (requires MyGeneInfo mapping)
    - Scenario 3: With INDRA validation (optional, requires INDRA access)
"""

from pathlib import Path
from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper


def example_1_symbol_to_symbol():
    """
    Scenario 1: Both datasets use gene symbols.

    This is the fastest path - direct set intersection without any ID conversion.
    """
    print("\n" + "=" * 70)
    print("Example 1: Both datasets use gene symbols")
    print("=" * 70)

    # Proteomics data (gene symbols)
    protein_ids = [
        'PGK1', 'BRCA1', 'TP53', 'MDM2', 'CDKN1A',
        'BAX', 'BBC3', 'PUMA', 'GAPDH', 'ACTB'
    ]

    # RNA-seq data (also gene symbols)
    rna_ids = [
        'PGK1', 'BRCA1', 'TP53', 'CDKN1A', 'BAX',
        'GAPDH', 'ACTB', 'SOD1', 'MYC', 'JUN'
    ]

    # Create mapper (no INDRA needed for simple mapping)
    mapper = CrossModalIDMapper(use_indra=False)

    # Unify IDs
    mapping = mapper.unify_ids(
        protein_ids=protein_ids,
        rna_ids=rna_ids,
        rna_id_type='symbol'  # Both use symbols
    )

    # Print results
    print(f"\nCommon genes ({len(mapping.common_genes)}):")
    print(f"  {sorted(mapping.common_genes)}")
    print(f"\nProteomics-only genes ({len(mapping.protein_only)}):")
    print(f"  {sorted(mapping.protein_only)}")
    print(f"\nRNA-only genes ({len(mapping.rna_only)}):")
    print(f"  {sorted(mapping.rna_only)}")
    print(f"\nCross-modal overlap rate: {mapping.overlap_rate:.1%}")


def example_2_ensembl_to_symbol():
    """
    Scenario 2: RNA uses Ensembl IDs, proteomics uses symbols.

    Requires MyGeneInfo API to convert Ensembl -> Symbol.
    Results are cached to avoid redundant API calls.
    """
    print("\n" + "=" * 70)
    print("Example 2: RNA uses Ensembl IDs, proteomics uses symbols")
    print("=" * 70)

    # Proteomics data (gene symbols)
    protein_ids = [
        'PGK1', 'BRCA1', 'TP53', 'MDM2', 'GAPDH'
    ]

    # RNA-seq data (Ensembl gene IDs)
    rna_ids = [
        'ENSG00000102144',  # PGK1
        'ENSG00000012048',  # BRCA1
        'ENSG00000141510',  # TP53
        'ENSG00000111640',  # GAPDH
        'ENSG00000134061',  # CD274 (not in proteomics)
    ]

    # Create mapper with caching
    cache_dir = Path.home() / '.cache/biocore/cross_modal_example'
    mapper = CrossModalIDMapper(
        use_indra=False,
        cache_dir=cache_dir
    )

    print(f"\nCache directory: {cache_dir}")
    print("(MyGeneInfo queries will be cached here)")

    # Unify IDs - this will query MyGeneInfo API
    mapping = mapper.unify_ids(
        protein_ids=protein_ids,
        rna_ids=rna_ids,
        rna_id_type='ensembl_gene'  # RNA uses Ensembl IDs
    )

    # Print detailed summary
    print(f"\n{mapping.summary()}")


def example_3_with_indra_validation():
    """
    Scenario 3: With INDRA CoGEx validation.

    Optionally validates that gene symbols are recognized by INDRA's knowledge graph.
    This provides additional confidence in gene name quality.

    Note: Requires INDRA CoGEx access and credentials.
    """
    print("\n" + "=" * 70)
    print("Example 3: With INDRA validation (optional)")
    print("=" * 70)

    # Check if INDRA is available
    try:
        from cliquefinder.knowledge.cogex import CoGExClient

        # Proteomics data
        protein_ids = [
            'TP53', 'MDM2', 'CDKN1A', 'BAX', 'BBC3',
            'GAPDH', 'ACTB', 'FAKEGENE123'  # Invalid gene
        ]

        # RNA data
        rna_ids = [
            'TP53', 'MDM2', 'CDKN1A', 'BAX',
            'GAPDH', 'ACTB', 'MYC'
        ]

        # Create mapper with INDRA validation
        # Note: This requires INDRA_NEO4J_URL, INDRA_NEO4J_USER, INDRA_NEO4J_PASSWORD
        # in environment or .env file
        mapper = CrossModalIDMapper(use_indra=True)

        # Unify IDs with validation
        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='symbol'
        )

        print(f"\n{mapping.summary()}")

        # Check INDRA validation statistics
        if 'indra_validated' in mapping.mapping_stats:
            validated = mapping.mapping_stats['indra_validated']
            total = len(mapping.common_genes)
            print(f"\nINDRA validated {validated}/{total} common genes")
            print("(Invalid gene names filtered out)")

    except ImportError:
        print("\nINDRA CoGEx not available.")
        print("Install with: pip install git+https://github.com/indralab/indra_cogex.git")
    except ValueError as e:
        print(f"\nINDRA credentials not configured: {e}")
        print("Set INDRA_NEO4J_URL, INDRA_NEO4J_USER, INDRA_NEO4J_PASSWORD")


def example_4_filter_to_common():
    """
    Scenario 4: Convenience method - just get common genes.

    For simple use cases where you only need the intersection.
    """
    print("\n" + "=" * 70)
    print("Example 4: Convenience method - filter_to_common()")
    print("=" * 70)

    protein_ids = ['PGK1', 'BRCA1', 'TP53', 'MDM2']
    rna_ids = ['PGK1', 'BRCA1', 'TP53', 'SOD1']

    mapper = CrossModalIDMapper()

    # Get only common genes (shortcut)
    common = mapper.filter_to_common(
        protein_ids=protein_ids,
        rna_ids=rna_ids,
        rna_id_type='symbol'
    )

    print(f"\nCommon genes: {sorted(common)}")
    print("(Use this when you only need the intersection)")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Cross-Modal ID Mapping Examples")
    print("=" * 70)
    print("\nThese examples demonstrate ID mapping between proteomics and RNA-seq data.")
    print("Note: Example 2 requires internet connection for MyGeneInfo API.")
    print("      Example 3 requires INDRA CoGEx credentials.")

    # Run examples
    example_1_symbol_to_symbol()

    # Uncomment to run additional examples:
    # example_2_ensembl_to_symbol()  # Requires internet for MyGeneInfo
    # example_3_with_indra_validation()  # Requires INDRA credentials
    # example_4_filter_to_common()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
