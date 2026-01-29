"""
AnswerALS Genetic Subtype Comparisons.

This example demonstrates how to use the permutation testing framework
to compare genetic subtypes of ALS, such as C9orf72 vs Sporadic cases.

The MetadataDerivedDesign enables creation of custom comparisons by
deriving condition labels from existing metadata columns.

Example Usage:
    >>> import pandas as pd
    >>> from examples.als.genetic_contrasts import create_c9_vs_sporadic_design
    >>>
    >>> # Create the experimental design
    >>> design = create_c9_vs_sporadic_design(blocking_column="subject_id")
    >>>
    >>> # Use with PermutationTestEngine
    >>> from cliquefinder.stats.permutation_framework import (
    ...     PermutationTestEngine,
    ...     MedianPolishSummarizer,
    ...     MixedModelStatistic,
    ... )
    >>>
    >>> engine = PermutationTestEngine(
    ...     data=proteomics_matrix,  # features × samples
    ...     feature_ids=protein_ids,
    ...     metadata=sample_metadata,
    ...     summarizer=MedianPolishSummarizer(),
    ...     test=MixedModelStatistic(),
    ... )
    >>>
    >>> results = engine.run_competitive_test(
    ...     feature_sets=protein_cliques,
    ...     design=design,
    ...     feature_pool=all_regulated_proteins,
    ...     n_permutations=1000,
    ... )
"""

from __future__ import annotations

import pandas as pd

from cliquefinder.stats.permutation_framework import MetadataDerivedDesign


def create_c9_vs_sporadic_design(blocking_column: str = "subject_id") -> MetadataDerivedDesign:
    """
    Create experimental design for C9orf72 vs Sporadic ALS comparison.

    This function creates a MetadataDerivedDesign that derives genetic subtype
    labels from AnswerALS metadata columns. It demonstrates how to create
    complex experimental comparisons beyond simple two-group designs.

    Metadata Requirements:
        - 'ClinReport_Mutations_Details': Contains mutation information
        - 'phenotype': Contains CASE/CTRL labels
        - blocking_column: Typically 'subject_id' for paired/longitudinal data

    Genetic Subtype Definitions:
        - C9orf72: Cases where ClinReport_Mutations_Details contains "C9orf72"
        - Sporadic: Cases with phenotype='CASE' and no reported mutations

    Args:
        blocking_column: Column name for blocking variable (e.g., "subject_id")
                        Used for mixed models to account for repeated measures

    Returns:
        MetadataDerivedDesign configured for C9 vs Sporadic comparison

    Example:
        >>> design = create_c9_vs_sporadic_design()
        >>>
        >>> # Preview which samples will be included
        >>> metadata = pd.read_csv("sample_metadata.csv")
        >>> mask = design.sample_mask(metadata)
        >>> included_samples = metadata[mask]
        >>> print(included_samples['derived_condition'].value_counts())
        C9orf72     45
        Sporadic    123
    """
    def derive_genetic_subtype(row: pd.Series) -> str | None:
        """
        Derive genetic subtype label from metadata row.

        Args:
            row: Pandas Series representing one sample's metadata

        Returns:
            'C9orf72', 'Sporadic', or None (excluded from analysis)
        """
        # C9orf72 carrier
        mutations = str(row.get('ClinReport_Mutations_Details', ''))
        if 'C9orf72' in mutations:
            return 'C9orf72'

        # Sporadic: ALS case with no known mutations
        phenotype = row.get('phenotype', '')
        if phenotype == 'CASE' and (pd.isna(row.get('ClinReport_Mutations_Details')) or mutations == ''):
            return 'Sporadic'

        # Exclude other mutation carriers and controls
        return None

    return MetadataDerivedDesign(
        derivation_fn=derive_genetic_subtype,
        test_condition="C9orf72",
        reference_condition="Sporadic",
        blocking_column=blocking_column,
    )


def create_sod1_vs_sporadic_design(blocking_column: str = "subject_id") -> MetadataDerivedDesign:
    """
    Create experimental design for SOD1 vs Sporadic ALS comparison.

    Similar to C9orf72 comparison but for SOD1 mutations.

    Args:
        blocking_column: Column name for blocking variable

    Returns:
        MetadataDerivedDesign configured for SOD1 vs Sporadic comparison
    """
    def derive_genetic_subtype(row: pd.Series) -> str | None:
        mutations = str(row.get('ClinReport_Mutations_Details', ''))
        if 'SOD1' in mutations:
            return 'SOD1'

        phenotype = row.get('phenotype', '')
        if phenotype == 'CASE' and (pd.isna(row.get('ClinReport_Mutations_Details')) or mutations == ''):
            return 'Sporadic'

        return None

    return MetadataDerivedDesign(
        derivation_fn=derive_genetic_subtype,
        test_condition="SOD1",
        reference_condition="Sporadic",
        blocking_column=blocking_column,
    )


def create_familial_vs_sporadic_design(blocking_column: str = "subject_id") -> MetadataDerivedDesign:
    """
    Create experimental design for Familial vs Sporadic ALS comparison.

    Familial ALS: Any known mutation (C9orf72, SOD1, FUS, TARDBP, etc.)
    Sporadic ALS: No known mutations

    Args:
        blocking_column: Column name for blocking variable

    Returns:
        MetadataDerivedDesign configured for Familial vs Sporadic comparison
    """
    def derive_genetic_subtype(row: pd.Series) -> str | None:
        mutations = str(row.get('ClinReport_Mutations_Details', ''))
        phenotype = row.get('phenotype', '')

        # Must be ALS case
        if phenotype != 'CASE':
            return None

        # Familial: any mutation reported
        if not pd.isna(row.get('ClinReport_Mutations_Details')) and mutations != '':
            return 'Familial'

        # Sporadic: no mutations
        if pd.isna(row.get('ClinReport_Mutations_Details')) or mutations == '':
            return 'Sporadic'

        return None

    return MetadataDerivedDesign(
        derivation_fn=derive_genetic_subtype,
        test_condition="Familial",
        reference_condition="Sporadic",
        blocking_column=blocking_column,
    )


if __name__ == "__main__":
    """
    Demonstration of genetic subtype designs with sample data.
    """
    # Create sample metadata to demonstrate
    sample_metadata = pd.DataFrame({
        'sample_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
        'subject_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'phenotype': ['CASE', 'CASE', 'CASE', 'CASE', 'CTRL'],
        'ClinReport_Mutations_Details': ['C9orf72', '', 'SOD1 A4V', pd.NA, ''],
    })

    print("Sample Metadata:")
    print(sample_metadata)
    print("\n" + "="*70 + "\n")

    # C9orf72 vs Sporadic
    c9_design = create_c9_vs_sporadic_design()
    metadata_with_derived = sample_metadata.copy()
    metadata_with_derived['derived_condition'] = c9_design.derive_conditions(sample_metadata)
    c9_mask = c9_design.sample_mask(sample_metadata)

    print("C9orf72 vs Sporadic Design:")
    print(metadata_with_derived[['sample_id', 'phenotype', 'ClinReport_Mutations_Details', 'derived_condition']])
    print(f"\nIncluded samples (n={c9_mask.sum()}):")
    print(metadata_with_derived[c9_mask][['sample_id', 'derived_condition']])
    print("\n" + "="*70 + "\n")

    # Familial vs Sporadic
    fam_design = create_familial_vs_sporadic_design()
    metadata_with_fam = sample_metadata.copy()
    metadata_with_fam['derived_condition'] = fam_design.derive_conditions(sample_metadata)
    fam_mask = fam_design.sample_mask(sample_metadata)

    print("Familial vs Sporadic Design:")
    print(metadata_with_fam[['sample_id', 'phenotype', 'ClinReport_Mutations_Details', 'derived_condition']])
    print(f"\nIncluded samples (n={fam_mask.sum()}):")
    print(metadata_with_fam[fam_mask][['sample_id', 'derived_condition']])
