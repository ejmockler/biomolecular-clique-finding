"""
PreparedCliqueExperiment and the ``prepare_experiment()`` factory function.

A ``PreparedCliqueExperiment`` is an immutable snapshot of preprocessed data
that ALL statistical methods receive, ensuring fair comparison and
reproducibility. Create one via :func:`prepare_experiment`.

These types are re-exported from ``method_comparison`` for backward
compatibility -- prefer importing from there in application code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


# =============================================================================
# PreparedCliqueExperiment
# =============================================================================


@dataclass(frozen=True)
class PreparedCliqueExperiment:
    """
    Immutable snapshot of preprocessed data ready for testing.

    ALL methods receive the same PreparedCliqueExperiment, ensuring:
        1. No preprocessing-induced differences between methods
        2. Fair comparison (same input data)
        3. Reproducibility (frozen state)

    This class should be created via prepare_experiment() factory function.

    Attributes:
        data: Expression matrix (n_features, n_samples) as float64.
            This array should be treated as immutable after creation.
        feature_ids: Tuple of feature identifiers (immutable).
        feature_to_idx: Mapping from feature ID to row index.
            Note: dict is mutable but frozen class prevents reassignment.
        sample_metadata: DataFrame with sample annotations.
        condition_column: Name of the condition column in metadata.
        subject_column: Name of the subject column (None for no random effects).
        conditions: Ordered tuple of unique conditions.
        n_samples: Number of samples.
        cliques: Tuple of CliqueDefinition objects (immutable).
        clique_to_feature_indices: Mapping from clique_id to feature row indices.
        contrast: Tuple of (test_condition, reference_condition).
        contrast_name: Human-readable contrast name (e.g., "ALS_vs_Control").
        preprocessing_params: Dict capturing preprocessing provenance.
        creation_timestamp: ISO format timestamp of creation.

    Example:
        >>> experiment = prepare_experiment(
        ...     data=expression_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_metadata=metadata,
        ...     cliques=clique_definitions,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ... )
        >>> print(f"Features: {experiment.n_features}, Cliques: {experiment.n_cliques}")
        >>> clique_data, clique_ids = experiment.get_clique_data("TP53")
    """

    # Expression data (preprocessed)
    data: NDArray[np.float64]
    feature_ids: tuple[str, ...]
    feature_to_idx: dict[str, int]

    # Sample information
    sample_metadata: object  # pd.DataFrame - using object for frozen compatibility
    condition_column: str
    subject_column: str | None
    conditions: tuple[str, ...]
    n_samples: int

    # Clique definitions
    cliques: tuple[object, ...]  # tuple[CliqueDefinition, ...]
    clique_to_feature_indices: dict[str, tuple[int, ...]]

    # ID mapping (symbol -> feature_id for UniProt/Ensembl translation)
    symbol_to_feature: dict[str, str]

    # Contrast
    contrast: tuple[str, str]
    contrast_name: str

    # Provenance
    preprocessing_params: dict[str, object]
    creation_timestamp: str

    def __post_init__(self):
        """Enforce true immutability for mutable fields."""
        from types import MappingProxyType
        # Make NDArray read-only
        data_copy = self.data.copy()
        data_copy.flags.writeable = False
        object.__setattr__(self, 'data', data_copy)
        # Wrap dicts as read-only MappingProxyType
        for attr in ('feature_to_idx', 'clique_to_feature_indices',
                     'symbol_to_feature', 'preprocessing_params'):
            val = getattr(self, attr)
            if isinstance(val, dict) and not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))

    @property
    def n_features(self) -> int:
        """Number of features (proteins/genes) in the data."""
        return self.data.shape[0]

    @property
    def n_cliques(self) -> int:
        """Number of clique definitions."""
        return len(self.cliques)

    def get_clique_data(
        self, clique_id: str
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Extract expression data for a specific clique.

        Args:
            clique_id: Identifier of the clique to extract

        Returns:
            Tuple of (data_subset, feature_ids_subset) where:
                - data_subset has shape (n_proteins_found, n_samples)
                - feature_ids_subset lists the feature IDs included
                - Returns empty array and list if clique has no proteins in data
        """
        indices = self.clique_to_feature_indices.get(clique_id, ())
        if not indices:
            return np.array([]).reshape(0, self.n_samples), []

        data_subset = self.data[list(indices), :]
        ids_subset = [self.feature_ids[i] for i in indices]
        return data_subset, ids_subset

    def get_condition_mask(self, condition: str) -> NDArray[np.bool_]:
        """
        Get boolean mask for samples in given condition.

        Args:
            condition: Condition label to match

        Returns:
            Boolean array of shape (n_samples,) where True indicates
            the sample belongs to the specified condition.

        Raises:
            TypeError: If sample_metadata is not a pandas DataFrame
        """
        import pandas as pd

        if isinstance(self.sample_metadata, pd.DataFrame):
            return (self.sample_metadata[self.condition_column] == condition).values
        raise TypeError("sample_metadata must be a pandas DataFrame")

    def get_design_matrix(self) -> NDArray[np.float64]:
        """
        Build design matrix for OLS/LMM.

        Constructs a design matrix with an intercept column and dummy-coded
        condition variables. The first condition (alphabetically) is used as
        the reference level and is dropped.

        Returns:
            Design matrix with shape (n_samples, n_conditions).
            First column is the intercept (constant 1).

        Raises:
            TypeError: If sample_metadata is not a pandas DataFrame
            ImportError: If statsmodels is not installed
        """
        import pandas as pd

        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError(
                "statsmodels is required for design matrix construction. "
                "Install with: pip install statsmodels"
            )

        if not isinstance(self.sample_metadata, pd.DataFrame):
            raise TypeError("sample_metadata must be a pandas DataFrame")

        condition_cat = pd.Categorical(
            self.sample_metadata[self.condition_column],
            categories=list(self.conditions)
        )
        X = pd.get_dummies(
            pd.DataFrame({'condition': condition_cat}),
            drop_first=True,
            dtype=float
        )
        X = sm.add_constant(X)
        return X.values


# =============================================================================
# Factory Function: prepare_experiment()
# =============================================================================


def prepare_experiment(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: object,  # pd.DataFrame
    cliques: list[object],    # list[CliqueDefinition]
    condition_column: str,
    contrast: tuple[str, str],
    subject_column: str | None = None,
    normalization_method: str = "median",
    imputation_method: str = "min_feature",
    map_ids: bool = True,
    verbose: bool = True,
    precomputed_symbol_map: dict[str, str] | None = None,
) -> PreparedCliqueExperiment:
    """
    Prepare data for multi-method comparison.

    This is the SINGLE preprocessing entry point. All methods receive
    the same prepared data, ensuring fair comparison.

    Preprocessing pipeline:
        1. Copy data (never modify input)
        2. Normalize (median centering, quantile, etc.)
        3. Impute missing values
        4. Map feature IDs (UniProt -> Symbol if needed)
        5. Build clique -> feature index mapping
        6. Validate contrast

    Args:
        data: Expression matrix (n_features, n_samples) of log2 intensities.
        feature_ids: Feature identifiers (UniProt, Ensembl, or Symbol).
        sample_metadata: DataFrame with sample annotations. Must contain
            the condition_column and optionally subject_column.
        cliques: List of CliqueDefinition objects to analyze.
        condition_column: Metadata column name for condition labels.
        contrast: Tuple of (test_condition, reference_condition).
        subject_column: Optional metadata column for biological replicates
            (required for LMM).
        normalization_method: Normalization to apply. Options: "none",
            "median", "quantile", "global_standards", "vsn".
        imputation_method: Imputation for missing values. Options: "none",
            "min_feature", "min_global", "min_sample", "aft", "knn", "qrilc".
        map_ids: Whether to map feature IDs to gene symbols for clique
            matching. Set to False if IDs are already gene symbols.
        verbose: Print progress messages.

    Returns:
        PreparedCliqueExperiment (frozen, immutable snapshot).

    Raises:
        ValueError: If contrast conditions not found in data.
        ValueError: If no cliques have proteins in data.

    Example:
        >>> from cliquefinder.stats.method_comparison import prepare_experiment
        >>> experiment = prepare_experiment(
        ...     data=expression_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_metadata=metadata,
        ...     cliques=clique_definitions,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ...     subject_column='subject_id',
        ... )
        >>> print(f"Prepared: {experiment.n_features} features, {experiment.n_cliques} cliques")
    """
    import pandas as pd
    from datetime import datetime

    # Import preprocessing modules
    from .normalization import NormalizationMethod, normalize
    from .missing import ImputationMethod, impute_missing_values
    from .clique_analysis import map_feature_ids_to_symbols

    # Convert string to enum if needed
    if isinstance(normalization_method, str):
        norm_method = NormalizationMethod(normalization_method)
    else:
        norm_method = normalization_method

    if isinstance(imputation_method, str):
        imp_method = ImputationMethod(imputation_method)
    else:
        imp_method = imputation_method

    if verbose:
        print("Preparing experiment for method comparison")
        print(f"  Data: {data.shape[0]} features x {data.shape[1]} samples")
        print(f"  Cliques: {len(cliques)}")
        print(f"  Contrast: {contrast[0]} vs {contrast[1]}")

    # 1. Copy and convert to float64 (never modify input)
    work_data = data.astype(np.float64).copy()
    work_ids = list(feature_ids)

    # 2. Normalize
    if norm_method != NormalizationMethod.NONE:
        if verbose:
            print(f"  Normalizing: {norm_method.value}")
        norm_result = normalize(work_data, method=norm_method)
        work_data = norm_result.data

    # 3. Impute missing values
    if imp_method != ImputationMethod.NONE:
        n_missing_before = int(np.sum(np.isnan(work_data)))
        if verbose and n_missing_before > 0:
            missing_rate = n_missing_before / work_data.size
            print(f"  Missing values: {n_missing_before} ({missing_rate:.1%})")
            print(f"  Imputing: {imp_method.value}")
        imp_result = impute_missing_values(work_data, method=imp_method)
        work_data = imp_result.data

    # 4. ID mapping (UniProt/Ensembl -> Symbol)
    # If precomputed_symbol_map is provided, use it directly (critical for bootstrap efficiency)
    symbol_to_feature: dict[str, str] = {}
    if precomputed_symbol_map is not None:
        symbol_to_feature = precomputed_symbol_map
        if verbose:
            print(f"  Using precomputed symbol map ({len(symbol_to_feature)} mappings)")
    elif map_ids and len(cliques) > 0:
        # Check if mapping is needed by sampling clique proteins
        sample_proteins: list[str] = []
        for clique in cliques[:10]:
            # Access protein_ids attribute safely
            if hasattr(clique, 'protein_ids'):
                sample_proteins.extend(clique.protein_ids[:3])

        matches = sum(1 for p in sample_proteins if p in work_ids)
        if sample_proteins and matches < len(sample_proteins) * 0.5:
            if verbose:
                print("  Mapping feature IDs to symbols...")
            # Uses module-level cache for efficiency across bootstrap iterations
            symbol_to_feature = map_feature_ids_to_symbols(work_ids, verbose=verbose)

    # 5. Build feature index map
    feature_to_idx: dict[str, int] = {fid: i for i, fid in enumerate(work_ids)}

    # Also add symbol -> idx mapping if we did ID conversion
    if symbol_to_feature:
        for symbol, feature_id in symbol_to_feature.items():
            if feature_id in feature_to_idx:
                feature_to_idx[symbol] = feature_to_idx[feature_id]

    # 6. Build clique -> feature indices mapping
    clique_to_indices: dict[str, tuple[int, ...]] = {}
    n_cliques_with_data = 0

    for clique in cliques:
        indices: list[int] = []
        # Access protein_ids attribute safely
        protein_ids = getattr(clique, 'protein_ids', [])
        clique_id = getattr(clique, 'clique_id', str(clique))

        for pid in protein_ids:
            # Try direct match first
            if pid in feature_to_idx:
                indices.append(feature_to_idx[pid])
            # Try symbol mapping (already added to feature_to_idx if available)
            elif symbol_to_feature and pid in symbol_to_feature:
                mapped = symbol_to_feature[pid]
                if mapped in feature_to_idx:
                    indices.append(feature_to_idx[mapped])

        if indices:
            clique_to_indices[clique_id] = tuple(indices)
            n_cliques_with_data += 1

    if n_cliques_with_data == 0:
        # Get sample info for error message
        sample_clique_proteins: list[str] = []
        if cliques and hasattr(cliques[0], 'protein_ids'):
            sample_clique_proteins = cliques[0].protein_ids[:5]
        raise ValueError(
            "No cliques have any proteins in the data. "
            "Check that protein IDs match between clique definitions and feature_ids. "
            f"Sample clique proteins: {sample_clique_proteins}, "
            f"Sample feature IDs: {work_ids[:5]}"
        )

    if verbose:
        print(f"  Cliques with data: {n_cliques_with_data}/{len(cliques)}")

    # 7. Validate conditions and contrast
    if not isinstance(sample_metadata, pd.DataFrame):
        raise TypeError("sample_metadata must be a pandas DataFrame")

    available_conditions = sorted(
        sample_metadata[condition_column].dropna().unique().tolist()
    )
    test_condition, ref_condition = contrast

    if test_condition not in available_conditions:
        raise ValueError(
            f"Test condition '{test_condition}' not found in data. "
            f"Available conditions: {available_conditions}"
        )
    if ref_condition not in available_conditions:
        raise ValueError(
            f"Reference condition '{ref_condition}' not found in data. "
            f"Available conditions: {available_conditions}"
        )

    # 8. Build preprocessing params for provenance
    preprocessing_params: dict[str, object] = {
        'normalization': norm_method.value,
        'imputation': imp_method.value,
        'map_ids': map_ids,
        'n_features_original': len(feature_ids),
        'n_samples': work_data.shape[1],
        'n_cliques_total': len(cliques),
        'n_cliques_with_data': n_cliques_with_data,
        'n_symbols_mapped': len(symbol_to_feature),
    }

    # 9. Build and return the frozen experiment
    creation_ts = datetime.now().isoformat()
    contrast_name = f"{test_condition}_vs_{ref_condition}"

    if verbose:
        print(f"  Creation timestamp: {creation_ts}")
        print()

    return PreparedCliqueExperiment(
        data=work_data,
        feature_ids=tuple(work_ids),
        feature_to_idx=feature_to_idx,
        sample_metadata=sample_metadata.copy(),
        condition_column=condition_column,
        subject_column=subject_column,
        conditions=tuple(available_conditions),
        n_samples=work_data.shape[1],
        cliques=tuple(cliques),
        clique_to_feature_indices=clique_to_indices,
        symbol_to_feature=symbol_to_feature,
        contrast=contrast,
        contrast_name=contrast_name,
        preprocessing_params=preprocessing_params,
        creation_timestamp=creation_ts,
    )


__all__ = [
    "PreparedCliqueExperiment",
    "prepare_experiment",
]
