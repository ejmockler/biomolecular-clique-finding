"""
Protein/clique summarization methods inspired by MSstats.

Implements Tukey's Median Polish (TMP) and alternative aggregation strategies
for summarizing feature-level data to protein or clique level.

The key insight from MSstats: robust summarization using medians rather than
means provides resistance to outliers while preserving biological signal.

References:
    - Choi et al. (2014) MSstats: Bioinformatics 30(17):2524-2526
    - Tukey (1977) Exploratory Data Analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class SummarizationMethod(Enum):
    """Available summarization methods."""

    TUKEY_MEDIAN_POLISH = "tmp"
    MEDIAN = "median"
    MEAN = "mean"
    LOGSUM = "logsum"
    PCA = "pca"


@dataclass(frozen=True)
class MedianPolishResult:
    """Result of Tukey's Median Polish algorithm.

    Attributes:
        overall: Grand effect (overall level)
        row_effects: Per-feature (row) effects
        col_effects: Per-sample (column) effects - these are the summarized values
        residuals: Residual matrix after decomposition
        iterations: Number of iterations until convergence
        converged: Whether algorithm converged within max_iter
    """

    overall: float
    row_effects: NDArray[np.float64]
    col_effects: NDArray[np.float64]
    residuals: NDArray[np.float64]
    iterations: int
    converged: bool

    @property
    def sample_abundances(self) -> NDArray[np.float64]:
        """Get the summarized sample-level abundances.

        The column effects represent sample-level deviations from the
        overall mean, which combined with the overall effect gives
        the protein/clique abundance per sample.
        """
        return self.overall + self.col_effects


def tukey_median_polish(
    data: NDArray[np.float64],
    max_iter: int = 10,
    eps: float = 0.01,
    na_rm: bool = True,
) -> MedianPolishResult:
    """
    Tukey's Median Polish for robust two-way decomposition.

    Decomposes a features × samples matrix into additive components:
        X[i,j] = overall + row_effect[i] + col_effect[j] + residual[i,j]

    The column effects (+ overall) represent the summarized protein/clique
    abundance per sample, robust to feature-level outliers.

    This is the core summarization method in MSstats for aggregating
    peptide/transition intensities to protein level.

    Args:
        data: 2D array of shape (n_features, n_samples) with log-transformed
              intensities. Features are rows, samples are columns.
        max_iter: Maximum iterations for convergence.
        eps: Convergence tolerance - stop when max median adjustment < eps.
        na_rm: If True, use nanmedian to handle missing values.

    Returns:
        MedianPolishResult with decomposition components.

    Algorithm:
        1. Initialize row_effects = 0, col_effects = 0, overall = 0
        2. Repeat until convergence:
           a. Compute row medians, subtract from data, add to row_effects
           b. Compute col medians, subtract from data, add to col_effects
        3. Extract overall from col_effects

    Note:
        The algorithm is symmetric - alternating row/column sweeps ensures
        both dimensions are treated equally. The order (row-first vs col-first)
        doesn't affect the final result for well-behaved data.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    n_features, n_samples = data.shape

    # Work on a copy to avoid modifying input
    residuals = data.astype(np.float64).copy()
    row_effects = np.zeros(n_features, dtype=np.float64)
    col_effects = np.zeros(n_samples, dtype=np.float64)
    overall = 0.0

    median_fn = np.nanmedian if na_rm else np.median

    converged = False
    for iteration in range(max_iter):
        # Row sweep: subtract row medians
        row_medians = median_fn(residuals, axis=1)
        # Handle all-NaN rows
        row_medians = np.nan_to_num(row_medians, nan=0.0)
        residuals = residuals - row_medians[:, np.newaxis]
        row_effects = row_effects + row_medians

        # Column sweep: subtract column medians
        col_medians = median_fn(residuals, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        residuals = residuals - col_medians[np.newaxis, :]
        col_effects = col_effects + col_medians

        # Check convergence
        max_adjustment = max(
            np.max(np.abs(row_medians)),
            np.max(np.abs(col_medians))
        )

        if max_adjustment < eps:
            converged = True
            break

    # Extract overall effect from row effects (could also use col effects)
    overall = median_fn(row_effects)
    if np.isnan(overall):
        overall = 0.0
    row_effects = row_effects - overall

    return MedianPolishResult(
        overall=overall,
        row_effects=row_effects,
        col_effects=col_effects,
        residuals=residuals,
        iterations=iteration + 1,
        converged=converged,
    )


def summarize_to_protein(
    feature_data: NDArray[np.float64],
    method: SummarizationMethod | str = SummarizationMethod.TUKEY_MEDIAN_POLISH,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Summarize feature-level data to protein/clique level.

    Args:
        feature_data: 2D array (n_features, n_samples) of log-intensities.
        method: Summarization method to use.
        **kwargs: Additional arguments passed to the method.

    Returns:
        1D array of shape (n_samples,) with summarized abundances.
    """
    if isinstance(method, str):
        method = SummarizationMethod(method)

    if feature_data.ndim == 1:
        # Single feature - return as-is
        return feature_data.copy()

    if method == SummarizationMethod.TUKEY_MEDIAN_POLISH:
        result = tukey_median_polish(feature_data, **kwargs)
        return result.sample_abundances

    elif method == SummarizationMethod.MEDIAN:
        return np.nanmedian(feature_data, axis=0)

    elif method == SummarizationMethod.MEAN:
        return np.nanmean(feature_data, axis=0)

    elif method == SummarizationMethod.LOGSUM:
        # Sum in original space, return in log space
        # log(sum(exp(x))) = logsumexp(x)
        from scipy.special import logsumexp
        return logsumexp(feature_data, axis=0)

    elif method == SummarizationMethod.PCA:
        # First principal component
        from sklearn.decomposition import PCA

        # Handle missing values by mean imputation for PCA
        data_imputed = feature_data.copy()
        col_means = np.nanmean(data_imputed, axis=0)
        for j in range(data_imputed.shape[1]):
            mask = np.isnan(data_imputed[:, j])
            data_imputed[mask, j] = col_means[j]

        # Transpose: PCA expects (n_samples, n_features)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(data_imputed.T).flatten()

        # Ensure direction is consistent (positive correlation with mean)
        if np.corrcoef(pc1, np.nanmean(feature_data, axis=0))[0, 1] < 0:
            pc1 = -pc1

        return pc1

    else:
        raise ValueError(f"Unknown summarization method: {method}")


@dataclass
class CliqueSummary:
    """Summary statistics for a protein clique.

    Attributes:
        clique_id: Identifier for the clique (e.g., regulator name)
        sample_abundances: Summarized abundance per sample
        n_proteins: Number of proteins in the clique
        protein_ids: List of protein identifiers in the clique
        method: Summarization method used
        coherence: Mean pairwise correlation within clique (if computed)
    """

    clique_id: str
    sample_abundances: NDArray[np.float64]
    n_proteins: int
    protein_ids: list[str]
    method: str
    coherence: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            'clique_id': self.clique_id,
            'n_proteins': self.n_proteins,
            'method': self.method,
            'coherence': self.coherence,
            'mean_abundance': float(np.nanmean(self.sample_abundances)),
            'std_abundance': float(np.nanstd(self.sample_abundances)),
        }


def summarize_clique(
    protein_data: NDArray[np.float64],
    protein_ids: list[str],
    clique_id: str,
    method: SummarizationMethod | str = SummarizationMethod.TUKEY_MEDIAN_POLISH,
    compute_coherence: bool = True,
) -> CliqueSummary:
    """
    Summarize a protein clique to a single abundance vector.

    This extends MSstats' peptide→protein summarization to the
    protein→clique level, treating proteins as "features" of the clique.

    Args:
        protein_data: 2D array (n_proteins, n_samples) of log-intensities.
        protein_ids: List of protein identifiers.
        clique_id: Identifier for this clique.
        method: Summarization method.
        compute_coherence: Whether to compute mean pairwise correlation.

    Returns:
        CliqueSummary with aggregated abundance and metadata.
    """
    if isinstance(method, str):
        method = SummarizationMethod(method)

    abundances = summarize_to_protein(protein_data, method)

    coherence = None
    if compute_coherence and protein_data.shape[0] > 1:
        # Compute mean pairwise correlation
        # Use complete cases only
        valid_mask = ~np.any(np.isnan(protein_data), axis=0)
        if np.sum(valid_mask) >= 3:
            valid_data = protein_data[:, valid_mask]
            corr_matrix = np.corrcoef(valid_data)
            # Extract upper triangle (excluding diagonal)
            n = corr_matrix.shape[0]
            upper_tri = corr_matrix[np.triu_indices(n, k=1)]
            coherence = float(np.mean(upper_tri))

    return CliqueSummary(
        clique_id=clique_id,
        sample_abundances=abundances,
        n_proteins=len(protein_ids),
        protein_ids=protein_ids,
        method=method.value if isinstance(method, SummarizationMethod) else method,
        coherence=coherence,
    )


def parallel_clique_summarization(
    data_matrix: NDArray[np.float64],
    feature_ids: NDArray | list[str],
    clique_definitions: dict[str, list[str]],
    method: SummarizationMethod | str = SummarizationMethod.TUKEY_MEDIAN_POLISH,
    n_jobs: int = -1,
) -> dict[str, CliqueSummary]:
    """
    Summarize multiple cliques in parallel.

    Args:
        data_matrix: Full data matrix (n_features, n_samples).
        feature_ids: Feature identifiers matching data_matrix rows.
        clique_definitions: Dict mapping clique_id → list of protein IDs.
        method: Summarization method.
        n_jobs: Number of parallel jobs (-1 for all CPUs).

    Returns:
        Dict mapping clique_id → CliqueSummary.
    """
    from joblib import Parallel, delayed

    feature_id_list = list(feature_ids) if not isinstance(feature_ids, list) else feature_ids
    feature_to_idx = {fid: i for i, fid in enumerate(feature_id_list)}

    def process_clique(clique_id: str, protein_ids: list[str]) -> tuple[str, CliqueSummary | None]:
        # Find indices for proteins in this clique
        indices = []
        valid_proteins = []
        for pid in protein_ids:
            if pid in feature_to_idx:
                indices.append(feature_to_idx[pid])
                valid_proteins.append(pid)

        if len(indices) < 1:
            return clique_id, None

        clique_data = data_matrix[indices, :]
        summary = summarize_clique(
            clique_data,
            valid_proteins,
            clique_id,
            method=method,
        )
        return clique_id, summary

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_clique)(cid, pids)
        for cid, pids in clique_definitions.items()
    )

    return {cid: summary for cid, summary in results if summary is not None}
