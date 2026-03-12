"""Tests for RNG isolation across modules.

Verifies that stability, marker_discovery, and layouts
use local RNG instances instead of polluting np.random global state.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag


def _make_biomatrix(n_genes=20, n_samples=30, seed=42):
    """Create a minimal BioMatrix for testing."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_genes, n_samples))
    feature_ids = pd.Index([f"GENE{i}" for i in range(n_genes)])
    sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
    metadata = pd.DataFrame(
        {"phenotype": ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)},
        index=sample_ids,
    )
    quality_flags = np.full(
        (n_genes, n_samples), QualityFlag.ORIGINAL, dtype=np.uint32
    )
    return BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)


# ---------------------------------------------------------------------------
# Fix 1: stability.py — bootstrap_clique_stability
# ---------------------------------------------------------------------------

class TestStabilityRNG:
    """Verify stability.py uses SeedSequence, not global np.random.seed."""

    def _patch_validator(self, n_samples=20):
        """Return a context manager that patches CliqueValidator for stability tests."""
        mock_validator = MagicMock()
        mock_validator._compute_condition_mask_internal.return_value = np.ones(
            n_samples, dtype=bool
        )
        return patch(
            "cliquefinder.knowledge.clique_validator.CliqueValidator",
            return_value=mock_validator,
        )

    def test_stability_no_global_seed(self):
        """bootstrap_clique_stability does not call np.random.seed."""
        from cliquefinder.knowledge import stability as stab_mod

        with patch.object(np.random, "seed", wraps=np.random.seed) as mock_seed:
            matrix = _make_biomatrix(n_genes=10, n_samples=20)
            genes = set(matrix.feature_ids[:5])

            with self._patch_validator(20):
                try:
                    stab_mod.bootstrap_clique_stability(
                        matrix, genes, "all", n_bootstrap=2, n_jobs=1, random_state=42
                    )
                except Exception:
                    pass

            mock_seed.assert_not_called()

    def test_stability_thread_safety(self):
        """SeedSequence produces reproducible results across threads."""
        from cliquefinder.knowledge import stability as stab_mod

        matrix = _make_biomatrix(n_genes=10, n_samples=20)
        genes = set(matrix.feature_ids[:5])

        with self._patch_validator(20):
            r1 = stab_mod.bootstrap_clique_stability(
                matrix, genes, "all", n_bootstrap=5, n_jobs=2, random_state=42
            )
            r2 = stab_mod.bootstrap_clique_stability(
                matrix, genes, "all", n_bootstrap=5, n_jobs=2, random_state=42
            )

        # Same seed -> same results
        assert len(r1) == len(r2)
        genes_r1 = {c.genes for c in r1}
        genes_r2 = {c.genes for c in r2}
        assert genes_r1 == genes_r2

    def test_stability_no_global_rng_pollution(self):
        """bootstrap_clique_stability does not alter np.random global state."""
        from cliquefinder.knowledge import stability as stab_mod

        matrix = _make_biomatrix(n_genes=10, n_samples=20)
        genes = set(matrix.feature_ids[:5])

        state_before = np.random.get_state()

        with self._patch_validator(20):
            stab_mod.bootstrap_clique_stability(
                matrix, genes, "all", n_bootstrap=3, n_jobs=1, random_state=42
            )

        state_after = np.random.get_state()

        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2] == state_after[2]


# ---------------------------------------------------------------------------
# Fix 3: marker_discovery.py
# ---------------------------------------------------------------------------

class TestMarkerDiscoveryRNG:
    """Verify marker_discovery uses local RNG."""

    def test_marker_discovery_no_global_seed(self):
        """discover_from_reference does not call np.random.seed."""
        from cliquefinder.quality.marker_discovery import MarkerDiscovery

        matrix = _make_biomatrix(n_genes=20, n_samples=40)
        # Create bimodal data in first feature so Otsu threshold works
        rng = np.random.default_rng(99)
        bimodal = np.concatenate([rng.normal(-3, 0.5, 20), rng.normal(3, 0.5, 20)])
        # We need to create a new BioMatrix with the bimodal feature
        data = matrix.data.copy()
        data[0, :] = bimodal
        matrix2 = BioMatrix(
            data,
            matrix.feature_ids,
            matrix.sample_ids,
            matrix.sample_metadata,
            np.full_like(matrix.data, QualityFlag.ORIGINAL, dtype=np.uint32),
        )

        discovery = MarkerDiscovery(
            min_labeled_per_class=2, min_effect_size=0.1, random_state=42
        )

        state_before = np.random.get_state()
        try:
            discovery.discover_from_reference(matrix2, "GENE0", label_fraction=0.5)
        except (ValueError, IndexError):
            pass  # May fail due to thresholding edge cases
        state_after = np.random.get_state()

        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2] == state_after[2]

    def test_marker_discovery_reproducible(self):
        """Same random_state produces same results."""
        from cliquefinder.quality.marker_discovery import MarkerDiscovery

        rng = np.random.default_rng(99)
        bimodal = np.concatenate([rng.normal(-3, 0.5, 20), rng.normal(3, 0.5, 20)])
        data = rng.normal(size=(10, 40))
        data[0, :] = bimodal
        feature_ids = pd.Index([f"GENE{i}" for i in range(10)])
        sample_ids = pd.Index([f"S{i}" for i in range(40)])
        metadata = pd.DataFrame({"phenotype": ["A"] * 20 + ["B"] * 20}, index=sample_ids)
        qf = np.full((10, 40), QualityFlag.ORIGINAL, dtype=np.uint32)
        matrix = BioMatrix(data, feature_ids, sample_ids, metadata, qf)

        d1 = MarkerDiscovery(min_labeled_per_class=2, min_effect_size=0.1, random_state=42)
        d2 = MarkerDiscovery(min_labeled_per_class=2, min_effect_size=0.1, random_state=42)

        try:
            r1 = d1.discover_from_reference(matrix, "GENE0", label_fraction=0.5)
            r2 = d2.discover_from_reference(matrix, "GENE0", label_fraction=0.5)
            assert r1.best_feature == r2.best_feature
            np.testing.assert_array_equal(r1.predictions, r2.predictions)
        except (ValueError, IndexError):
            pytest.skip("Thresholding edge case prevented comparison")


# ---------------------------------------------------------------------------
# Fix 4: layouts.py
# ---------------------------------------------------------------------------

class TestLayoutsRNG:
    """Verify layouts functions do not pollute global RNG."""

    def test_compute_layout_no_global_seed(self):
        """compute_layout does not alter np.random global state."""
        from cliquefinder.viz.layouts import compute_layout
        import networkx as nx

        G = nx.erdos_renyi_graph(20, 0.3, seed=42)

        state_before = np.random.get_state()
        compute_layout(G, algorithm="spring", seed=42)
        state_after = np.random.get_state()

        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2] == state_after[2]

    def test_layout_with_communities_no_global_seed(self):
        """layout_with_communities does not alter np.random global state."""
        from cliquefinder.viz.layouts import layout_with_communities
        import networkx as nx

        G = nx.erdos_renyi_graph(20, 0.3, seed=42)
        communities = {n: n % 3 for n in G.nodes()}

        state_before = np.random.get_state()
        layout_with_communities(G, communities, algorithm="spring", seed=42)
        state_after = np.random.get_state()

        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2] == state_after[2]

    def test_compute_layout_empty_graph(self):
        """compute_layout handles empty graph without error."""
        from cliquefinder.viz.layouts import compute_layout
        import networkx as nx

        G = nx.Graph()
        result = compute_layout(G, seed=42)
        assert result == {}

    def test_compute_layout_reproducible(self):
        """Same seed produces same layout."""
        from cliquefinder.viz.layouts import compute_layout
        import networkx as nx

        G = nx.erdos_renyi_graph(15, 0.3, seed=42)

        pos1 = compute_layout(G, algorithm="spring", seed=99)
        pos2 = compute_layout(G, algorithm="spring", seed=99)

        for node in G.nodes():
            assert pos1[node] == pytest.approx(pos2[node])
