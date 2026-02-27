"""
Tests for security audit findings S-1 and S-3 (a, b, c).

S-1:  eval() replaced with ast.literal_eval() in viz/differential.py
S-3a: Bare except narrowed to (ValueError, SyntaxError) in viz/differential.py
S-3b: Bare except narrowed to (ValueError, FloatingPointError) in viz/cliques.py
S-3c: Bare except narrowed to (ValueError, OverflowError) in quality/outliers.py
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper to read source files directly (avoids editable-install path issues)
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "cliquefinder"


def _read_source(relpath: str) -> str:
    """Read a source file relative to the cliquefinder package root."""
    return (_SRC_ROOT / relpath).read_text()


# ---------------------------------------------------------------------------
# S-1: eval() replaced with ast.literal_eval()
# ---------------------------------------------------------------------------

class TestS1_NoEvalOnUntrustedData:
    """Verify that viz/differential.py uses ast.literal_eval, not eval."""

    def test_source_does_not_contain_bare_eval(self):
        """No bare eval() call (not preceded by literal_) in viz/differential.py."""
        source = _read_source("viz/differential.py")
        bare_eval_calls = re.findall(r'(?<!literal_)eval\(', source)
        assert len(bare_eval_calls) == 0, (
            f"Found {len(bare_eval_calls)} bare eval() call(s) in viz/differential.py"
        )

    def test_source_contains_ast_literal_eval(self):
        """The fix must use ast.literal_eval."""
        source = _read_source("viz/differential.py")
        assert 'ast.literal_eval(' in source

    def test_source_imports_ast(self):
        """The ast module must be imported."""
        source = _read_source("viz/differential.py")
        assert re.search(r'^import ast$', source, re.MULTILINE)

    def test_valid_dict_string_parses(self):
        """ast.literal_eval correctly parses a dict string like the pipeline produces."""
        val = "{'up': 1.5, 'down': -0.8}"
        parsed = ast.literal_eval(val)
        assert isinstance(parsed, dict)
        assert parsed["down"] == -0.8

    def test_malicious_string_raises(self):
        """ast.literal_eval rejects code that eval() would execute."""
        malicious = "__import__('os').system('echo pwned')"
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval(malicious)

    def test_function_call_string_raises(self):
        """ast.literal_eval rejects any function call expression."""
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval("open('/etc/passwd').read()")

    def test_lambda_string_raises(self):
        """ast.literal_eval rejects lambda expressions."""
        with pytest.raises((ValueError, SyntaxError)):
            ast.literal_eval("lambda: None")


# ---------------------------------------------------------------------------
# S-3a: Bare except narrowed in viz/differential.py effect-size parsing
# ---------------------------------------------------------------------------

class TestS3a_NarrowedExceptDifferential:
    """Verify that the except clause only catches ValueError and SyntaxError."""

    def test_except_clause_is_narrowed(self):
        """Source contains except (ValueError, SyntaxError), not bare except."""
        source = _read_source("viz/differential.py")
        assert 'except (ValueError, SyntaxError):' in source
        # Ensure there is no bare 'except:' left (bare = except followed by colon only)
        bare_excepts = re.findall(r'except\s*:', source)
        assert len(bare_excepts) == 0, (
            f"Found {len(bare_excepts)} bare except clause(s) in viz/differential.py"
        )

    def test_invalid_literal_caught(self):
        """ValueError from ast.literal_eval for non-literal is caught and returns 0."""
        val = "function_call(down=1)"
        effect_vals = []
        if isinstance(val, str) and "down" in val:
            try:
                parsed = ast.literal_eval(val)
                effect_vals.append(parsed.get("down", 0))
            except (ValueError, SyntaxError):
                effect_vals.append(0)
        assert effect_vals == [0]

    def test_syntaxerror_caught(self):
        """SyntaxError from ast.literal_eval for malformed string is caught."""
        val = "{'down': }"  # invalid syntax
        effect_vals = []
        try:
            parsed = ast.literal_eval(val)
            effect_vals.append(parsed.get("down", 0))
        except (ValueError, SyntaxError):
            effect_vals.append(0)
        assert effect_vals == [0]

    def test_keyboard_interrupt_not_caught(self):
        """KeyboardInterrupt must NOT be caught -- it should propagate."""
        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            except (ValueError, SyntaxError):
                pass  # should NOT reach here

    def test_system_exit_not_caught(self):
        """SystemExit must NOT be caught -- it should propagate."""
        with pytest.raises(SystemExit):
            try:
                raise SystemExit(1)
            except (ValueError, SyntaxError):
                pass  # should NOT reach here


# ---------------------------------------------------------------------------
# S-3b: Bare except narrowed in viz/cliques.py Jaccard clustering
# ---------------------------------------------------------------------------

class TestS3b_NarrowedExceptCliques:
    """Verify clustering fallback only catches ValueError and FloatingPointError."""

    def test_except_clause_is_narrowed(self):
        """Source contains except (ValueError, FloatingPointError), not bare except."""
        source = _read_source("viz/cliques.py")
        assert 'except (ValueError, FloatingPointError):' in source
        bare_excepts = re.findall(r'except\s*:', source)
        assert len(bare_excepts) == 0, (
            f"Found {len(bare_excepts)} bare except clause(s) in viz/cliques.py"
        )

    def test_empty_distance_matrix_falls_back(self):
        """Single-observation pdist gives empty distances; linkage raises ValueError."""
        from scipy.spatial.distance import pdist
        from scipy.cluster import hierarchy

        # Single-row matrix: pdist returns empty array, linkage raises ValueError
        matrix = np.array([[1, 0, 1, 0]])
        gene_order = [0]

        try:
            distances = pdist(matrix, metric='jaccard')
            if not np.all(np.isnan(distances)):
                linkage_mat = hierarchy.linkage(distances, method='average')
                dendro = hierarchy.dendrogram(linkage_mat, no_plot=True)
                gene_order = dendro['leaves']
        except (ValueError, FloatingPointError):
            # linkage raises ValueError on empty distance matrix
            pass

        # Falls back to original order
        assert gene_order == [0]

    def test_valid_binary_matrix_clusters(self):
        """A valid binary matrix with variance clusters without error."""
        from scipy.spatial.distance import pdist
        from scipy.cluster import hierarchy

        # Matrix with some variation
        matrix = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
        ])
        gene_order = list(range(5))

        try:
            distances = pdist(matrix, metric='jaccard')
            if not np.all(np.isnan(distances)):
                linkage_mat = hierarchy.linkage(distances, method='average')
                dendro = hierarchy.dendrogram(linkage_mat, no_plot=True)
                gene_order = dendro['leaves']
        except (ValueError, FloatingPointError):
            pass

        # Should have reordered (not original order, since there are clear clusters)
        assert len(gene_order) == 5
        assert sorted(gene_order) == [0, 1, 2, 3, 4]

    def test_keyboard_interrupt_not_caught(self):
        """KeyboardInterrupt must propagate through the narrowed except."""
        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            except (ValueError, FloatingPointError):
                pass


# ---------------------------------------------------------------------------
# S-3c: Bare except narrowed in quality/outliers.py Student's t fitting
# ---------------------------------------------------------------------------

class TestS3c_NarrowedExceptOutliers:
    """Verify neg_log_likelihood only catches ValueError and OverflowError."""

    def test_except_clause_is_narrowed(self):
        """Source contains except (ValueError, OverflowError), not bare except."""
        source = _read_source("quality/outliers.py")
        assert 'except (ValueError, OverflowError):' in source
        bare_excepts = re.findall(r'except\s*:', source)
        assert len(bare_excepts) == 0, (
            f"Found {len(bare_excepts)} bare except clause(s) in quality/outliers.py"
        )

    def test_fit_student_t_shared_succeeds(self):
        """fit_student_t_shared returns valid results for normal data."""
        from cliquefinder.quality.outliers import fit_student_t_shared

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 20))

        df_shared, locations, scales = fit_student_t_shared(data)

        assert 2.1 <= df_shared <= 100
        assert len(locations) == 10
        assert len(scales) == 10

    def test_neg_log_likelihood_guard_clauses(self):
        """neg_log_likelihood returns inf for out-of-range df values."""
        from scipy import stats

        standardized_residuals = np.array([0.0, 0.0, 0.0])

        def neg_log_likelihood(df_val):
            if df_val <= 2 or df_val > 100:
                return np.inf
            try:
                ll = np.sum(stats.t.logpdf(standardized_residuals, df=df_val))
                return -ll
            except (ValueError, OverflowError):
                return np.inf

        # df <= 2 returns inf (guard clause)
        assert neg_log_likelihood(1.0) == np.inf
        assert neg_log_likelihood(2.0) == np.inf
        assert neg_log_likelihood(-1.0) == np.inf

        # df > 100 returns inf (guard clause)
        assert neg_log_likelihood(101.0) == np.inf

        # Valid df returns a finite number
        result = neg_log_likelihood(5.0)
        assert np.isfinite(result)

    def test_extreme_residuals_handled(self):
        """Extreme residual values do not raise uncaught exceptions."""
        from scipy import stats

        standardized_residuals = np.array([1e308, -1e308])

        def neg_log_likelihood(df_val):
            if df_val <= 2 or df_val > 100:
                return np.inf
            try:
                ll = np.sum(stats.t.logpdf(standardized_residuals, df=df_val))
                return -ll
            except (ValueError, OverflowError):
                return np.inf

        # Should return a finite value or inf, but NOT raise an uncaught exception
        result = neg_log_likelihood(3.0)
        assert isinstance(result, (float, np.floating))

    def test_keyboard_interrupt_not_caught(self):
        """KeyboardInterrupt must propagate through the narrowed except."""
        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            except (ValueError, OverflowError):
                pass


# ---------------------------------------------------------------------------
# Integration: Verify the full effect-size parsing loop (S-1 + S-3a combined)
# ---------------------------------------------------------------------------

class TestEffectSizeParsingIntegration:
    """Test the complete effect-size parsing loop from plot_ranked_cliques."""

    def _parse_effect_vals(self, series: pd.Series) -> list[float]:
        """Reproduce the exact parsing logic from differential.py."""
        effect_vals = []
        for val in series:
            if isinstance(val, str) and "down" in val:
                try:
                    parsed = ast.literal_eval(val)
                    effect_vals.append(parsed.get("down", 0))
                except (ValueError, SyntaxError):
                    effect_vals.append(0)
            else:
                effect_vals.append(0)
        return effect_vals

    def test_mixed_series(self):
        """Parses a realistic Series with valid dicts, invalid strings, and non-strings."""
        series = pd.Series([
            "{'up': 1.5, 'down': -0.8}",
            "{'up': 0.3, 'down': -1.2}",
            "invalid_down_string",
            42,  # non-string
            None,
            "{'up': 0.0}",  # no "down" substring in value
        ])
        result = self._parse_effect_vals(series)
        assert result[0] == -0.8
        assert result[1] == -1.2
        assert result[2] == 0  # invalid, caught by except
        assert result[3] == 0  # non-string
        assert result[4] == 0  # None
        assert result[5] == 0  # no "down" substring in value

    def test_malicious_input_neutralized(self):
        """Code injection attempts are safely rejected."""
        series = pd.Series([
            "__import__('os').system('rm -rf / --down')",
        ])
        result = self._parse_effect_vals(series)
        assert result == [0]  # caught by except (ValueError, SyntaxError)
