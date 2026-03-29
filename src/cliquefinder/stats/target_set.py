"""Canonical serialization for INDRA-derived target gene sets.

This module provides a single source of truth for the gene set that flows
from the INDRA query (experiment) into the validation pipeline.  Without
it, validation re-queries INDRA live and can silently diverge from the
experimental gene set.

Usage — analysis side (writes):
    ts = TargetSet.from_query(
        targets_in_data={"TP53": "P04637", ...},
        gene_symbol="C9orf72",
        min_evidence=3,
        n_hops=1,
        min_intermediaries=1,
        n_indra_edges_raw=115,
    )
    ts.save(output_dir / "indra_targets.json")

Usage — validation side (reads):
    ts = TargetSet.load(target_set_path)
    target_gene_ids = list(ts.targets.values())
    adjacency = ts.adjacency
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class TargetSet:
    """Frozen snapshot of an INDRA-derived target gene set."""

    # Core mapping: {gene_symbol: feature_id}
    targets: dict[str, str]

    # INDRA query parameters — provenance
    gene_symbol: str
    min_evidence: int
    n_hops: int
    min_intermediaries: int = 1

    # Optional metadata
    n_indra_edges_raw: Optional[int] = None
    query_timestamp: str = ""

    # Phase 5b adjacency: {regulator_symbol: [target_symbols]}
    # Populated by attach_adjacency() after discover_regulators.
    adjacency: dict[str, list[str]] = field(default_factory=dict)
    adjacency_min_evidence: Optional[int] = None
    adjacency_min_targets: Optional[int] = None

    # Per-target edge metadata (v2): {symbol: [{regulation_type, sources, evidence_count}]}
    # Populated by query_network_targets() when edge info is available.
    edge_metadata: dict[str, list[dict]] = field(default_factory=dict)
    min_sources: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.query_timestamp:
            self.query_timestamp = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_query(
        cls,
        targets_in_data: dict[str, str],
        gene_symbol: str,
        min_evidence: int,
        n_hops: int,
        min_intermediaries: int = 1,
        n_indra_edges_raw: int | None = None,
        edge_metadata: dict[str, list[dict]] | None = None,
        min_sources: int | None = None,
    ) -> TargetSet:
        return cls(
            targets=dict(targets_in_data),
            gene_symbol=gene_symbol,
            min_evidence=min_evidence,
            n_hops=n_hops,
            min_intermediaries=min_intermediaries,
            n_indra_edges_raw=n_indra_edges_raw,
            edge_metadata=edge_metadata or {},
            min_sources=min_sources,
        )

    def attach_adjacency(
        self,
        adjacency: dict[str, list[str]],
        min_evidence: int,
        min_targets: int = 2,
    ) -> None:
        """Attach the graph permutation adjacency dict (Phase 5b)."""
        self.adjacency = {k: sorted(v) for k, v in adjacency.items()}
        self.adjacency_min_evidence = min_evidence
        self.adjacency_min_targets = min_targets

    # ── Serialization ──────────────────────────────────────────────

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        version = 2 if self.edge_metadata else 1
        blob = {
            "format_version": version,
            "gene_symbol": self.gene_symbol,
            "min_evidence": self.min_evidence,
            "n_hops": self.n_hops,
            "min_intermediaries": self.min_intermediaries,
            "n_indra_edges_raw": self.n_indra_edges_raw,
            "query_timestamp": self.query_timestamp,
            "n_targets": len(self.targets),
            "targets": self.targets,
        }
        if self.edge_metadata:
            blob["min_sources"] = self.min_sources
            blob["edge_metadata"] = self.edge_metadata
        if self.adjacency:
            blob["adjacency"] = {
                "min_evidence": self.adjacency_min_evidence,
                "min_targets": self.adjacency_min_targets,
                "n_regulators": len(self.adjacency),
                "regulators": self.adjacency,
            }
        # Atomic write: write to temp then rename to avoid partial files
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".target_set_"
        )
        try:
            with open(tmp_fd, "w") as f:
                json.dump(blob, f, indent=2, sort_keys=False)
                f.write("\n")
            Path(tmp_path).replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: Path | str) -> TargetSet:
        path = Path(path)
        blob = json.loads(path.read_text())

        version = blob.get("format_version", 0)
        if version not in (1, 2):
            raise ValueError(
                f"Unsupported target set format version {version} "
                f"(expected 1 or 2) in {path}"
            )

        ts = cls(
            targets=blob["targets"],
            gene_symbol=blob["gene_symbol"],
            min_evidence=blob["min_evidence"],
            n_hops=blob["n_hops"],
            min_intermediaries=blob.get("min_intermediaries", 1),
            n_indra_edges_raw=blob.get("n_indra_edges_raw"),
            query_timestamp=blob.get("query_timestamp", ""),
        )

        adj_blob = blob.get("adjacency")
        if adj_blob:
            ts.adjacency = adj_blob["regulators"]
            ts.adjacency_min_evidence = adj_blob.get("min_evidence")
            ts.adjacency_min_targets = adj_blob.get("min_targets")

        ts.edge_metadata = blob.get("edge_metadata", {})
        ts.min_sources = blob.get("min_sources")

        return ts

    # ── Convenience ────────────────────────────────────────────────

    @property
    def feature_ids(self) -> list[str]:
        """UniProt/Ensembl feature IDs in the target set."""
        return list(self.targets.values())

    @property
    def symbols(self) -> list[str]:
        """Gene symbols in the target set."""
        return list(self.targets.keys())

    def get_unambiguous_targets(
        self, loss_of_function: bool = True,
    ) -> dict[str, str]:
        """Classify targets by predicted direction from edge types.

        For loss-of-function (default): activation → predicted DOWN,
        repression → predicted UP.  Targets with both activation AND
        repression edges are excluded (ambiguous).  Targets with only
        phosphorylation edges are also excluded (no clear directional
        prediction for loss-of-function).

        Returns:
            Dict mapping {gene_symbol: "predicted_down" | "predicted_up"}
            for targets with unambiguous activation or repression edges.
        """
        result: dict[str, str] = {}
        for sym, edges in self.edge_metadata.items():
            if sym not in self.targets:
                continue
            reg_types = {e.get("regulation_type") for e in edges}
            has_act = "activation" in reg_types
            has_rep = "repression" in reg_types
            if has_act and has_rep:
                continue  # mixed — ambiguous
            if has_act:
                result[sym] = "predicted_down" if loss_of_function else "predicted_up"
            elif has_rep:
                result[sym] = "predicted_up" if loss_of_function else "predicted_down"
        return result

    def get_mixed_targets(self) -> set[str]:
        """Targets with both activation and repression edges."""
        mixed = set()
        for sym, edges in self.edge_metadata.items():
            if sym not in self.targets:
                continue
            reg_types = {e.get("regulation_type") for e in edges}
            if "activation" in reg_types and "repression" in reg_types:
                mixed.add(sym)
        return mixed

    def filter_by_min_sources(self, min_sources: int) -> dict[str, str]:
        """Return targets where at least one edge has >= min_sources.

        Returns:
            Dict mapping {gene_symbol: feature_id} for passing targets.
        """
        if not self.edge_metadata:
            return dict(self.targets)  # no metadata → can't filter
        result: dict[str, str] = {}
        for sym, fid in self.targets.items():
            edges = self.edge_metadata.get(sym, [])
            if any(len(e.get("sources", [])) >= min_sources for e in edges):
                result[sym] = fid
        return result

    def evidence_weights(self) -> dict[str, float]:
        """Compute per-target weights based on INDRA source diversity.

        Weighting scheme (principled, not tuned):
          - ≥2 distinct sources (e.g., reach + sparser, or any curated DB):
            weight = 1.0 (independently corroborated)
          - 1 source, evidence_count ≥ 3: weight = 0.5
            (multiple extractions from one reader — moderate confidence)
          - 1 source, evidence_count < 3: weight = 0.2
            (minimal evidence — included but downweighted)

        When no edge_metadata is available, all targets get weight 1.0.

        Returns:
            Dict mapping {gene_symbol: weight} for each target.
        """
        if not self.edge_metadata:
            return {sym: 1.0 for sym in self.targets}

        weights: dict[str, float] = {}
        for sym in self.targets:
            edges = self.edge_metadata.get(sym, [])
            if not edges:
                weights[sym] = 0.2
                continue
            # Max source count across all edges for this target
            max_sources = max(len(e.get("sources", [])) for e in edges)
            max_evidence = max(e.get("evidence_count", 1) for e in edges)
            if max_sources >= 2:
                weights[sym] = 1.0
            elif max_evidence >= 3:
                weights[sym] = 0.5
            else:
                weights[sym] = 0.2
        return weights

    def to_weighted_feature_ids(self) -> tuple[list[str], list[float]]:
        """Return (feature_ids, weights) for ROAST WeightedFeatureSet.

        Returns feature IDs (UniProt) and corresponding evidence weights,
        suitable for passing to RotationTestEngine.test_gene_set(gene_set=ids, weights=w).
        """
        import numpy as np
        w = self.evidence_weights()
        ids = []
        weights = []
        for sym, fid in self.targets.items():
            ids.append(fid)
            weights.append(w.get(sym, 0.2))
        return ids, weights

    def __repr__(self) -> str:
        adj_str = f", {len(self.adjacency)} regulators" if self.adjacency else ""
        return (
            f"TargetSet({self.gene_symbol}, {len(self.targets)} targets, "
            f"{self.n_hops}-hop, min_ev={self.min_evidence}{adj_str})"
        )
