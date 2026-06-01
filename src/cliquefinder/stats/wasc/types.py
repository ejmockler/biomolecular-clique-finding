"""Data types for the WASC (Within-cluster Anchor-Slope Concordance) analysis.

See memory/wasc_spec.md for the full statistical specification.
M1 scope: only the enum + edge dataclass — fit/null/FDR types come later.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Theme(str, Enum):
    """The three pre-registered biological clusters.

    Frozen at pre-registration; matches scripts/viz/common.py::TERMS grouping.
    """
    SPLICING = "Splicing"
    CHROMATIN = "Chromatin"
    TRANSPORT = "Transport"


class Network(str, Enum):
    """Source graph for hop-1 adjacency.

    INDRA is the primary regulatory subgraph; STRING is the pre-registered
    physical-PPI negative control (per spec §7).
    """
    INDRA = "INDRA"
    STRING = "STRING"


@dataclass(frozen=True)
class WascEdge:
    """One unordered (anchor, target) pair in E_WASC.

    Lexicographic ordering: ``anchor_uniprot <= target_uniprot``.
    Use :py:meth:`WascEdge.make` to construct with automatic ordering.

    Notes
    -----
    The "anchor/target" naming is conventional (regression has anchor
    on the RHS, target on the LHS) but the edge itself is undirected;
    the Cochran-Q invariance test is symmetric in the swap to within
    numerical precision (verified in M1 numerical-identity gate per spec
    "Required modifications" item 6).
    """
    anchor_uniprot: str
    target_uniprot: str
    theme: Theme
    network: Network
    anchor_symbol: str = ""
    target_symbol: str = ""
    # Optional per-edge metadata (filled when available from the source graph)
    evidence_count: int | None = None
    stmt_types: tuple[str, ...] | None = None  # raw INDRA stmt_type tokens

    def __post_init__(self) -> None:
        if self.anchor_uniprot > self.target_uniprot:
            raise ValueError(
                f"WascEdge: anchor must be lex-smaller than target "
                f"(got {self.anchor_uniprot!r} > {self.target_uniprot!r}); "
                "use WascEdge.make() for automatic ordering."
            )
        if self.anchor_uniprot == self.target_uniprot:
            raise ValueError(
                f"WascEdge: self-loop not allowed "
                f"(uniprot={self.anchor_uniprot!r})"
            )

    @property
    def edge_id(self) -> str:
        """Canonical edge identifier: ``f'{anchor}|{target}'``."""
        return f"{self.anchor_uniprot}|{self.target_uniprot}"

    @classmethod
    def make(
        cls,
        u1: str,
        u2: str,
        theme: Theme,
        network: Network = Network.INDRA,
        *,
        anchor_symbol: str = "",
        target_symbol: str = "",
        evidence_count: int | None = None,
        stmt_types: tuple[str, ...] | None = None,
    ) -> "WascEdge":
        """Construct with automatic lex-ordering of ``u1`` vs ``u2``.

        Symbols are swapped to match if the UniProts are swapped.
        """
        if u1 == u2:
            raise ValueError(f"WascEdge.make: self-loop ({u1})")
        if u1 > u2:
            u1, u2 = u2, u1
            anchor_symbol, target_symbol = target_symbol, anchor_symbol
        return cls(
            anchor_uniprot=u1,
            target_uniprot=u2,
            theme=theme,
            network=network,
            anchor_symbol=anchor_symbol,
            target_symbol=target_symbol,
            evidence_count=evidence_count,
            stmt_types=stmt_types,
        )
