"""Panel design — the locked manifest for a multi-seed gradient panel.

A ``PanelDesign`` is a frozen specification of which seeds to test under
which contrast, partitioned into named biological strata.  It is the
load-bearing artifact for reproducibility: every panel run carries its
design alongside its results, so any panel can be re-derived from its
manifest alone.

Design decisions (Wave 24e):

- **Frozen everywhere**: the design cannot mutate after construction.
  Tuples (not lists) for collections to enforce immutability through
  the type system as well.
- **YAML round-trippable**: ``save_yaml`` / ``load_yaml`` are exact
  inverses on the supported field types.  The on-disk format is the
  canonical form; the dataclass is its in-memory projection.
- **Edge scope is NOT a parameter**: the gradient pipeline is
  committed to ``ALL_REGULATORY_TYPES`` (Wave 24d).  A panel inherits
  that commitment by construction — there is nothing to choose.
- **Target seed is its own implicit stratum**: it is named separately
  from the panel strata.  Stratum-vs-stratum tests compare the panel
  strata; the target's position is reported as an empirical rank.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cliquefinder.utils.fileio import atomic_write_text

from ._intensity import LOG2_TRANSFORM, RAW_TRANSFORM, VALID_TRANSFORMS

# Reserved sentinel for the implicit target stratum.  See
# ``cliquefinder.panels.analysis.TARGET_STRATUM_LABEL``; defined here
# to break a circular import while keeping the constant in one place.
_RESERVED_STRATUM_NAMES: frozenset[str] = frozenset({"<target>", "<feature>"})


@dataclass(frozen=True)
class PanelStratum:
    """A named subgroup of seeds drawn from one biological category."""

    name: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        # Coerce list inputs to tuples so the "frozen" guarantee
        # extends to immutable collection types, not just rebinding.
        if not isinstance(self.members, tuple):
            object.__setattr__(self, "members", tuple(self.members))

        if not self.name:
            raise ValueError("PanelStratum.name must be non-empty")
        if self.name in _RESERVED_STRATUM_NAMES:
            raise ValueError(
                f"PanelStratum.name {self.name!r} is reserved for the "
                f"implicit target stratum; choose another label."
            )
        if len(self.members) == 0:
            raise ValueError(
                f"PanelStratum '{self.name}' must have at least one member"
            )
        if len(set(self.members)) != len(self.members):
            raise ValueError(
                f"PanelStratum '{self.name}' has duplicate members: "
                f"{self.members}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "members": list(self.members)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PanelStratum:
        return cls(
            name=str(data["name"]),
            members=tuple(str(m) for m in data["members"]),
        )


@dataclass(frozen=True)
class PanelDesign:
    """Locked specification for a multi-seed gradient panel.

    Attributes
    ----------
    target_seed
        The primary seed whose gradient is the subject of inference
        (e.g., ``"C9orf72"``).  Reported as its own implicit stratum
        with empirical rank within the panel.
    strata
        Tuple of named ``PanelStratum`` containing the panel members.
        The target seed is NOT included in any stratum.
    contrast
        Two-tuple of condition labels in (case, control) order
        (e.g., ``("C9ORF72", "SPORADIC")``).
    max_hops
        BFS depth for shortest-path shell construction.  In the
        current dataset this saturates at 2 (Wave 23).
    n_permutations
        Degree-binned label-permutation null sample size.  Sets the
        floor of the empirical p-value at ``1 / (n_permutations + 1)``.
    covariates
        Metadata column names included in the ROAST design matrix
        (e.g., ``("Sex",)``).
    selection_rng_seed
        RNG seed used by ``select_panel`` to make stratum membership
        reproducible.  Stored on the design so a regenerated panel
        can be checked against the original.
    transform
        Intensity scale for the moderated-t fit: ``"log2"`` (default,
        ``log2(x+1)``) or ``"raw"`` (linear, the historical default).
        Applied before the engine via the shared ``_intensity`` helper,
        same as the landscape path; serialized so panel manifests are
        self-describing.
    description
        Free-text note explaining the panel's intent.  Optional.

    Notes
    -----
    Edge scope is implicit — fixed to ``ALL_REGULATORY_TYPES`` by the
    gradient pipeline.  See ``cliquefinder.knowledge.cogex``.
    """

    target_seed: str
    strata: tuple[PanelStratum, ...]
    contrast: tuple[str, str]
    max_hops: int
    n_permutations: int
    covariates: tuple[str, ...]
    selection_rng_seed: int
    transform: str = LOG2_TRANSFORM
    description: str = ""

    def __post_init__(self) -> None:
        # Coerce list/tuple inputs to tuples (frozen-ness extends to
        # collection immutability, not just rebinding).
        if not isinstance(self.strata, tuple):
            object.__setattr__(self, "strata", tuple(self.strata))
        if not isinstance(self.contrast, tuple):
            object.__setattr__(self, "contrast", tuple(self.contrast))
        if not isinstance(self.covariates, tuple):
            object.__setattr__(self, "covariates", tuple(self.covariates))

        if self.transform not in VALID_TRANSFORMS:
            raise ValueError(
                f"PanelDesign.transform must be one of "
                f"{sorted(VALID_TRANSFORMS)}, got {self.transform!r}"
            )

        if not self.target_seed:
            raise ValueError("PanelDesign.target_seed must be non-empty")
        if len(self.strata) == 0:
            raise ValueError("PanelDesign must have at least one stratum")
        if len(self.contrast) != 2 or self.contrast[0] == self.contrast[1]:
            raise ValueError(
                f"PanelDesign.contrast must be a 2-tuple of distinct labels, "
                f"got {self.contrast!r}"
            )
        if self.max_hops < 1:
            raise ValueError(
                f"PanelDesign.max_hops must be >= 1, got {self.max_hops}"
            )
        if self.n_permutations < 1:
            raise ValueError(
                f"PanelDesign.n_permutations must be >= 1, got {self.n_permutations}"
            )

        stratum_names = [s.name for s in self.strata]
        if len(set(stratum_names)) != len(stratum_names):
            raise ValueError(
                f"PanelDesign has duplicate stratum names: {stratum_names}"
            )

        all_members = self.selected_seeds()
        if len(set(all_members)) != len(all_members):
            duplicates = sorted(
                {m for m in all_members if all_members.count(m) > 1}
            )
            raise ValueError(
                f"PanelDesign has seeds appearing in multiple strata: "
                f"{duplicates}"
            )
        if self.target_seed in all_members:
            raise ValueError(
                f"PanelDesign.target_seed {self.target_seed!r} must not appear "
                f"in any stratum (target is an implicit stratum of one)"
            )

    def selected_seeds(self) -> tuple[str, ...]:
        """Return the panel seeds (excluding the target), preserving stratum order."""
        return tuple(g for s in self.strata for g in s.members)

    def stratum_for(self, seed: str) -> str:
        """Return the stratum name containing ``seed``.

        Raises
        ------
        KeyError
            If ``seed`` is not a panel member.  The target seed is
            never a stratum member; pass the result of
            ``selected_seeds()`` to enumerate valid inputs.
        """
        for s in self.strata:
            if seed in s.members:
                return s.name
        raise KeyError(
            f"Seed {seed!r} is not a panel member. "
            f"Valid seeds: {self.selected_seeds()}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a YAML-safe dict (canonical form)."""
        return {
            "target_seed": self.target_seed,
            "strata": [s.to_dict() for s in self.strata],
            "contrast": list(self.contrast),
            "max_hops": int(self.max_hops),
            "n_permutations": int(self.n_permutations),
            "covariates": list(self.covariates),
            "selection_rng_seed": int(self.selection_rng_seed),
            "transform": str(self.transform),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PanelDesign:
        contrast = data["contrast"]
        if not isinstance(contrast, (list, tuple)) or len(contrast) != 2:
            raise ValueError(
                f"PanelDesign.contrast must be a 2-element sequence, "
                f"got {contrast!r}"
            )
        return cls(
            target_seed=str(data["target_seed"]),
            strata=tuple(
                PanelStratum.from_dict(s) for s in data["strata"]
            ),
            contrast=(str(contrast[0]), str(contrast[1])),
            max_hops=int(data["max_hops"]),
            n_permutations=int(data["n_permutations"]),
            covariates=tuple(str(c) for c in data.get("covariates", [])),
            selection_rng_seed=int(data["selection_rng_seed"]),
            # Back-compat: a manifest without "transform" was written
            # before the field existed, i.e. a RAW run — do not relabel it
            # log2 (the constructor default).  Mirrors LandscapeDesign.
            transform=str(data.get("transform", RAW_TRANSFORM)),
            description=str(data.get("description", "")),
        )

    def save_yaml(self, path: Path | str) -> None:
        """Write the design to ``path`` as YAML.  Round-trips with ``load_yaml``.

        Atomic via :func:`cliquefinder.utils.fileio.atomic_write_text` —
        a partially written manifest is not visible to readers.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            default_flow_style=False,
        )
        atomic_write_text(path, text)

    @classmethod
    def load_yaml(cls, path: Path | str) -> PanelDesign:
        """Read a design from a YAML file.  Inverse of ``save_yaml``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Panel manifest not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Panel manifest must be a YAML mapping, got "
                f"{type(raw).__name__} in {path}"
            )
        return cls.from_dict(raw)
