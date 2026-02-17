"""
Declarative cohort resolution for biomolecular studies.

Separates cohort definitions (which samples belong to which group) from
application logic. Definitions are YAML-serializable dataclasses; the
CohortResolver applies them to metadata and produces labeled DataFrames
with an audit trail.

Groups are evaluated in priority order — each sample is assigned to the
FIRST matching group. Place more specific groups before general ones.

Usage:
    # From YAML
    definition = CohortResolver.from_yaml("cohorts/c9orf72_vs_sporadic.yaml")
    result = CohortResolver.resolve(metadata, definition)
    result.print_summary()

    # Programmatic
    definition = CohortDefinition(
        name="C9orf72_vs_Sporadic",
        base_filter=CohortGroup(label="cases", criteria=[
            CohortCriterion(column="phenotype", operator="eq", value="CASE"),
        ]),
        groups=[...],
        contrast=("C9ORF72", "SPORADIC"),
    )
    result = CohortResolver.resolve(metadata, definition)

    # CLI helper (shared by differential.py and compare.py)
    metadata, condition_col, contrast_list = resolve_cohort_from_args(
        metadata, cohort_config=path, genetic_contrast=None,
    )
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Column name used for cohort-derived condition labels
COHORT_CONDITION_COLUMN = "genetic_phenotype"

# Known familial ALS mutations for sporadic exclusion.
# Source: ALSoD (https://alsod.ac.uk/) and OMIM, current as of 2025.
# This list is used by from_genetic_contrast() for auto-generated definitions.
# For precise control, use explicit YAML cohort definitions instead.
KNOWN_ALS_MUTATIONS = [
    "C9orf72",
    "SOD1",
    "FUS",
    "TARDBP",
    "TARDBP (TDP43)",
    "SETX",
    "OPTN",
    "VCP",
    "UBQLN2",
    "TBK1",
    "NEK1",
    "CHCHD10",
    "SQSTM1",
    "KIF5A",
    "ANXA11",
    "Multiple",
    "Other",
]

# Gene-specific pathogenic repeat expansion thresholds.
# Only genes with validated thresholds are included.
REPEAT_EXPANSION_THRESHOLDS: Dict[str, int] = {
    "c9orf72": 30,   # GGGGCC hexanucleotide; DeJesus-Hernandez et al. 2011
    "atxn2": 32,     # CAG trinucleotide; ALS risk alleles 27-32
}


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class CohortCriterion:
    """Single predicate on a metadata column.

    Operators:
        eq       — column == value
        ne       — column != value
        in       — column in value (list)
        not_in   — column not in value (list)
        gte      — column >= value
        gt       — column > value
        lte      — column <= value
        lt       — column < value
        isna     — column is NaN (value ignored)
        notna    — column is not NaN (value ignored)

    Parameters:
        allow_na: If True, rows where the column is NaN pass this criterion.
            This is essential for exclusion-style criteria where absence of data
            should not disqualify a sample (e.g., a sporadic ALS patient with
            no repeat length measurement should not be excluded from the sporadic
            group just because the column is NaN).
    """

    column: str
    operator: str
    value: Any = None
    allow_na: bool = False

    _VALID_OPERATORS = frozenset(
        {"eq", "ne", "in", "not_in", "gte", "gt", "lte", "lt", "isna", "notna"}
    )

    def __post_init__(self):
        if self.operator not in self._VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}' for criterion on column "
                f"'{self.column}'. Valid operators: {sorted(self._VALID_OPERATORS)}"
            )

    def evaluate(self, series: pd.Series) -> pd.Series:
        """Evaluate this criterion against a metadata column Series.

        Parameters:
            series: A pandas Series representing one metadata column,
                indexed by sample IDs.

        Returns:
            Boolean Series (True = criterion satisfied).
        """
        col = series
        na_mask = col.isna()

        if self.operator == "isna":
            return na_mask
        if self.operator == "notna":
            return ~na_mask

        # Compute on valid (non-NA) entries using numpy arrays
        # to avoid pandas FutureWarning about incompatible dtype.
        result_arr = np.zeros(len(col), dtype=bool)
        valid_mask = ~na_mask
        valid_idx = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask

        if valid_idx.any():
            valid_col = col[valid_mask]
            if self.operator == "eq":
                matched = (valid_col == self.value).values
            elif self.operator == "ne":
                matched = (valid_col != self.value).values
            elif self.operator == "in":
                matched = valid_col.isin(self.value).values
            elif self.operator == "not_in":
                matched = (~valid_col.isin(self.value)).values
            elif self.operator == "gte":
                matched = (valid_col.astype(float) >= float(self.value)).values
            elif self.operator == "gt":
                matched = (valid_col.astype(float) > float(self.value)).values
            elif self.operator == "lte":
                matched = (valid_col.astype(float) <= float(self.value)).values
            elif self.operator == "lt":
                matched = (valid_col.astype(float) < float(self.value)).values
            else:
                matched = np.zeros(valid_idx.sum(), dtype=bool)
            result_arr[valid_idx] = matched

        if self.allow_na:
            na_idx = na_mask.values if hasattr(na_mask, 'values') else na_mask
            result_arr[na_idx] = True

        return pd.Series(result_arr, index=col.index, dtype=bool)


@dataclass
class CohortGroup:
    """A named cohort defined by one or more criteria.

    Parameters:
        label: The group label assigned to matching samples.
        criteria: List of CohortCriterion to evaluate.
        logic: "any" (OR — at least one criterion must match) or
               "all" (AND — all criteria must match).
    """

    label: str
    criteria: List[CohortCriterion] = field(default_factory=list)
    logic: str = "any"

    def __post_init__(self):
        if self.logic not in ("any", "all"):
            raise ValueError(f"logic must be 'any' or 'all', got '{self.logic}'")

    def evaluate(
        self,
        metadata: pd.DataFrame,
        strict: bool = True,
    ) -> pd.Series:
        """Return boolean mask of samples matching this group's criteria.

        Parameters:
            metadata: Sample metadata DataFrame.
            strict: If True, raise ValueError on missing columns instead
                of logging a warning. Prevents silent misclassification
                from column name typos.
        """
        if not self.criteria:
            return pd.Series(True, index=metadata.index)

        masks = []
        for criterion in self.criteria:
            if criterion.column not in metadata.columns:
                if strict:
                    raise ValueError(
                        f"Column '{criterion.column}' required by cohort group "
                        f"'{self.label}' not found in metadata. "
                        f"Available columns: {sorted(metadata.columns.tolist())}"
                    )
                logger.warning(
                    f"Column '{criterion.column}' not found in metadata; "
                    f"criterion skipped for group '{self.label}'"
                )
                if self.logic == "all":
                    return pd.Series(False, index=metadata.index)
                continue
            masks.append(criterion.evaluate(metadata[criterion.column]))

        if not masks:
            return pd.Series(False, index=metadata.index)

        combined = masks[0]
        for m in masks[1:]:
            if self.logic == "any":
                combined = combined | m
            else:
                combined = combined & m

        return combined


@dataclass
class CohortDefinition:
    """Full cohort specification.

    Parameters:
        name: Human-readable name for this cohort definition.
        description: Optional longer description.
        base_filter: If provided, only samples matching this group are
            considered for cohort assignment. Typically used to restrict
            to CASE samples before genetic subtyping.
        groups: List of CohortGroup definitions. Samples are assigned to
            the first matching group (priority order — place specific
            groups before general ones).
        contrast: Optional (test, reference) tuple for automatic contrast
            setup in differential analysis. This is metadata for downstream
            consumers; CohortResolver.resolve() does not use it.
    """

    name: str
    description: str = ""
    base_filter: Optional[CohortGroup] = None
    groups: List[CohortGroup] = field(default_factory=list)
    contrast: Optional[Tuple[str, str]] = None

    def __post_init__(self):
        labels = [g.label for g in self.groups]
        seen = set()
        duplicates = []
        for label in labels:
            if label in seen:
                duplicates.append(label)
            seen.add(label)
        if duplicates:
            raise ValueError(
                f"Duplicate group labels in CohortDefinition '{self.name}': "
                f"{duplicates}. Each group must have a unique label."
            )


# =============================================================================
# Resolution Result
# =============================================================================


@dataclass
class CohortResult:
    """Output of cohort resolution with full audit trail.

    Attributes:
        metadata: Filtered metadata with cohort column added.
        group_counts: {label: sample_count} for each group.
        audit: Per-group audit info (criteria match counts, etc.).
        definition: The CohortDefinition that produced this result.
        excluded_count: Samples in base filter that matched no group.
        base_filtered_count: Samples excluded by the base filter.
    """

    metadata: pd.DataFrame
    group_counts: Dict[str, int]
    audit: Dict[str, Dict[str, Any]]
    definition: CohortDefinition
    excluded_count: int
    base_filtered_count: int = 0

    def print_summary(self) -> None:
        """Print a human-readable summary of cohort resolution."""
        defn = self.definition
        print(f"\nCohort Resolution: {defn.name}")
        if defn.description:
            print(f"  {defn.description}")
        print()

        if self.base_filtered_count > 0:
            print(f"  Base filter: {self.base_filtered_count} samples excluded")

        for label, count in self.group_counts.items():
            audit = self.audit.get(label, {})
            print(f"  {label}: n={count}")
            for crit_info in audit.get("criteria", []):
                print(
                    f"    {crit_info['column']} {crit_info['operator']} "
                    f"{crit_info['value']}: {crit_info['n_matched']} matches"
                    + (" (allow_na)" if crit_info['allow_na'] else "")
                )

        if self.excluded_count > 0:
            print(f"  Unassigned: {self.excluded_count} samples (excluded)")

        total = sum(self.group_counts.values())
        print(f"  Total assigned: {total}")

        if defn.contrast:
            print(f"  Contrast: {defn.contrast[0]} vs {defn.contrast[1]}")


# =============================================================================
# Resolver
# =============================================================================


class CohortResolver:
    """Applies a CohortDefinition to metadata, producing labeled samples."""

    @staticmethod
    def resolve(
        metadata: pd.DataFrame,
        definition: CohortDefinition,
        cohort_column: str = "cohort",
        strict: bool = True,
    ) -> CohortResult:
        """Resolve cohort assignments from metadata.

        Parameters:
            metadata: Sample metadata DataFrame (indexed by sample ID).
            definition: The CohortDefinition to apply.
            cohort_column: Name of the output column for cohort labels.
            strict: If True (default), raise ValueError when criteria
                reference columns not in metadata. If False, log warnings
                and skip missing columns.

        Returns:
            CohortResult with labeled metadata and audit trail.
        """
        original_count = len(metadata)

        # 1. Apply base filter
        if definition.base_filter is not None:
            base_mask = definition.base_filter.evaluate(metadata, strict=strict)
            working = metadata[base_mask].copy()
            base_filtered_count = original_count - len(working)
            logger.info(
                f"Base filter '{definition.base_filter.label}': "
                f"{len(working)}/{original_count} samples pass"
            )
        else:
            working = metadata.copy()
            base_filtered_count = 0

        # 2. Assign groups (priority order — first match wins)
        #    Cache masks for reuse in overlap check.
        working[cohort_column] = None
        audit = {}
        group_masks: Dict[str, pd.Series] = {}

        for group in definition.groups:
            unassigned = working[cohort_column].isna()
            if not unassigned.any():
                break

            mask = group.evaluate(working, strict=strict)
            group_masks[group.label] = mask

            assign_mask = mask & unassigned
            n_assigned = assign_mask.sum()
            working.loc[assign_mask, cohort_column] = group.label

            # Audit: per-criterion match counts
            criterion_audit = []
            for criterion in group.criteria:
                if criterion.column in working.columns:
                    crit_mask = criterion.evaluate(working[criterion.column])
                    criterion_audit.append({
                        "column": criterion.column,
                        "operator": criterion.operator,
                        "value": criterion.value,
                        "allow_na": criterion.allow_na,
                        "n_matched": int(crit_mask.sum()),
                    })

            audit[group.label] = {
                "n_assigned": int(n_assigned),
                "criteria": criterion_audit,
                "logic": group.logic,
            }

            logger.info(f"  Group '{group.label}': {n_assigned} samples assigned")

        # 3. Filter to assigned samples only
        assigned = working[working[cohort_column].notna()].copy()
        excluded_count = len(working) - len(assigned)

        group_counts = assigned[cohort_column].value_counts().to_dict()

        # 4. Overlap check (reuses cached masks)
        group_labels = list(group_masks.keys())
        for i, label1 in enumerate(group_labels):
            for label2 in group_labels[i + 1:]:
                overlap = (group_masks[label1] & group_masks[label2]).sum()
                if overlap > 0:
                    logger.warning(
                        f"Overlap: {overlap} samples match both "
                        f"'{label1}' and '{label2}' criteria. "
                        f"First-match priority applied."
                    )

        return CohortResult(
            metadata=assigned,
            group_counts=group_counts,
            audit=audit,
            definition=definition,
            excluded_count=excluded_count,
            base_filtered_count=base_filtered_count,
        )

    @staticmethod
    def from_yaml(path: Path) -> CohortDefinition:
        """Load a CohortDefinition from a YAML file.

        Parameters:
            path: Path to YAML cohort definition file.

        Returns:
            Parsed CohortDefinition.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML is malformed or missing required fields.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cohort config not found: {path}")

        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in cohort config '{path}': {e}"
            ) from e

        if raw is None:
            raise ValueError(f"Empty cohort config file: {path}")

        if not isinstance(raw, dict):
            raise ValueError(
                f"Cohort config must be a YAML mapping, got {type(raw).__name__} "
                f"in '{path}'"
            )

        return CohortResolver._parse_definition(raw)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> CohortDefinition:
        """Parse a CohortDefinition from a dictionary."""
        return CohortResolver._parse_definition(raw)

    @staticmethod
    def _parse_definition(raw: Dict[str, Any]) -> CohortDefinition:
        """Parse raw dict into CohortDefinition."""

        if "name" not in raw:
            raise ValueError(
                "Cohort definition must have a 'name' field. "
                f"Got keys: {sorted(raw.keys())}"
            )

        def parse_criterion(d: Dict[str, Any]) -> CohortCriterion:
            if "column" not in d:
                raise ValueError(
                    f"Criterion must have a 'column' field. Got: {d}"
                )

            column = d["column"]
            allow_na = d.get("allow_na", False)

            # Find the operator — it's any key that's not 'column' or 'allow_na'
            op_keys = set(d.keys()) - {"column", "allow_na"}
            if len(op_keys) != 1:
                raise ValueError(
                    f"Criterion on column '{column}' must have exactly one "
                    f"operator key, got: {op_keys}. Valid operators: "
                    f"{sorted(CohortCriterion._VALID_OPERATORS)}"
                )

            operator = op_keys.pop()
            value = d[operator]

            return CohortCriterion(
                column=column,
                operator=operator,
                value=value,
                allow_na=allow_na,
            )

        def parse_group(d: Dict[str, Any]) -> CohortGroup:
            if "label" not in d:
                raise ValueError(
                    f"Cohort group must have a 'label' field. Got: {d}"
                )
            criteria_raw = d.get("criteria") or []  # Handle None
            return CohortGroup(
                label=d["label"],
                criteria=[parse_criterion(c) for c in criteria_raw],
                logic=d.get("logic", "any"),
            )

        base_filter = None
        if "base_filter" in raw:
            base_filter = parse_group(raw["base_filter"])

        groups = [parse_group(g) for g in (raw.get("groups") or [])]

        contrast = None
        if "contrast" in raw:
            c = raw["contrast"]
            if not isinstance(c, (list, tuple)) or len(c) != 2:
                raise ValueError(
                    f"'contrast' must be a list of exactly 2 group labels, "
                    f"got: {c}"
                )
            contrast = (c[0], c[1])

        return CohortDefinition(
            name=raw["name"],
            description=raw.get("description", ""),
            base_filter=base_filter,
            groups=groups,
            contrast=contrast,
        )

    @staticmethod
    def from_genetic_contrast(
        mutation: str,
        metadata: pd.DataFrame,
        mutation_col: str = "ClinReport_Mutations_Details",
        phenotype_col: str = "phenotype",
    ) -> CohortDefinition:
        """Build a CohortDefinition by introspecting metadata columns.

        Backward-compatible with the old --genetic-contrast pattern, but
        auto-detects genomic columns (e.g., repeat length) to produce a
        biologically accurate carrier definition.

        Parameters:
            mutation: Mutation name (e.g., "C9orf72", "SOD1").
            metadata: Sample metadata — inspected for available columns.
            mutation_col: Column with clinical mutation annotations.
            phenotype_col: Column with CASE/CTRL labels.

        Returns:
            CohortDefinition with auto-detected criteria.
        """
        # Base filter: CASE samples only
        base_filter = CohortGroup(
            label="ALS_cases",
            criteria=[CohortCriterion(column=phenotype_col, operator="eq", value="CASE")],
        )

        # --- Carrier criteria ---
        carrier_criteria = [
            CohortCriterion(column=mutation_col, operator="eq", value=mutation),
        ]

        # Auto-detect genomic columns for this mutation
        mutation_lower = mutation.lower().replace("-", "").replace("_", "")

        # Look up gene-specific repeat expansion threshold
        threshold = REPEAT_EXPANSION_THRESHOLDS.get(mutation_lower)

        # Look for repeat length columns (e.g., C9orf72_repeat_length)
        for col in metadata.columns:
            col_lower = col.lower().replace("-", "").replace("_", "")
            if mutation_lower in col_lower and "repeat" in col_lower and "length" in col_lower:
                if threshold is None:
                    logger.warning(
                        f"Found repeat length column '{col}' for {mutation}, "
                        f"but no validated pathogenic threshold is configured. "
                        f"Skipping auto-detection. Use an explicit YAML cohort "
                        f"definition to set the correct threshold."
                    )
                    continue

                carrier_criteria.append(
                    CohortCriterion(column=col, operator="gte", value=threshold)
                )
                logger.info(
                    f"Auto-detected genomic column '{col}' for {mutation} — "
                    f"adding repeat length >= {threshold} to carrier criteria"
                )

        # NOTE: ExpansionHunter status columns (EH_*) are intentionally NOT
        # auto-detected. In AnswerALS and similar datasets, EH_<gene> == "Yes"
        # means "was tested by ExpansionHunter", NOT "has pathogenic expansion".
        # The repeat length column is the reliable genomic indicator.
        # Use explicit YAML cohort definitions for datasets with non-standard
        # column semantics.

        carrier_group = CohortGroup(
            label=mutation.upper(),
            criteria=carrier_criteria,
            logic="any",  # OR — any criterion makes you a carrier
        )

        # --- Sporadic criteria ---
        sporadic_criteria = [
            CohortCriterion(
                column=mutation_col,
                operator="not_in",
                value=KNOWN_ALS_MUTATIONS,
                allow_na=True,
            ),
        ]

        # Exclude by repeat length if column detected
        for crit in carrier_criteria:
            if crit.operator == "gte" and "repeat" in crit.column.lower():
                sporadic_criteria.append(
                    CohortCriterion(
                        column=crit.column,
                        operator="lt",
                        value=crit.value,
                        allow_na=True,
                    )
                )

        sporadic_group = CohortGroup(
            label="SPORADIC",
            criteria=sporadic_criteria,
            logic="all",  # AND — must satisfy all exclusion criteria
        )

        carrier_label = mutation.upper()
        return CohortDefinition(
            name=f"{carrier_label}_vs_Sporadic",
            description=(
                f"Auto-generated: {carrier_label} carriers vs sporadic ALS. "
                f"Carrier criteria auto-detected from metadata columns."
            ),
            base_filter=base_filter,
            groups=[carrier_group, sporadic_group],
            contrast=(carrier_label, "SPORADIC"),
        )


# =============================================================================
# CLI Helper
# =============================================================================


def resolve_cohort_from_args(
    metadata: pd.DataFrame,
    cohort_config: Optional[Path] = None,
    genetic_contrast: Optional[str] = None,
    mutation_col: str = "ClinReport_Mutations_Details",
    phenotype_col: str = "phenotype",
    condition_col: str = "phenotype",
) -> Tuple[pd.DataFrame, str, Optional[List]]:
    """Unified cohort resolution for CLI commands.

    Handles --cohort-config and --genetic-contrast flags, returning
    resolved metadata with the appropriate condition column and contrast
    specification. Used by both differential.py and compare.py.

    Parameters:
        metadata: Sample metadata DataFrame.
        cohort_config: Path to YAML cohort definition (--cohort-config).
        genetic_contrast: Mutation name for auto-detection (--genetic-contrast).
        mutation_col: Column with clinical mutation annotations.
        phenotype_col: Column with CASE/CTRL labels.
        condition_col: Default condition column if no cohort resolution.

    Returns:
        (metadata, condition_col, contrast_list) where:
        - metadata has the cohort column added (if resolved)
        - condition_col is the column to use for differential analysis
        - contrast_list is [(name, cond1, cond2)] or None
    """
    if cohort_config and genetic_contrast:
        raise ValueError(
            "Cannot specify both --cohort-config and --genetic-contrast. "
            "Use --cohort-config for declarative YAML definitions, or "
            "--genetic-contrast for auto-detected definitions."
        )

    if cohort_config:
        definition = CohortResolver.from_yaml(cohort_config)
        result = CohortResolver.resolve(
            metadata, definition, cohort_column=COHORT_CONDITION_COLUMN,
        )
        result.print_summary()

        metadata = result.metadata
        condition_col = COHORT_CONDITION_COLUMN

        if definition.contrast:
            carrier_label, sporadic_label = definition.contrast
        else:
            labels = list(result.group_counts.keys())
            if len(labels) < 2:
                raise ValueError(
                    f"Cohort definition '{definition.name}' produced fewer "
                    f"than 2 groups ({labels}). Cannot form a contrast."
                )
            carrier_label, sporadic_label = labels[0], labels[1]

        contrast_list = [(
            f"{carrier_label}_vs_{sporadic_label}",
            carrier_label,
            sporadic_label,
        )]
        return metadata, condition_col, contrast_list

    elif genetic_contrast:
        if mutation_col not in metadata.columns:
            raise ValueError(
                f"Mutation column '{mutation_col}' not found in metadata. "
                f"Available columns: {', '.join(sorted(metadata.columns))}"
            )

        definition = CohortResolver.from_genetic_contrast(
            mutation=genetic_contrast,
            metadata=metadata,
            mutation_col=mutation_col,
            phenotype_col=phenotype_col,
        )
        result = CohortResolver.resolve(
            metadata, definition, cohort_column=COHORT_CONDITION_COLUMN,
        )
        result.print_summary()

        carrier_label, sporadic_label = definition.contrast

        if carrier_label not in result.group_counts:
            raise ValueError(
                f"No carriers found for mutation '{genetic_contrast}'. "
                f"Available mutations: "
                f"{metadata[metadata[phenotype_col] == 'CASE'][mutation_col].value_counts().to_dict()}"
            )
        if sporadic_label not in result.group_counts:
            raise ValueError("No sporadic ALS samples found.")

        n_carriers = result.group_counts.get(carrier_label, 0)
        n_sporadic = result.group_counts.get(sporadic_label, 0)
        if n_carriers < 30 or n_sporadic < 30:
            print(f"  WARNING: Small sample size detected. Statistical power may be limited.")
            if n_carriers < 10 or n_sporadic < 10:
                print(f"  WARNING: Very small sample size (n<10). Results should be interpreted with caution.")

        metadata = result.metadata
        condition_col = COHORT_CONDITION_COLUMN
        contrast_list = [(
            f"{carrier_label}_vs_{sporadic_label}",
            carrier_label,
            sporadic_label,
        )]
        return metadata, condition_col, contrast_list

    # No cohort resolution — use default condition column
    return metadata, condition_col, None
