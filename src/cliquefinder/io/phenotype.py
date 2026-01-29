"""
Phenotype inference for biomolecular datasets.

This module provides study-specific phenotype inference logic, enabling separation
of study-specific business rules from generic data processing pipelines.

Classes:
    PhenotypeInferencer: Abstract base class for phenotype inference
    AnswerALSPhenotypeInferencer: ALS AnswerALS-specific phenotype logic
    GenericPhenotypeInferencer: Generic phenotype inference from metadata column

Design Philosophy:
    - Study-specific logic is encapsulated in dedicated classes
    - Generic pipelines can swap out inferencer implementations
    - Phenotypes map to standard categories: CASE, CTRL, EXCLUDE
    - Inference provenance is tracked for quality assurance

Example:
    >>> import pandas as pd
    >>> from cliquefinder.io.phenotype import AnswerALSPhenotypeInferencer
    >>>
    >>> # Load clinical metadata
    >>> clinical_df = pd.read_csv("portal_metadata.csv")
    >>>
    >>> # Create inferencer with ALS-specific logic
    >>> inferencer = AnswerALSPhenotypeInferencer(
    ...     subject_group_col="SUBJECT_GROUP",
    ...     case_values=["ALS"],
    ...     ctrl_values=["Healthy Control"],
    ...     exclude_values=["Non-ALS MND", "Asymptomatic"],
    ...     sample_id_pattern=r"(?:CASE|CTRL)_([A-Z0-9]+)",
    ... )
    >>>
    >>> # Infer phenotypes
    >>> sample_ids = pd.Index(["CASE_NEUAA295HHE-9014-P_D3", "CTRL_NEUBB123ABC-1234-P_D3"])
    >>> phenotypes = inferencer.infer(sample_ids, clinical_df)
    >>> print(phenotypes)
    CASE_NEUAA295HHE-9014-P_D3    CASE
    CTRL_NEUBB123ABC-1234-P_D3    CTRL
    dtype: object
"""

import re
from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd


class PhenotypeInferencer(ABC):
    """
    Base class for study-specific phenotype inference.

    Subclasses implement study-specific logic for mapping sample IDs to phenotypes
    (CASE/CTRL/EXCLUDE) using clinical metadata and sample ID patterns.

    Methods:
        infer: Map sample IDs to phenotypes
        get_inference_provenance: Return provenance tracking for each sample
    """

    @abstractmethod
    def infer(
        self,
        sample_ids: pd.Index,
        clinical_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """
        Infer phenotypes for a set of sample IDs.

        Args:
            sample_ids: Sample IDs to infer phenotypes for
            clinical_df: Clinical metadata DataFrame (study-specific columns)

        Returns:
            Series mapping sample_id -> phenotype (CASE/CTRL/EXCLUDE)
        """
        pass

    def get_inference_provenance(
        self,
        sample_ids: pd.Index,
        clinical_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Return provenance information for phenotype inference.

        Args:
            sample_ids: Sample IDs to infer phenotypes for
            clinical_df: Clinical metadata DataFrame (study-specific columns)

        Returns:
            DataFrame with columns:
                - sample_id: Sample identifier
                - phenotype: Inferred phenotype (CASE/CTRL/EXCLUDE)
                - source: Inference method used (e.g., 'metadata', 'sample_id_fallback')
        """
        phenotypes = self.infer(sample_ids, clinical_df)

        # Default implementation: mark all as 'metadata' source
        # Subclasses can override for more detailed provenance
        return pd.DataFrame({
            'sample_id': phenotypes.index,
            'phenotype': phenotypes.values,
            'source': 'metadata',
        })


class AnswerALSPhenotypeInferencer(PhenotypeInferencer):
    """
    AnswerALS study-specific phenotype inference.

    Implements the following logic:
    1. Extract participant GUID from sample ID using regex pattern
    2. Look up SUBJECT_GROUP in clinical metadata
    3. Map SUBJECT_GROUP to CASE/CTRL/EXCLUDE based on configuration
    4. Fallback to sample ID prefix (CASE_/CTRL_) if metadata lookup fails
    5. Mark as EXCLUDE if no match found

    Args:
        subject_group_col: Column name for subject group in clinical metadata
            (default: "SUBJECT_GROUP" for AnswerALS)
        case_values: Subject group values to map to CASE (default: ["ALS"]
            for AnswerALS, customize for other studies)
        ctrl_values: Subject group values to map to CTRL (default: ["Healthy Control"]
            for AnswerALS, customize for other studies)
        exclude_values: Subject group values to explicitly exclude (default:
            ["Non-ALS MND", "Asymptomatic"] for AnswerALS, customize for other studies)
        sample_id_pattern: Regex pattern for extracting participant ID (GUID)
            from sample ID. If None, uses default AnswerALS pattern
            r"^(?:CASE|CTRL)_([A-Z0-9]+)". The pattern should have one capture
            group for the subject ID.
        subject_id_col: Column name for subject/participant ID in clinical metadata
            (default: "GUID" for AnswerALS, customize for other studies)

    Note:
        This class is designed for AnswerALS and uses ALS-specific defaults.
        For other studies, either:
        1. Provide all parameters explicitly to match your study
        2. Use GenericPhenotypeInferencer for simpler phenotype mapping
        3. Subclass PhenotypeInferencer for complex custom logic

    Example:
        >>> # AnswerALS with defaults
        >>> inferencer = AnswerALSPhenotypeInferencer()
        >>> phenotypes = inferencer.infer(sample_ids, clinical_df)
        >>>
        >>> # Custom study with explicit parameters
        >>> inferencer = AnswerALSPhenotypeInferencer(
        ...     subject_group_col="disease_type",
        ...     case_values=["Parkinsons", "PD"],
        ...     ctrl_values=["Control", "Healthy"],
        ...     exclude_values=["Other"],
        ...     sample_id_pattern=r"SAMPLE_(\d+)",
        ...     subject_id_col="patient_id",
        ... )
    """

    def __init__(
        self,
        subject_group_col: str = "SUBJECT_GROUP",
        case_values: list[str] | None = None,
        ctrl_values: list[str] | None = None,
        exclude_values: list[str] | None = None,
        sample_id_pattern: str | None = None,
        subject_id_col: str = "GUID",
    ):
        """
        Initialize AnswerALS phenotype inferencer.

        Note:
            All default values are AnswerALS-specific:
            - case_values: ["ALS"] (ALS-specific disease group)
            - ctrl_values: ["Healthy Control"] (ALS-specific control group)
            - exclude_values: ["Non-ALS MND", "Asymptomatic"] (ALS-specific exclusions)
            - subject_id_col: "GUID" (AnswerALS subject identifier column)
            - sample_id_pattern: r"^(?:CASE|CTRL)_([A-Z0-9]+)" (AnswerALS sample ID format)

            For other studies, explicitly provide these parameters to match your data.
        """
        self.subject_group_col = subject_group_col
        self.case_values = case_values or ["ALS"]
        self.ctrl_values = ctrl_values or ["Healthy Control"]
        self.exclude_values = exclude_values or ["Non-ALS MND", "Asymptomatic"]
        self.subject_id_col = subject_id_col

        # Default AnswerALS pattern: extract GUID from "CASE_NEUAA295HHE-9014-P_D3"
        # Pattern extracts the part between first underscore and first hyphen
        if sample_id_pattern is None:
            # Extract the participant ID component (e.g., NEUAA295HHE from CASE_NEUAA295HHE-9014-P_D3)
            self.sample_id_pattern = r"^(?:CASE|CTRL)_([A-Z0-9]+)"
        else:
            self.sample_id_pattern = sample_id_pattern

        self._compiled_pattern = re.compile(self.sample_id_pattern)

    def _extract_subject_id(self, sample_id: str) -> str | None:
        """
        Extract participant/subject ID (GUID) from sample ID.

        Args:
            sample_id: Sample identifier (e.g., "CASE_NEUAA295HHE-9014-P_D3")

        Returns:
            Extracted subject ID (e.g., "NEUAA295HHE") or None if no match
        """
        match = self._compiled_pattern.search(sample_id)
        if match:
            return match.group(1)
        return None

    def infer(
        self,
        sample_ids: pd.Index,
        clinical_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """
        Infer phenotypes for AnswerALS samples.

        Args:
            sample_ids: Sample IDs to infer phenotypes for
            clinical_df: Clinical metadata with GUID and SUBJECT_GROUP columns

        Returns:
            Series mapping sample_id -> phenotype (CASE/CTRL/EXCLUDE)
        """
        phenotypes = pd.Series(index=sample_ids, dtype=object)

        # Build lookup dictionary from clinical metadata
        guid_to_group = {}
        if clinical_df is not None:
            if self.subject_id_col in clinical_df.columns and self.subject_group_col in clinical_df.columns:
                guid_to_group = dict(
                    zip(clinical_df[self.subject_id_col], clinical_df[self.subject_group_col])
                )

        for sample_id in sample_ids:
            # Extract GUID
            guid = self._extract_subject_id(sample_id)

            # Primary: metadata lookup
            if guid and guid in guid_to_group:
                group = guid_to_group[guid]

                if group in self.case_values:
                    phenotypes[sample_id] = 'CASE'
                elif group in self.ctrl_values:
                    phenotypes[sample_id] = 'CTRL'
                elif group in self.exclude_values:
                    phenotypes[sample_id] = 'EXCLUDE'
                else:
                    # Unknown group from metadata
                    phenotypes[sample_id] = 'EXCLUDE'

            # Fallback: sample ID prefix
            elif sample_id.startswith('CTRL_'):
                phenotypes[sample_id] = 'CTRL'
            elif sample_id.startswith('CASE_'):
                phenotypes[sample_id] = 'CASE'
            else:
                phenotypes[sample_id] = 'EXCLUDE'

        return phenotypes

    def get_inference_provenance(
        self,
        sample_ids: pd.Index,
        clinical_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Return detailed provenance for AnswerALS phenotype inference.

        Returns:
            DataFrame with columns:
                - sample_id: Sample identifier
                - phenotype: Inferred phenotype
                - source: Inference method (metadata/sample_id_fallback/no_match)
                - subject_id: Extracted subject/participant ID (GUID)
                - subject_group: Raw SUBJECT_GROUP value from metadata (if available)
        """
        phenotypes = pd.Series(index=sample_ids, dtype=object)
        sources = pd.Series(index=sample_ids, dtype=object)
        subject_ids = pd.Series(index=sample_ids, dtype=object)
        subject_groups = pd.Series(index=sample_ids, dtype=object)

        # Build lookup dictionaries from clinical metadata
        guid_to_group = {}
        if clinical_df is not None:
            if self.subject_id_col in clinical_df.columns and self.subject_group_col in clinical_df.columns:
                guid_to_group = dict(
                    zip(clinical_df[self.subject_id_col], clinical_df[self.subject_group_col])
                )

        for sample_id in sample_ids:
            # Extract GUID
            guid = self._extract_subject_id(sample_id)
            subject_ids[sample_id] = guid

            # Primary: metadata lookup
            if guid and guid in guid_to_group:
                group = guid_to_group[guid]
                subject_groups[sample_id] = group

                if group in self.case_values:
                    phenotypes[sample_id] = 'CASE'
                    sources[sample_id] = 'metadata'
                elif group in self.ctrl_values:
                    phenotypes[sample_id] = 'CTRL'
                    sources[sample_id] = 'metadata'
                elif group in self.exclude_values:
                    phenotypes[sample_id] = 'EXCLUDE'
                    sources[sample_id] = 'metadata'
                else:
                    phenotypes[sample_id] = 'EXCLUDE'
                    sources[sample_id] = 'metadata'

            # Fallback: sample ID prefix
            elif sample_id.startswith('CTRL_'):
                phenotypes[sample_id] = 'CTRL'
                sources[sample_id] = 'sample_id_fallback'
            elif sample_id.startswith('CASE_'):
                phenotypes[sample_id] = 'CASE'
                sources[sample_id] = 'sample_id_fallback'
            else:
                phenotypes[sample_id] = 'EXCLUDE'
                sources[sample_id] = 'no_match'

        return pd.DataFrame({
            'sample_id': sample_ids,
            'phenotype': phenotypes.values,
            'source': sources.values,
            'subject_id': subject_ids.values,
            'subject_group': subject_groups.values,
        })


class GenericPhenotypeInferencer(PhenotypeInferencer):
    """
    Generic phenotype inference from a metadata column.

    Simple lookup-based inference that maps metadata column values to CASE/CTRL/EXCLUDE.
    Useful for studies with straightforward phenotype annotations.

    Args:
        phenotype_col: Column name in clinical metadata containing phenotype info
        case_values: Values to map to CASE (required, study-specific)
        ctrl_values: Values to map to CTRL (required, study-specific)
        exclude_values: Values to explicitly exclude (default: all other values
            are marked as EXCLUDE)
        sample_id_col: Column name for sample ID in clinical metadata
            (default: "sample_id", customize if your data uses different naming)

    Note:
        Unlike AnswerALSPhenotypeInferencer, this class has no study-specific
        defaults. You must explicitly provide case_values and ctrl_values that
        match your study's phenotype column values.

    Example:
        >>> # Cancer study
        >>> inferencer = GenericPhenotypeInferencer(
        ...     phenotype_col="disease_status",
        ...     case_values=["tumor", "cancer"],
        ...     ctrl_values=["normal", "healthy"],
        ... )
        >>> phenotypes = inferencer.infer(sample_ids, clinical_df)
        >>>
        >>> # Neurological study with custom sample ID column
        >>> inferencer = GenericPhenotypeInferencer(
        ...     phenotype_col="diagnosis",
        ...     case_values=["AD", "Alzheimers"],
        ...     ctrl_values=["Control"],
        ...     sample_id_col="specimen_id",
        ... )
    """

    def __init__(
        self,
        phenotype_col: str,
        case_values: list[str],
        ctrl_values: list[str],
        exclude_values: list[str] | None = None,
        sample_id_col: str = "sample_id",
    ):
        self.phenotype_col = phenotype_col
        self.case_values = case_values
        self.ctrl_values = ctrl_values
        self.exclude_values = exclude_values or []
        self.sample_id_col = sample_id_col

    def infer(
        self,
        sample_ids: pd.Index,
        clinical_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """
        Infer phenotypes from metadata column.

        Args:
            sample_ids: Sample IDs to infer phenotypes for
            clinical_df: Clinical metadata with sample_id and phenotype columns

        Returns:
            Series mapping sample_id -> phenotype (CASE/CTRL/EXCLUDE)
        """
        phenotypes = pd.Series(index=sample_ids, dtype=object)

        # Build lookup dictionary
        sample_to_value = {}
        if clinical_df is not None:
            if self.sample_id_col in clinical_df.columns and self.phenotype_col in clinical_df.columns:
                sample_to_value = dict(
                    zip(clinical_df[self.sample_id_col], clinical_df[self.phenotype_col])
                )

        for sample_id in sample_ids:
            if sample_id in sample_to_value:
                value = sample_to_value[sample_id]

                if value in self.case_values:
                    phenotypes[sample_id] = 'CASE'
                elif value in self.ctrl_values:
                    phenotypes[sample_id] = 'CTRL'
                else:
                    # Either in exclude_values or unknown
                    phenotypes[sample_id] = 'EXCLUDE'
            else:
                # Sample not found in metadata
                phenotypes[sample_id] = 'EXCLUDE'

        return phenotypes

    def get_inference_provenance(
        self,
        sample_ids: pd.Index,
        clinical_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Return provenance for generic phenotype inference.

        Returns:
            DataFrame with columns:
                - sample_id: Sample identifier
                - phenotype: Inferred phenotype
                - source: Always 'metadata' for generic inference
                - raw_value: Raw value from phenotype column
        """
        phenotypes = self.infer(sample_ids, clinical_df)
        raw_values = pd.Series(index=sample_ids, dtype=object)

        # Build lookup dictionary
        sample_to_value = {}
        if clinical_df is not None:
            if self.sample_id_col in clinical_df.columns and self.phenotype_col in clinical_df.columns:
                sample_to_value = dict(
                    zip(clinical_df[self.sample_id_col], clinical_df[self.phenotype_col])
                )

        for sample_id in sample_ids:
            if sample_id in sample_to_value:
                raw_values[sample_id] = sample_to_value[sample_id]

        return pd.DataFrame({
            'sample_id': sample_ids,
            'phenotype': phenotypes.values,
            'source': 'metadata',
            'raw_value': raw_values.values,
        })
