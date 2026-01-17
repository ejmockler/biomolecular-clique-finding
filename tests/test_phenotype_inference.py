"""
Tests for phenotype inference modules.

These tests demonstrate the functionality of the new phenotype inference
and data filtering classes.
"""

import pandas as pd
import pytest


class TestMetadataRowFilter:
    """Tests for MetadataRowFilter class."""

    def test_basic_filtering(self):
        """Test basic pattern-based filtering."""
        from cliquefinder.io.data_filters import MetadataRowFilter

        # Create sample feature IDs
        feature_ids = pd.Index([
            "PROTEIN1",
            "PROTEIN2",
            "nFragment",
            "nPeptide",
            "PROTEIN3",
            "iRT_protein",
        ])

        # Create filter
        filter = MetadataRowFilter(patterns=["nFragment", "nPeptide", "iRT_protein"])

        # Apply filter
        filtered = filter.filter(feature_ids)

        # Verify results
        assert len(filtered) == 3
        assert "PROTEIN1" in filtered
        assert "PROTEIN2" in filtered
        assert "PROTEIN3" in filtered
        assert "nFragment" not in filtered
        assert "nPeptide" not in filtered
        assert "iRT_protein" not in filtered

        # Verify statistics
        assert filter.n_filtered_ == 3

    def test_case_insensitive_default(self):
        """Test case-insensitive matching (default behavior)."""
        from cliquefinder.io.data_filters import MetadataRowFilter

        feature_ids = pd.Index(["NFRAGMENT", "nfragment", "NFragment", "PROTEIN1"])

        filter = MetadataRowFilter(patterns=["nFragment"])
        filtered = filter.filter(feature_ids)

        # All variations should be filtered (case-insensitive)
        assert len(filtered) == 1
        assert "PROTEIN1" in filtered

    def test_case_sensitive(self):
        """Test case-sensitive matching."""
        from cliquefinder.io.data_filters import MetadataRowFilter

        feature_ids = pd.Index(["NFRAGMENT", "nfragment", "NFragment", "PROTEIN1"])

        filter = MetadataRowFilter(patterns=["nfragment"], case_sensitive=True)
        filtered = filter.filter(feature_ids)

        # Only exact match should be filtered
        assert len(filtered) == 3
        assert "NFRAGMENT" in filtered
        assert "NFragment" in filtered
        assert "PROTEIN1" in filtered
        assert "nfragment" not in filtered

    def test_get_filtered_ids(self):
        """Test retrieval of excluded IDs."""
        from cliquefinder.io.data_filters import MetadataRowFilter

        feature_ids = pd.Index(["PROTEIN1", "nFragment", "PROTEIN2", "nPeptide"])

        filter = MetadataRowFilter(patterns=["nFragment", "nPeptide"])
        excluded = filter.get_filtered_ids(feature_ids)

        assert len(excluded) == 2
        assert "nFragment" in excluded
        assert "nPeptide" in excluded


class TestRegexMetadataRowFilter:
    """Tests for RegexMetadataRowFilter class."""

    def test_regex_pattern_matching(self):
        """Test regex-based filtering."""
        from cliquefinder.io.data_filters import RegexMetadataRowFilter

        feature_ids = pd.Index([
            "iRT_protein_1",
            "iRT_protein_2",
            "PROTEIN_QC",
            "PROTEIN1",
            "PROTEIN2",
        ])

        # Filter features starting with "iRT" or ending with "_QC"
        filter = RegexMetadataRowFilter(patterns=[r"^iRT", r"_QC$"])
        filtered = filter.filter(feature_ids)

        assert len(filtered) == 2
        assert "PROTEIN1" in filtered
        assert "PROTEIN2" in filtered


class TestAnswerALSPhenotypeInferencer:
    """Tests for AnswerALS-specific phenotype inference."""

    def test_basic_inference(self):
        """Test basic phenotype inference from clinical metadata."""
        from cliquefinder.io.phenotype import AnswerALSPhenotypeInferencer

        # Create sample IDs
        sample_ids = pd.Index([
            "CASE_NEUAA295HHE-9014-P_D3",
            "CTRL_NEUBB123ABC-1234-P_D3",
            "CASE_NEUCC456DEF-5678-P_D3",
        ])

        # Create mock clinical metadata
        clinical_df = pd.DataFrame({
            "GUID": ["NEUAA295HHE", "NEUBB123ABC", "NEUCC456DEF"],
            "SUBJECT_GROUP": ["ALS", "Healthy Control", "ALS"],
        })

        # Create inferencer
        inferencer = AnswerALSPhenotypeInferencer(
            subject_group_col="SUBJECT_GROUP",
            case_values=["ALS"],
            ctrl_values=["Healthy Control"],
        )

        # Infer phenotypes
        phenotypes = inferencer.infer(sample_ids, clinical_df)

        # Verify results
        assert phenotypes["CASE_NEUAA295HHE-9014-P_D3"] == "CASE"
        assert phenotypes["CTRL_NEUBB123ABC-1234-P_D3"] == "CTRL"
        assert phenotypes["CASE_NEUCC456DEF-5678-P_D3"] == "CASE"

    def test_exclusion_logic(self):
        """Test exclusion of Non-ALS MND and Asymptomatic samples."""
        from cliquefinder.io.phenotype import AnswerALSPhenotypeInferencer

        sample_ids = pd.Index([
            "CASE_NEUAA111AAA-1111-P_D3",
            "CASE_NEUBB222BBB-2222-P_D3",
            "CASE_NEUCC333CCC-3333-P_D3",
        ])

        clinical_df = pd.DataFrame({
            "GUID": ["NEUAA111AAA", "NEUBB222BBB", "NEUCC333CCC"],
            "SUBJECT_GROUP": ["ALS", "Non-ALS MND", "Asymptomatic"],
        })

        inferencer = AnswerALSPhenotypeInferencer(
            subject_group_col="SUBJECT_GROUP",
            case_values=["ALS"],
            ctrl_values=["Healthy Control"],
            exclude_values=["Non-ALS MND", "Asymptomatic"],
        )

        phenotypes = inferencer.infer(sample_ids, clinical_df)

        assert phenotypes["CASE_NEUAA111AAA-1111-P_D3"] == "CASE"
        assert phenotypes["CASE_NEUBB222BBB-2222-P_D3"] == "EXCLUDE"
        assert phenotypes["CASE_NEUCC333CCC-3333-P_D3"] == "EXCLUDE"

    def test_fallback_to_sample_id(self):
        """Test fallback to sample ID prefix when metadata unavailable."""
        from cliquefinder.io.phenotype import AnswerALSPhenotypeInferencer

        sample_ids = pd.Index([
            "CASE_NEUAA999ZZZ-9999-P_D3",
            "CTRL_NEUBB888YYY-8888-P_D3",
        ])

        # Empty clinical metadata
        clinical_df = pd.DataFrame({
            "GUID": [],
            "SUBJECT_GROUP": [],
        })

        inferencer = AnswerALSPhenotypeInferencer()

        phenotypes = inferencer.infer(sample_ids, clinical_df)

        # Should fall back to sample ID prefix
        assert phenotypes["CASE_NEUAA999ZZZ-9999-P_D3"] == "CASE"
        assert phenotypes["CTRL_NEUBB888YYY-8888-P_D3"] == "CTRL"

    def test_provenance_tracking(self):
        """Test detailed provenance tracking."""
        from cliquefinder.io.phenotype import AnswerALSPhenotypeInferencer

        sample_ids = pd.Index([
            "CASE_NEUAA111AAA-1111-P_D3",
            "CTRL_NEUBB999ZZZ-9999-P_D3",  # Not in metadata
        ])

        clinical_df = pd.DataFrame({
            "GUID": ["NEUAA111AAA"],
            "SUBJECT_GROUP": ["ALS"],
        })

        inferencer = AnswerALSPhenotypeInferencer()

        provenance = inferencer.get_inference_provenance(sample_ids, clinical_df)

        # Verify provenance structure
        assert "sample_id" in provenance.columns
        assert "phenotype" in provenance.columns
        assert "source" in provenance.columns
        assert "subject_id" in provenance.columns
        assert "subject_group" in provenance.columns

        # Verify provenance values
        row1 = provenance[provenance["sample_id"] == "CASE_NEUAA111AAA-1111-P_D3"].iloc[0]
        assert row1["phenotype"] == "CASE"
        assert row1["source"] == "metadata"
        assert row1["subject_id"] == "NEUAA111AAA"
        assert row1["subject_group"] == "ALS"

        row2 = provenance[provenance["sample_id"] == "CTRL_NEUBB999ZZZ-9999-P_D3"].iloc[0]
        assert row2["phenotype"] == "CTRL"
        assert row2["source"] == "sample_id_fallback"
        assert row2["subject_id"] == "NEUBB999ZZZ"


class TestGenericPhenotypeInferencer:
    """Tests for generic phenotype inference."""

    def test_basic_inference(self):
        """Test simple metadata column-based inference."""
        from cliquefinder.io.phenotype import GenericPhenotypeInferencer

        sample_ids = pd.Index(["SAMPLE1", "SAMPLE2", "SAMPLE3"])

        clinical_df = pd.DataFrame({
            "sample_id": ["SAMPLE1", "SAMPLE2", "SAMPLE3"],
            "disease_status": ["tumor", "normal", "tumor"],
        })

        inferencer = GenericPhenotypeInferencer(
            phenotype_col="disease_status",
            case_values=["tumor"],
            ctrl_values=["normal"],
        )

        phenotypes = inferencer.infer(sample_ids, clinical_df)

        assert phenotypes["SAMPLE1"] == "CASE"
        assert phenotypes["SAMPLE2"] == "CTRL"
        assert phenotypes["SAMPLE3"] == "CASE"

    def test_multiple_values_per_category(self):
        """Test inference with multiple values mapping to same phenotype."""
        from cliquefinder.io.phenotype import GenericPhenotypeInferencer

        sample_ids = pd.Index(["S1", "S2", "S3", "S4"])

        clinical_df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3", "S4"],
            "status": ["primary_tumor", "metastatic", "normal", "healthy"],
        })

        inferencer = GenericPhenotypeInferencer(
            phenotype_col="status",
            case_values=["primary_tumor", "metastatic"],
            ctrl_values=["normal", "healthy"],
        )

        phenotypes = inferencer.infer(sample_ids, clinical_df)

        assert phenotypes["S1"] == "CASE"
        assert phenotypes["S2"] == "CASE"
        assert phenotypes["S3"] == "CTRL"
        assert phenotypes["S4"] == "CTRL"

    def test_unknown_values_excluded(self):
        """Test that unknown values are marked as EXCLUDE."""
        from cliquefinder.io.phenotype import GenericPhenotypeInferencer

        sample_ids = pd.Index(["S1", "S2", "S3"])

        clinical_df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "status": ["tumor", "normal", "unknown"],
        })

        inferencer = GenericPhenotypeInferencer(
            phenotype_col="status",
            case_values=["tumor"],
            ctrl_values=["normal"],
        )

        phenotypes = inferencer.infer(sample_ids, clinical_df)

        assert phenotypes["S1"] == "CASE"
        assert phenotypes["S2"] == "CTRL"
        assert phenotypes["S3"] == "EXCLUDE"

    def test_provenance_tracking(self):
        """Test generic inferencer provenance tracking."""
        from cliquefinder.io.phenotype import GenericPhenotypeInferencer

        sample_ids = pd.Index(["S1", "S2"])

        clinical_df = pd.DataFrame({
            "sample_id": ["S1", "S2"],
            "status": ["tumor", "normal"],
        })

        inferencer = GenericPhenotypeInferencer(
            phenotype_col="status",
            case_values=["tumor"],
            ctrl_values=["normal"],
        )

        provenance = inferencer.get_inference_provenance(sample_ids, clinical_df)

        # Verify structure
        assert "sample_id" in provenance.columns
        assert "phenotype" in provenance.columns
        assert "source" in provenance.columns
        assert "raw_value" in provenance.columns

        # Verify values
        row1 = provenance[provenance["sample_id"] == "S1"].iloc[0]
        assert row1["phenotype"] == "CASE"
        assert row1["raw_value"] == "tumor"
