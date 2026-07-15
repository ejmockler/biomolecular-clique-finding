"""Tests for network-query integration in the differential CLI."""

import argparse
import inspect
from pathlib import Path


def test_cli_imports():
    """Differential CLI exposes the network-query helper and parser setup."""
    from cliquefinder.cli.differential import query_network_targets, setup_parser

    assert callable(query_network_targets)
    assert callable(setup_parser)


def test_helper_function_signature():
    """The network-query helper retains its public calling convention."""
    from cliquefinder.cli.differential import query_network_targets

    params = list(inspect.signature(query_network_targets).parameters)
    assert params == [
        "gene_symbol",
        "feature_ids",
        "min_evidence",
        "min_sources",
        "env_file",
        "verbose",
        "output_dir",
    ]


def test_parser_arguments():
    """The differential parser accepts all network-query options."""
    from cliquefinder.cli.differential import setup_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)

    args = parser.parse_args(
        [
            "differential",
            "--data",
            "dummy.csv",
            "--metadata",
            "dummy.csv",
            "--cliques",
            "dummy.csv",
            "--output",
            "dummy",
            "--network-query",
            "C9ORF72",
            "--min-evidence",
            "2",
            "--indra-env-file",
            "/path/to/.env",
        ]
    )

    assert args.network_query == "C9ORF72"
    assert args.min_evidence == 2
    assert args.indra_env_file == Path("/path/to/.env")


def test_indra_knowledge_source_imports_without_optional_runtime():
    """The lazy INDRA adapter itself remains importable in a base install."""
    from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

    assert INDRAKnowledgeSource.__name__ == "INDRAKnowledgeSource"


def test_id_mapping_function_is_public():
    """The ID-mapping helper used by the CLI remains importable."""
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols

    assert callable(map_feature_ids_to_symbols)
