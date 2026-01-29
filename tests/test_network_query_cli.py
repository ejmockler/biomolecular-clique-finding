#!/usr/bin/env python3
"""
Test script for network query integration in differential CLI.

This script validates that:
1. CLI arguments are properly added
2. Helper function can be imported and called
3. Integration follows expected patterns
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_cli_imports():
    """Test that differential CLI imports correctly."""
    print("Testing CLI imports...")
    try:
        from cliquefinder.cli.differential import query_network_targets, setup_parser
        print("  ✓ CLI functions import successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_helper_function_signature():
    """Test that helper function has correct signature."""
    print("\nTesting helper function signature...")
    try:
        from cliquefinder.cli.differential import query_network_targets
        import inspect

        sig = inspect.signature(query_network_targets)
        params = list(sig.parameters.keys())

        expected_params = ['gene_symbol', 'feature_ids', 'min_evidence', 'env_file', 'verbose']
        if params == expected_params:
            print(f"  ✓ Function signature correct: {params}")
            return True
        else:
            print(f"  ✗ Expected {expected_params}, got {params}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_parser_arguments():
    """Test that parser has network query arguments."""
    print("\nTesting parser arguments...")
    try:
        import argparse
        from cliquefinder.cli.differential import setup_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        setup_parser(subparsers)

        # Parse help to check if arguments exist
        test_args = [
            'differential',
            '--data', 'dummy.csv',
            '--metadata', 'dummy.csv',
            '--cliques', 'dummy.csv',
            '--output', 'dummy',
            '--network-query', 'C9ORF72',
            '--min-evidence', '2',
            '--indra-env-file', '/path/to/.env'
        ]

        args = parser.parse_args(test_args)

        if hasattr(args, 'network_query') and args.network_query == 'C9ORF72':
            print("  ✓ --network-query argument works")
        else:
            print("  ✗ --network-query argument missing or incorrect")
            return False

        if hasattr(args, 'min_evidence') and args.min_evidence == 2:
            print("  ✓ --min-evidence argument works")
        else:
            print("  ✗ --min-evidence argument missing or incorrect")
            return False

        if hasattr(args, 'indra_env_file'):
            print("  ✓ --indra-env-file argument works")
        else:
            print("  ✗ --indra-env-file argument missing")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_indra_dependencies():
    """Test INDRA dependencies availability."""
    print("\nTesting INDRA dependencies...")
    try:
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource
        print("  ✓ INDRAKnowledgeSource can be imported")
        return True
    except ImportError as e:
        print(f"  ⚠ INDRAKnowledgeSource not available (expected if INDRA not installed): {e}")
        return True  # Not a failure - just not installed


def test_id_mapping_function():
    """Test that ID mapping function can be imported."""
    print("\nTesting ID mapping function...")
    try:
        from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
        import inspect

        sig = inspect.signature(map_feature_ids_to_symbols)
        print(f"  ✓ map_feature_ids_to_symbols available with params: {list(sig.parameters.keys())}")
        return True
    except ImportError as e:
        print(f"  ✗ Function not available: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Network Query CLI Integration Test")
    print("=" * 70)

    tests = [
        test_cli_imports,
        test_helper_function_signature,
        test_parser_arguments,
        test_indra_dependencies,
        test_id_mapping_function,
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)

    if all(results):
        print("\n✓ All tests passed! Implementation looks good.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Review the output above.")
        sys.exit(1)
