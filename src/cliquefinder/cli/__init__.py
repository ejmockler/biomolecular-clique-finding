"""
CliqueFinder CLI - Command-line interface for co-expression module discovery.

Commands:
    cliquefinder impute        - Detect and impute outliers in expression data
    cliquefinder analyze       - Regulatory validation with INDRA CoGEx (knowledge-guided)
    cliquefinder discover      - De novo co-expression discovery (data-driven)
    cliquefinder differential  - Clique-level differential abundance (MSstats-inspired)
    cliquefinder sensitivity   - MAD-Z threshold sensitivity analysis (methodological rigor)
    cliquefinder viz           - Generate visualizations and reports
"""

import argparse
import sys
from typing import Optional, List


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI dispatcher for cliquefinder."""
    parser = argparse.ArgumentParser(
        prog="cliquefinder",
        description="Regulatory clique discovery for ALS transcriptomics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  impute        Detect and impute outliers in expression data
  analyze       Regulatory validation with INDRA CoGEx (knowledge-guided)
  discover      De novo co-expression module discovery (data-driven)
  differential  Clique-level differential abundance (MSstats-inspired)
  sensitivity   MAD-Z threshold sensitivity analysis (methodological rigor)
  viz           Generate visualizations and reports

Examples:
  cliquefinder impute --input data.csv --output results/imputed
  cliquefinder analyze --input results/imputed.data.csv --discover --workers 6
  cliquefinder discover --input data.csv --n-genes 5000 --min-correlation 0.8
  cliquefinder sensitivity --input data.csv --output results/sensitivity
        """
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import and register subcommands
    from cliquefinder.cli import impute, analyze, discover, differential, sensitivity, viz
    impute.register_parser(subparsers)
    analyze.register_parser(subparsers)
    discover.register_parser(subparsers)
    differential.setup_parser(subparsers)
    sensitivity.register_parser(subparsers)
    viz.register_parser(subparsers)

    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    # Dispatch to subcommand
    return parsed_args.func(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
