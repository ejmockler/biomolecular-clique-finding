"""
CliqueFinder CLI - Command-line interface for co-expression module discovery.

Commands:
    cliquefinder impute        - Detect and impute outliers in expression data
    cliquefinder analyze       - Regulatory validation with INDRA CoGEx (knowledge-guided)
    cliquefinder discover      - De novo co-expression discovery (data-driven)
    cliquefinder differential  - Clique-level differential abundance (MSstats-inspired)
    cliquefinder compare       - Cross-method differential abundance comparison
    cliquefinder sensitivity   - MAD-Z threshold sensitivity analysis (methodological rigor)
    cliquefinder viz           - Generate visualizations and reports
"""

import argparse
import logging
import sys
import traceback
from typing import Optional, List

logger = logging.getLogger(__name__)


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
  compare       Cross-method differential abundance comparison
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

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show full tracebacks on error",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import and register subcommands
    from cliquefinder.cli import impute, analyze, discover, differential, compare, sensitivity, viz, validate_baselines
    impute.register_parser(subparsers)
    analyze.register_parser(subparsers)
    discover.register_parser(subparsers)
    differential.setup_parser(subparsers)
    compare.setup_parser(subparsers)
    sensitivity.register_parser(subparsers)
    viz.register_parser(subparsers)
    validate_baselines.register_parser(subparsers)

    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    # CLI-14: Catch unhandled exceptions and present user-friendly errors.
    # Full tracebacks are shown only in verbose mode.
    try:
        return parsed_args.func(parsed_args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        verbose = getattr(parsed_args, "verbose", False)
        if verbose:
            traceback.print_exc()
        else:
            print(
                f"Error: {exc}\n\nRe-run with --verbose for the full traceback.",
                file=sys.stderr,
            )
        logger.debug("Unhandled exception in CLI", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
