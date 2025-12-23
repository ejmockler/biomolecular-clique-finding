#!/usr/bin/env python3
"""
Quick validation runner for integration tests.

This script runs all test suites and generates a summary report.
Use this before merging optimizations to verify everything works.

Usage:
    python tests/run_validation.py              # Run all tests
    python tests/run_validation.py --quick      # Skip slow performance tests
    python tests/run_validation.py --report     # Generate detailed report
"""

import sys
import argparse
import subprocess
from pathlib import Path
import time


def run_pytest(test_file, verbose=True, show_output=False):
    """
    Run pytest on a specific test file.

    Returns:
        (passed, failed, duration)
    """
    cmd = ["pytest", test_file, "-v"]
    if show_output:
        cmd.append("-s")

    # Add JSON report for parsing
    cmd.extend(["--tb=short"])

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from repo root
        )
        duration = time.time() - start

        # Parse output for pass/fail counts
        output = result.stdout if result.stdout else ""

        # Simple parsing: look for "X passed" or "X failed"
        passed = 0
        failed = 0

        for line in output.split('\n'):
            if 'passed' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'passed' in part.lower() and i > 0:
                        try:
                            passed = int(parts[i-1])
                        except:
                            pass

            if 'failed' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'failed' in part.lower() and i > 0:
                        try:
                            failed = int(parts[i-1])
                        except:
                            pass

        return (passed, failed, duration, result.returncode == 0)

    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return (0, 0, 0, False)


def main():
    parser = argparse.ArgumentParser(description="Run integration test validation")
    parser.add_argument("--quick", action="store_true",
                       help="Skip slow performance tests")
    parser.add_argument("--report", action="store_true",
                       help="Generate detailed validation report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")

    args = parser.parse_args()

    # Test files to run
    test_files = [
        ("Numerical Equivalence", "tests/test_optimization_equivalence.py", False),
        ("Code Cleanup", "tests/test_legacy_removed.py", False),
        ("Full Pipeline Integration", "tests/test_full_pipeline.py", False),
    ]

    if not args.quick:
        test_files.append(
            ("Performance Benchmarks", "tests/test_optimization_performance.py", True)
        )

    print("=" * 70)
    print("INTEGRATION TEST VALIDATION")
    print("=" * 70)
    print()

    results = {}
    total_passed = 0
    total_failed = 0
    total_duration = 0
    all_success = True

    for name, test_file, show_output in test_files:
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print(f"{'=' * 70}\n")

        passed, failed, duration, success = run_pytest(
            test_file,
            verbose=args.verbose,
            show_output=show_output
        )

        results[name] = {
            'passed': passed,
            'failed': failed,
            'duration': duration,
            'success': success
        }

        total_passed += passed
        total_failed += failed
        total_duration += duration
        all_success = all_success and success

        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{status} - {name}")
        print(f"  Tests: {passed} passed, {failed} failed")
        print(f"  Time: {duration:.1f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}\n")

    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name:40s} {result['passed']:3d} passed  {result['duration']:6.1f}s")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed in {total_duration:.1f}s")
    print(f"{'=' * 70}\n")

    if all_success:
        print("✅ ALL VALIDATIONS PASSED\n")
        print("Ready to merge optimizations:")
        print("  - Numerical equivalence verified")
        print("  - Code quality validated")
        print("  - Integration tests successful")
        if not args.quick:
            print("  - Performance benchmarks complete")
        return 0
    else:
        print("❌ VALIDATION FAILED\n")
        print("Fix failing tests before merging optimizations.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
