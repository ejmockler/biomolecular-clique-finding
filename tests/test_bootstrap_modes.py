"""
Test demonstrating the difference between bootstrap modes.

This shows how bootstrap_controls=True (true bootstrap) differs from
bootstrap_controls=False (subsampling) in terms of sample reuse.
"""

import numpy as np
import pandas as pd
from collections import Counter


def demonstrate_bootstrap_modes():
    """Show the difference between bootstrap and subsampling modes."""

    # Set up test data
    rng = np.random.default_rng(42)

    case_samples = [f"case_{i}" for i in range(10)]
    ctrl_samples = [f"ctrl_{i}" for i in range(5)]

    n_bootstraps = 5
    target_n_cases = 6
    n_ctrls = len(ctrl_samples)

    print("=" * 70)
    print("BOOTSTRAP MODE COMPARISON")
    print("=" * 70)
    print(f"Available: {len(case_samples)} cases, {len(ctrl_samples)} controls")
    print(f"Target per bootstrap: {target_n_cases} cases, {n_ctrls} controls")
    print()

    # Mode 1: TRUE BOOTSTRAP (replace=True for both)
    print("MODE 1: TRUE BOOTSTRAP (bootstrap_controls=True)")
    print("-" * 70)
    print("Sampling WITH replacement from both groups")
    print()

    case_reuse_counts = []
    ctrl_reuse_counts = []

    for b in range(n_bootstraps):
        selected_cases = rng.choice(case_samples, size=target_n_cases, replace=True)
        selected_ctrls = rng.choice(ctrl_samples, size=n_ctrls, replace=True)

        case_counter = Counter(selected_cases)
        ctrl_counter = Counter(selected_ctrls)

        # Track how many times samples appear multiple times
        case_reuses = sum(1 for count in case_counter.values() if count > 1)
        ctrl_reuses = sum(1 for count in ctrl_counter.values() if count > 1)

        case_reuse_counts.append(case_reuses)
        ctrl_reuse_counts.append(ctrl_reuses)

        print(f"Bootstrap {b+1}:")
        print(f"  Cases: {list(selected_cases)}")
        print(f"  Reused cases: {case_reuses} (samples appearing >1 time)")
        print(f"  Controls: {list(selected_ctrls)}")
        print(f"  Reused controls: {ctrl_reuses} (samples appearing >1 time)")
        print()

    print(f"Summary - True Bootstrap Mode:")
    print(f"  Average case reuse per bootstrap: {np.mean(case_reuse_counts):.1f}")
    print(f"  Average control reuse per bootstrap: {np.mean(ctrl_reuse_counts):.1f}")
    print(f"  → This captures uncertainty in BOTH populations")
    print()

    # Mode 2: SUBSAMPLING (replace=False for cases, all controls)
    print()
    print("MODE 2: SUBSAMPLING (bootstrap_controls=False)")
    print("-" * 70)
    print("Sampling WITHOUT replacement for cases, ALL controls every time")
    print()

    rng = np.random.default_rng(42)  # Reset for fair comparison

    ctrl_variety = []

    for b in range(n_bootstraps):
        selected_cases = rng.choice(case_samples, size=target_n_cases, replace=False)
        bootstrap_samples = list(selected_cases) + ctrl_samples

        # Count controls
        ctrl_in_sample = [s for s in bootstrap_samples if s.startswith('ctrl_')]
        ctrl_variety.append(len(set(ctrl_in_sample)))

        print(f"Bootstrap {b+1}:")
        print(f"  Cases: {selected_cases}")
        print(f"  Reused cases: 0 (no replacement)")
        print(f"  Controls: {ctrl_samples}")
        print(f"  Reused controls: 0 (but SAME {len(ctrl_samples)} controls every iteration)")
        print()

    print(f"Summary - Subsampling Mode:")
    print(f"  Case reuse: NEVER (sampled without replacement)")
    print(f"  Controls: IDENTICAL across all {n_bootstraps} bootstraps")
    print(f"  → Control uncertainty is NOT captured")
    print()

    # Statistical implications
    print()
    print("=" * 70)
    print("STATISTICAL IMPLICATIONS")
    print("=" * 70)
    print()
    print("TRUE BOOTSTRAP (bootstrap_controls=True):")
    print("  ✓ Captures sampling variability in both groups")
    print("  ✓ Confidence intervals reflect uncertainty in cases AND controls")
    print("  ✓ More conservative (wider CIs) but statistically proper")
    print("  ✓ Recommended when control uncertainty matters")
    print()
    print("SUBSAMPLING (bootstrap_controls=False):")
    print("  ✓ Cleaner interpretation (same controls, varying cases)")
    print("  ✓ Narrower CIs (less conservative)")
    print("  ✗ Underestimates uncertainty (assumes controls are fixed)")
    print("  ✓ Useful when controls are well-characterized/numerous")
    print()
    print("RECOMMENDATION:")
    print("  Use bootstrap_controls=True (default) for proper uncertainty")
    print("  quantification, especially when:")
    print("    - Control sample size is small (n < 50)")
    print("    - Control population is heterogeneous")
    print("    - You need conservative confidence intervals")
    print()


if __name__ == "__main__":
    demonstrate_bootstrap_modes()
