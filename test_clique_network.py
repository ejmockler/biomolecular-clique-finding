#!/usr/bin/env python
"""
Test script for plot_clique_network() method.

This script demonstrates the clique network visualization with MAPT regulator.
"""

import pandas as pd
from pathlib import Path
from cliquefinder.viz.cliques import CliqueVisualizer

def test_clique_network():
    """Test the clique network visualization."""

    # Load data
    df = pd.read_csv("results/cliques/cliques.csv")
    expr = pd.read_csv("results/proteomics_imputed.data.csv", index_col=0)

    # Create visualizer
    viz = CliqueVisualizer(style="paper")

    # Test 1: Network with correlations
    print("Creating MAPT clique network with correlations...")
    fig1 = viz.plot_clique_network(
        "MAPT",
        df,
        correlation_data=expr,
        correlation_threshold=0.4
    )

    # Save figure
    output_dir = Path("figures/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig1.save(output_dir / "mapt_network_with_correlations.pdf")
    fig1.save(output_dir / "mapt_network_with_correlations.png")
    print(f"  Saved to {output_dir}")
    print(f"  Metadata: {fig1.metadata}")

    # Test 2: Network without correlations (membership only)
    print("\nCreating MAPT clique network (membership only)...")
    fig2 = viz.plot_clique_network(
        "MAPT",
        df,
        correlation_data=None
    )
    fig2.save(output_dir / "mapt_network_membership_only.pdf")
    print(f"  Saved to {output_dir}")

    # Test 3: Condition-specific network
    print("\nCreating CASE_F condition network...")
    fig3 = viz.plot_clique_network(
        "MAPT",
        df,
        correlation_data=expr,
        condition_focus="CASE_F"
    )
    fig3.save(output_dir / "mapt_network_case_f.pdf")
    print(f"  Saved to {output_dir}")

    # Test 4: Different regulator (SLC2A1)
    print("\nCreating SLC2A1 clique network...")
    fig4 = viz.plot_clique_network(
        "SLC2A1",
        df,
        correlation_data=expr,
        correlation_threshold=0.3
    )
    fig4.save(output_dir / "slc2a1_network.pdf")
    print(f"  Saved to {output_dir}")
    print(f"  Metadata: {fig4.metadata}")

    print("\nAll tests completed successfully!")
    print(f"Figures saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    test_clique_network()
