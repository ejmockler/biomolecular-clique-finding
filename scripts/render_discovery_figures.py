#!/usr/bin/env python3
"""Render the five recursive discovery narrative figures from existing JSON data.

Usage:
    python scripts/render_discovery_figures.py [--output DIR] [--format png|pdf|svg]

Data sources:
    output/validation/c9orf72_phase2/discovery_results.json  (primary: 5-hop discovery)
    output/validation/specificity_triangle/                   (3-contrast comparison)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for rendering

from cliquefinder.viz.validation import ValidationVisualizer


def load_discovery_data(base: Path) -> dict | None:
    """Load the primary discovery results (5-hop run)."""
    candidates = [
        base / "validation" / "c9orf72_phase2" / "discovery_results.json",
        base / "validation" / "c9orf72_complete" / "discovery_results.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            print(f"  Loaded discovery: {p} ({data.get('max_hops_reached', '?')} hops, "
                  f"{data.get('n_significant_pathways', '?')} significant)")
            return data
    return None


def load_specificity_contrasts(base: Path) -> dict[str, dict] | None:
    """Load the three-contrast specificity triangle data."""
    tri_dir = base / "validation" / "specificity_triangle"
    if not tri_dir.exists():
        return None

    contrasts = {}
    mapping = {
        "C9 vs Sporadic": "discovery_C9_vs_SPORADIC.json",
        "C9 vs Control": "discovery_C9_vs_CTRL.json",
        "Sporadic vs Control": "discovery_SPORADIC_vs_CTRL.json",
    }
    for name, filename in mapping.items():
        p = tri_dir / filename
        if p.exists():
            with open(p) as f:
                contrasts[name] = json.load(f)
            hops = contrasts[name].get("hops", [])
            n_sig = sum(h.get("n_significant", 0) for h in hops)
            print(f"  Loaded contrast: {name} ({len(hops)} hops, {n_sig} total significant)")

    return contrasts if contrasts else None


def main():
    parser = argparse.ArgumentParser(description="Render recursive discovery figures")
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/validation/discovery_figures"),
        help="Output directory for figures",
    )
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--style", default="paper", choices=["paper", "presentation", "notebook"],
    )
    args = parser.parse_args()

    base = Path("output")

    print("Loading data...")
    discovery = load_discovery_data(base)
    contrasts = load_specificity_contrasts(base)

    if discovery is None and contrasts is None:
        print("ERROR: No discovery data found. Run the validation pipeline first.")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    viz = ValidationVisualizer(style=args.style)

    figures_rendered = []

    # Figure 11: Cascade Staircase
    if discovery:
        print("\nRendering 11_cascade_staircase...")
        fig = viz.plot_cascade_staircase(discovery)
        path = fig.save(args.output / f"11_cascade_staircase.{args.format}", dpi=args.dpi)
        fig.close()
        figures_rendered.append(path)
        print(f"  → {path}")

    # Figure 12: Hop 2 Intermediaries
    if discovery:
        print("Rendering 12_hop2_intermediaries...")
        fig = viz.plot_hop2_intermediaries(discovery)
        path = fig.save(args.output / f"12_hop2_intermediaries.{args.format}", dpi=args.dpi)
        fig.close()
        figures_rendered.append(path)
        print(f"  → {path}")

    # Figure 13: π₀ Convergence
    if discovery:
        print("Rendering 13_pi0_convergence...")
        fig = viz.plot_pi0_convergence(discovery)
        path = fig.save(args.output / f"13_pi0_convergence.{args.format}", dpi=args.dpi)
        fig.close()
        figures_rendered.append(path)
        print(f"  → {path}")

    # Figure 14: Specificity Triangle
    if contrasts:
        print("Rendering 14_specificity_triangle...")
        fig = viz.plot_specificity_triangle(contrasts)
        path = fig.save(args.output / f"14_specificity_triangle.{args.format}", dpi=args.dpi)
        fig.close()
        figures_rendered.append(path)
        print(f"  → {path}")

    # Figure 15: Hop 2 Specificity Heatmap
    if contrasts:
        print("Rendering 15_hop2_specificity_heatmap...")
        fig = viz.plot_hop2_specificity_heatmap(contrasts)
        path = fig.save(args.output / f"15_hop2_specificity_heatmap.{args.format}", dpi=args.dpi)
        fig.close()
        figures_rendered.append(path)
        print(f"  → {path}")

    print(f"\nDone: {len(figures_rendered)} figures → {args.output}/")


if __name__ == "__main__":
    main()
