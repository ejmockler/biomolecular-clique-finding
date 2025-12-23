"""
Consistent visual styles for biomolecular visualizations.

This module defines color palettes, typography, and matplotlib/seaborn
configuration following domain conventions and perceptual engineering principles.

Domain Conventions
------------------
- CASE = Blue (#2563eb), CTRL = Orange (#f97316)
- Male = Teal (#0d9488), Female = Violet (#7c3aed) [gender-neutral colors]
- Outliers = Orange (#fb923c), Imputed = Red (#ef4444)
- Correlations = RdBu_r diverging colormap
- Gene symbols italicized (use $gene$ in matplotlib)
- All colorblind-safe palettes

Perceptual Principles
---------------------
- High contrast for key comparisons (CASE vs CTRL)
- Semantic color usage (red = warning/removed, blue = normal/kept)
- Consistent across all visualizations (builds spatial memory)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


@dataclass(frozen=True)
class Palette:
    """
    Color palette for biomolecular visualizations.

    Attributes
    ----------
    case : str
        Color for CASE (ALS) samples
    ctrl : str
        Color for CTRL (healthy) samples
    male : str
        Color for male samples
    female : str
        Color for female samples
    outlier : str
        Color for outlier values
    imputed : str
        Color for imputed values
    missing : str
        Color for missing data
    highlight : str
        Color for highlighted elements
    neutral : str
        Color for non-differentiated elements
    diverging : str
        Colormap name for diverging data (correlations)
    sequential : str
        Colormap name for sequential data (counts, magnitudes)
    """
    case: str = "#2563eb"        # Blue-600
    ctrl: str = "#f97316"        # Orange-500
    male: str = "#0d9488"        # Teal-600 (gender-neutral)
    female: str = "#7c3aed"      # Violet-600 (gender-neutral)
    outlier: str = "#fb923c"     # Orange-400
    imputed: str = "#ef4444"     # Red-500
    missing: str = "#9ca3af"     # Gray-400
    highlight: str = "#059669"   # Emerald-600 (visible on white)
    neutral: str = "#6b7280"     # Gray-500
    diverging: str = "RdBu_r"    # Red-Blue reversed (blue=positive)
    sequential: str = "viridis"  # Perceptually uniform

    @property
    def phenotype(self) -> dict[str, str]:
        """Color mapping for phenotype values."""
        return {"CASE": self.case, "CTRL": self.ctrl, "ALS": self.case, "Healthy Control": self.ctrl}

    @property
    def sex(self) -> dict[str, str]:
        """Color mapping for sex values."""
        return {"M": self.male, "F": self.female, "Male": self.male, "Female": self.female}

    @property
    def quality(self) -> dict[str, str]:
        """Color mapping for quality flags."""
        return {"outlier": self.outlier, "imputed": self.imputed, "missing": self.missing, "clean": self.neutral}

    def for_groups(self, groups: list[str]) -> list[str]:
        """
        Get colors for a list of groups.

        Automatically maps phenotype and sex labels to appropriate colors.
        Falls back to a categorical palette for unknown groups.
        """
        all_mappings = {**self.phenotype, **self.sex, **self.quality}
        colors = []
        fallback_idx = 0
        fallback_colors = sns.color_palette("Set2", 8).as_hex()

        for group in groups:
            if group in all_mappings:
                colors.append(all_mappings[group])
            else:
                colors.append(fallback_colors[fallback_idx % len(fallback_colors)])
                fallback_idx += 1

        return colors


# Predefined palettes
PALETTES = {
    "default": Palette(),
    "colorblind": Palette(
        case="#0077bb",      # Blue
        ctrl="#ee7733",      # Orange
        male="#009988",      # Teal (gender-neutral)
        female="#aa3377",    # Purple (gender-neutral)
        outlier="#ee7733",
        imputed="#cc3311",   # Red
        missing="#bbbbbb",
        highlight="#009988", # Teal
        neutral="#999999",
    ),
    "print": Palette(
        case="#1a1a1a",      # Near-black
        ctrl="#666666",      # Dark gray
        male="#333333",
        female="#4d4d4d",
        outlier="#999999",
        imputed="#b3b3b3",
        missing="#e6e6e6",
        highlight="#000000",
        neutral="#808080",
        diverging="RdGy",    # Red-Gray for B&W printing
        sequential="Greys",
    ),
}


def configure_style(
    style: Literal["paper", "presentation", "notebook"] = "paper",
    palette: str | Palette = "default",
    font_scale: float = 1.0
) -> Palette:
    """
    Configure matplotlib and seaborn for consistent visualization style.

    Parameters
    ----------
    style : {"paper", "presentation", "notebook"}
        Target medium:
        - paper: High DPI, publication-quality, minimal decoration
        - presentation: Large fonts, bold colors, high contrast
        - notebook: Interactive-friendly, moderate sizes
    palette : str or Palette
        Color palette name or Palette instance.
    font_scale : float
        Multiplier for all font sizes.

    Returns
    -------
    Palette
        The configured color palette.

    Examples
    --------
    >>> from cliquefinder.viz.styles import configure_style
    >>> palette = configure_style("paper")
    >>> # Now all matplotlib/seaborn plots use paper style
    """
    # Get palette
    if isinstance(palette, str):
        palette = PALETTES.get(palette, PALETTES["default"])

    # Base style parameters
    base_params = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "text.color": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.autolayout": True,
    }

    # Style-specific parameters
    if style == "paper":
        style_params = {
            "font.size": 10 * font_scale,
            "axes.titlesize": 11 * font_scale,
            "axes.labelsize": 10 * font_scale,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "legend.fontsize": 9 * font_scale,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.8,
        }
        context = "paper"
    elif style == "presentation":
        style_params = {
            "font.size": 14 * font_scale,
            "axes.titlesize": 18 * font_scale,
            "axes.labelsize": 14 * font_scale,
            "xtick.labelsize": 12 * font_scale,
            "ytick.labelsize": 12 * font_scale,
            "legend.fontsize": 12 * font_scale,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "lines.linewidth": 2.0,
            "axes.linewidth": 1.5,
        }
        context = "talk"
    else:  # notebook
        style_params = {
            "font.size": 11 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "axes.labelsize": 11 * font_scale,
            "xtick.labelsize": 10 * font_scale,
            "ytick.labelsize": 10 * font_scale,
            "legend.fontsize": 10 * font_scale,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "lines.linewidth": 1.5,
            "axes.linewidth": 1.0,
        }
        context = "notebook"

    # Apply seaborn style
    sns.set_theme(style="whitegrid", context=context, font_scale=font_scale)

    # Override with our parameters
    all_params = {**base_params, **style_params}
    plt.rcParams.update(all_params)

    return palette


def italicize_gene(gene: str) -> str:
    """
    Format gene symbol for matplotlib (italicized per biology convention).

    Parameters
    ----------
    gene : str
        Gene symbol (e.g., "TP53", "SOD1")

    Returns
    -------
    str
        Matplotlib-formatted string with italics.

    Examples
    --------
    >>> italicize_gene("SOD1")
    '$\\\\mathit{SOD1}$'
    """
    return f"$\\mathit{{{gene}}}$"


def format_pvalue(p: float) -> str:
    """
    Format p-value with appropriate precision and notation.

    Parameters
    ----------
    p : float
        P-value

    Returns
    -------
    str
        Formatted string (e.g., "p < 0.001", "p = 0.034")
    """
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.2f}"
    else:
        return f"p = {p:.2f}"


def significance_stars(p: float) -> str:
    """
    Convert p-value to significance stars.

    Parameters
    ----------
    p : float
        P-value

    Returns
    -------
    str
        Significance notation (*** / ** / * / ns)
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# Convenience function for stratification colors
def get_stratum_colors(
    strata: list[str],
    palette: Optional[Palette] = None
) -> dict[str, str]:
    """
    Get color mapping for phenotype×sex strata.

    Parameters
    ----------
    strata : list[str]
        Stratum labels (e.g., ["CASE_M", "CASE_F", "CTRL_M", "CTRL_F"])
    palette : Palette, optional
        Color palette. If None, uses default.

    Returns
    -------
    dict[str, str]
        Stratum -> color mapping

    Examples
    --------
    >>> colors = get_stratum_colors(["CASE_M", "CASE_F", "CTRL_M", "CTRL_F"])
    >>> colors["CASE_M"]
    '#2563eb'  # Blue with full saturation
    """
    if palette is None:
        palette = PALETTES["default"]

    # Define stratum color logic
    colors = {}
    for stratum in strata:
        stratum_upper = stratum.upper()

        # Determine base color from phenotype
        if "CASE" in stratum_upper or "ALS" in stratum_upper:
            base = palette.case
        elif "CTRL" in stratum_upper or "HEALTHY" in stratum_upper:
            base = palette.ctrl
        else:
            base = palette.neutral

        # Adjust for sex (lighter for female)
        if "_F" in stratum_upper or "FEMALE" in stratum_upper:
            # Lighten the color
            base = _lighten_hex(base, 0.3)

        colors[stratum] = base

    return colors


def _lighten_hex(hex_color: str, factor: float) -> str:
    """Lighten a hex color by mixing with white."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)

    return f"#{r:02x}{g:02x}{b:02x}"
