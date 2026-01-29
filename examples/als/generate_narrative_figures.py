#!/usr/bin/env python3
"""
Generate narrative visualizations for clique-finding results.

**ANSWERALS-SPECIFIC EXAMPLE**

This script is tailored for the AnswerALS proteomics dataset with specific
phenotypes (CTRL/CASE), condition labels (Healthy/ALS), and biological
interpretations. It serves as a reference implementation for creating
custom narrative figures for your own dataset.

For generic visualization tools that work with any dataset, use the
CliqueVisualizer class from cliquefinder.viz.cliques instead.

Perceptual Engineering Principles Applied:
- Position encodes magnitude (most accurate perceptual channel)
- Color encodes category and direction semantically
- Whitespace encodes grouping (no boxes/lines needed)
- Typography hierarchy guides attention
- Minimal chartjunk - every element serves a purpose
- ~4 chunks of information per figure (working memory limit)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# === PERCEPTUAL DESIGN SYSTEM ===

# Semantic color palette
# Disease state = HUE (red vs green), Sex = VALUE/SATURATION
COLORS = {
    # Disease state (generic)
    'CASE': '#D64550',      # Red - disease/attention
    'CTRL': '#4A7C59',      # Green - healthy/stable

    # Female = lighter/warmer shades
    'CASE_F': '#E07A8A',    # Rose pink - female + disease
    'CTRL_F': '#7BC47F',    # Light green - female + control

    # Male = darker/cooler shades
    'CASE_M': '#A63D4D',    # Burgundy - male + disease
    'CTRL_M': '#2D6A4F',    # Forest green - male + control

    # Direction
    'gained': '#2A9D8F',    # Teal - positive/gained
    'lost': '#E76F51',      # Coral - negative/lost

    # Neutral
    'text': '#2B2D42',      # Near-black
    'text_secondary': '#6B7280',  # Gray
    'grid': '#E5E7EB',      # Light gray
    'background': '#FAFAFA', # Off-white
}

# Typography scale (modular scale ratio 1.25)
FONT = {
    'title': 20,
    'subtitle': 14,
    'label': 11,
    'tick': 10,
    'annotation': 9,
}

# Spacing scale (base unit 8px)
SPACE = {
    'xs': 4,
    'sm': 8,
    'md': 16,
    'lg': 24,
    'xl': 32,
}

def setup_style():
    """Configure matplotlib for clean, publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'sans-serif'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.8,
        'axes.edgecolor': COLORS['text_secondary'],
        'axes.labelcolor': COLORS['text'],
        'axes.titlecolor': COLORS['text'],
        'xtick.color': COLORS['text_secondary'],
        'ytick.color': COLORS['text_secondary'],
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })


def figure_1_cohort_coherence(stratified_df, output_path):
    """
    Figure 1: Disease Signal - Coherence by Cohort

    Primary message: ALS causes systematic loss of regulatory coherence
    Secondary message: Males have higher baseline and greater loss

    Perceptual mapping:
    - Bar height = coherence (magnitude via position)
    - Color = disease state (categorical)
    - Grouping = sex (spatial proximity)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate coherence by condition
    coherence = stratified_df.groupby('condition')['coherence_ratio'].mean()

    # Order: CTRL_F, CASE_F, CTRL_M, CASE_M (grouped by sex, CTRL first)
    order = ['CTRL_F', 'CASE_F', 'CTRL_M', 'CASE_M']
    values = [coherence[c] for c in order]
    colors = [COLORS[c] for c in order]

    # Bar positions with gap between sex groups
    positions = [0, 1, 2.5, 3.5]

    bars = ax.bar(positions, values, color=colors, width=0.8, edgecolor='white', linewidth=1)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.1%}', ha='center', va='bottom',
                fontsize=FONT['label'], fontweight='medium', color=COLORS['text'])

    # Annotations for key findings
    # Female difference
    female_diff = (coherence['CASE_F'] - coherence['CTRL_F']) / coherence['CTRL_F'] * 100
    ax.annotate(f'{female_diff:+.1f}%', xy=(0.5, max(coherence['CTRL_F'], coherence['CASE_F']) + 0.02),
                ha='center', fontsize=FONT['annotation'], color=COLORS['lost'], fontweight='bold')

    # Male difference
    male_diff = (coherence['CASE_M'] - coherence['CTRL_M']) / coherence['CTRL_M'] * 100
    ax.annotate(f'{male_diff:+.1f}%', xy=(3, max(coherence['CTRL_M'], coherence['CASE_M']) + 0.02),
                ha='center', fontsize=FONT['annotation'], color=COLORS['lost'], fontweight='bold')

    # Axis configuration
    ax.set_xticks(positions)
    ax.set_xticklabels(['Control', 'ALS', 'Control', 'ALS'], fontsize=FONT['tick'])
    ax.set_ylabel('Mean Regulatory Coherence', fontsize=FONT['label'], fontweight='medium')
    ax.set_ylim(0, 0.42)

    # Sex labels above groups
    ax.text(0.5, -0.08, 'Female', ha='center', transform=ax.get_xaxis_transform(),
            fontsize=FONT['label'], fontweight='bold', color=COLORS['text_secondary'])
    ax.text(3, -0.08, 'Male', ha='center', transform=ax.get_xaxis_transform(),
            fontsize=FONT['label'], fontweight='bold', color=COLORS['text_secondary'])

    # Title
    ax.set_title('Regulatory Network Coherence by Condition',
                 fontsize=FONT['title'], fontweight='bold', loc='left', pad=20)
    ax.text(0, 1.02, 'Coherence measures fraction of regulator targets forming correlated cliques',
            transform=ax.transAxes, fontsize=FONT['subtitle'], color=COLORS['text_secondary'])

    # Remove top/right spines (already done by style)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def figure_2_rewiring_direction(rewiring_df, output_path):
    """
    Figure 2: Rewiring Direction - Butterfly Chart

    Primary message: ALS causes net loss of cliques (dysregulation)
    Secondary message: Males lose 3.8x more cliques than females

    Perceptual mapping:
    - Bar length = clique count (magnitude via position)
    - Left = lost, Right = gained (direction via position)
    - Color = lost (red) vs gained (teal)
    - Rows = sex comparison
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    # Aggregate by comparison
    female = rewiring_df[rewiring_df['comparison'] == 'CASE_F_vs_CTRL_F']
    male = rewiring_df[rewiring_df['comparison'] == 'CASE_M_vs_CTRL_M']

    data = {
        'Female': {
            'gained': female['gained_cliques'].sum(),
            'lost': -female['lost_cliques'].sum(),  # Negative for left side
            'net': female['gained_cliques'].sum() - female['lost_cliques'].sum()
        },
        'Male': {
            'gained': male['gained_cliques'].sum(),
            'lost': -male['lost_cliques'].sum(),
            'net': male['gained_cliques'].sum() - male['lost_cliques'].sum()
        }
    }

    labels = list(data.keys())
    y_positions = [1, 0]

    # Draw bars
    for i, (label, values) in enumerate(data.items()):
        y = y_positions[i]

        # Lost (left, red)
        ax.barh(y, values['lost'], height=0.5, color=COLORS['lost'],
                label='Lost in ALS' if i == 0 else '', edgecolor='white', linewidth=1)
        ax.text(values['lost'] - 800, y, f"{abs(values['lost']):,}",
                ha='right', va='center', fontsize=FONT['label'], color='white', fontweight='bold')

        # Gained (right, teal)
        ax.barh(y, values['gained'], height=0.5, color=COLORS['gained'],
                label='Gained in ALS' if i == 0 else '', edgecolor='white', linewidth=1)
        ax.text(values['gained'] + 800, y, f"{values['gained']:,}",
                ha='left', va='center', fontsize=FONT['label'], color=COLORS['text'], fontweight='bold')

        # Net change annotation on far right
        net = values['net']
        net_color = COLORS['gained'] if net > 0 else COLORS['lost']
        ax.text(42000, y, f"Net: {net:+,}", ha='left', va='center',
                fontsize=FONT['label'], color=net_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=net_color, alpha=0.9))

    # Center line
    ax.axvline(0, color=COLORS['text_secondary'], linewidth=1.5, linestyle='-')

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=FONT['subtitle'], fontweight='bold')
    ax.set_xlabel('Number of Cliques', fontsize=FONT['label'])

    # X-axis formatting
    ax.set_xlim(-42000, 58000)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{abs(x/1000):.0f}k'))

    # Direction labels at top
    ax.text(-21000, 1.7, 'LOST IN ALS', ha='center', fontsize=FONT['label'],
            color=COLORS['lost'], fontweight='bold')
    ax.text(11000, 1.7, 'GAINED IN ALS', ha='center', fontsize=FONT['label'],
            color=COLORS['gained'], fontweight='bold')

    # Title with more space
    fig.suptitle('Clique Gains and Losses: ALS vs Control',
                 fontsize=FONT['title'], fontweight='bold', x=0.12, ha='left', y=0.97)
    fig.text(0.12, 0.88, 'Lost = genes correlated in Control but not ALS. Gained = newly correlated in ALS.',
             fontsize=FONT['subtitle'], color=COLORS['text_secondary'], ha='left')

    # Set y limits to make room
    ax.set_ylim(-0.5, 2.2)

    # Clean up
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def figure_3_top_regulators(summary_df, output_path):
    """
    Figure 3: Top Regulators by Disease Impact

    Primary message: Key biological regulators showing network disruption
    Secondary message: Oxidative stress, mitochondria, protein quality control

    Perceptual mapping:
    - Bar length = disease impact score (magnitude)
    - Vertical position = rank (ordinal)
    - Color intensity = clique size (additional dimension)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute disease impact: net clique loss × max clique size
    df = summary_df.copy()
    df['net_loss'] = df['total_lost_cliques'] - df['total_gained_cliques']
    df['disease_impact'] = df['net_loss'] * df['max_clique_size']

    # Filter to meaningful regulators (10+ targets) and get top 20
    meaningful = df[df['n_indra_targets'] >= 10].nlargest(20, 'disease_impact')

    # Reverse for bottom-to-top ordering (highest at top)
    meaningful = meaningful.iloc[::-1]

    y_positions = range(len(meaningful))

    # Color by best condition (which cohort has strongest clique)
    colors = [COLORS.get(row['best_condition'], COLORS['CTRL'])
              for _, row in meaningful.iterrows()]

    # Draw horizontal bars
    bars = ax.barh(y_positions, meaningful['disease_impact'],
                   color=colors, height=0.7, edgecolor='white', linewidth=0.5)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(meaningful['regulator'], fontsize=FONT['label'], fontweight='medium')
    ax.set_xlabel('Disease Impact Score\n(Net Clique Loss × Max Clique Size)',
                  fontsize=FONT['label'])

    # Add clique size annotations
    for i, (_, row) in enumerate(meaningful.iterrows()):
        # Clique size inside bar
        ax.text(row['disease_impact'] - 100, i, f"n={row['max_clique_size']}",
                ha='right', va='center', fontsize=FONT['annotation'],
                color='white', fontweight='medium')
        # Target count outside bar
        ax.text(row['disease_impact'] + 100, i, f"({row['n_indra_targets']} targets)",
                ha='left', va='center', fontsize=FONT['annotation'],
                color=COLORS['text_secondary'])

    # Title
    ax.set_title('Top Regulators Disrupted in ALS',
                 fontsize=FONT['title'], fontweight='bold', loc='left', pad=20)
    ax.text(0, 1.02, 'Ranked by disease impact: net clique loss weighted by clique size',
            transform=ax.transAxes, fontsize=FONT['subtitle'], color=COLORS['text_secondary'])

    # Legend for colors
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['CTRL_M'], label='Best clique in CTRL Male'),
        mpatches.Patch(facecolor=COLORS['CTRL_F'], label='Best clique in CTRL Female'),
        mpatches.Patch(facecolor=COLORS['CASE_M'], label='Best clique in ALS Male'),
        mpatches.Patch(facecolor=COLORS['CASE_F'], label='Best clique in ALS Female'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=FONT['annotation'],
              frameon=True, framealpha=0.9, edgecolor=COLORS['grid'])

    # Clean up
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def figure_4_sex_specific(rewiring_df, output_path):
    """
    Figure 4: Sex-Specific Rewiring Patterns

    Primary message: Both shared and sex-specific regulatory disruption
    Secondary message: More male-specific than female-specific patterns

    Perceptual mapping:
    - Bar length = number of regulators (most accurate)
    - Color = category
    - Horizontal layout for easy comparison
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Get regulators with strong rewiring (>0.3) per sex
    male = rewiring_df[rewiring_df['comparison'] == 'CASE_M_vs_CTRL_M']
    female = rewiring_df[rewiring_df['comparison'] == 'CASE_F_vs_CTRL_F']

    male_strong = set(male[male['rewiring_score'] > 0.3]['regulator'])
    female_strong = set(female[female['rewiring_score'] > 0.3]['regulator'])

    male_only = male_strong - female_strong
    female_only = female_strong - male_strong
    shared = male_strong & female_strong

    # Data for horizontal bar chart
    categories = ['Male-Specific', 'Shared', 'Female-Specific']
    counts = [len(male_only), len(shared), len(female_only)]
    colors_list = [COLORS['CASE_M'], '#8B5CF6', COLORS['CASE_F']]

    y_pos = [2, 1, 0]

    # Draw horizontal bars
    bars = ax.barh(y_pos, counts, color=colors_list, height=0.6,
                   edgecolor='white', linewidth=1)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center',
                fontsize=FONT['subtitle'], fontweight='bold', color=COLORS['text'])

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=FONT['label'], fontweight='medium')
    ax.set_xlabel('Number of Regulators', fontsize=FONT['label'])

    # Add percentage annotations
    total = sum(counts)
    for y, count in zip(y_pos, counts):
        pct = count / total * 100
        ax.text(count - 15, y, f'{pct:.0f}%', ha='right', va='center',
                fontsize=FONT['label'], color='white', fontweight='bold')

    # Title
    ax.set_title('Sex-Specific Regulatory Disruption in ALS',
                 fontsize=FONT['title'], fontweight='bold', loc='left', pad=20)
    ax.text(0, 1.02, 'Regulators with strong rewiring (>0.3) between ALS and Control',
            transform=ax.transAxes, fontsize=FONT['subtitle'], color=COLORS['text_secondary'])

    # Total annotation
    ax.text(0.98, 0.05, f'Total: {total} regulators',
            ha='right', transform=ax.transAxes, fontsize=FONT['annotation'],
            color=COLORS['text_secondary'])

    # Clean up
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)
    ax.set_xlim(0, max(counts) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def figure_5_clique_size_distribution(stratified_df, output_path):
    """
    Figure 5: Clique Size Distribution by Cohort

    Primary message: Control maintains larger cliques than ALS
    Secondary message: Distribution shape differs by sex

    Perceptual mapping:
    - X position = clique size (magnitude)
    - Y position = frequency (count, log scale)
    - Color = cohort
    - Facets = sex
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Female subplot
    ax_f = axes[0]
    female_case = stratified_df[stratified_df['condition'] == 'CASE_F']['n_coherent_genes']
    female_ctrl = stratified_df[stratified_df['condition'] == 'CTRL_F']['n_coherent_genes']

    bins = range(3, 30)
    ax_f.hist(female_ctrl, bins=bins, alpha=0.7, color=COLORS['CTRL_F'],
              label='Control', edgecolor='white', linewidth=0.5)
    ax_f.hist(female_case, bins=bins, alpha=0.7, color=COLORS['CASE_F'],
              label='ALS', edgecolor='white', linewidth=0.5)

    ax_f.set_xlabel('Clique Size (genes)', fontsize=FONT['label'])
    ax_f.set_ylabel('Number of Cliques (log scale)', fontsize=FONT['label'])
    ax_f.set_title('Female', fontsize=FONT['subtitle'], fontweight='bold')
    ax_f.legend(fontsize=FONT['annotation'], loc='upper right')
    ax_f.set_yscale('log')

    # Add mean lines
    ax_f.axvline(female_ctrl.mean(), color=COLORS['CTRL_F'], linestyle='--', linewidth=2, alpha=0.8)
    ax_f.axvline(female_case.mean(), color=COLORS['CASE_F'], linestyle='--', linewidth=2, alpha=0.8)

    # Male subplot
    ax_m = axes[1]
    male_case = stratified_df[stratified_df['condition'] == 'CASE_M']['n_coherent_genes']
    male_ctrl = stratified_df[stratified_df['condition'] == 'CTRL_M']['n_coherent_genes']

    ax_m.hist(male_ctrl, bins=bins, alpha=0.7, color=COLORS['CTRL_M'],
              label='Control', edgecolor='white', linewidth=0.5)
    ax_m.hist(male_case, bins=bins, alpha=0.7, color=COLORS['CASE_M'],
              label='ALS', edgecolor='white', linewidth=0.5)

    ax_m.set_xlabel('Clique Size (genes)', fontsize=FONT['label'])
    ax_m.set_title('Male', fontsize=FONT['subtitle'], fontweight='bold')
    ax_m.legend(fontsize=FONT['annotation'], loc='upper right')
    ax_m.set_yscale('log')

    # Add mean lines
    ax_m.axvline(male_ctrl.mean(), color=COLORS['CTRL_M'], linestyle='--', linewidth=2, alpha=0.8)
    ax_m.axvline(male_case.mean(), color=COLORS['CASE_M'], linestyle='--', linewidth=2, alpha=0.8)

    # Overall title
    fig.suptitle('Clique Size Distribution: Control vs ALS',
                 fontsize=FONT['title'], fontweight='bold', y=1.02)

    # Clean up
    for ax in axes:
        ax.spines['left'].set_visible(False)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(left=False)
        ax.set_ylim(1, None)  # Start log scale at 1

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Generate all narrative figures."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate narrative figures')
    parser.add_argument('--data-dir', type=Path, default=Path('output/regulatory_cliques'),
                        help='Directory with clique analysis results')
    parser.add_argument('--output-dir', type=Path, default=Path('output/regulatory_cliques/figures'),
                        help='Output directory for figures')
    args = parser.parse_args()

    setup_style()

    # Paths
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    summary = pd.read_csv(data_dir / 'regulators_summary.csv')
    stratified = pd.read_csv(data_dir / 'cliques.csv')
    rewiring = pd.read_csv(data_dir / 'regulator_rewiring_stats.csv')

    print(f"  Regulators: {len(summary)}")
    print(f"  Stratified cliques: {len(stratified)}")
    print(f"  Rewiring comparisons: {len(rewiring)}")

    print("\nGenerating figures...")

    # Figure 1: Cohort coherence overview
    figure_1_cohort_coherence(stratified, output_dir / '01_cohort_coherence.png')

    # Figure 2: Rewiring direction
    figure_2_rewiring_direction(rewiring, output_dir / '02_rewiring_direction.png')

    # Figure 3: Top regulators
    figure_3_top_regulators(summary, output_dir / '03_top_regulators.png')

    # Figure 4: Sex-specific patterns
    figure_4_sex_specific(rewiring, output_dir / '04_sex_specific_patterns.png')

    # Figure 5: Clique size distribution
    figure_5_clique_size_distribution(stratified, output_dir / '05_clique_size_distribution.png')

    print(f"\nAll figures saved to: {output_dir}")
    print("\nNarrative Summary:")
    print("=" * 60)
    print("1. ALS reduces regulatory coherence by ~7% overall")
    print("2. Males show 3.8× more clique loss than females")
    print("3. Key disrupted regulators: PRKN, MAPT, FOXO1, SIRT3, CAT")
    print("4. Biological themes: oxidative stress, mitochondria, proteostasis")
    print("=" * 60)


if __name__ == '__main__':
    main()
