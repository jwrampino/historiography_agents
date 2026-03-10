"""
Visualization tools for experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_experiment_overview(results_csv: Path, output_path: Optional[Path] = None):
    """Create overview visualization of all experiments."""
    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment Overview', fontsize=16, fontweight='bold')

    # 1. Turn count distribution
    axes[0, 0].hist(df['turn_count'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Turns')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Dialogue Length Distribution')
    axes[0, 0].axvline(df['turn_count'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["turn_count"].mean():.1f}')
    axes[0, 0].legend()

    # 2. Sources used
    axes[0, 1].hist(df['n_sources_used'], bins=15, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Number of Sources Used')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Source Usage Distribution')
    axes[0, 1].axvline(df['n_sources_used'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["n_sources_used"].mean():.1f}')
    axes[0, 1].legend()

    # 3. Consensus rate
    consensus_counts = df['consensus_reached'].value_counts()
    axes[1, 0].bar(['No Consensus', 'Consensus'],
                   [consensus_counts.get(False, 0), consensus_counts.get(True, 0)],
                   color=['red', 'green'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Number of Experiments')
    axes[1, 0].set_title('Consensus Achievement')
    axes[1, 0].text(0, consensus_counts.get(False, 0) + 1,
                    f"{consensus_counts.get(False, 0)}", ha='center')
    axes[1, 0].text(1, consensus_counts.get(True, 0) + 1,
                    f"{consensus_counts.get(True, 0)}", ha='center')

    # 4. Output length
    axes[1, 1].scatter(df['question_length'], df['abstract_length'], alpha=0.6)
    axes[1, 1].set_xlabel('Question Length (chars)')
    axes[1, 1].set_ylabel('Abstract Length (chars)')
    axes[1, 1].set_title('Output Lengths')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_persona_effects(results_csv: Path, output_path: Optional[Path] = None):
    """Visualize effects of different persona characteristics."""
    df = pd.read_csv(results_csv)

    # Get persona columns
    agent_cols = [c for c in df.columns if c.startswith('agent_')]

    # Focus on first agent for simplicity
    field_col = 'agent_0_field'
    method_col = 'agent_0_method'
    orientation_col = 'agent_0_orientation'

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Persona Characteristics vs Outcomes', fontsize=16, fontweight='bold')

    # 1. Field vs Sources Used
    if field_col in df.columns:
        field_sources = df.groupby(field_col)['n_sources_used'].mean().sort_values()
        axes[0, 0].barh(range(len(field_sources)), field_sources.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_yticks(range(len(field_sources)))
        axes[0, 0].set_yticklabels(field_sources.index)
        axes[0, 0].set_xlabel('Average Sources Used')
        axes[0, 0].set_title('Historical Field vs Source Usage')

    # 2. Method vs Turn Count
    if method_col in df.columns:
        method_turns = df.groupby(method_col)['turn_count'].mean().sort_values()
        axes[0, 1].barh(range(len(method_turns)), method_turns.values, color='lightcoral', edgecolor='black')
        axes[0, 1].set_yticks(range(len(method_turns)))
        axes[0, 1].set_yticklabels(method_turns.index)
        axes[0, 1].set_xlabel('Average Turn Count')
        axes[0, 1].set_title('Methodological Approach vs Dialogue Length')

    # 3. Orientation vs Consensus
    if orientation_col in df.columns:
        orientation_consensus = df.groupby(orientation_col)['consensus_reached'].mean().sort_values()
        axes[1, 0].barh(range(len(orientation_consensus)), orientation_consensus.values,
                       color='lightgreen', edgecolor='black')
        axes[1, 0].set_yticks(range(len(orientation_consensus)))
        axes[1, 0].set_yticklabels(orientation_consensus.index)
        axes[1, 0].set_xlabel('Consensus Rate')
        axes[1, 0].set_title('Theoretical Orientation vs Consensus')
        axes[1, 0].set_xlim([0, 1])

    # 4. Heatmap: Field vs Method
    if field_col in df.columns and method_col in df.columns:
        pivot = df.pivot_table(values='turn_count', index=field_col, columns=method_col, aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Avg Turns'})
        axes[1, 1].set_title('Field × Method Interaction')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_source_selection_patterns(output_dir: Path, output_path: Optional[Path] = None):
    """Visualize source selection patterns across experiments."""
    from utils import analyze_source_usage

    df = analyze_source_usage(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Source Selection Patterns', fontsize=16, fontweight='bold')

    # 1. Top sources
    top_sources = df.head(15)
    axes[0].barh(range(len(top_sources)), top_sources['count'].values, color='steelblue', edgecolor='black')
    axes[0].set_yticks(range(len(top_sources)))
    axes[0].set_yticklabels([t[:40] + '...' if len(t) > 40 else t for t in top_sources['title']], fontsize=8)
    axes[0].set_xlabel('Usage Count')
    axes[0].set_title('Most Frequently Used Sources')
    axes[0].invert_yaxis()

    # 2. Usage distribution
    axes[1].hist(df['count'], bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Number of Times Used')
    axes[1].set_ylabel('Number of Sources')
    axes[1].set_title('Source Usage Distribution')
    axes[1].set_yscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_causal_analysis(analysis_report: Path, output_path: Optional[Path] = None):
    """Visualize causal analysis results."""
    with open(analysis_report, 'r') as f:
        report = json.load(f)

    rq1 = report['rq1_source_selection']
    rq2 = report['rq2_optimal_configurations']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('Causal Analysis Results', fontsize=16, fontweight='bold')

    # RQ1: Significant effects by dimension
    ax1 = fig.add_subplot(gs[0, :])
    if rq1['n_significant'] > 0:
        sig_effects = pd.DataFrame(rq1['significant_effects'])
        dimension_counts = sig_effects['dimension'].value_counts()
        ax1.bar(dimension_counts.index, dimension_counts.values, color='purple', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Significant Effects')
        ax1.set_title(f'RQ1: Significant Causal Effects by Persona Dimension (p < 0.05)')
        ax1.set_xlabel('Persona Dimension')
    else:
        ax1.text(0.5, 0.5, 'No significant effects found', ha='center', va='center', fontsize=12)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

    # RQ2: Feature importance for each outcome
    row = 1
    for i, (outcome, data) in enumerate(rq2.items()):
        ax = fig.add_subplot(gs[row + (i // 2), i % 2])

        importance_df = pd.DataFrame(data['feature_importance']).head(10)
        ax.barh(range(len(importance_df)), importance_df['importance'].values,
               color='teal', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels([f[:20] for f in importance_df['feature']], fontsize=8)
        ax.set_xlabel('Importance')
        ax.set_title(f'{outcome}\n(R² = {data["r2_score"]:.3f})')
        ax.invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def create_all_visualizations(output_dir: Path = Path("outputs")):
    """Generate all visualizations and save to outputs/figures/."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Generating visualizations...")

    # 1. Experiment overview
    if (output_dir / "results.csv").exists():
        print("  - Experiment overview...")
        plot_experiment_overview(
            output_dir / "results.csv",
            figures_dir / "experiment_overview.png"
        )

        print("  - Persona effects...")
        plot_persona_effects(
            output_dir / "results.csv",
            figures_dir / "persona_effects.png"
        )

    # 2. Source patterns
    if len(list(output_dir.glob("exp_*.json"))) > 0:
        print("  - Source selection patterns...")
        plot_source_selection_patterns(
            output_dir,
            figures_dir / "source_patterns.png"
        )

    # 3. Causal analysis
    if (output_dir / "analysis_report.json").exists():
        print("  - Causal analysis...")
        plot_causal_analysis(
            output_dir / "analysis_report.json",
            figures_dir / "causal_analysis.png"
        )

    print(f"\nAll visualizations saved to {figures_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "overview":
            plot_experiment_overview(Path("outputs/results.csv"))

        elif command == "personas":
            plot_persona_effects(Path("outputs/results.csv"))

        elif command == "sources":
            plot_source_selection_patterns(Path("outputs"))

        elif command == "causal":
            plot_causal_analysis(Path("outputs/analysis_report.json"))

        elif command == "all":
            create_all_visualizations()

        else:
            print("Unknown command. Available commands:")
            print("  python visualize.py overview")
            print("  python visualize.py personas")
            print("  python visualize.py sources")
            print("  python visualize.py causal")
            print("  python visualize.py all")
    else:
        create_all_visualizations()
