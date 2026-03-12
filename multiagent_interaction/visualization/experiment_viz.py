"""
Experiment Visualizations: Overview of experimental results and outcomes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_experiment_overview(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Comprehensive overview of all experiments."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Experiment Overview Dashboard', fontsize=16, fontweight='bold')

    # 1. Turn count distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results_df['turn_count'], bins=25, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(results_df['turn_count'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {results_df["turn_count"].mean():.1f}')
    ax1.axvline(results_df['turn_count'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {results_df["turn_count"].median():.1f}')
    ax1.set_xlabel('Number of Turns')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Dialogue Length Distribution')
    ax1.legend(fontsize=8)

    # 2. Sources used
    ax2 = fig.add_subplot(gs[0, 1])
    if 'n_sources_used' in results_df.columns:
        ax2.hist(results_df['n_sources_used'], bins=20, edgecolor='black',
                alpha=0.7, color='coral')
        ax2.axvline(results_df['n_sources_used'].mean(), color='red',
                   linestyle='--', linewidth=2,
                   label=f'Mean: {results_df["n_sources_used"].mean():.1f}')
        ax2.set_xlabel('Number of Sources')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Source Usage Distribution')
        ax2.legend(fontsize=8)

    # 3. Consensus reached
    ax3 = fig.add_subplot(gs[0, 2])
    if 'consensus_reached' in results_df.columns:
        consensus_counts = results_df['consensus_reached'].value_counts()
        colors = ['red', 'green']
        labels = ['No Consensus', 'Consensus']
        ax3.pie([consensus_counts.get(False, 0), consensus_counts.get(True, 0)],
               labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Consensus Achievement Rate')

    # 4. Abstract length
    ax4 = fig.add_subplot(gs[1, 0])
    if 'abstract_length' in results_df.columns:
        ax4.hist(results_df['abstract_length'], bins=25, edgecolor='black',
                alpha=0.7, color='lightgreen')
        ax4.axvline(results_df['abstract_length'].mean(), color='red',
                   linestyle='--', linewidth=2,
                   label=f'Mean: {results_df["abstract_length"].mean():.0f}')
        ax4.set_xlabel('Abstract Length (words)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Abstract Length Distribution')
        ax4.legend(fontsize=8)

    # 5. Turns vs Sources
    ax5 = fig.add_subplot(gs[1, 1])
    if 'n_sources_used' in results_df.columns:
        scatter = ax5.scatter(results_df['turn_count'], results_df['n_sources_used'],
                             alpha=0.6, s=50, c=results_df.index, cmap='viridis')
        ax5.set_xlabel('Turn Count')
        ax5.set_ylabel('Sources Used')
        ax5.set_title('Turns vs Sources Used')

        # Add trend line
        z = np.polyfit(results_df['turn_count'], results_df['n_sources_used'], 1)
        p = np.poly1d(z)
        ax5.plot(results_df['turn_count'], p(results_df['turn_count']),
                "r--", alpha=0.8, linewidth=2, label='Trend')
        ax5.legend()

    # 6. Triangle geometry overview
    ax6 = fig.add_subplot(gs[1, 2])
    if 'geom_perimeter' in results_df.columns and 'geom_area' in results_df.columns:
        scatter = ax6.scatter(results_df['geom_perimeter'], results_df['geom_area'],
                             alpha=0.6, s=50, c=results_df['turn_count'], cmap='plasma')
        ax6.set_xlabel('Triangle Perimeter')
        ax6.set_ylabel('Triangle Area')
        ax6.set_title('Triangle Geometry\n(colored by turns)')
        plt.colorbar(scatter, ax=ax6, label='Turns')

    # 7. Perimeter distribution
    ax7 = fig.add_subplot(gs[2, 0])
    if 'geom_perimeter' in results_df.columns:
        ax7.hist(results_df['geom_perimeter'], bins=25, edgecolor='black',
                alpha=0.7, color='mediumpurple')
        ax7.axvline(results_df['geom_perimeter'].mean(), color='red',
                   linestyle='--', linewidth=2,
                   label=f'Mean: {results_df["geom_perimeter"].mean():.3f}')
        ax7.set_xlabel('Perimeter')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Triangle Perimeter Distribution')
        ax7.legend(fontsize=8)

    # 8. Area distribution
    ax8 = fig.add_subplot(gs[2, 1])
    if 'geom_area' in results_df.columns:
        ax8.hist(results_df['geom_area'], bins=25, edgecolor='black',
                alpha=0.7, color='gold')
        ax8.axvline(results_df['geom_area'].mean(), color='red',
                   linestyle='--', linewidth=2,
                   label=f'Mean: {results_df["geom_area"].mean():.4f}')
        ax8.set_xlabel('Area')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Triangle Area Distribution')
        ax8.legend(fontsize=8)

    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    stats_text = f"""
    SUMMARY STATISTICS

    Total Experiments: {len(results_df)}

    Turns:
      Mean: {results_df['turn_count'].mean():.1f}
      Std: {results_df['turn_count'].std():.1f}
    """

    if 'n_sources_used' in results_df.columns:
        stats_text += f"""
    Sources:
      Mean: {results_df['n_sources_used'].mean():.1f}
      Std: {results_df['n_sources_used'].std():.1f}
    """

    if 'consensus_reached' in results_df.columns:
        consensus_rate = results_df['consensus_reached'].mean()
        stats_text += f"""
    Consensus Rate: {consensus_rate:.1%}
    """

    ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_historian_participation(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Analyze which historians participated most frequently."""
    # Extract historian columns
    historian_cols = [c for c in results_df.columns if c.startswith('historian_')]

    if len(historian_cols) == 0:
        print("No historian columns found")
        return

    # Count participations
    all_historians = []
    for col in historian_cols:
        all_historians.extend(results_df[col].dropna().tolist())

    historian_counts = pd.Series(all_historians).value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Historian Participation Analysis', fontsize=14, fontweight='bold')

    # 1. Top historians
    top_n = min(20, len(historian_counts))
    top_historians = historian_counts.head(top_n)

    axes[0].barh(range(len(top_historians)), top_historians.values,
                color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(top_historians)))
    axes[0].set_yticklabels([name.split()[-1] for name in top_historians.index],
                           fontsize=9)  # Last names
    axes[0].set_xlabel('Number of Experiments')
    axes[0].set_title(f'Top {top_n} Most Frequent Historians')
    axes[0].invert_yaxis()

    # 2. Participation distribution
    axes[1].hist(historian_counts.values, bins=20, edgecolor='black',
                alpha=0.7, color='coral')
    axes[1].set_xlabel('Number of Experiments')
    axes[1].set_ylabel('Number of Historians')
    axes[1].set_title('Distribution of Participation Frequency')
    axes[1].axvline(historian_counts.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {historian_counts.mean():.1f}')
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_outcome_distributions(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot distributions of key outcome variables."""
    outcome_cols = {
        'turn_count': ('Dialogue Length', 'Turn Count', 'skyblue'),
        'n_sources_used': ('Source Usage', 'Number of Sources', 'coral'),
        'abstract_length': ('Abstract Length', 'Word Count', 'lightgreen'),
        'question_length': ('Question Length', 'Character Count', 'gold')
    }

    # Filter to existing columns
    existing_outcomes = {k: v for k, v in outcome_cols.items() if k in results_df.columns}

    n_outcomes = len(existing_outcomes)
    if n_outcomes == 0:
        print("No outcome columns found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Outcome Variable Distributions', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for idx, (col, (title, xlabel, color)) in enumerate(existing_outcomes.items()):
        if idx >= 4:
            break

        ax = axes[idx]
        data = results_df[col].dropna()

        # Histogram
        n, bins, patches = ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color=color)

        # Statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.1f}')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend(fontsize=8)

        # Add text box with stats
        stats_text = f'μ={mean_val:.1f}\nσ={std_val:.1f}\nn={len(data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused subplots
    for idx in range(n_outcomes, 4):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_pairwise_outcomes(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Pairplot of outcome variables."""
    outcome_cols = ['turn_count', 'n_sources_used', 'abstract_length']
    outcome_cols = [c for c in outcome_cols if c in results_df.columns]

    if len(outcome_cols) < 2:
        print("Need at least 2 outcome columns")
        return

    # Create pairplot
    g = sns.pairplot(
        results_df[outcome_cols + (['consensus_reached'] if 'consensus_reached' in results_df.columns else [])],
        hue='consensus_reached' if 'consensus_reached' in results_df.columns else None,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 50},
        corner=False
    )

    g.fig.suptitle('Pairwise Outcome Relationships', y=1.02, fontsize=14, fontweight='bold')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_geometry_vs_outcomes(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot triangle geometry features vs outcome variables."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Triangle Geometry vs Outcomes', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    geometry_outcome_pairs = [
        ('geom_perimeter', 'turn_count', 'Perimeter vs Turns'),
        ('geom_area', 'turn_count', 'Area vs Turns'),
        ('geom_perimeter', 'n_sources_used', 'Perimeter vs Sources'),
        ('geom_area', 'n_sources_used', 'Area vs Sources'),
        ('geom_regularity', 'turn_count', 'Regularity vs Turns'),
        ('geom_regularity', 'n_sources_used', 'Regularity vs Sources'),
    ]

    for idx, (geom_col, outcome_col, title) in enumerate(geometry_outcome_pairs):
        if idx >= 6:
            break

        ax = axes[idx]

        if geom_col in results_df.columns and outcome_col in results_df.columns:
            # Compute regularity if needed
            if geom_col == 'geom_regularity' and geom_col not in results_df.columns:
                avg_side = (results_df['geom_side_1'] + results_df['geom_side_2'] + results_df['geom_side_3']) / 3
                side_var = results_df[['geom_side_1', 'geom_side_2', 'geom_side_3']].var(axis=1)
                results_df['geom_regularity'] = 1 - side_var / (avg_side ** 2 + 1e-6)

            valid_mask = results_df[geom_col].notna() & results_df[outcome_col].notna()
            x = results_df.loc[valid_mask, geom_col]
            y = results_df.loc[valid_mask, outcome_col]

            ax.scatter(x, y, alpha=0.6, s=50, color=f'C{idx}')

            # Trend line
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

                # Correlation
                from scipy.stats import pearsonr
                r, p_val = pearsonr(x, y)
                ax.text(0.05, 0.95, f'r={r:.3f}\np={p_val:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel(geom_col.replace('geom_', '').replace('_', ' ').title())
            ax.set_ylabel(outcome_col.replace('_', ' ').title())
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_experiment_timeline(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot metrics over experiment sequence."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Experiment Timeline', fontsize=14, fontweight='bold')

    experiment_indices = np.arange(len(results_df))

    # 1. Turns over time
    axes[0, 0].plot(experiment_indices, results_df['turn_count'], alpha=0.6, color='blue')
    axes[0, 0].set_xlabel('Experiment Index')
    axes[0, 0].set_ylabel('Turn Count')
    axes[0, 0].set_title('Dialogue Length Over Time')
    axes[0, 0].grid(True, alpha=0.3)

    # Moving average
    window = 10
    if len(results_df) >= window:
        moving_avg = results_df['turn_count'].rolling(window=window).mean()
        axes[0, 0].plot(experiment_indices, moving_avg, color='red', linewidth=2,
                       label=f'{window}-exp moving avg')
        axes[0, 0].legend()

    # 2. Sources over time
    if 'n_sources_used' in results_df.columns:
        axes[0, 1].plot(experiment_indices, results_df['n_sources_used'],
                       alpha=0.6, color='coral')
        axes[0, 1].set_xlabel('Experiment Index')
        axes[0, 1].set_ylabel('Sources Used')
        axes[0, 1].set_title('Source Usage Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        if len(results_df) >= window:
            moving_avg = results_df['n_sources_used'].rolling(window=window).mean()
            axes[0, 1].plot(experiment_indices, moving_avg, color='red',
                           linewidth=2, label=f'{window}-exp moving avg')
            axes[0, 1].legend()

    # 3. Perimeter over time
    if 'geom_perimeter' in results_df.columns:
        axes[1, 0].plot(experiment_indices, results_df['geom_perimeter'],
                       alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Experiment Index')
        axes[1, 0].set_ylabel('Perimeter')
        axes[1, 0].set_title('Triangle Perimeter Over Time')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Area over time
    if 'geom_area' in results_df.columns:
        axes[1, 1].plot(experiment_indices, results_df['geom_area'],
                       alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Experiment Index')
        axes[1, 1].set_ylabel('Area')
        axes[1, 1].set_title('Triangle Area Over Time')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()
