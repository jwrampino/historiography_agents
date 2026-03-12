"""
Geometry Visualizations: Triangle geometry and embedding space plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_triangle_geometry_distribution(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot distributions of triangle geometry features."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Triangle Geometry Feature Distributions', fontsize=16, fontweight='bold')

    geometry_features = [
        ('geom_side_1', 'Side 1 (cosine distance)'),
        ('geom_side_2', 'Side 2 (cosine distance)'),
        ('geom_side_3', 'Side 3 (cosine distance)'),
        ('geom_perimeter', 'Perimeter'),
        ('geom_area', 'Area'),
        ('geom_min_angle', 'Min Angle (radians)'),
        ('geom_max_angle', 'Max Angle (radians)'),
        ('geom_angle_variance', 'Angle Variance')
    ]

    for idx, (col, label) in enumerate(geometry_features):
        row = idx // 4
        col_idx = idx % 4
        ax = axes[row, col_idx]

        if col in results_df.columns:
            data = results_df[col].dropna()
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color=f'C{idx}')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {data.median():.3f}')
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_triangle_shape_space(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot triangle shapes in 2D/3D space."""
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Triangle Shape Space', fontsize=16, fontweight='bold')

    # 1. Perimeter vs Area
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(
        results_df['geom_perimeter'],
        results_df['geom_area'],
        c=results_df['geom_angle_variance'],
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    ax1.set_xlabel('Perimeter')
    ax1.set_ylabel('Area')
    ax1.set_title('Perimeter vs Area\n(colored by angle variance)')
    plt.colorbar(scatter, ax=ax1, label='Angle Variance')

    # 2. Side lengths triangle plot
    ax2 = fig.add_subplot(132)
    ax2.scatter(
        results_df['geom_side_1'],
        results_df['geom_side_2'],
        c=results_df['geom_side_3'],
        cmap='plasma',
        alpha=0.6,
        s=50
    )
    ax2.set_xlabel('Side 1')
    ax2.set_ylabel('Side 2')
    ax2.set_title('Side Lengths Distribution\n(colored by Side 3)')
    plt.colorbar(scatter, ax=ax2, label='Side 3')

    # 3. Angle distribution
    ax3 = fig.add_subplot(133)
    ax3.scatter(
        results_df['geom_min_angle'],
        results_df['geom_max_angle'],
        c=results_df['geom_area'],
        cmap='coolwarm',
        alpha=0.6,
        s=50
    )
    ax3.plot([0, np.pi], [0, np.pi], 'k--', alpha=0.3, label='Equal angles')
    ax3.set_xlabel('Min Angle (radians)')
    ax3.set_ylabel('Max Angle (radians)')
    ax3.set_title('Angle Range\n(colored by area)')
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='Area')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_regularity_analysis(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Analyze triangle regularity (how equilateral)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Triangle Regularity Analysis', fontsize=16, fontweight='bold')

    # Compute regularity if not present
    if 'geom_regularity' not in results_df.columns:
        avg_side = (results_df['geom_side_1'] + results_df['geom_side_2'] + results_df['geom_side_3']) / 3
        side_var = results_df[['geom_side_1', 'geom_side_2', 'geom_side_3']].var(axis=1)
        results_df['geom_regularity'] = 1 - side_var / (avg_side ** 2 + 1e-6)

    # 1. Regularity distribution
    axes[0].hist(results_df['geom_regularity'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(results_df['geom_regularity'].mean(), color='red', linestyle='--',
                   label=f'Mean: {results_df["geom_regularity"].mean():.3f}')
    axes[0].set_xlabel('Regularity Score (1 = equilateral)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Triangle Regularity Distribution')
    axes[0].legend()

    # 2. Regularity vs Area
    axes[1].scatter(results_df['geom_regularity'], results_df['geom_area'],
                   alpha=0.6, s=50, color='coral')
    axes[1].set_xlabel('Regularity')
    axes[1].set_ylabel('Area')
    axes[1].set_title('Regularity vs Area')

    # 3. Regular vs Irregular comparison
    median_reg = results_df['geom_regularity'].median()
    regular = results_df[results_df['geom_regularity'] >= median_reg]
    irregular = results_df[results_df['geom_regularity'] < median_reg]

    metrics = ['geom_perimeter', 'geom_area']
    x = np.arange(len(metrics))
    width = 0.35

    regular_means = [regular[m].mean() for m in metrics]
    irregular_means = [irregular[m].mean() for m in metrics]

    axes[2].bar(x - width/2, regular_means, width, label='Regular', alpha=0.7, color='green')
    axes[2].bar(x + width/2, irregular_means, width, label='Irregular', alpha=0.7, color='red')
    axes[2].set_ylabel('Mean Value')
    axes[2].set_title('Regular vs Irregular Triangles')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['Perimeter', 'Area'])
    axes[2].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_embedding_space_2d(results_df: pd.DataFrame, historian_embeddings: np.ndarray,
                             historian_names: List[str], output_path: Optional[Path] = None):
    """Plot historian embedding space in 2D using PCA."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Historian Embedding Space (2D Projection)', fontsize=16, fontweight='bold')

    # PCA projection
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(historian_embeddings)

    # 1. All historians
    axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.6, color='steelblue')

    # Label some historians
    for i, name in enumerate(historian_names[:10]):  # Label first 10
        axes[0].annotate(
            name.split()[-1],  # Last name only
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8, alpha=0.7
        )

    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].set_title('All Historians (PCA)')
    axes[0].grid(True, alpha=0.3)

    # 2. Example triangles
    axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=50, alpha=0.3, color='gray')

    # Draw a few example triangles
    n_examples = min(5, len(results_df))
    colors = plt.cm.tab10(np.linspace(0, 1, n_examples))

    for idx, color in zip(range(n_examples), colors):
        # Get historian indices for this group (simplified - would need actual mapping)
        # For now, draw random triangles
        triangle_indices = np.random.choice(len(embeddings_2d), 3, replace=False)
        triangle_points = embeddings_2d[triangle_indices]

        # Draw triangle
        triangle = plt.Polygon(triangle_points, fill=False, edgecolor=color,
                              linewidth=2, alpha=0.7)
        axes[1].add_patch(triangle)

    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1].set_title('Example Historian Triangles')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_embedding_space_3d(historian_embeddings: np.ndarray, historian_names: List[str],
                            output_path: Optional[Path] = None):
    """Plot historian embedding space in 3D using PCA."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # PCA to 3D
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(historian_embeddings)

    # Scatter plot
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=np.arange(len(embeddings_3d)),
        cmap='viridis',
        s=100,
        alpha=0.6
    )

    # Label some historians
    for i, name in enumerate(historian_names[::2]):  # Every other
        ax.text(
            embeddings_3d[i*2, 0],
            embeddings_3d[i*2, 1],
            embeddings_3d[i*2, 2],
            name.split()[-1],
            fontsize=8
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title('Historian Embedding Space (3D PCA)', fontsize=14, fontweight='bold')

    plt.colorbar(scatter, ax=ax, label='Historian Index', pad=0.1)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_distance_heatmap(historian_embeddings: np.ndarray, historian_names: List[str],
                          output_path: Optional[Path] = None):
    """Plot pairwise distance heatmap between historians."""
    # Compute pairwise cosine distances
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(historian_embeddings)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Heatmap
    sns.heatmap(
        distances,
        xticklabels=[n.split()[-1] for n in historian_names],  # Last names
        yticklabels=[n.split()[-1] for n in historian_names],
        cmap='RdYlBu_r',
        square=True,
        cbar_kws={'label': 'Cosine Distance'},
        ax=ax
    )

    ax.set_title('Pairwise Cosine Distances Between Historians', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_geometry_correlations(results_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot correlation matrix of geometry features."""
    geometry_cols = [c for c in results_df.columns if c.startswith('geom_')]

    if len(geometry_cols) == 0:
        print("No geometry columns found")
        return

    corr_matrix = results_df[geometry_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )

    ax.set_title('Geometry Feature Correlations', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()
