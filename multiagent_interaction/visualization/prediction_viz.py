"""
Prediction Visualizations: Model performance and feature importance plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_feature_importance(feature_importance: Dict[str, float], model_name: str,
                            output_path: Optional[Path] = None):
    """Plot feature importance from prediction model."""
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features[:15])  # Top 15

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax.barh(range(len(features)), importance, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance: {model_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Add value labels
    for i, (feat, imp) in enumerate(zip(features, importance)):
        ax.text(imp + 0.01 * max(importance), i, f'{imp:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_prediction_results(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                            target_name: str, output_path: Optional[Path] = None):
    """Plot prediction results: actual vs predicted."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name}: Predicting {target_name}', fontsize=14, fontweight='bold')

    # 1. Scatter: Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue')

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    # Metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    axes[0].text(0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}',
                transform=axes[0].transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[0].set_xlabel('Actual', fontsize=11)
    axes[0].set_ylabel('Predicted', fontsize=11)
    axes[0].set_title('Actual vs Predicted')
    axes[0].legend()

    # 2. Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, color='coral')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Residuals', fontsize=11)
    axes[1].set_title('Residual Plot')

    # 3. Residual distribution
    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[2].set_xlabel('Residuals', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Residual Distribution')
    axes[2].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_model_comparison(prediction_results: Dict[str, Dict], target_name: str,
                         output_path: Optional[Path] = None):
    """Compare performance across different models."""
    models = list(prediction_results.keys())

    train_r2 = [prediction_results[m]['train_r2'] for m in models]
    test_r2 = [prediction_results[m]['test_r2'] for m in models]
    cv_r2 = [prediction_results[m]['cv_r2_mean'] for m in models]
    cv_std = [prediction_results[m].get('cv_r2_std', 0) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Model Comparison: {target_name}', fontsize=14, fontweight='bold')

    # 1. R² comparison
    x = np.arange(len(models))
    width = 0.25

    axes[0].bar(x - width, train_r2, width, label='Train R²', alpha=0.8, color='skyblue', edgecolor='black')
    axes[0].bar(x, test_r2, width, label='Test R²', alpha=0.8, color='coral', edgecolor='black')
    axes[0].bar(x + width, cv_r2, width, label='CV R² (mean)', alpha=0.8, color='lightgreen', edgecolor='black')

    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].set_title('R² Comparison Across Models')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3)

    # 2. MSE comparison
    train_mse = [prediction_results[m]['train_mse'] for m in models]
    test_mse = [prediction_results[m]['test_mse'] for m in models]

    axes[1].bar(x - width/2, train_mse, width, label='Train MSE', alpha=0.8, color='orange', edgecolor='black')
    axes[1].bar(x + width/2, test_mse, width, label='Test MSE', alpha=0.8, color='purple', edgecolor='black')

    axes[1].set_ylabel('MSE', fontsize=11)
    axes[1].set_title('MSE Comparison Across Models')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_cross_validation_scores(cv_scores: List[float], model_name: str,
                                 output_path: Optional[Path] = None):
    """Plot cross-validation fold scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Cross-Validation: {model_name}', fontsize=14, fontweight='bold')

    # 1. Fold scores
    folds = np.arange(1, len(cv_scores) + 1)
    axes[0].bar(folds, cv_scores, alpha=0.7, color='teal', edgecolor='black')
    axes[0].axhline(np.mean(cv_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(cv_scores):.3f}')
    axes[0].set_xlabel('Fold', fontsize=11)
    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].set_title('R² Score by Fold')
    axes[0].legend()
    axes[0].set_xticks(folds)

    # 2. Distribution
    axes[1].hist(cv_scores, bins=10, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1].axvline(np.mean(cv_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(cv_scores):.3f}')
    axes[1].axvline(np.median(cv_scores), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(cv_scores):.3f}')
    axes[1].set_xlabel('R² Score', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distribution of CV Scores')
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_correlation_matrix(correlation_df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot correlation matrix between geometry features and outcomes."""
    # Pivot: geometry features vs outcomes
    pivot = correlation_df.pivot(index='geometry_feature', columns='outcome', values='pearson_r')

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )

    ax.set_title('Geometry Features vs Outcomes: Correlation Matrix',
                fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_significant_correlations(correlation_df: pd.DataFrame, p_threshold: float = 0.05,
                                  output_path: Optional[Path] = None):
    """Plot only significant correlations."""
    significant = correlation_df[correlation_df['pearson_p'] < p_threshold].copy()
    significant = significant.sort_values('pearson_r', key=abs, ascending=False)

    if len(significant) == 0:
        print("No significant correlations found")
        return

    fig, ax = plt.subplots(figsize=(12, max(8, len(significant) * 0.3)))

    # Create labels
    labels = [f"{row['geometry_feature']}\nvs\n{row['outcome']}"
              for _, row in significant.iterrows()]

    # Color by correlation strength
    colors = ['green' if r > 0 else 'red' for r in significant['pearson_r']]

    bars = ax.barh(range(len(significant)), significant['pearson_r'],
                   color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(range(len(significant)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
    ax.set_title(f'Significant Correlations (p < {p_threshold})',
                fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.invert_yaxis()

    # Add p-value labels
    for i, (_, row) in enumerate(significant.iterrows()):
        x_pos = row['pearson_r'] + (0.02 if row['pearson_r'] > 0 else -0.02)
        ax.text(x_pos, i, f"p={row['pearson_p']:.3f}",
               va='center', ha='left' if row['pearson_r'] > 0 else 'right',
               fontsize=7, style='italic')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_learning_curves(train_sizes: np.ndarray, train_scores: np.ndarray,
                        test_scores: np.ndarray, model_name: str,
                        output_path: Optional[Path] = None):
    """Plot learning curves showing performance vs training set size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.plot(train_sizes, train_mean, 'o-', label='Training score',
           color='blue', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='blue')

    ax.plot(train_sizes, test_mean, 'o-', label='Test score',
           color='red', linewidth=2)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color='red')

    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title(f'Learning Curves: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()


def plot_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                              y_std: Optional[np.ndarray] = None,
                              output_path: Optional[Path] = None):
    """Plot predictions with uncertainty intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))

    indices = np.arange(len(y_true))

    ax.scatter(indices, y_true, label='Actual', color='blue', alpha=0.6, s=50)
    ax.scatter(indices, y_pred, label='Predicted', color='red', alpha=0.6, s=50)

    if y_std is not None:
        ax.fill_between(indices, y_pred - 1.96*y_std, y_pred + 1.96*y_std,
                       alpha=0.2, color='red', label='95% CI')

    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Predictions with Uncertainty', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
    else:
        plt.show()
