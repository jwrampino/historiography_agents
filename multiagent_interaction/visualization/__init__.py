"""
Visualization package for historiography multi-agent experiments.
"""

from .geometry_viz import (
    plot_triangle_geometry_distribution,
    plot_triangle_shape_space,
    plot_regularity_analysis,
    plot_embedding_space_2d,
    plot_embedding_space_3d,
    plot_distance_heatmap,
    plot_geometry_correlations
)

from .prediction_viz import (
    plot_feature_importance,
    plot_prediction_results,
    plot_model_comparison,
    plot_cross_validation_scores,
    plot_correlation_matrix,
    plot_significant_correlations,
    plot_learning_curves,
    plot_prediction_intervals
)

from .experiment_viz import (
    plot_experiment_overview,
    plot_historian_participation,
    plot_outcome_distributions,
    plot_pairwise_outcomes,
    plot_geometry_vs_outcomes,
    plot_experiment_timeline
)

__all__ = [
    # Geometry
    'plot_triangle_geometry_distribution',
    'plot_triangle_shape_space',
    'plot_regularity_analysis',
    'plot_embedding_space_2d',
    'plot_embedding_space_3d',
    'plot_distance_heatmap',
    'plot_geometry_correlations',
    # Prediction
    'plot_feature_importance',
    'plot_prediction_results',
    'plot_model_comparison',
    'plot_cross_validation_scores',
    'plot_correlation_matrix',
    'plot_significant_correlations',
    'plot_learning_curves',
    'plot_prediction_intervals',
    # Experiment
    'plot_experiment_overview',
    'plot_historian_participation',
    'plot_outcome_distributions',
    'plot_pairwise_outcomes',
    'plot_geometry_vs_outcomes',
    'plot_experiment_timeline',
]
