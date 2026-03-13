"""
Prediction Model: Ridge regression to predict convergence delta.
Uses triangle geometry features to predict how much the synthesis
converged toward the centroid, and analyzes historian bias.
"""
 
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
 
logger = logging.getLogger(__name__)
 
 
class ConvergencePredictionModel:
    """Predicts convergence delta for historian triads via ridge regression."""
 
    def __init__(self):
        """Initialize prediction model."""
        self.model = None
        self.feature_names = None
        self.scaler = None
 
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for regression.
 
        Args:
            df: DataFrame with triad and convergence data
 
        Returns:
            (feature_matrix, feature_names)
        """
        feature_names = [
            'perimeter',
            'area',
            'angle_variance',
            'mean_historian_distance'
        ]
 
        if 'abstract_distance_variance' in df.columns:
            feature_names.extend([
                'abstract_distance_variance',
                'mean_pairwise_abstract_similarity'
            ])
 
        # Add bias features if present
        for col in ['bias_score', 'dominant_historian_position']:
            if col in df.columns:
                feature_names.append(col)

        # Add source embedding features if present
        for col in ['mean_source_embedding_distance', 'source_embedding_variance']:
            if col in df.columns:
                feature_names.append(col)

        X = df[feature_names].values
        return X, feature_names
 
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Fit ridge regression model predicting convergence_delta.
 
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Continuous target — convergence_delta (n_samples,)
            feature_names: List of feature names
 
        Returns:
            Dict with training results
        """
        try:
            from sklearn.linear_model import RidgeCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score, mean_absolute_error
        except ImportError:
            raise RuntimeError(
                "scikit-learn required: pip install scikit-learn"
            )
 
        self.feature_names = feature_names
 
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
 
        # Fit ridge regression with cross-validated alpha
        self.model = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            cv=min(5, len(y))
        )
        self.model.fit(X_scaled, y)
 
        # In-sample metrics
        y_pred = self.model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
 
        results = {
            'n_samples': int(len(y)),
            'target': 'convergence_delta',
            'target_mean': float(np.mean(y)),
            'target_std': float(np.std(y)),
            'target_min': float(np.min(y)),
            'target_max': float(np.max(y)),
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'alpha_selected': float(self.model.alpha_),
            'coefficients': {
                name: float(coef)
                for name, coef in zip(feature_names, self.model.coef_)
            },
            'intercept': float(self.model.intercept_)
        }
 
        logger.info(
            f"Regression fitted: R²={r2:.3f}, MAE={mae:.4f}, "
            f"alpha={self.model.alpha_}"
        )
        return results
 
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict convergence delta for new triads.
 
        Args:
            X: Feature matrix
 
        Returns:
            Predicted convergence deltas
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
 
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
 
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze feature importance from standardized coefficients.
 
        Returns:
            DataFrame with feature importance sorted by absolute coefficient
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
 
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
 
        return importance_df
 
    def save_model(self, output_path: str = "data/agent_experiments/prediction_model.json"):
        """Save model parameters to JSON."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
 
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
 
        model_data = {
            'model_type': 'ridge_regression',
            'target': 'convergence_delta',
            'feature_names': self.feature_names,
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_),
            'alpha': float(self.model.alpha_),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist()
        }
 
        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)
 
        logger.info(f"Saved model to {output_path}")
 
 
def run_inference_analysis(df: pd.DataFrame) -> Dict:
    """
    Run inference analysis: test if geometric diversity predicts convergence delta.
    Uses OLS with triangle area as the key predictor of convergence_delta.
 
    Args:
        df: DataFrame with triangle geometry and convergence_delta
 
    Returns:
        Dict with inference results
    """
    if len(df) < 4:
        logger.warning("Too few samples for inference analysis")
        return {'error': 'insufficient_samples'}
 
    if 'convergence_delta' not in df.columns:
        logger.warning("convergence_delta not found in dataframe")
        return {'error': 'missing_convergence_delta'}
 
    y = df['convergence_delta'].values
 
    results = {
        'target': 'convergence_delta',
        'n_samples': len(df),
        'convergence_delta_mean': float(np.mean(y)),
        'convergence_delta_std': float(np.std(y)),
        'convergence_delta_min': float(np.min(y)),
        'convergence_delta_max': float(np.max(y)),
    }
 
    # Correlations between geometry features and convergence delta
    geometry_features = ['perimeter', 'area', 'angle_variance', 'mean_historian_distance']
    correlations = {}
    for feat in geometry_features:
        if feat in df.columns:
            corr = float(np.corrcoef(df[feat].values, y)[0, 1])
            correlations[feat] = corr
    results['correlations_with_delta'] = correlations
 
    # Add bias correlations if present
    if 'bias_score' in df.columns:
        bias_corr = float(np.corrcoef(df['bias_score'].values, y)[0, 1])
        results['bias_score_correlation_with_delta'] = bias_corr
 
    # OLS regression: area -> convergence_delta
    try:
        from scipy import stats
 
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['area'].values, y
        )
        results['ols_area_vs_delta'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err)
        }
 
        # Most correlated feature overall
        best_feat = max(correlations, key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_feat]
        results['strongest_predictor'] = best_feat
        results['strongest_correlation'] = best_corr
        results['interpretation'] = (
            f"'{best_feat}' is the strongest geometric predictor of convergence "
            f"(r={best_corr:.3f}). Higher convergence_delta means the synthesis "
            f"landed closer to the true centroid of the triad."
        )
 
        logger.info(
            f"Inference: delta_mean={np.mean(y):.4f}, "
            f"best_predictor={best_feat} (r={best_corr:.3f}), "
            f"area p={p_value:.3f}"
        )
 
    except ImportError:
        logger.warning("scipy not available")
        results['error'] = 'scipy_not_available'

    return results


def run_ablation_study(df: pd.DataFrame) -> Dict:
    """
    Ablation study: compare prediction with/without source embedding features.

    Tests whether source geometry improves convergence prediction beyond
    just historian triangle geometry and bias metrics.

    Args:
        df: DataFrame with all features

    Returns:
        Dict with ablation results
    """
    if len(df) < 4:
        logger.warning("Too few samples for ablation study")
        return {'error': 'insufficient_samples'}

    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        logger.warning("scikit-learn required for ablation study")
        return {'error': 'sklearn_not_available'}

    y = df['convergence_delta'].values

    # Feature sets
    baseline_features = ['perimeter', 'area', 'angle_variance', 'mean_historian_distance']

    extended_features = baseline_features.copy()
    if 'abstract_distance_variance' in df.columns:
        extended_features.extend(['abstract_distance_variance', 'mean_pairwise_abstract_similarity'])
    if 'bias_score' in df.columns:
        extended_features.extend(['bias_score', 'dominant_historian_position'])

    source_features = []
    if 'mean_source_embedding_distance' in df.columns:
        source_features.append('mean_source_embedding_distance')
    if 'source_embedding_variance' in df.columns:
        source_features.append('source_embedding_variance')

    full_features = extended_features + source_features

    # Filter available features
    baseline_features = [f for f in baseline_features if f in df.columns]
    extended_features = [f for f in extended_features if f in df.columns]
    full_features = [f for f in full_features if f in df.columns]

    results = {
        'n_samples': len(df),
        'baseline_features': baseline_features,
        'extended_features': extended_features,
        'source_features': source_features,
        'full_features': full_features
    }

    # Train models
    models = {
        'baseline': baseline_features,
        'extended': extended_features,
        'full': full_features
    }

    for model_name, features in models.items():
        if len(features) == 0:
            continue

        X = df[features].values

        # Remove rows with NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 4:
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Fit model
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=min(5, len(y_clean)))
        model.fit(X_scaled, y_clean)

        # Metrics
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y_clean, y_pred)
        mae = mean_absolute_error(y_clean, y_pred)
        rmse = float(np.sqrt(np.mean((y_clean - y_pred) ** 2)))

        # Cross-validation score
        cv_scores = cross_val_score(model, X_scaled, y_clean, cv=min(5, len(y_clean)), scoring='r2')

        results[f'{model_name}_model'] = {
            'n_features': len(features),
            'n_samples_used': int(len(y_clean)),
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'cv_r2_mean': float(cv_scores.mean()),
            'cv_r2_std': float(cv_scores.std()),
            'alpha': float(model.alpha_)
        }

    # Compute improvements
    if 'baseline_model' in results and 'extended_model' in results:
        results['extended_improvement'] = {
            'delta_r2': results['extended_model']['r2'] - results['baseline_model']['r2'],
            'delta_mae': results['extended_model']['mae'] - results['baseline_model']['mae'],
            'pct_r2_improvement': 100 * (results['extended_model']['r2'] - results['baseline_model']['r2']) / max(0.01, abs(results['baseline_model']['r2']))
        }

    if 'extended_model' in results and 'full_model' in results and source_features:
        results['source_improvement'] = {
            'delta_r2': results['full_model']['r2'] - results['extended_model']['r2'],
            'delta_mae': results['full_model']['mae'] - results['extended_model']['mae'],
            'pct_r2_improvement': 100 * (results['full_model']['r2'] - results['extended_model']['r2']) / max(0.01, abs(results['extended_model']['r2'])),
            'interpretation': 'Positive delta_r2 means source embeddings improve prediction'
        }

    logger.info(
        f"Ablation: baseline R²={results.get('baseline_model', {}).get('r2', 0):.3f}, "
        f"extended R²={results.get('extended_model', {}).get('r2', 0):.3f}, "
        f"full R²={results.get('full_model', {}).get('r2', 0):.3f}"
    )

    return results