"""
Prediction Analysis Module: Uses embedding geometry to predict outcomes.

Addresses:
- RQ1: Can embedding distances predict differential source selection?
- RQ2: Can embedding geometry predict abstract perplexity/quality?

Replaces causal inference with supervised prediction modeling.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass
import yaml

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

import sys
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PredictionResult:
    """Stores prediction model results."""
    model_name: str
    target_name: str
    train_r2: float
    test_r2: float
    train_mse: float
    test_mse: float
    cv_r2_mean: float
    cv_r2_std: float
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    metadata: Dict


class PerplexityCalculator:
    """Calculate perplexity of text using language model."""

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize perplexity calculator.

        Args:
            model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch

            self._tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._model.eval()

            # Add padding token if missing
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        except ImportError:
            raise RuntimeError(
                "transformers and torch required: "
                "pip install transformers torch"
            )

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text.

        Args:
            text: Input text

        Returns:
            Perplexity score (lower = more fluent/expected)
        """
        import torch

        self._load_model()

        if not text or len(text.strip()) == 0:
            return float('inf')

        # Tokenize
        encodings = self._tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        )

        # Compute loss
        with torch.no_grad():
            outputs = self._model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss

        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()

        return perplexity


class PredictionAnalyzer:
    """Analyzes experiments using prediction models based on embedding geometry."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.results_df: Optional[pd.DataFrame] = None
        self.detailed_results: List[Dict] = []
        self.perplexity_calc = None

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_results(self, results_path: Path):
        """Load experiment results from CSV."""
        self.results_df = pd.read_csv(results_path)
        print(f"✓ Loaded {len(self.results_df)} experiment results")

    def load_detailed_results(self, output_dir: Path):
        """Load detailed results from individual JSON files."""
        result_files = sorted(list(output_dir.glob("exp_*.json")))
        self.detailed_results = []

        for result_file in result_files:
            with open(result_file, 'r') as f:
                self.detailed_results.append(json.load(f))

        print(f"✓ Loaded {len(self.detailed_results)} detailed results")

    def compute_perplexity_scores(self) -> pd.DataFrame:
        """
        Compute perplexity of final abstracts.

        Returns:
            DataFrame with experiment_id and perplexity scores
        """
        print("\nComputing abstract perplexity scores...")

        if self.perplexity_calc is None:
            self.perplexity_calc = PerplexityCalculator()

        perplexity_scores = []

        for result in self.detailed_results:
            exp_id = result['experiment_id']
            abstract = result.get('final_abstract', '')

            try:
                perplexity = self.perplexity_calc.compute_perplexity(abstract)
            except Exception as e:
                print(f"  Warning: Failed to compute perplexity for {exp_id}: {e}")
                perplexity = np.nan

            perplexity_scores.append({
                'experiment_id': exp_id,
                'perplexity': perplexity,
                'abstract_length': len(abstract.split())
            })

        print(f"✓ Computed perplexity for {len(perplexity_scores)} abstracts")
        return pd.DataFrame(perplexity_scores)

    def compute_source_selection_features(self) -> pd.DataFrame:
        """
        Compute source selection features.

        Returns:
            DataFrame with source usage counts and patterns
        """
        print("\nComputing source selection features...")

        features = []

        for result in self.detailed_results:
            exp_id = result['experiment_id']
            sources_accessed = result.get('sources_accessed', [])

            # Count features
            n_sources = len(sources_accessed)
            unique_sources = len(set(s.get('source_id', '') for s in sources_accessed))

            # Source diversity
            source_diversity = unique_sources / max(n_sources, 1)

            features.append({
                'experiment_id': exp_id,
                'n_sources_accessed': n_sources,
                'n_unique_sources': unique_sources,
                'source_diversity': source_diversity,
                'sources_per_turn': n_sources / max(result.get('turn_count', 1), 1)
            })

        print(f"✓ Computed source features for {len(features)} experiments")
        return pd.DataFrame(features)

    def engineer_geometry_features(self) -> pd.DataFrame:
        """
        Extract and engineer features from triangle geometry.

        Returns:
            DataFrame with engineered geometry features
        """
        print("\nEngineering geometry features...")

        if self.results_df is None:
            raise RuntimeError("Must call load_results first")

        df = self.results_df.copy()

        # Derived features
        df['geom_avg_side'] = (
            df['geom_side_1'] + df['geom_side_2'] + df['geom_side_3']
        ) / 3

        df['geom_side_variance'] = df[[
            'geom_side_1', 'geom_side_2', 'geom_side_3'
        ]].var(axis=1)

        df['geom_angle_range'] = df['geom_max_angle'] - df['geom_min_angle']

        # Normalized features
        df['geom_perimeter_norm'] = df['geom_perimeter'] / df['geom_perimeter'].max()
        df['geom_area_norm'] = df['geom_area'] / df['geom_area'].max()

        # Shape regularity (closer to equilateral = higher)
        df['geom_regularity'] = 1 - df['geom_side_variance'] / (
            df['geom_avg_side'] ** 2 + 1e-6
        )

        print(f"✓ Engineered {len(df.columns)} geometry features")
        return df

    def predict_source_selection(
        self,
        target: str = 'n_sources_accessed',
        models: Optional[List[str]] = None
    ) -> Dict[str, PredictionResult]:
        """
        Predict source selection patterns from triangle geometry.

        Args:
            target: Target variable ('n_sources_accessed', 'source_diversity', etc.)
            models: List of model names to try (default: ['rf', 'gb', 'ridge'])

        Returns:
            Dictionary of model results
        """
        print("\n" + "="*60)
        print(f"RQ1: Predicting {target} from Triangle Geometry")
        print("="*60)

        # Prepare data
        df = self.engineer_geometry_features()
        source_features = self.compute_source_selection_features()
        df = pd.merge(df, source_features, on='experiment_id')

        # Feature matrix (geometry features only)
        geometry_cols = [c for c in df.columns if c.startswith('geom_')]
        X = df[geometry_cols].values
        y = df[target].values

        # Check for NaNs
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"\nFeatures: {len(geometry_cols)}")
        print(f"Samples: {len(X)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model definitions
        if models is None:
            models = ['rf', 'gb', 'ridge']

        model_dict = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }

        results = {}

        for model_name in models:
            print(f"\n--- {model_name.upper()} ---")

            model = model_dict[model_name]

            # Train
            model.fit(X_train_scaled, y_train)

            # Predict
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Evaluate
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='r2'
            )

            print(f"Train R²: {train_r2:.3f}")
            print(f"Test R²: {test_r2:.3f}")
            print(f"Test MSE: {test_mse:.3f}")
            print(f"CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                importance = np.zeros(len(geometry_cols))

            feature_importance = dict(zip(geometry_cols, importance))

            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            print("\nTop 5 features:")
            for feat, imp in sorted_features[:5]:
                print(f"  {feat}: {imp:.4f}")

            results[model_name] = PredictionResult(
                model_name=model_name,
                target_name=target,
                train_r2=train_r2,
                test_r2=test_r2,
                train_mse=train_mse,
                test_mse=test_mse,
                cv_r2_mean=cv_scores.mean(),
                cv_r2_std=cv_scores.std(),
                feature_importance=feature_importance,
                predictions=y_test_pred,
                metadata={'n_train': len(X_train), 'n_test': len(X_test)}
            )

        return results

    def predict_abstract_perplexity(
        self,
        models: Optional[List[str]] = None
    ) -> Dict[str, PredictionResult]:
        """
        Predict abstract perplexity from triangle geometry.

        Args:
            models: List of model names to try

        Returns:
            Dictionary of model results
        """
        print("\n" + "="*60)
        print("RQ2: Predicting Abstract Perplexity from Triangle Geometry")
        print("="*60)

        # Prepare data
        df = self.engineer_geometry_features()
        perplexity_df = self.compute_perplexity_scores()
        df = pd.merge(df, perplexity_df, on='experiment_id')

        # Feature matrix
        geometry_cols = [c for c in df.columns if c.startswith('geom_')]
        X = df[geometry_cols].values
        y = df['perplexity'].values

        # Remove invalid perplexities
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]

        # Log-transform perplexity (often log-normal)
        y_log = np.log1p(y)

        print(f"\nFeatures: {len(geometry_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Perplexity range: {y.min():.2f} to {y.max():.2f}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models
        if models is None:
            models = ['rf', 'gb', 'ridge']

        model_dict = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }

        results = {}

        for model_name in models:
            print(f"\n--- {model_name.upper()} ---")

            model = model_dict[model_name]
            model.fit(X_train_scaled, y_train)

            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)

            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='r2'
            )

            print(f"Train R²: {train_r2:.3f}")
            print(f"Test R²: {test_r2:.3f}")
            print(f"Test MSE: {test_mse:.3f}")
            print(f"CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                importance = np.zeros(len(geometry_cols))

            feature_importance = dict(zip(geometry_cols, importance))

            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            print("\nTop 5 features:")
            for feat, imp in sorted_features[:5]:
                print(f"  {feat}: {imp:.4f}")

            results[model_name] = PredictionResult(
                model_name=model_name,
                target_name='log_perplexity',
                train_r2=train_r2,
                test_r2=test_r2,
                train_mse=train_mse,
                test_mse=test_mse,
                cv_r2_mean=cv_scores.mean(),
                cv_r2_std=cv_scores.std(),
                feature_importance=feature_importance,
                predictions=y_test_pred,
                metadata={'n_train': len(X_train), 'n_test': len(X_test)}
            )

        return results

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Compute correlations between geometry features and outcomes.

        Returns:
            DataFrame of correlation coefficients
        """
        print("\n" + "="*60)
        print("Correlation Analysis: Geometry vs Outcomes")
        print("="*60)

        df = self.engineer_geometry_features()
        source_features = self.compute_source_selection_features()
        perplexity_df = self.compute_perplexity_scores()

        df = pd.merge(df, source_features, on='experiment_id')
        df = pd.merge(df, perplexity_df, on='experiment_id')

        geometry_cols = [c for c in df.columns if c.startswith('geom_')]
        outcome_cols = [
            'n_sources_accessed', 'source_diversity', 'perplexity',
            'abstract_length', 'turn_count'
        ]

        correlations = []

        for geom_col in geometry_cols:
            for outcome_col in outcome_cols:
                # Skip if missing
                if outcome_col not in df.columns:
                    continue

                valid_mask = ~(df[geom_col].isna() | df[outcome_col].isna())
                if valid_mask.sum() < 10:
                    continue

                x = df.loc[valid_mask, geom_col]
                y = df.loc[valid_mask, outcome_col]

                # Pearson correlation
                r_pearson, p_pearson = pearsonr(x, y)

                # Spearman correlation
                r_spearman, p_spearman = spearmanr(x, y)

                correlations.append({
                    'geometry_feature': geom_col,
                    'outcome': outcome_col,
                    'pearson_r': r_pearson,
                    'pearson_p': p_pearson,
                    'spearman_r': r_spearman,
                    'spearman_p': p_spearman,
                    'n': valid_mask.sum()
                })

        corr_df = pd.DataFrame(correlations)

        # Show significant correlations
        significant = corr_df[corr_df['pearson_p'] < 0.05].sort_values(
            'pearson_r', key=abs, ascending=False
        )

        print(f"\nSignificant correlations (p < 0.05): {len(significant)}")
        if len(significant) > 0:
            print("\nTop 10 strongest correlations:")
            print(significant.head(10)[[
                'geometry_feature', 'outcome', 'pearson_r', 'pearson_p'
            ]].to_string(index=False))

        return corr_df

    def generate_report(self, output_path: Path):
        """Generate comprehensive prediction analysis report."""
        print("\n" + "="*60)
        print("Generating Prediction Analysis Report")
        print("="*60)

        # Run all analyses
        source_results = self.predict_source_selection()
        perplexity_results = self.predict_abstract_perplexity()
        correlation_df = self.correlation_analysis()

        # Compile report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_experiments': len(self.results_df),
            'rq1_source_prediction': {
                model_name: {
                    'test_r2': result.test_r2,
                    'test_mse': result.test_mse,
                    'cv_r2_mean': result.cv_r2_mean,
                    'top_features': dict(sorted(
                        result.feature_importance.items(),
                        key=lambda x: x[1], reverse=True
                    )[:5])
                }
                for model_name, result in source_results.items()
            },
            'rq2_perplexity_prediction': {
                model_name: {
                    'test_r2': result.test_r2,
                    'test_mse': result.test_mse,
                    'cv_r2_mean': result.cv_r2_mean,
                    'top_features': dict(sorted(
                        result.feature_importance.items(),
                        key=lambda x: x[1], reverse=True
                    )[:5])
                }
                for model_name, result in perplexity_results.items()
            },
            'correlations': {
                'n_significant': len(correlation_df[correlation_df['pearson_p'] < 0.05]),
                'strongest': correlation_df.nlargest(10, 'pearson_r', key=abs).to_dict('records')
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved to {output_path}")

        # Save correlation matrix
        corr_path = output_path.parent / "correlations.csv"
        correlation_df.to_csv(corr_path, index=False)
        print(f"✓ Correlations saved to {corr_path}")

        return report


if __name__ == "__main__":
    # Example usage
    analyzer = PredictionAnalyzer()

    # Load results
    output_dir = Path("../outputs")
    if (output_dir / "results.csv").exists():
        analyzer.load_results(output_dir / "results.csv")
        analyzer.load_detailed_results(output_dir)

        # Run analysis
        report = analyzer.generate_report(output_dir / "prediction_report.json")
    else:
        print("No results found. Run experiments first.")
