"""
Causal Analysis Module: Implements SAE and double debiasing methods
to determine causal effects of historian personas on source selection
and thesis quality.

Addresses:
- RQ1: How do historian groups differentially select primary sources?
- RQ2: Which configurations produce novel, perplexing, high-quality theses?
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass
import yaml

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import bootstrap
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('..')


@dataclass
class CausalEstimate:
    """Stores causal effect estimates."""
    effect_name: str
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    metadata: Dict


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning latent representations
    of source selection patterns and thesis characteristics.
    """

    def __init__(self, input_dim: int, latent_dim: int, sparsity_weight: float = 0.1):
        super(SparseAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def loss_function(self, x, reconstructed, latent):
        # Reconstruction loss
        recon_loss = nn.MSELoss()(reconstructed, x)

        # Sparsity penalty (KL divergence)
        rho = 0.05  # Target sparsity
        rho_hat = torch.mean(latent, dim=0)
        kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        sparsity_loss = torch.sum(kl_div)

        return recon_loss + self.sparsity_weight * sparsity_loss


class DoubleDebiasingEstimator:
    """
    Implements double debiasing (double machine learning) for causal inference.
    Reduces bias in treatment effect estimation.
    """

    def __init__(
        self,
        treatment_model=None,
        outcome_model=None,
        n_splits: int = 5
    ):
        self.treatment_model = treatment_model or RandomForestClassifier(n_estimators=100)
        self.outcome_model = outcome_model or RandomForestRegressor(n_estimators=100)
        self.n_splits = n_splits

    def estimate_ate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> CausalEstimate:
        """
        Estimate Average Treatment Effect (ATE) using double debiasing.

        Args:
            X: Covariates (confounders)
            treatment: Binary treatment indicator
            outcome: Outcome variable

        Returns:
            CausalEstimate with ATE and confidence intervals
        """
        n = len(X)
        residuals_y = np.zeros(n)
        residuals_t = np.zeros(n)

        # Cross-fitting
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            t_train, t_test = treatment[train_idx], treatment[test_idx]
            y_train, y_test = outcome[train_idx], outcome[test_idx]

            # Fit outcome model
            self.outcome_model.fit(X_train, y_train)
            y_pred = self.outcome_model.predict(X_test)
            residuals_y[test_idx] = y_test - y_pred

            # Fit treatment model (propensity score)
            self.treatment_model.fit(X_train, t_train)
            t_pred = self.treatment_model.predict_proba(X_test)[:, 1]
            residuals_t[test_idx] = t_test - t_pred

        # Double debiased estimator
        ate = np.mean((residuals_y * residuals_t) / np.mean(residuals_t ** 2))

        # Bootstrap for standard errors
        def ate_bootstrap(data):
            indices = np.random.choice(n, size=n, replace=True)
            res_y = residuals_y[indices]
            res_t = residuals_t[indices]
            return np.mean((res_y * res_t) / np.mean(res_t ** 2))

        bootstrap_samples = [ate_bootstrap(None) for _ in range(1000)]
        std_error = np.std(bootstrap_samples)

        # Confidence intervals
        ci_lower = np.percentile(bootstrap_samples, 2.5)
        ci_upper = np.percentile(bootstrap_samples, 97.5)

        # P-value (two-tailed test)
        z_score = ate / std_error
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return CausalEstimate(
            effect_name="ATE",
            estimate=ate,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            metadata={}
        )


class ExperimentAnalyzer:
    """Analyzes experiment results to answer research questions."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.results_df: Optional[pd.DataFrame] = None
        self.source_patterns: Optional[np.ndarray] = None
        self.sae: Optional[SparseAutoencoder] = None

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_results(self, results_path: Path):
        """Load experiment results from CSV."""
        if results_path.suffix == '.csv':
            self.results_df = pd.read_csv(results_path)
        else:
            raise ValueError("Results must be in CSV format")

        print(f"Loaded {len(self.results_df)} experiment results")

    def load_detailed_results(self, output_dir: Path):
        """Load detailed results from individual JSON files."""
        result_files = list(output_dir.glob("exp_*.json"))
        self.detailed_results = []

        for result_file in result_files:
            with open(result_file, 'r') as f:
                self.detailed_results.append(json.load(f))

        print(f"Loaded {len(self.detailed_results)} detailed results")

    def extract_source_selection_patterns(self) -> np.ndarray:
        """
        Extract source selection patterns from detailed results.
        Creates a matrix of [n_experiments, n_sources] indicating usage.
        """
        if not hasattr(self, 'detailed_results'):
            raise RuntimeError("Must call load_detailed_results first")

        # Get all unique sources
        all_sources = set()
        for result in self.detailed_results:
            for source in result.get('sources_accessed', []):
                all_sources.add(source['source_id'])

        source_list = sorted(list(all_sources))
        source_to_idx = {s: i for i, s in enumerate(source_list)}

        # Create pattern matrix
        n_experiments = len(self.detailed_results)
        n_sources = len(source_list)
        pattern_matrix = np.zeros((n_experiments, n_sources))

        for i, result in enumerate(self.detailed_results):
            for source in result.get('sources_accessed', []):
                source_idx = source_to_idx[source['source_id']]
                pattern_matrix[i, source_idx] = 1

        self.source_patterns = pattern_matrix
        self.source_list = source_list

        print(f"Extracted source patterns: {pattern_matrix.shape}")
        return pattern_matrix

    def train_sae(self, latent_dim: Optional[int] = None):
        """Train Sparse Autoencoder on source selection patterns."""
        if self.source_patterns is None:
            self.extract_source_selection_patterns()

        if latent_dim is None:
            latent_dim = self.config['analysis']['sae_latent_dim']

        input_dim = self.source_patterns.shape[1]
        self.sae = SparseAutoencoder(input_dim, latent_dim)

        # Convert to PyTorch tensors
        X = torch.FloatTensor(self.source_patterns)

        # Training
        optimizer = optim.Adam(self.sae.parameters(), lr=0.001)
        n_epochs = 100

        print("Training Sparse Autoencoder...")
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            reconstructed, latent = self.sae(X)
            loss = self.sae.loss_function(X, reconstructed, latent)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

        print("SAE training complete")

    def get_latent_representations(self) -> np.ndarray:
        """Get latent representations from trained SAE."""
        if self.sae is None:
            self.train_sae()

        X = torch.FloatTensor(self.source_patterns)
        with torch.no_grad():
            _, latent = self.sae(X)

        return latent.numpy()

    def analyze_rq1_source_selection(self) -> Dict:
        """
        RQ1: How do specific groups of historians differentially select sources?

        Returns causal estimates of persona characteristics on source selection.
        """
        print("\n" + "="*60)
        print("Analyzing RQ1: Source Selection Patterns")
        print("="*60)

        if self.source_patterns is None:
            self.extract_source_selection_patterns()

        # Get latent representations
        latent_repr = self.get_latent_representations()

        # Prepare treatment variables (persona characteristics)
        results = []

        # Test effect of each persona dimension
        for agent_idx in range(3):  # 3 agents per group
            for dimension in ['field', 'method', 'era', 'orientation']:
                col_name = f'agent_{agent_idx}_{dimension}'
                if col_name not in self.results_df.columns:
                    continue

                # Encode categorical variable
                le = LabelEncoder()
                treatment = le.fit_transform(self.results_df[col_name])

                # For each latent dimension
                for latent_dim in range(latent_repr.shape[1]):
                    outcome = latent_repr[:, latent_dim]

                    # Prepare confounders (other persona characteristics)
                    confounder_cols = [
                        c for c in self.results_df.columns
                        if c.startswith('agent_') and c != col_name
                    ]

                    X_confounders = self.results_df[confounder_cols].copy()
                    for col in X_confounders.columns:
                        if X_confounders[col].dtype == 'object':
                            X_confounders[col] = LabelEncoder().fit_transform(X_confounders[col])

                    X_confounders = StandardScaler().fit_transform(X_confounders)

                    # Estimate causal effect
                    estimator = DoubleDebiasingEstimator()

                    # For continuous outcome, convert to binary treatment
                    # Split treatment at median for binary
                    treatment_binary = (treatment > np.median(treatment)).astype(int)

                    try:
                        causal_est = estimator.estimate_ate(
                            X_confounders,
                            treatment_binary,
                            outcome
                        )

                        results.append({
                            'agent': agent_idx,
                            'dimension': dimension,
                            'latent_dim': latent_dim,
                            'effect': causal_est.estimate,
                            'std_error': causal_est.std_error,
                            'p_value': causal_est.p_value,
                            'significant': causal_est.p_value < 0.05
                        })
                    except:
                        continue

        results_df = pd.DataFrame(results)

        # Summarize significant effects
        significant = results_df[results_df['significant']]
        print(f"\nFound {len(significant)} significant effects (p < 0.05)")

        summary = {
            'n_tests': len(results_df),
            'n_significant': len(significant),
            'significant_effects': significant.to_dict('records'),
            'effect_by_dimension': results_df.groupby('dimension')['effect'].mean().to_dict()
        }

        return summary

    def compute_thesis_quality_metrics(self) -> pd.DataFrame:
        """
        Compute outcome metrics for thesis quality:
        - Novelty (embedding-based)
        - Perplexity
        - Quality scores
        """
        if not hasattr(self, 'detailed_results'):
            raise RuntimeError("Must call load_detailed_results first")

        metrics = []

        # Load sentence transformer for novelty
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')

        # Embed all theses
        all_questions = [r['final_question'] for r in self.detailed_results]
        all_abstracts = [r['final_abstract'] for r in self.detailed_results]

        question_embeddings = model.encode(all_questions)
        abstract_embeddings = model.encode(all_abstracts)

        for i, result in enumerate(self.detailed_results):
            # Novelty: average distance to other theses
            q_dists = np.linalg.norm(
                question_embeddings - question_embeddings[i], axis=1
            )
            novelty_question = np.mean(q_dists[q_dists > 0])  # Exclude self

            a_dists = np.linalg.norm(
                abstract_embeddings - abstract_embeddings[i], axis=1
            )
            novelty_abstract = np.mean(a_dists[a_dists > 0])

            # Quality proxies
            complexity = len(result['final_abstract'].split())
            diversity = len(result['sources_accessed'])

            metrics.append({
                'experiment_id': result['experiment_id'],
                'novelty_question': novelty_question,
                'novelty_abstract': novelty_abstract,
                'novelty_combined': (novelty_question + novelty_abstract) / 2,
                'complexity': complexity,
                'source_diversity': diversity,
                'quality_score': (novelty_abstract * 0.5 + complexity * 0.001 + diversity * 0.1)
            })

        return pd.DataFrame(metrics)

    def analyze_rq2_optimal_configurations(self) -> Dict:
        """
        RQ2: Which configurations produce the most novel, perplexing,
        and high-quality theses?
        """
        print("\n" + "="*60)
        print("Analyzing RQ2: Optimal Configurations")
        print("="*60)

        # Compute quality metrics
        quality_df = self.compute_thesis_quality_metrics()

        # Merge with persona characteristics
        merged_df = pd.merge(
            self.results_df,
            quality_df,
            on='experiment_id'
        )

        results = {}

        # For each outcome metric
        for outcome in ['novelty_combined', 'complexity', 'quality_score']:
            print(f"\nAnalyzing outcome: {outcome}")

            outcome_values = merged_df[outcome].values

            # Feature importance via random forest
            feature_cols = [c for c in merged_df.columns if c.startswith('agent_')]
            X_features = merged_df[feature_cols].copy()

            for col in X_features.columns:
                if X_features[col].dtype == 'object':
                    X_features[col] = LabelEncoder().fit_transform(X_features[col])

            X_features = StandardScaler().fit_transform(X_features)

            # Fit random forest
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_features, outcome_values)

            # Feature importance
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"Top 5 features for {outcome}:")
            print(importance_df.head())

            results[outcome] = {
                'feature_importance': importance_df.to_dict('records'),
                'r2_score': rf.score(X_features, outcome_values)
            }

            # Identify best configurations
            top_experiments = merged_df.nlargest(5, outcome)
            results[outcome]['top_configurations'] = top_experiments[[
                'experiment_id', outcome
            ] + feature_cols[:6]].to_dict('records')

        return results

    def generate_report(self, output_path: Path):
        """Generate comprehensive analysis report."""
        print("\n" + "="*60)
        print("Generating Analysis Report")
        print("="*60)

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_experiments': len(self.results_df),
            'rq1_source_selection': self.analyze_rq1_source_selection(),
            'rq2_optimal_configurations': self.analyze_rq2_optimal_configurations()
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to {output_path}")

        return report


if __name__ == "__main__":
    # Example usage
    analyzer = ExperimentAnalyzer()

    # Load results
    output_dir = Path("../outputs")
    if (output_dir / "results.csv").exists():
        analyzer.load_results(output_dir / "results.csv")
        analyzer.load_detailed_results(output_dir)

        # Run analysis
        report = analyzer.generate_report(output_dir / "analysis_report.json")
    else:
        print("No results found. Run experiments first.")
