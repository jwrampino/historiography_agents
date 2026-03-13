"""
Perplexity Analysis: Compute, analyze, and visualize perplexity metrics for experiment outputs.

Analyzes how linguistic complexity (perplexity) relates to convergence, source diversity,
and historian triad geometry.

Usage:
    python -m agents.perplexity_analysis --data-dir data/agent_experiments
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json

logger = logging.getLogger(__name__)


class PerplexityAnalyzer:
    """Compute and analyze perplexity of LLM-generated texts."""

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize perplexity analyzer.

        Args:
            model_name: HuggingFace model for perplexity computation (default: gpt2)
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                logger.info(f"Loading {self.model_name} for perplexity computation...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self._model.eval()

                # Use GPU if available
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    logger.info("Using GPU for perplexity computation")

                logger.info("Model loaded successfully")
            except ImportError:
                raise RuntimeError(
                    "transformers and torch required: pip install transformers torch"
                )
        return self._model

    @property
    def tokenizer(self):
        """Get tokenizer (triggers model load if needed)."""
        _ = self.model  # Ensure model is loaded
        return self._tokenizer

    def compute_perplexity(self, text: str, max_length: int = 512) -> float:
        """
        Compute perplexity of a text.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Perplexity score (lower = more predictable/simpler)
        """
        import torch

        if not text or len(text.strip()) == 0:
            return np.nan

        try:
            # Tokenize
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )

            # Move to same device as model
            if next(self.model.parameters()).is_cuda:
                encodings = {k: v.cuda() for k, v in encodings.items()}

            # Compute loss
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss

            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()

            return float(perplexity)

        except Exception as e:
            logger.warning(f"Failed to compute perplexity: {e}")
            return np.nan

    def compute_batch_perplexity(self, texts: List[str]) -> List[float]:
        """
        Compute perplexity for multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of perplexity scores
        """
        perplexities = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                logger.info(f"Computing perplexity {i + 1}/{len(texts)}...")
            ppl = self.compute_perplexity(text)
            perplexities.append(ppl)
        return perplexities


def load_experiment_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load experiment data from CSVs.

    Args:
        data_dir: Directory containing CSV exports

    Returns:
        Dict of DataFrames
    """
    logger.info(f"Loading data from {data_dir}...")

    data = {}
    for csv_name in ['triads', 'proposals', 'synthesis', 'convergence_results', 'llm_interactions']:
        csv_path = data_dir / f"{csv_name}.csv"
        if csv_path.exists():
            data[csv_name] = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(data[csv_name])} rows from {csv_name}.csv")
        else:
            logger.warning(f"{csv_name}.csv not found")

    return data


def compute_perplexity_features(data: Dict[str, pd.DataFrame], analyzer: PerplexityAnalyzer) -> pd.DataFrame:
    """
    Compute perplexity features for all texts.

    Args:
        data: Experiment data dict
        analyzer: PerplexityAnalyzer instance

    Returns:
        DataFrame with perplexity features per triad
    """
    logger.info("Computing perplexity features...")

    results = []

    for triad_id in data['triads']['triad_id'].unique():
        # Get proposals for this triad
        proposals = data['proposals'][data['proposals']['triad_id'] == triad_id]
        synthesis = data['synthesis'][data['synthesis']['triad_id'] == triad_id]

        if len(proposals) == 0 or len(synthesis) == 0:
            continue

        # Compute perplexity for each proposal abstract
        proposal_perplexities = []
        for _, prop in proposals.iterrows():
            ppl = analyzer.compute_perplexity(prop['abstract'])
            proposal_perplexities.append(ppl)

        # Compute perplexity for synthesis
        synthesis_text = synthesis.iloc[0]['final_abstract']
        synthesis_ppl = analyzer.compute_perplexity(synthesis_text)

        # Aggregate metrics
        results.append({
            'triad_id': triad_id,
            'proposal_ppl_mean': np.mean(proposal_perplexities),
            'proposal_ppl_std': np.std(proposal_perplexities),
            'proposal_ppl_min': np.min(proposal_perplexities),
            'proposal_ppl_max': np.max(proposal_perplexities),
            'synthesis_ppl': synthesis_ppl,
            'ppl_delta': synthesis_ppl - np.mean(proposal_perplexities),  # How much simpler/complex is synthesis
        })

    df = pd.DataFrame(results)
    logger.info(f"Computed perplexity for {len(df)} triads")
    return df


def merge_with_convergence_data(
    perplexity_df: pd.DataFrame,
    convergence_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge perplexity features with convergence data.

    Args:
        perplexity_df: Perplexity features
        convergence_df: Convergence results

    Returns:
        Merged DataFrame
    """
    merged = convergence_df.merge(perplexity_df, on='triad_id', how='inner')
    logger.info(f"Merged data: {len(merged)} rows with {len(merged.columns)} columns")
    return merged


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlations between perplexity and other features.

    Args:
        df: Merged DataFrame

    Returns:
        DataFrame of correlations
    """
    logger.info("Analyzing correlations...")

    perplexity_cols = [
        'proposal_ppl_mean', 'proposal_ppl_std', 'synthesis_ppl', 'ppl_delta'
    ]

    target_cols = [
        'convergence_delta', 'bias_score', 'mean_historian_distance',
        'mean_source_embedding_distance', 'source_embedding_variance'
    ]

    # Filter to available columns
    perplexity_cols = [c for c in perplexity_cols if c in df.columns]
    target_cols = [c for c in target_cols if c in df.columns]

    correlations = []
    for ppl_col in perplexity_cols:
        for target_col in target_cols:
            if df[ppl_col].notna().sum() > 0 and df[target_col].notna().sum() > 0:
                corr = df[ppl_col].corr(df[target_col])
                correlations.append({
                    'perplexity_feature': ppl_col,
                    'target_feature': target_col,
                    'correlation': corr
                })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    return corr_df


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create visualization plots.

    Args:
        df: Merged DataFrame
        output_dir: Output directory for plots
    """
    logger.info("Creating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # 1. Perplexity distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Perplexity Distributions', fontsize=16)

    axes[0, 0].hist(df['proposal_ppl_mean'].dropna(), bins=20, edgecolor='black')
    axes[0, 0].set_xlabel('Mean Proposal Perplexity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Proposal Perplexity Distribution')

    axes[0, 1].hist(df['synthesis_ppl'].dropna(), bins=20, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Synthesis Perplexity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Synthesis Perplexity Distribution')

    axes[1, 0].hist(df['ppl_delta'].dropna(), bins=20, edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Perplexity Delta (synthesis - proposals)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Perplexity Change Distribution')
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)

    axes[1, 1].scatter(df['proposal_ppl_mean'], df['synthesis_ppl'], alpha=0.6)
    axes[1, 1].set_xlabel('Mean Proposal Perplexity')
    axes[1, 1].set_ylabel('Synthesis Perplexity')
    axes[1, 1].set_title('Proposal vs Synthesis Perplexity')
    # Add diagonal line
    min_val = min(df['proposal_ppl_mean'].min(), df['synthesis_ppl'].min())
    max_val = max(df['proposal_ppl_mean'].max(), df['synthesis_ppl'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'perplexity_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved perplexity_distributions.png")

    # 2. Perplexity vs Convergence
    if 'convergence_delta' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Perplexity vs Convergence Delta', fontsize=16)

        axes[0].scatter(df['proposal_ppl_mean'], df['convergence_delta'], alpha=0.6)
        axes[0].set_xlabel('Mean Proposal Perplexity')
        axes[0].set_ylabel('Convergence Delta')
        axes[0].set_title('Proposal Perplexity')
        if df['proposal_ppl_mean'].notna().sum() > 0:
            corr = df['proposal_ppl_mean'].corr(df['convergence_delta'])
            axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        axes[1].scatter(df['synthesis_ppl'], df['convergence_delta'], alpha=0.6, color='orange')
        axes[1].set_xlabel('Synthesis Perplexity')
        axes[1].set_ylabel('Convergence Delta')
        axes[1].set_title('Synthesis Perplexity')
        if df['synthesis_ppl'].notna().sum() > 0:
            corr = df['synthesis_ppl'].corr(df['convergence_delta'])
            axes[1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        axes[2].scatter(df['ppl_delta'], df['convergence_delta'], alpha=0.6, color='green')
        axes[2].set_xlabel('Perplexity Delta')
        axes[2].set_ylabel('Convergence Delta')
        axes[2].set_title('Perplexity Change')
        axes[2].axvline(0, color='red', linestyle='--', alpha=0.3)
        if df['ppl_delta'].notna().sum() > 0:
            corr = df['ppl_delta'].corr(df['convergence_delta'])
            axes[2].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[2].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        plt.savefig(output_dir / 'perplexity_vs_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved perplexity_vs_convergence.png")

    # 3. Perplexity vs Source Diversity
    if 'mean_source_embedding_distance' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Perplexity vs Source Diversity', fontsize=16)

        axes[0].scatter(df['mean_source_embedding_distance'], df['proposal_ppl_mean'], alpha=0.6)
        axes[0].set_xlabel('Mean Source Embedding Distance')
        axes[0].set_ylabel('Mean Proposal Perplexity')
        axes[0].set_title('Source Diversity vs Proposal Complexity')
        if df['mean_source_embedding_distance'].notna().sum() > 0:
            corr = df['mean_source_embedding_distance'].corr(df['proposal_ppl_mean'])
            axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        axes[1].scatter(df['mean_source_embedding_distance'], df['synthesis_ppl'], alpha=0.6, color='orange')
        axes[1].set_xlabel('Mean Source Embedding Distance')
        axes[1].set_ylabel('Synthesis Perplexity')
        axes[1].set_title('Source Diversity vs Synthesis Complexity')
        if df['mean_source_embedding_distance'].notna().sum() > 0:
            corr = df['mean_source_embedding_distance'].corr(df['synthesis_ppl'])
            axes[1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        plt.savefig(output_dir / 'perplexity_vs_source_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved perplexity_vs_source_diversity.png")

    # 4. Correlation heatmap
    ppl_cols = ['proposal_ppl_mean', 'proposal_ppl_std', 'synthesis_ppl', 'ppl_delta']
    other_cols = ['convergence_delta', 'bias_score', 'mean_historian_distance']
    if 'mean_source_embedding_distance' in df.columns:
        other_cols.append('mean_source_embedding_distance')

    # Filter to available columns
    ppl_cols = [c for c in ppl_cols if c in df.columns]
    other_cols = [c for c in other_cols if c in df.columns]
    all_cols = ppl_cols + other_cols

    if len(all_cols) > 1:
        corr_matrix = df[all_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix: Perplexity and Convergence Features', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved correlation_heatmap.png")


def run_analysis(data_dir: Path, output_dir: Path, model_name: str = "gpt2"):
    """
    Run complete perplexity analysis.

    Args:
        data_dir: Directory with experiment data
        output_dir: Output directory for results
        model_name: Model for perplexity computation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_experiment_data(data_dir)

    if 'proposals' not in data or 'synthesis' not in data:
        logger.error("Missing required data files (proposals.csv, synthesis.csv)")
        return

    # Initialize analyzer
    analyzer = PerplexityAnalyzer(model_name=model_name)

    # Compute perplexity features
    perplexity_df = compute_perplexity_features(data, analyzer)

    # Save perplexity features
    perplexity_csv = output_dir / 'perplexity_features.csv'
    perplexity_df.to_csv(perplexity_csv, index=False)
    logger.info(f"Saved perplexity features to {perplexity_csv}")

    # Merge with convergence data if available
    if 'convergence_results' in data:
        # Load convergence data with geometry
        convergence_df = data['triads'].merge(
            data['convergence_results'],
            on='triad_id',
            how='inner'
        )
        merged_df = merge_with_convergence_data(perplexity_df, convergence_df)

        # Analyze correlations
        corr_df = analyze_correlations(merged_df)
        corr_csv = output_dir / 'perplexity_correlations.csv'
        corr_df.to_csv(corr_csv, index=False)
        logger.info(f"Saved correlations to {corr_csv}")

        print("\n" + "=" * 60)
        print("TOP CORRELATIONS")
        print("=" * 60)
        print(corr_df.head(10).to_string(index=False))
        print("=" * 60 + "\n")

        # Create visualizations
        create_visualizations(merged_df, output_dir)

        # Save merged data
        merged_csv = output_dir / 'merged_perplexity_data.csv'
        merged_df.to_csv(merged_csv, index=False)
        logger.info(f"Saved merged data to {merged_csv}")

        # Summary statistics
        summary = {
            'n_triads': len(merged_df),
            'mean_proposal_ppl': float(merged_df['proposal_ppl_mean'].mean()),
            'mean_synthesis_ppl': float(merged_df['synthesis_ppl'].mean()),
            'mean_ppl_delta': float(merged_df['ppl_delta'].mean()),
            'ppl_delta_correlation_with_convergence': float(
                merged_df['ppl_delta'].corr(merged_df['convergence_delta'])
            ) if 'convergence_delta' in merged_df.columns else None,
        }

        summary_json = output_dir / 'perplexity_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_json}")

        print("\n" + "=" * 60)
        print("PERPLEXITY SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2))
        print("=" * 60 + "\n")

    logger.info("Analysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Perplexity Analysis for Multi-Agent Experiments"
    )
    parser.add_argument(
        "--data-dir",
        default="data/agent_experiments",
        help="Directory containing experiment CSV files"
    )
    parser.add_argument(
        "--output-dir",
        default="data/perplexity_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Model for perplexity computation (default: gpt2)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Run analysis
    run_analysis(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model
    )


if __name__ == "__main__":
    main()
