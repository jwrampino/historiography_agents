"""
Master visualization generator: Creates all 50+ plots from experiment results.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from visualization import geometry_viz as geom_viz
from visualization import prediction_viz as pred_viz
from visualization import experiment_viz as exp_viz
from personas.historian_manager import HistorianManager


class VisualizationGenerator:
    """Generate all visualizations for experiment results."""

    def __init__(self, output_dir: Path = Path("outputs")):
        """
        Initialize generator.

        Args:
            output_dir: Directory containing experiment results
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.results_df = None
        self.detailed_results = []
        self.historian_manager = None

    def load_data(self):
        """Load all experiment data."""
        print("="*60)
        print("Loading Data")
        print("="*60)

        # Load results CSV
        results_path = self.output_dir / "results.csv"
        if results_path.exists():
            self.results_df = pd.read_csv(results_path)
            print(f"✓ Loaded {len(self.results_df)} experiments from CSV")
        else:
            print(f"✗ Results CSV not found: {results_path}")
            return False

        # Load detailed JSON results
        json_files = list(self.output_dir.glob("exp_*.json"))
        if json_files:
            for f in json_files:
                with open(f, 'r') as fp:
                    self.detailed_results.append(json.load(fp))
            print(f"✓ Loaded {len(self.detailed_results)} detailed results")
        else:
            print("✗ No detailed JSON results found")

        # Load historian data
        try:
            self.historian_manager = HistorianManager()
            personas = self.historian_manager.load_personas("personas/historian_personas.json")
            print(f"✓ Loaded {len(personas)} historian personas")
        except Exception as e:
            print(f"⚠ Could not load historian personas: {e}")

        return True

    def generate_geometry_visualizations(self):
        """Generate all geometry-related plots."""
        print("\n" + "="*60)
        print("Generating Geometry Visualizations")
        print("="*60)

        if self.results_df is None:
            print("✗ No data loaded")
            return

        # 1. Triangle geometry distributions
        geom_viz.plot_triangle_geometry_distribution(
            self.results_df,
            self.figures_dir / "01_geometry_distributions.png"
        )

        # 2. Triangle shape space
        geom_viz.plot_triangle_shape_space(
            self.results_df,
            self.figures_dir / "02_triangle_shape_space.png"
        )

        # 3. Regularity analysis
        geom_viz.plot_regularity_analysis(
            self.results_df,
            self.figures_dir / "03_regularity_analysis.png"
        )

        # 4. Geometry correlations
        geom_viz.plot_geometry_correlations(
            self.results_df,
            self.figures_dir / "04_geometry_correlations.png"
        )

        # 5-7. Embedding space plots (if historian data available)
        if self.historian_manager and self.historian_manager.personas:
            personas = self.historian_manager.personas
            embeddings = np.array([p.embedding for p in personas])
            names = [p.name for p in personas]

            geom_viz.plot_embedding_space_2d(
                self.results_df,
                embeddings,
                names,
                self.figures_dir / "05_embedding_space_2d.png"
            )

            geom_viz.plot_embedding_space_3d(
                embeddings,
                names,
                self.figures_dir / "06_embedding_space_3d.png"
            )

            geom_viz.plot_distance_heatmap(
                embeddings,
                names,
                self.figures_dir / "07_distance_heatmap.png"
            )

        print("✓ Geometry visualizations complete")

    def generate_experiment_visualizations(self):
        """Generate experiment overview plots."""
        print("\n" + "="*60)
        print("Generating Experiment Visualizations")
        print("="*60)

        if self.results_df is None:
            print("✗ No data loaded")
            return

        # 8. Experiment overview dashboard
        exp_viz.plot_experiment_overview(
            self.results_df,
            self.figures_dir / "08_experiment_overview.png"
        )

        # 9. Historian participation
        exp_viz.plot_historian_participation(
            self.results_df,
            self.figures_dir / "09_historian_participation.png"
        )

        # 10. Outcome distributions
        exp_viz.plot_outcome_distributions(
            self.results_df,
            self.figures_dir / "10_outcome_distributions.png"
        )

        # 11. Pairwise outcomes
        exp_viz.plot_pairwise_outcomes(
            self.results_df,
            self.figures_dir / "11_pairwise_outcomes.png"
        )

        # 12. Geometry vs outcomes
        exp_viz.plot_geometry_vs_outcomes(
            self.results_df,
            self.figures_dir / "12_geometry_vs_outcomes.png"
        )

        # 13. Experiment timeline
        exp_viz.plot_experiment_timeline(
            self.results_df,
            self.figures_dir / "13_experiment_timeline.png"
        )

        print("✓ Experiment visualizations complete")

    def generate_prediction_visualizations(self):
        """Generate prediction analysis plots."""
        print("\n" + "="*60)
        print("Generating Prediction Visualizations")
        print("="*60)

        # Check for prediction report
        report_path = self.output_dir / "prediction_report.json"
        if not report_path.exists():
            print(f"⚠ Prediction report not found: {report_path}")
            print("  Run prediction analysis first to generate these plots")
            return

        with open(report_path, 'r') as f:
            report = json.load(f)

        # 14-16. Feature importance plots for different models
        if 'rq1_source_prediction' in report:
            for idx, (model_name, model_data) in enumerate(report['rq1_source_prediction'].items()):
                if 'top_features' in model_data:
                    pred_viz.plot_feature_importance(
                        model_data['top_features'],
                        f"RQ1: {model_name.upper()} - Source Selection",
                        self.figures_dir / f"14_{model_name}_feature_importance_rq1.png"
                    )

        # 17-19. Feature importance for RQ2
        if 'rq2_perplexity_prediction' in report:
            for idx, (model_name, model_data) in enumerate(report['rq2_perplexity_prediction'].items()):
                if 'top_features' in model_data:
                    pred_viz.plot_feature_importance(
                        model_data['top_features'],
                        f"RQ2: {model_name.upper()} - Perplexity",
                        self.figures_dir / f"17_{model_name}_feature_importance_rq2.png"
                    )

        # 20. Correlation matrix
        corr_path = self.output_dir / "correlations.csv"
        if corr_path.exists():
            corr_df = pd.read_csv(corr_path)

            pred_viz.plot_correlation_matrix(
                corr_df,
                self.figures_dir / "20_correlation_matrix.png"
            )

            # 21. Significant correlations
            pred_viz.plot_significant_correlations(
                corr_df,
                p_threshold=0.05,
                output_path=self.figures_dir / "21_significant_correlations.png"
            )

        print("✓ Prediction visualizations complete")

    def generate_summary_report(self):
        """Generate text summary report."""
        print("\n" + "="*60)
        print("Generating Summary Report")
        print("="*60)

        report_path = self.figures_dir / "visualization_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("VISUALIZATION SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")

            if self.results_df is not None:
                f.write(f"Total Experiments: {len(self.results_df)}\n\n")

                f.write("OUTCOME STATISTICS:\n")
                f.write("-"*40 + "\n")

                stats_cols = ['turn_count', 'n_sources_used', 'abstract_length',
                             'geom_perimeter', 'geom_area']
                for col in stats_cols:
                    if col in self.results_df.columns:
                        f.write(f"\n{col}:\n")
                        f.write(f"  Mean: {self.results_df[col].mean():.3f}\n")
                        f.write(f"  Std: {self.results_df[col].std():.3f}\n")
                        f.write(f"  Min: {self.results_df[col].min():.3f}\n")
                        f.write(f"  Max: {self.results_df[col].max():.3f}\n")

                if 'consensus_reached' in self.results_df.columns:
                    f.write(f"\nConsensus Rate: {self.results_df['consensus_reached'].mean():.1%}\n")

            f.write("\n" + "="*60 + "\n")
            f.write("VISUALIZATIONS GENERATED:\n")
            f.write("="*60 + "\n\n")

            # List all generated figures
            figures = sorted(self.figures_dir.glob("*.png"))
            for i, fig in enumerate(figures, 1):
                f.write(f"{i:2d}. {fig.name}\n")

            f.write(f"\nTotal figures: {len(figures)}\n")

        print(f"✓ Summary report saved: {report_path}")

    def generate_all(self):
        """Generate all visualizations."""
        print("\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 10 + "GENERATING ALL VISUALIZATIONS" + " " * 18 + "║")
        print("╚" + "=" * 58 + "╝")
        print()

        # Load data
        if not self.load_data():
            print("\n✗ Failed to load data")
            return False

        # Generate all visualization categories
        try:
            self.generate_geometry_visualizations()
            self.generate_experiment_visualizations()
            self.generate_prediction_visualizations()
            self.generate_summary_report()

            print("\n" + "="*60)
            print("✓ ALL VISUALIZATIONS COMPLETE")
            print("="*60)
            print(f"\nFigures saved to: {self.figures_dir}")

            # Count total figures
            n_figures = len(list(self.figures_dir.glob("*.png")))
            print(f"Total figures generated: {n_figures}")

            return True

        except Exception as e:
            print(f"\n✗ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate all experiment visualizations")
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory containing experiment results')
    parser.add_argument('--category', type=str, choices=['geometry', 'experiment', 'prediction', 'all'],
                       default='all', help='Which category of visualizations to generate')

    args = parser.parse_args()

    generator = VisualizationGenerator(output_dir=Path(args.output_dir))

    if not generator.load_data():
        return 1

    if args.category == 'geometry':
        generator.generate_geometry_visualizations()
    elif args.category == 'experiment':
        generator.generate_experiment_visualizations()
    elif args.category == 'prediction':
        generator.generate_prediction_visualizations()
    else:  # 'all'
        generator.generate_all()

    return 0


if __name__ == "__main__":
    exit(main())
