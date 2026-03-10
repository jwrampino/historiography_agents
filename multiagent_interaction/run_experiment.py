"""
Main entry point for running multi-agent historian experiments.

Usage:
    python run_experiment.py --mode full
    python run_experiment.py --mode setup-only
    python run_experiment.py --mode analyze-only
"""

import argparse
from pathlib import Path
import sys

from personas.persona_manager import PersonaManager
from sources.source_library import SourceLibrary, create_example_library
from experiments.experiment_runner import ExperimentRunner
from analysis.causal_analysis import ExperimentAnalyzer


def setup_experiment(args):
    """Setup personas and source library."""
    print("="*70)
    print("SETUP: Initializing Experiment Components")
    print("="*70)

    # Initialize persona manager
    persona_manager = PersonaManager(args.config)

    # Create and save personas
    print("\nCreating historian personas...")
    personas = persona_manager.create_persona_grid()
    persona_manager.save_personas_to_storage()
    print(f"Created and saved {len(personas)} historian personas")

    # Create source library
    print("\nCreating source library...")
    if args.existing_library:
        library = SourceLibrary(args.config)
        library.load_index(Path(args.existing_library))
    else:
        library = create_example_library(n_sources=args.n_sources)
        library.save_index()

    print(f"Source library ready with {len(library.sources)} sources")
    print("\nSetup complete!")


def run_experiments(args):
    """Run the full experimental pipeline."""
    print("="*70)
    print("EXPERIMENTS: Running Multi-Agent Dialogue")
    print("="*70)

    # Initialize runner
    runner = ExperimentRunner(args.config)

    # Setup personas
    runner.setup_personas(
        strategy=args.strategy,
        n_samples=args.n_samples
    )

    # Setup source library
    runner.setup_source_library(args.existing_library)

    # Run experiments
    print(f"\nStarting {len(runner.groups)} experiments...")
    runner.run_all_experiments(
        max_experiments=args.max_experiments,
        start_from=args.start_from
    )

    # Export results
    df = runner.get_results_dataframe()
    output_path = runner.output_dir / "results.csv"
    df.to_csv(output_path, index=False)

    print(f"\nExperiments complete!")
    print(f"Results saved to: {output_path}")


def analyze_results(args):
    """Analyze experimental results."""
    print("="*70)
    print("ANALYSIS: Causal Inference and Pattern Detection")
    print("="*70)

    # Initialize analyzer
    analyzer = ExperimentAnalyzer(args.config)

    # Load results
    output_dir = Path(args.output_dir)
    results_file = output_dir / "results.csv"

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Run experiments first with: python run_experiment.py --mode full")
        sys.exit(1)

    print(f"\nLoading results from {output_dir}...")
    analyzer.load_results(results_file)
    analyzer.load_detailed_results(output_dir)

    # Run analysis
    print("\nPerforming causal analysis...")
    report = analyzer.generate_report(output_dir / "analysis_report.json")

    print("\nAnalysis complete!")
    print(f"Report saved to: {output_dir / 'analysis_report.json'}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    rq1 = report['rq1_source_selection']
    print(f"\nRQ1: Source Selection Patterns")
    print(f"  - Total tests: {rq1['n_tests']}")
    print(f"  - Significant effects: {rq1['n_significant']}")

    rq2 = report['rq2_optimal_configurations']
    print(f"\nRQ2: Optimal Configurations")
    for metric, data in rq2.items():
        print(f"  - {metric}: R² = {data['r2_score']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Historian Experiment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (setup + run + analyze)
  python run_experiment.py --mode full --n-samples 20

  # Setup only
  python run_experiment.py --mode setup-only --n-sources 500

  # Run experiments only
  python run_experiment.py --mode run-only --strategy stratified --n-samples 50

  # Analyze existing results
  python run_experiment.py --mode analyze-only

Research Questions:
  RQ1: How do specific groups of historians differentially select primary sources?
  RQ2: Which configurations produce the most novel, perplexing, high-quality theses?
        """
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['full', 'setup-only', 'run-only', 'analyze-only'],
        default='full',
        help='Execution mode (default: full)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    # Setup options
    parser.add_argument(
        '--n-sources',
        type=int,
        default=200,
        help='Number of sources in library (for setup)'
    )
    parser.add_argument(
        '--existing-library',
        type=str,
        default=None,
        help='Path to existing source library'
    )

    # Experiment options
    parser.add_argument(
        '--strategy',
        choices=['full', 'stratified'],
        default='stratified',
        help='Sampling strategy for persona groups'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=50,
        help='Number of groups to sample (stratified strategy)'
    )
    parser.add_argument(
        '--max-experiments',
        type=int,
        default=None,
        help='Maximum number of experiments to run'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Experiment index to start from (for resuming)'
    )

    # Analysis options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory containing experiment outputs'
    )

    args = parser.parse_args()

    # Execute based on mode
    try:
        if args.mode in ['full', 'setup-only']:
            setup_experiment(args)

        if args.mode in ['full', 'run-only']:
            run_experiments(args)

        if args.mode in ['full', 'analyze-only']:
            analyze_results(args)

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
