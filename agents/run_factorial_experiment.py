"""
Full Factorial Experiment Runner: Run all possible triad combinations.

Instead of sampling triads, this exhaustively tests all possible combinations
of 3 historians from the pool (or a filtered subset based on geometry constraints).

Usage:
    python -m agents.run_factorial_experiment --output-dir data/factorial_experiments
    python -m agents.run_factorial_experiment --max-triads 100 --min-area 0.001
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple
from itertools import combinations

from agents.historian_manager import HistorianManager
from agents.source_retrieval import SourceRetriever
from agents.agent_llm import AgentLLM
from agents.interaction_pipeline import InteractionPipeline, TriadExperimentResult
from agents.convergence_analysis import ConvergenceAnalyzer
from agents.storage import ExperimentStorage
from agents.prediction_model import ConvergencePredictionModel, run_inference_analysis

from historian_pipeline.storage.corpus_store import CorpusStore
from historian_pipeline.embeddings.faiss_index import CorpusIndex

logger = logging.getLogger(__name__)


class FactorialExperimentRunner:
    """Runs experiments on all possible triad combinations."""

    def __init__(
        self,
        output_dir: str = "data/factorial_experiments",
        openai_api_key: str = None,
        min_area: float = 0.001,
        max_triads: int = None
    ):
        """
        Initialize factorial experiment runner.

        Args:
            output_dir: Output directory
            openai_api_key: OpenAI API key
            min_area: Minimum triangle area filter (default: 0.001)
            max_triads: Maximum number of triads to run (None = all)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_area = min_area
        self.max_triads = max_triads

        logger.info("Initializing factorial experiment components...")

        # Compute embeddings BEFORE FAISS loads
        self.historian_manager = HistorianManager(n_historians=25)
        self.historian_manager.create_historian_personas()
        self.historian_manager.compute_historian_embeddings()

        # Now safe to load FAISS
        self.corpus_store = CorpusStore()
        self.corpus_index = CorpusIndex()
        self.source_retriever = SourceRetriever(
            corpus_store=self.corpus_store,
            corpus_index=self.corpus_index
        )

        # Delete existing database to start fresh
        db_path = self.output_dir / "experiments.duckdb"
        if db_path.exists():
            db_path.unlink()
            logger.info("Deleted existing experiments database")

        # Initialize storage before AgentLLM for checkpoint logging
        self.storage = ExperimentStorage(
            db_path=str(self.output_dir / "experiments.duckdb")
        )

        self.agent_llm = AgentLLM(api_key=openai_api_key, storage=self.storage)
        self.interaction_pipeline = InteractionPipeline(
            source_retriever=self.source_retriever,
            agent_llm=self.agent_llm
        )
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.prediction_model = ConvergencePredictionModel()

        logger.info("All components initialized")

    def generate_all_triads(self) -> List[Tuple]:
        """
        Generate all possible 3-combinations of historians.

        Returns:
            List of (triad, geometry) tuples
        """
        logger.info("Generating all possible historian triads...")

        personas = self.historian_manager.personas
        n_personas = len(personas)

        # Generate all combinations
        all_combinations = list(combinations(range(n_personas), 3))
        logger.info(f"Total possible triads: {len(all_combinations)}")

        # Filter by geometry constraints
        valid_triads = []
        for combo in all_combinations:
            triad = tuple([personas[i] for i in combo])
            geometry = self.historian_manager.compute_triangle_geometry(triad)

            # Apply filters
            if geometry['area'] >= self.min_area:
                valid_triads.append((triad, geometry))

        logger.info(f"Valid triads after filtering (area >= {self.min_area}): {len(valid_triads)}")

        # Sort by area (largest first for most interesting cases)
        valid_triads.sort(key=lambda x: x[1]['area'], reverse=True)

        # Limit if requested
        if self.max_triads and len(valid_triads) > self.max_triads:
            logger.info(f"Limiting to {self.max_triads} triads")
            valid_triads = valid_triads[:self.max_triads]

        return valid_triads

    def run(self):
        """Run the full factorial experiment."""
        t0 = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"FULL FACTORIAL EXPERIMENT")
        logger.info(f"{'='*60}\n")

        # Generate all valid triads
        triads_with_geometry = self.generate_all_triads()

        if len(triads_with_geometry) == 0:
            logger.error("No valid triads found!")
            return {'error': 'no_triads_found'}

        logger.info(f"Step 1: Running {len(triads_with_geometry)} triad experiments...")
        results = self._run_triad_experiments(triads_with_geometry)
        logger.info(f"Completed {len(results)} experiments\n")

        logger.info("Step 2: Analyzing convergence...")
        self._analyze_convergence(results)
        logger.info(f"Convergence analysis complete\n")

        logger.info("Step 3: Exporting data...")
        self.storage.export_to_csv(str(self.output_dir))
        logger.info(f"Data exported to {self.output_dir}\n")

        logger.info("Step 4: Training prediction model...")
        prediction_results = self._train_prediction_model()
        logger.info(f"Prediction model trained\n")

        logger.info("Step 5: Running inference analysis...")
        inference_results = self._run_inference_analysis()
        logger.info(f"Inference analysis complete\n")

        elapsed = time.time() - t0
        summary = {
            'n_triads_attempted': len(triads_with_geometry),
            'n_triads_successful': len([r for r in results if r.success]),
            'elapsed_seconds': round(elapsed, 1),
            'output_dir': str(self.output_dir),
            'min_area_filter': self.min_area,
            'max_triads_limit': self.max_triads,
            'prediction_results': prediction_results,
            'inference_results': inference_results
        }

        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"FACTORIAL EXPERIMENT COMPLETE")
        logger.info(f"Elapsed time: {elapsed:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*60}\n")

        return summary

    def _run_triad_experiments(self, triads_with_geometry: List[Tuple]) -> List[TriadExperimentResult]:
        """Run experiments for all triads."""
        from agents.run_experiment import ExperimentRunner

        results = []
        for i, (triad, geometry) in enumerate(triads_with_geometry, 1):
            logger.info(f"\n--- Triad {i}/{len(triads_with_geometry)} ---")
            result = self.interaction_pipeline.run_triad_experiment(
                triad_id=i,
                historians=triad,
                geometry=geometry,
                n_text_sources=3,
                n_image_sources=2
            )
            if result.success:
                self._store_triad_result(result)
            results.append(result)
            self.storage.export_to_csv(str(self.output_dir))
            time.sleep(1)
        return results

    def _store_triad_result(self, result: TriadExperimentResult):
        """Store triad result in database."""
        from agents.run_experiment import ExperimentRunner

        self.storage.insert_triad(
            triad_id=result.triad_id,
            historian_names=result.historian_names,
            historian_ids=[h.historian_id for h in result.historians],
            geometry=result.geometry,
            retrieval_query=result.retrieval_query
        )

        # Collect all source IDs
        all_text_ids = []
        all_image_ids = []

        for i, (name, proposal, packet) in enumerate(
            zip(result.historian_names, result.proposals, result.source_packets), 1
        ):
            # Extract source IDs from packet
            text_source_ids = [src['source_id'] for src in packet['text_sources']]
            image_source_ids = [src['source_id'] for src in packet['image_sources']]

            all_text_ids.extend(text_source_ids)
            all_image_ids.extend(image_source_ids)

            self.storage.insert_proposal(
                triad_id=result.triad_id,
                historian_name=name,
                position=i,
                proposal=proposal,
                n_text_sources=len(packet['text_sources']),
                n_image_sources=len(packet['image_sources']),
                text_source_ids=text_source_ids,
                image_source_ids=image_source_ids
            )

        self.storage.insert_synthesis(
            triad_id=result.triad_id,
            synthesis=result.synthesis,
            all_text_source_ids=all_text_ids,
            all_image_source_ids=all_image_ids
        )

    def _analyze_convergence(self, results: List[TriadExperimentResult]):
        """Analyze convergence for all results."""
        from agents.run_experiment import ExperimentRunner

        for result in results:
            if not result.success:
                continue
            try:
                historian_embeddings = tuple([h.embedding for h in result.historians])
                individual_abstracts = [p['abstract'] for p in result.proposals]
                final_abstract = result.synthesis['final_abstract']

                metrics = self.convergence_analyzer.compute_convergence_metrics(
                    historian_embeddings=historian_embeddings,
                    individual_abstracts=individual_abstracts,
                    final_abstract=final_abstract
                )
                additional_stats = self.convergence_analyzer.compute_embedding_stats(metrics)

                # Compute comprehensive source geometry
                from agents.run_experiment import ExperimentRunner
                # Use the instance method from run_experiment
                runner = ExperimentRunner(n_triads=1, output_dir=str(self.output_dir))
                (simple_stats, source_ids, distance_matrix, detailed_stats) = runner._compute_source_geometry(
                    result.triad_id, result.source_packets
                )
                additional_stats.update(simple_stats)

                # Store convergence results
                self.storage.insert_convergence_result(
                    triad_id=result.triad_id,
                    metrics=metrics.to_dict(),
                    additional_stats=additional_stats
                )

                # Store detailed source geometry
                if source_ids and distance_matrix:
                    self.storage.insert_source_geometry(
                        triad_id=result.triad_id,
                        source_ids=source_ids,
                        distance_matrix=distance_matrix,
                        stats=detailed_stats
                    )

                logger.info(
                    f"  Triad {result.triad_id}: "
                    f"delta={additional_stats['convergence_delta']:.4f}, "
                    f"bias_score={additional_stats['bias_score']:.3f}, "
                    f"dominant=historian_{additional_stats['dominant_historian_position']}"
                )

            except Exception as e:
                logger.error(f"Convergence analysis failed for triad {result.triad_id}: {e}", exc_info=True)

    def _train_prediction_model(self):
        """Train prediction model."""
        df = self.storage.get_convergence_data()

        if len(df) < 3:
            logger.warning("Insufficient data for prediction model")
            return {'error': 'insufficient_data'}

        X, feature_names = self.prediction_model.extract_features(df)
        y = df['convergence_delta'].values

        results = self.prediction_model.fit(X, y, feature_names)

        self.prediction_model.save_model(
            str(self.output_dir / "prediction_model.json")
        )

        importance_df = self.prediction_model.analyze_feature_importance()
        importance_df.to_csv(
            self.output_dir / "feature_importance.csv",
            index=False
        )

        logger.info(f"  R²={results['r2']:.3f}, MAE={results['mae']:.4f}, "
                    f"RMSE={results['rmse']:.4f}")
        return results

    def _run_inference_analysis(self):
        """Run inference analysis."""
        df = self.storage.get_convergence_data()

        if len(df) < 4:
            logger.warning("Insufficient data for inference analysis")
            return {'error': 'insufficient_data'}

        results = run_inference_analysis(df)

        with open(self.output_dir / "inference_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        if 'interpretation' in results:
            logger.info(f"  {results['interpretation']}")

        return results

    def close(self):
        """Clean up resources."""
        self.storage.close()
        self.corpus_store.close()


def setup_logging(log_level: str = "INFO", log_dir: Path = None):
    """Setup logging configuration."""
    if log_dir is None:
        log_dir = Path("data/factorial_experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "factorial_experiment.log")
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Full Factorial Multi-Agent Historical Synthesis Experiment"
    )
    parser.add_argument(
        "--output-dir",
        default="data/factorial_experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.001,
        help="Minimum triangle area threshold (default: 0.001)"
    )
    parser.add_argument(
        "--max-triads",
        type=int,
        default=None,
        help="Maximum number of triads to run (default: None = all)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    setup_logging(args.log_level, Path(args.output_dir) / "logs")

    runner = FactorialExperimentRunner(
        output_dir=args.output_dir,
        openai_api_key=args.openai_api_key,
        min_area=args.min_area,
        max_triads=args.max_triads
    )

    try:
        summary = runner.run()
        print("\n" + "="*60)
        print("FACTORIAL EXPERIMENT SUMMARY")
        print("="*60)
        print(json.dumps(summary, indent=2))
        print("="*60 + "\n")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
