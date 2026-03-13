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
from typing import Dict, List, Tuple
from itertools import combinations

from agents.historian_manager import HistorianManager
from agents.source_retrieval import SourceRetriever
from agents.agent_llm import AgentLLM
from agents.interaction_pipeline import InteractionPipeline, TriadExperimentResult
from agents.visualization import SynthesisAnalyzer
from agents.storage import ExperimentStorage
from agents.visualization import ConvergencePredictionModel, run_inference_analysis, SynthesisAnalyzer

from sources.storage.corpus_store import CorpusStore
from sources.embeddings.faiss_index import CorpusIndex

logger = logging.getLogger(__name__)


class FactorialExperimentRunner:
    """Runs experiments on all possible triad combinations."""

    def __init__(
        self,
        output_dir: str = "data/factorial_experiments",
        openai_api_key: str = None,
        min_area: float = 0.001,
        max_triads: int = None,
        reanalyze_only: bool = False
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

        # Delete existing database to start fresh (skip if reanalyzing)
        db_path = self.output_dir / "experiments.duckdb"
        if db_path.exists() and not reanalyze_only:
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
        self.synthesis_analyzer = SynthesisAnalyzer()
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

        logger.info("Step 6: Running ablation study (source embeddings)...")
        ablation_results = self._run_ablation_study()
        logger.info(f"Ablation study complete\n")

        elapsed = time.time() - t0
        summary = {
            'n_triads_attempted': len(triads_with_geometry),
            'n_triads_successful': len([r for r in results if r.success]),
            'elapsed_seconds': round(elapsed, 1),
            'output_dir': str(self.output_dir),
            'min_area_filter': self.min_area,
            'max_triads_limit': self.max_triads,
            'prediction_results': prediction_results,
            'inference_results': inference_results,
            'ablation_results': ablation_results
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
            zip(result.historian_names, result.hypotheses, result.source_packets), 1
        ):
            # Extract source IDs from packet
            text_source_ids = [src['source_id'] for src in packet['text_sources']]
            image_source_ids = [src['source_id'] for src in packet['image_sources']]

            all_text_ids.extend(text_source_ids)
            all_image_ids.extend(image_source_ids)

            # Remap hypothesis key to match DB schema (research_question column)
            proposal_for_storage = dict(proposal)
            if 'hypothesis' in proposal_for_storage and 'research_question' not in proposal_for_storage:
                proposal_for_storage['research_question'] = proposal_for_storage.pop('hypothesis')

            self.storage.insert_proposal(
                triad_id=result.triad_id,
                historian_name=name,
                position=i,
                proposal=proposal_for_storage,
                n_text_sources=len(packet['text_sources']),
                n_image_sources=len(packet['image_sources']),
                text_source_ids=text_source_ids,
                image_source_ids=image_source_ids
            )

        # Remap synthesis keys to match DB schema
        synthesis_for_storage = dict(result.synthesis)
        if 'final_hypothesis' in synthesis_for_storage and 'final_research_question' not in synthesis_for_storage:
            synthesis_for_storage['final_research_question'] = synthesis_for_storage.pop('final_hypothesis')

        self.storage.insert_synthesis(
            triad_id=result.triad_id,
            synthesis=synthesis_for_storage,
            all_text_source_ids=all_text_ids,
            all_image_source_ids=all_image_ids
        )

    def _analyze_convergence(self, results: List[TriadExperimentResult]):
        """Analyze convergence for all results."""
        for result in results:
            if not result.success:
                continue
            try:
                historian_embeddings = tuple([h.embedding for h in result.historians])
                individual_abstracts = [p['abstract'] for p in result.hypotheses]
                final_abstract = result.synthesis['final_abstract']

                metrics = self.synthesis_analyzer.compute_synthesis_metrics(
                    historian_embeddings=historian_embeddings,
                    individual_abstracts=individual_abstracts,
                    final_abstract=final_abstract
                )
                additional_stats = self.synthesis_analyzer.compute_embedding_stats(metrics)

                # Compute comprehensive source geometry
                (simple_stats, source_ids, distance_matrix, detailed_stats) = self._compute_source_geometry(
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



    def _compute_source_geometry(
        self, triad_id: int, source_packets: List[Dict]
    ) -> Tuple[Dict[str, float], List[str], List[List[float]], Dict[str, float]]:
        """
        Compute comprehensive source geometry including full distance matrix and statistics.
 
        Args:
            triad_id: Triad ID
            source_packets: List of 3 source packet dicts with text_sources and image_sources
 
        Returns:
            Tuple of (simple_stats, source_ids, distance_matrix, detailed_stats)
            - simple_stats: Dict with mean_source_embedding_distance and source_embedding_variance (for backward compat)
            - source_ids: List of all source IDs
            - distance_matrix: NxN distance matrix
            - detailed_stats: Dict with distribution and within/between stats
        """
        import numpy as np
 
        try:
            # Collect source IDs by historian
            historian_sources = [
                [src['source_id'] for src in packet['text_sources']] +
                [src['source_id'] for src in packet['image_sources']]
                for packet in source_packets
            ]
 
            all_source_ids = [sid for hist_sources in historian_sources for sid in hist_sources]
 
            # Load embeddings for all sources from disk
            embeddings = []
            valid_source_ids = []
            embeddings_dir = Path("data/embeddings")
 
            for source_id in all_source_ids:
                # Embeddings are stored as data/embeddings/XX/source_id.npy
                # where XX is first 2 chars of source_id
                emb_path = embeddings_dir / source_id[:2] / f"{source_id}.npy"
                if emb_path.exists():
                    emb = np.load(emb_path)
                    embeddings.append(emb)
                    valid_source_ids.append(source_id)
 
            if len(embeddings) < 2:
                return (
                    {'mean_source_embedding_distance': None, 'source_embedding_variance': None},
                    [],
                    [],
                    {}
                )
 
            # Compute pairwise cosine distances
            embeddings = np.array(embeddings)
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
 
            # Compute all pairwise cosine similarities
            similarities = np.dot(embeddings, embeddings.T)
            # Convert to distances (1 - similarity)
            distances = 1.0 - similarities
 
            # Get upper triangle (excluding diagonal) for summary stats
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            pairwise_distances = distances[mask]
 
            # Simple stats for backward compatibility
            simple_stats = {
                'mean_source_embedding_distance': float(np.mean(pairwise_distances)),
                'source_embedding_variance': float(np.var(pairwise_distances))
            }
 
            # Distribution statistics
            detailed_stats = {
                'distance_mean': float(np.mean(pairwise_distances)),
                'distance_std': float(np.std(pairwise_distances)),
                'distance_min': float(np.min(pairwise_distances)),
                'distance_max': float(np.max(pairwise_distances)),
                'distance_p10': float(np.percentile(pairwise_distances, 10)),
                'distance_p25': float(np.percentile(pairwise_distances, 25)),
                'distance_p50': float(np.percentile(pairwise_distances, 50)),
                'distance_p75': float(np.percentile(pairwise_distances, 75)),
                'distance_p90': float(np.percentile(pairwise_distances, 90)),
            }
 
            # Within vs between historian distances
            # Build mapping of source_id to index
            id_to_idx = {sid: idx for idx, sid in enumerate(valid_source_ids)}
 
            # Get indices for each historian's sources
            hist_indices = []
            for hist_sources in historian_sources:
                indices = [id_to_idx[sid] for sid in hist_sources if sid in id_to_idx]
                hist_indices.append(indices)
 
            # Compute within-historian distances
            within_distances = [[], [], []]
            for h_idx, indices in enumerate(hist_indices):
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        within_distances[h_idx].append(distances[indices[i], indices[j]])
 
            detailed_stats['within_hist1_mean'] = float(np.mean(within_distances[0])) if within_distances[0] else None
            detailed_stats['within_hist2_mean'] = float(np.mean(within_distances[1])) if within_distances[1] else None
            detailed_stats['within_hist3_mean'] = float(np.mean(within_distances[2])) if within_distances[2] else None
 
            # Compute between-historian distances
            between_12 = []
            between_13 = []
            between_23 = []
 
            for i in hist_indices[0]:
                for j in hist_indices[1]:
                    between_12.append(distances[i, j])
 
            for i in hist_indices[0]:
                for j in hist_indices[2]:
                    between_13.append(distances[i, j])
 
            for i in hist_indices[1]:
                for j in hist_indices[2]:
                    between_23.append(distances[i, j])
 
            detailed_stats['between_hist12_mean'] = float(np.mean(between_12)) if between_12 else None
            detailed_stats['between_hist13_mean'] = float(np.mean(between_13)) if between_13 else None
            detailed_stats['between_hist23_mean'] = float(np.mean(between_23)) if between_23 else None
 
            # Overall within vs between
            all_within = [d for within in within_distances for d in within]
            all_between = between_12 + between_13 + between_23
 
            detailed_stats['within_mean'] = float(np.mean(all_within)) if all_within else None
            detailed_stats['between_mean'] = float(np.mean(all_between)) if all_between else None
 
            if detailed_stats['between_mean'] and detailed_stats['between_mean'] > 0:
                detailed_stats['within_between_ratio'] = detailed_stats['within_mean'] / detailed_stats['between_mean']
            else:
                detailed_stats['within_between_ratio'] = None
 
            # Convert distance matrix to list of lists for JSON serialization
            distance_matrix = distances.tolist()
 
            return (simple_stats, valid_source_ids, distance_matrix, detailed_stats)
 
        except Exception as e:
            logger.warning(f"Failed to compute source geometry: {e}", exc_info=True)
            return (
                {'mean_source_embedding_distance': None, 'source_embedding_variance': None},
                [],
                [],
                {}
            )
 
    def _reanalyze_from_db(self):
        """Re-run convergence analysis using stored abstracts from DB (no API calls)."""
        import numpy as np

        con = self.storage.con

        # Load triads
        triads_df = con.execute("SELECT * FROM triads ORDER BY triad_id").fetchdf()
        logger.info(f"Re-analyzing {len(triads_df)} triads from DB...")

        # Load proposals (abstracts per historian per triad)
        proposals_df = con.execute(
            "SELECT triad_id, historian_position, abstract FROM proposals ORDER BY triad_id, historian_position"
        ).fetchdf()

        # Load synthesis abstracts
        synthesis_df = con.execute(
            "SELECT triad_id, final_abstract FROM synthesis ORDER BY triad_id"
        ).fetchdf()

        success = 0
        for _, triad_row in triads_df.iterrows():
            triad_id = int(triad_row['triad_id'])
            try:
                # Get historian names and recompute embeddings
                names = [
                    triad_row['historian_1_name'],
                    triad_row['historian_2_name'],
                    triad_row['historian_3_name'],
                ]
                historians = [
                    self.historian_manager.get_persona_by_name(n) for n in names
                ]
                if any(h is None for h in historians):
                    logger.warning(f"Triad {triad_id}: could not find all historians, skipping")
                    continue

                historian_embeddings = tuple(h.embedding for h in historians)

                # Get individual abstracts in position order
                triad_proposals = proposals_df[proposals_df['triad_id'] == triad_id].sort_values('historian_position')
                if len(triad_proposals) < 3:
                    logger.warning(f"Triad {triad_id}: only {len(triad_proposals)} proposals found, skipping")
                    continue
                individual_abstracts = triad_proposals['abstract'].tolist()

                # Get final abstract
                triad_synthesis = synthesis_df[synthesis_df['triad_id'] == triad_id]
                if triad_synthesis.empty:
                    logger.warning(f"Triad {triad_id}: no synthesis found, skipping")
                    continue
                final_abstract = triad_synthesis.iloc[0]['final_abstract']

                # Compute metrics
                metrics = self.synthesis_analyzer.compute_synthesis_metrics(
                    historian_embeddings=historian_embeddings,
                    individual_abstracts=individual_abstracts,
                    final_abstract=final_abstract
                )
                additional_stats = self.synthesis_analyzer.compute_embedding_stats(metrics)

                # Store
                self.storage.insert_convergence_result(
                    triad_id=triad_id,
                    metrics=metrics.to_dict(),
                    additional_stats=additional_stats
                )
                success += 1
                logger.info(
                    f"  Triad {triad_id}: delta={additional_stats['convergence_delta']:.4f}, "
                    f"bias_score={additional_stats['bias_score']:.3f}"
                )

            except Exception as e:
                logger.error(f"Triad {triad_id} reanalysis failed: {e}", exc_info=True)

        logger.info(f"Re-analysis complete: {success}/{len(triads_df)} triads stored")

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

    def _run_ablation_study(self) -> dict:
        df = self.storage.get_convergence_data()

        if len(df) < 4:
            logger.warning("Insufficient data for ablation study")
            return {'error': 'insufficient_data'}

        from agents.visualization import run_ablation_study
        results = run_ablation_study(df)

        with open(self.output_dir / "ablation_study.json", 'w') as f:
            json.dump(results, f, indent=2)

        if 'source_improvement' in results:
            logger.info(f"  Source features ΔR²={results['source_improvement']['delta_r2']:.4f}")

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
    parser.add_argument(
        "--reanalyze-only",
        action="store_true",
        help="Skip experiment; re-run convergence analysis + downstream steps on stored DB data"
    )
    args = parser.parse_args()

    setup_logging(args.log_level, Path(args.output_dir) / "logs")

    runner = FactorialExperimentRunner(
        output_dir=args.output_dir,
        openai_api_key=args.openai_api_key,
        min_area=args.min_area,
        max_triads=args.max_triads,
        reanalyze_only=args.reanalyze_only
    )

    try:
        if args.reanalyze_only:
            # Safety check — abort if DB is empty to avoid wiping CSVs
            n_triads = runner.storage.con.execute("SELECT COUNT(*) FROM triads").fetchone()[0]
            if n_triads == 0:
                print("ERROR: DB is empty — nothing to reanalyze. Re-run without --reanalyze-only.")
                runner.close()
                return
            runner._reanalyze_from_db()
            runner.storage.export_to_csv(str(runner.output_dir))
            prediction_results = runner._train_prediction_model()
            inference_results  = runner._run_inference_analysis()
            ablation_results   = runner._run_ablation_study()
            summary = {
                'mode': 'reanalyze_only',
                'output_dir': str(runner.output_dir),
                'prediction_results': prediction_results,
                'inference_results':  inference_results,
                'ablation_results':   ablation_results,
            }
        else:
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