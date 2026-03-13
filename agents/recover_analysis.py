"""
Recovery Analysis Script: Compute missing convergence analysis from checkpointed data.

Use this when an experiment fails midway - recovers raw triad data and computes
all missing analyses (convergence, source geometry, prediction models).

Usage:
    python -m agents.recover_analysis --data-dir data/factorial_experiments
    python -m agents.recover_analysis --data-dir data/agent_experiments --force
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from agents.storage import ExperimentStorage
from agents.convergence_analysis import ConvergenceAnalyzer
from agents.prediction_model import ConvergencePredictionModel, run_inference_analysis, run_ablation_study
from agents.historian_manager import HistorianManager, HistorianPersona
from historian_pipeline.storage.corpus_store import CorpusStore
from historian_pipeline.embeddings.faiss_index import CorpusIndex

logger = logging.getLogger(__name__)


class RecoveryAnalyzer:
    """Recovers and completes analysis from interrupted experiments."""

    def __init__(self, data_dir: Path, force: bool = False):
        """
        Initialize recovery analyzer.

        Args:
            data_dir: Directory with experiment database
            force: If True, recomputes even if convergence data exists
        """
        self.data_dir = Path(data_dir)
        self.force = force

        db_path = self.data_dir / "experiments.duckdb"
        if not db_path.exists():
            raise FileNotFoundError(f"No database found at {db_path}")

        logger.info(f"Loading database from {db_path}...")
        self.storage = ExperimentStorage(db_path=str(db_path))

        # Initialize analysis components
        logger.info("Initializing analysis components...")
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.prediction_model = ConvergencePredictionModel()

        # Load corpus for source geometry
        self.corpus_store = CorpusStore()
        self.corpus_index = CorpusIndex()
        try:
            self.corpus_index.load()
            logger.info("Loaded corpus index")
        except Exception as e:
            logger.warning(f"Could not load corpus index: {e}")

    def check_status(self) -> Dict:
        """
        Check which data exists and what needs recovery.

        Returns:
            Dict with status information
        """
        # Count rows in each table
        triads_df = self.storage.con.execute("SELECT COUNT(*) as n FROM triads").fetchdf()
        proposals_df = self.storage.con.execute("SELECT COUNT(*) as n FROM proposals").fetchdf()
        synthesis_df = self.storage.con.execute("SELECT COUNT(*) as n FROM synthesis").fetchdf()
        convergence_df = self.storage.con.execute("SELECT COUNT(*) as n FROM convergence_results").fetchdf()
        source_geom_df = self.storage.con.execute("SELECT COUNT(*) as n FROM source_geometry").fetchdf()
        llm_df = self.storage.con.execute("SELECT COUNT(*) as n FROM llm_interactions").fetchdf()

        n_triads = int(triads_df['n'].iloc[0])
        n_convergence = int(convergence_df['n'].iloc[0])
        n_source_geom = int(source_geom_df['n'].iloc[0])

        status = {
            'n_triads': n_triads,
            'n_proposals': int(proposals_df['n'].iloc[0]),
            'n_synthesis': int(synthesis_df['n'].iloc[0]),
            'n_convergence': n_convergence,
            'n_source_geometry': n_source_geom,
            'n_llm_interactions': int(llm_df['n'].iloc[0]),
            'needs_recovery': n_triads > 0 and (n_convergence < n_triads or n_source_geom < n_triads),
            'missing_convergence': max(0, n_triads - n_convergence),
            'missing_source_geometry': max(0, n_triads - n_source_geom)
        }

        return status

    def recover_triad_data(self, triad_id: int) -> Tuple[List[HistorianPersona], List[Dict], Dict, List[Dict]]:
        """
        Reconstruct triad data from database.

        Args:
            triad_id: Triad ID to recover

        Returns:
            Tuple of (historians, proposals, synthesis, source_packets)
        """
        # Get triad info
        triad_df = self.storage.con.execute(
            "SELECT * FROM triads WHERE triad_id = ?", [triad_id]
        ).fetchdf()

        if len(triad_df) == 0:
            raise ValueError(f"Triad {triad_id} not found")

        triad = triad_df.iloc[0]

        # Reconstruct historian personas (minimal - just need embeddings and names)
        # Load from historian manager to get embeddings
        hist_manager = HistorianManager(n_historians=25)
        hist_manager.create_historian_personas()
        hist_manager.compute_historian_embeddings()

        # Match by historian_id
        historian_ids = [triad['historian_1_id'], triad['historian_2_id'], triad['historian_3_id']]
        historians = []
        for hist_id in historian_ids:
            # Find matching persona
            matching = [p for p in hist_manager.personas if p.historian_id == hist_id]
            if matching:
                historians.append(matching[0])
            else:
                logger.warning(f"Could not find historian {hist_id}")
                return None, None, None, None

        # Get proposals
        proposals_df = self.storage.con.execute(
            "SELECT * FROM proposals WHERE triad_id = ? ORDER BY historian_position",
            [triad_id]
        ).fetchdf()

        proposals = []
        source_packets = []
        for _, prop in proposals_df.iterrows():
            proposals.append({
                'research_question': prop['research_question'],
                'abstract': prop['abstract'],
                'selected_sources': prop['selected_sources']
            })

            # Reconstruct source packets from stored IDs
            text_ids = prop['text_source_ids'].split(',') if prop['text_source_ids'] else []
            image_ids = prop['image_source_ids'].split(',') if prop['image_source_ids'] else []

            source_packets.append({
                'text_sources': [{'source_id': sid} for sid in text_ids if sid],
                'image_sources': [{'source_id': sid} for sid in image_ids if sid]
            })

        # Get synthesis
        synthesis_df = self.storage.con.execute(
            "SELECT * FROM synthesis WHERE triad_id = ?", [triad_id]
        ).fetchdf()

        if len(synthesis_df) == 0:
            logger.warning(f"No synthesis found for triad {triad_id}")
            return None, None, None, None

        synthesis = {
            'final_research_question': synthesis_df.iloc[0]['final_research_question'],
            'final_abstract': synthesis_df.iloc[0]['final_abstract'],
            'final_sources': synthesis_df.iloc[0]['final_sources']
        }

        return historians, proposals, synthesis, source_packets

    def compute_source_geometry(self, triad_id: int, source_packets: List[Dict]) -> Tuple[Dict, List, List, Dict]:
        """Compute source geometry for a triad."""
        # Use same logic as run_experiment
        try:
            historian_sources = [
                [src['source_id'] for src in packet['text_sources']] +
                [src['source_id'] for src in packet['image_sources']]
                for packet in source_packets
            ]

            all_source_ids = [sid for hist_sources in historian_sources for sid in hist_sources]

            # Load embeddings
            embeddings = []
            valid_source_ids = []
            embeddings_dir = Path("data/embeddings")

            for source_id in all_source_ids:
                emb_path = embeddings_dir / source_id[:2] / f"{source_id}.npy"
                if emb_path.exists():
                    emb = np.load(emb_path)
                    embeddings.append(emb)
                    valid_source_ids.append(source_id)

            if len(embeddings) < 2:
                return (
                    {'mean_source_embedding_distance': None, 'source_embedding_variance': None},
                    [], [], {}
                )

            # Compute distances
            embeddings = np.array(embeddings)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = np.dot(embeddings, embeddings.T)
            distances = 1.0 - similarities

            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            pairwise_distances = distances[mask]

            simple_stats = {
                'mean_source_embedding_distance': float(np.mean(pairwise_distances)),
                'source_embedding_variance': float(np.var(pairwise_distances))
            }

            # Distribution stats
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

            # Within/between stats
            id_to_idx = {sid: idx for idx, sid in enumerate(valid_source_ids)}
            hist_indices = [[id_to_idx[sid] for sid in hist_sources if sid in id_to_idx]
                           for hist_sources in historian_sources]

            within_distances = [[], [], []]
            for h_idx, indices in enumerate(hist_indices):
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        within_distances[h_idx].append(distances[indices[i], indices[j]])

            detailed_stats['within_hist1_mean'] = float(np.mean(within_distances[0])) if within_distances[0] else None
            detailed_stats['within_hist2_mean'] = float(np.mean(within_distances[1])) if within_distances[1] else None
            detailed_stats['within_hist3_mean'] = float(np.mean(within_distances[2])) if within_distances[2] else None

            between_12, between_13, between_23 = [], [], []
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

            all_within = [d for within in within_distances for d in within]
            all_between = between_12 + between_13 + between_23

            detailed_stats['within_mean'] = float(np.mean(all_within)) if all_within else None
            detailed_stats['between_mean'] = float(np.mean(all_between)) if all_between else None

            if detailed_stats['between_mean'] and detailed_stats['between_mean'] > 0:
                detailed_stats['within_between_ratio'] = detailed_stats['within_mean'] / detailed_stats['between_mean']
            else:
                detailed_stats['within_between_ratio'] = None

            distance_matrix = distances.tolist()

            return (simple_stats, valid_source_ids, distance_matrix, detailed_stats)

        except Exception as e:
            logger.warning(f"Failed to compute source geometry for triad {triad_id}: {e}")
            return (
                {'mean_source_embedding_distance': None, 'source_embedding_variance': None},
                [], [], {}
            )

    def recover_convergence_analysis(self):
        """Recover convergence analysis for all triads missing it."""
        # Get triads that need analysis
        if self.force:
            query = "SELECT triad_id FROM triads ORDER BY triad_id"
            # Delete existing convergence data if forcing
            self.storage.con.execute("DELETE FROM convergence_results")
            self.storage.con.execute("DELETE FROM source_geometry")
            self.storage.con.commit()
        else:
            query = """
                SELECT t.triad_id
                FROM triads t
                LEFT JOIN convergence_results c ON t.triad_id = c.triad_id
                WHERE c.triad_id IS NULL
                ORDER BY t.triad_id
            """

        triad_ids_df = self.storage.con.execute(query).fetchdf()
        triad_ids = triad_ids_df['triad_id'].tolist()

        if len(triad_ids) == 0:
            logger.info("No triads need recovery")
            return

        logger.info(f"Recovering convergence analysis for {len(triad_ids)} triads...")

        for i, triad_id in enumerate(triad_ids, 1):
            logger.info(f"  Processing triad {i}/{len(triad_ids)} (ID={triad_id})...")

            try:
                # Reconstruct triad data
                historians, proposals, synthesis, source_packets = self.recover_triad_data(triad_id)

                if historians is None:
                    logger.warning(f"  Skipping triad {triad_id} - incomplete data")
                    continue

                # Compute convergence metrics
                historian_embeddings = tuple([h.embedding for h in historians])
                individual_abstracts = [p['abstract'] for p in proposals]
                final_abstract = synthesis['final_abstract']

                metrics = self.convergence_analyzer.compute_convergence_metrics(
                    historian_embeddings=historian_embeddings,
                    individual_abstracts=individual_abstracts,
                    final_abstract=final_abstract
                )
                additional_stats = self.convergence_analyzer.compute_embedding_stats(metrics)

                # Compute source geometry
                (simple_stats, source_ids, distance_matrix, detailed_stats) = self.compute_source_geometry(
                    triad_id, source_packets
                )
                additional_stats.update(simple_stats)

                # Store results
                self.storage.insert_convergence_result(
                    triad_id=triad_id,
                    metrics=metrics.to_dict(),
                    additional_stats=additional_stats
                )

                if source_ids and distance_matrix:
                    self.storage.insert_source_geometry(
                        triad_id=triad_id,
                        source_ids=source_ids,
                        distance_matrix=distance_matrix,
                        stats=detailed_stats
                    )

                logger.info(f"  OK Triad {triad_id} recovered")

            except Exception as e:
                logger.error(f"  Failed to recover triad {triad_id}: {e}", exc_info=True)

        logger.info("Convergence recovery complete")

    def run_analyses(self, output_dir: Path):
        """Run all downstream analyses on recovered data."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export to CSV
        logger.info("Exporting to CSV...")
        self.storage.export_to_csv(str(output_dir))

        # Get convergence data
        df = self.storage.get_convergence_data()

        if len(df) < 3:
            logger.warning("Insufficient data for analyses (need at least 3 triads)")
            return

        # Prediction model
        logger.info("Training prediction model...")
        try:
            X, feature_names = self.prediction_model.extract_features(df)
            y = df['convergence_delta'].values
            results = self.prediction_model.fit(X, y, feature_names)

            self.prediction_model.save_model(str(output_dir / "prediction_model.json"))

            importance_df = self.prediction_model.analyze_feature_importance()
            importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

            logger.info(f"  R²={results['r2']:.3f}, MAE={results['mae']:.4f}")
        except Exception as e:
            logger.error(f"Prediction model failed: {e}")

        # Inference analysis
        logger.info("Running inference analysis...")
        try:
            inference_results = run_inference_analysis(df)
            with open(output_dir / "inference_results.json", 'w') as f:
                json.dump(inference_results, f, indent=2)
        except Exception as e:
            logger.error(f"Inference analysis failed: {e}")

        # Ablation study
        logger.info("Running ablation study...")
        try:
            ablation_results = run_ablation_study(df)
            with open(output_dir / "ablation_study.json", 'w') as f:
                json.dump(ablation_results, f, indent=2)

            if 'source_improvement' in ablation_results:
                logger.info(f"  Source features ΔR²={ablation_results['source_improvement']['delta_r2']:.4f}")
        except Exception as e:
            logger.error(f"Ablation study failed: {e}")

        logger.info("All analyses complete")

    def close(self):
        """Clean up resources."""
        self.storage.close()
        self.corpus_store.close()


def main():
    parser = argparse.ArgumentParser(
        description="Recover convergence analysis from interrupted experiments"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing experiments.duckdb"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute convergence even if it exists"
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
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    data_dir = Path(args.data_dir)

    try:
        # Initialize recovery
        analyzer = RecoveryAnalyzer(data_dir, force=args.force)

        # Check status
        status = analyzer.check_status()

        print("\n" + "="*60)
        print("RECOVERY STATUS")
        print("="*60)
        print(f"Triads: {status['n_triads']}")
        print(f"Proposals: {status['n_proposals']}")
        print(f"Synthesis: {status['n_synthesis']}")
        print(f"Convergence: {status['n_convergence']}")
        print(f"Source Geometry: {status['n_source_geometry']}")
        print(f"LLM Interactions: {status['n_llm_interactions']}")
        print(f"\nMissing convergence: {status['missing_convergence']}")
        print(f"Missing source geometry: {status['missing_source_geometry']}")
        print("="*60 + "\n")

        if not status['needs_recovery'] and not args.force:
            print("No recovery needed! All data complete.")
            return

        # Recover convergence analysis
        analyzer.recover_convergence_analysis()

        # Run analyses
        analyzer.run_analyses(data_dir)

        print("\n" + "="*60)
        print("RECOVERY COMPLETE")
        print("="*60)
        print(f"Results saved to: {data_dir}")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Recovery failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'analyzer' in locals():
            analyzer.close()


if __name__ == "__main__":
    main()
