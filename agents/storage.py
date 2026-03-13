"""
Storage: DuckDB tables and CSV export for experiment data.
Stores triads, proposals, convergence results, and predictions.
"""
 
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import duckdb
import pandas as pd
 
logger = logging.getLogger(__name__)
 
 
class ExperimentStorage:
    """Manages storage for multi-agent experiments."""
 
    def __init__(self, db_path: str = "data/db/agent_experiments.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()
 
    def _init_schema(self):
        """Create database tables."""
 
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS proposal_id_seq START 1;")
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS synthesis_id_seq START 1;")
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS result_id_seq START 1;")
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS llm_interaction_id_seq START 1;")
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS source_geometry_id_seq START 1;")
 
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS triads (
                triad_id INTEGER PRIMARY KEY,
                historian_1_name VARCHAR,
                historian_2_name VARCHAR,
                historian_3_name VARCHAR,
                historian_1_id VARCHAR,
                historian_2_id VARCHAR,
                historian_3_id VARCHAR,
                side_1 DOUBLE,
                side_2 DOUBLE,
                side_3 DOUBLE,
                perimeter DOUBLE,
                area DOUBLE,
                min_angle DOUBLE,
                max_angle DOUBLE,
                angle_variance DOUBLE,
                retrieval_query VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
 
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS proposals (
                proposal_id INTEGER PRIMARY KEY DEFAULT nextval('proposal_id_seq'),
                triad_id INTEGER,
                historian_name VARCHAR,
                historian_position INTEGER,
                research_question VARCHAR,
                abstract TEXT,
                selected_sources VARCHAR,
                text_source_ids VARCHAR,
                image_source_ids VARCHAR,
                n_text_sources INTEGER,
                n_image_sources INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (triad_id) REFERENCES triads(triad_id)
            );
        """)
 
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS synthesis (
                synthesis_id INTEGER PRIMARY KEY DEFAULT nextval('synthesis_id_seq'),
                triad_id INTEGER,
                final_research_question VARCHAR,
                final_abstract TEXT,
                final_sources VARCHAR,
                all_text_source_ids VARCHAR,
                all_image_source_ids VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (triad_id) REFERENCES triads(triad_id)
            );
        """)
 
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS convergence_results (
                result_id INTEGER PRIMARY KEY DEFAULT nextval('result_id_seq'),
                triad_id INTEGER,
 
                -- Historian distances to centroid
                distance_hist1_to_centroid DOUBLE,
                distance_hist2_to_centroid DOUBLE,
                distance_hist3_to_centroid DOUBLE,
                mean_historian_distance DOUBLE,
 
                -- Abstract distances to centroid
                distance_abstract1_to_centroid DOUBLE,
                distance_abstract2_to_centroid DOUBLE,
                distance_abstract3_to_centroid DOUBLE,
                mean_abstract_distance DOUBLE,
 
                -- Final abstract distance
                distance_final_to_centroid DOUBLE,
 
                -- Convergence outcome
                converged BOOLEAN,
                convergence_delta DOUBLE,
 
                -- Additional stats
                abstract_distance_variance DOUBLE,
                mean_pairwise_abstract_similarity DOUBLE,
 
                -- Bias analysis
                bias_weight_1 DOUBLE,
                bias_weight_2 DOUBLE,
                bias_weight_3 DOUBLE,
                dominant_historian_position INTEGER,
                bias_score DOUBLE,

                -- Source embedding similarity features
                mean_source_embedding_distance DOUBLE,
                source_embedding_variance DOUBLE,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (triad_id) REFERENCES triads(triad_id)
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS llm_interactions (
                interaction_id INTEGER PRIMARY KEY DEFAULT nextval('llm_interaction_id_seq'),
                triad_id INTEGER,
                interaction_type VARCHAR,
                historian_name VARCHAR,
                historian_position INTEGER,
                system_prompt TEXT,
                user_prompt TEXT,
                llm_response TEXT,
                model_name VARCHAR,
                temperature DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS source_geometry (
                geometry_id INTEGER PRIMARY KEY DEFAULT nextval('source_geometry_id_seq'),
                triad_id INTEGER,

                -- Source IDs
                source_ids JSON,  -- List of all source_ids
                n_sources INTEGER,

                -- Full pairwise distance matrix
                distance_matrix JSON,  -- NxN matrix of cosine distances

                -- Distribution statistics
                distance_mean DOUBLE,
                distance_std DOUBLE,
                distance_min DOUBLE,
                distance_max DOUBLE,
                distance_p10 DOUBLE,
                distance_p25 DOUBLE,
                distance_p50 DOUBLE,
                distance_p75 DOUBLE,
                distance_p90 DOUBLE,

                -- Within vs between historian distances
                within_hist1_mean DOUBLE,
                within_hist2_mean DOUBLE,
                within_hist3_mean DOUBLE,
                between_hist12_mean DOUBLE,
                between_hist13_mean DOUBLE,
                between_hist23_mean DOUBLE,
                within_mean DOUBLE,  -- Average of within-historian distances
                between_mean DOUBLE,  -- Average of between-historian distances
                within_between_ratio DOUBLE,  -- within_mean / between_mean

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (triad_id) REFERENCES triads(triad_id)
            );
        """)

        self.con.commit()
        logger.info(f"Initialized experiment database at {self.db_path}")
 
    def insert_triad(
        self,
        triad_id: int,
        historian_names: List[str],
        historian_ids: List[str],
        geometry: Dict,
        retrieval_query: str
    ) -> None:
        self.con.execute("""
            INSERT INTO triads (
                triad_id,
                historian_1_name, historian_2_name, historian_3_name,
                historian_1_id, historian_2_id, historian_3_id,
                side_1, side_2, side_3, perimeter, area,
                min_angle, max_angle, angle_variance,
                retrieval_query
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            triad_id,
            historian_names[0], historian_names[1], historian_names[2],
            historian_ids[0], historian_ids[1], historian_ids[2],
            geometry['side_1'], geometry['side_2'], geometry['side_3'],
            geometry['perimeter'], geometry['area'],
            geometry['min_angle'], geometry['max_angle'],
            geometry['angle_variance'],
            retrieval_query
        ])
        self.con.commit()
 
    def insert_proposal(
        self,
        triad_id: int,
        historian_name: str,
        position: int,
        proposal: Dict,
        n_text_sources: int,
        n_image_sources: int,
        text_source_ids: Optional[List[str]] = None,
        image_source_ids: Optional[List[str]] = None
    ) -> None:
        # Convert source ID lists to comma-separated strings
        text_ids_str = ','.join(text_source_ids) if text_source_ids else ''
        image_ids_str = ','.join(image_source_ids) if image_source_ids else ''

        self.con.execute("""
            INSERT INTO proposals (
                triad_id, historian_name, historian_position,
                research_question, abstract, selected_sources,
                text_source_ids, image_source_ids,
                n_text_sources, n_image_sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            triad_id, historian_name, position,
            proposal['research_question'],
            proposal['abstract'],
            proposal['selected_sources'],
            text_ids_str, image_ids_str,
            n_text_sources, n_image_sources
        ])
        self.con.commit()
 
    def insert_synthesis(
        self,
        triad_id: int,
        synthesis: Dict,
        all_text_source_ids: Optional[List[str]] = None,
        all_image_source_ids: Optional[List[str]] = None
    ) -> None:
        # Convert source ID lists to comma-separated strings
        text_ids_str = ','.join(all_text_source_ids) if all_text_source_ids else ''
        image_ids_str = ','.join(all_image_source_ids) if all_image_source_ids else ''

        self.con.execute("""
            INSERT INTO synthesis (
                triad_id,
                final_research_question,
                final_abstract,
                final_sources,
                all_text_source_ids,
                all_image_source_ids
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, [
            triad_id,
            synthesis['final_research_question'],
            synthesis['final_abstract'],
            synthesis['final_sources'],
            text_ids_str,
            image_ids_str
        ])
        self.con.commit()
 
    def insert_convergence_result(
        self,
        triad_id: int,
        metrics: Dict,
        additional_stats: Dict
    ) -> None:
        self.con.execute("""
            INSERT INTO convergence_results (
                triad_id,
                distance_hist1_to_centroid,
                distance_hist2_to_centroid,
                distance_hist3_to_centroid,
                mean_historian_distance,
                distance_abstract1_to_centroid,
                distance_abstract2_to_centroid,
                distance_abstract3_to_centroid,
                mean_abstract_distance,
                distance_final_to_centroid,
                converged,
                convergence_delta,
                abstract_distance_variance,
                mean_pairwise_abstract_similarity,
                bias_weight_1,
                bias_weight_2,
                bias_weight_3,
                dominant_historian_position,
                bias_score,
                mean_source_embedding_distance,
                source_embedding_variance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            triad_id,
            metrics['distance_hist1_to_centroid'],
            metrics['distance_hist2_to_centroid'],
            metrics['distance_hist3_to_centroid'],
            metrics['mean_historian_distance'],
            metrics['distance_abstract1_to_centroid'],
            metrics['distance_abstract2_to_centroid'],
            metrics['distance_abstract3_to_centroid'],
            metrics['mean_abstract_distance'],
            metrics['distance_final_to_centroid'],
            metrics['converged'],
            additional_stats['convergence_delta'],
            additional_stats['abstract_distance_variance'],
            additional_stats['mean_pairwise_abstract_similarity'],
            additional_stats['bias_weight_1'],
            additional_stats['bias_weight_2'],
            additional_stats['bias_weight_3'],
            additional_stats['dominant_historian_position'],
            additional_stats['bias_score'],
            additional_stats.get('mean_source_embedding_distance', None),
            additional_stats.get('source_embedding_variance', None)
        ])
        self.con.commit()

    def insert_llm_interaction(
        self,
        triad_id: int,
        interaction_type: str,
        historian_name: str,
        historian_position: int,
        system_prompt: str,
        user_prompt: str,
        llm_response: str,
        model_name: str,
        temperature: float
    ) -> None:
        """Log an LLM interaction for checkpoint tracking."""
        self.con.execute("""
            INSERT INTO llm_interactions (
                triad_id, interaction_type, historian_name, historian_position,
                system_prompt, user_prompt, llm_response,
                model_name, temperature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            triad_id, interaction_type, historian_name, historian_position,
            system_prompt, user_prompt, llm_response,
            model_name, temperature
        ])
        self.con.commit()

    def insert_source_geometry(
        self,
        triad_id: int,
        source_ids: List[str],
        distance_matrix: List[List[float]],
        stats: Dict[str, float]
    ) -> None:
        """Store detailed source geometry information."""
        import json

        self.con.execute("""
            INSERT INTO source_geometry (
                triad_id, source_ids, n_sources,
                distance_matrix,
                distance_mean, distance_std, distance_min, distance_max,
                distance_p10, distance_p25, distance_p50, distance_p75, distance_p90,
                within_hist1_mean, within_hist2_mean, within_hist3_mean,
                between_hist12_mean, between_hist13_mean, between_hist23_mean,
                within_mean, between_mean, within_between_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            triad_id,
            json.dumps(source_ids),
            len(source_ids),
            json.dumps(distance_matrix),
            stats.get('distance_mean'),
            stats.get('distance_std'),
            stats.get('distance_min'),
            stats.get('distance_max'),
            stats.get('distance_p10'),
            stats.get('distance_p25'),
            stats.get('distance_p50'),
            stats.get('distance_p75'),
            stats.get('distance_p90'),
            stats.get('within_hist1_mean'),
            stats.get('within_hist2_mean'),
            stats.get('within_hist3_mean'),
            stats.get('between_hist12_mean'),
            stats.get('between_hist13_mean'),
            stats.get('between_hist23_mean'),
            stats.get('within_mean'),
            stats.get('between_mean'),
            stats.get('within_between_ratio')
        ])
        self.con.commit()

    def export_to_csv(self, output_dir: str = "data/agent_experiments"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for table in ['triads', 'proposals', 'synthesis', 'convergence_results', 'llm_interactions', 'source_geometry']:
            df = self.con.execute(f"SELECT * FROM {table}").fetchdf()
            output_path = output_dir / f"{table}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} rows to {output_path}")
 
    def get_convergence_data(self) -> pd.DataFrame:
        query = """
            SELECT
                t.triad_id,
                t.perimeter,
                t.area,
                t.angle_variance,
                c.mean_historian_distance,
                c.mean_abstract_distance,
                c.distance_final_to_centroid,
                c.converged,
                c.convergence_delta,
                c.abstract_distance_variance,
                c.mean_pairwise_abstract_similarity,
                c.bias_weight_1,
                c.bias_weight_2,
                c.bias_weight_3,
                c.dominant_historian_position,
                c.bias_score,
                c.mean_source_embedding_distance,
                c.source_embedding_variance
            FROM triads t
            JOIN convergence_results c ON t.triad_id = c.triad_id
            ORDER BY t.triad_id
        """
        return self.con.execute(query).fetchdf()
 
    def get_full_experiment_data(self) -> pd.DataFrame:
        query = """
            SELECT t.*, c.*
            FROM triads t
            LEFT JOIN convergence_results c ON t.triad_id = c.triad_id
            ORDER BY t.triad_id
        """
        return self.con.execute(query).fetchdf()
 
    def close(self):
        self.con.close()
 
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        self.close()