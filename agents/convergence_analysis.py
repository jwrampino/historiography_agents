"""
Convergence Analysis: Computes embeddings and convergence metrics.
Determines whether a triad converged toward a shared interpretation,
and computes bias scores showing which historian the synthesis leaned toward.
"""
 
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class ConvergenceMetrics:
    """Container for convergence analysis results."""
    # Historian embeddings
    historian_1_embedding: np.ndarray
    historian_2_embedding: np.ndarray
    historian_3_embedding: np.ndarray
    centroid_embedding: np.ndarray
 
    # Individual abstract embeddings
    abstract_1_embedding: np.ndarray
    abstract_2_embedding: np.ndarray
    abstract_3_embedding: np.ndarray
 
    # Final synthesis embedding
    final_abstract_embedding: np.ndarray
 
    # Distances to centroid
    distance_hist1_to_centroid: float
    distance_hist2_to_centroid: float
    distance_hist3_to_centroid: float
    mean_historian_distance: float
 
    distance_abstract1_to_centroid: float
    distance_abstract2_to_centroid: float
    distance_abstract3_to_centroid: float
    mean_abstract_distance: float
 
    distance_final_to_centroid: float
 
    # Convergence outcome
    converged: bool
 
    # Bias: similarity of final abstract to each individual abstract [w1, w2, w3], sums to 1
    bias_weights: np.ndarray = field(default_factory=lambda: np.array([1/3, 1/3, 1/3]))
 
    # Which historian dominated (1, 2, or 3), and by how much above equal share
    dominant_historian_position: int = 0
    bias_score: float = 0.0  # max weight - 1/3; 0 = perfectly balanced, 0.67 = fully dominated
 
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding embeddings for serialization)."""
        return {
            'distance_hist1_to_centroid': float(self.distance_hist1_to_centroid),
            'distance_hist2_to_centroid': float(self.distance_hist2_to_centroid),
            'distance_hist3_to_centroid': float(self.distance_hist3_to_centroid),
            'mean_historian_distance': float(self.mean_historian_distance),
            'distance_abstract1_to_centroid': float(self.distance_abstract1_to_centroid),
            'distance_abstract2_to_centroid': float(self.distance_abstract2_to_centroid),
            'distance_abstract3_to_centroid': float(self.distance_abstract3_to_centroid),
            'mean_abstract_distance': float(self.mean_abstract_distance),
            'distance_final_to_centroid': float(self.distance_final_to_centroid),
            'converged': bool(self.converged),
            'bias_weight_1': float(self.bias_weights[0]),
            'bias_weight_2': float(self.bias_weights[1]),
            'bias_weight_3': float(self.bias_weights[2]),
            'dominant_historian_position': int(self.dominant_historian_position),
            'bias_score': float(self.bias_score)
        }
 
 
class ConvergenceAnalyzer:
    """Analyzes convergence and bias in historian triad interactions."""
 
    # Convergence threshold: final abstract must be within this fraction
    # of mean individual abstract distance to count as converged.
    # 0.6 requires genuine centering rather than trivial averaging.
    CONVERGENCE_THRESHOLD = 0.6
 
    def __init__(self):
        """Initialize analyzer with text embedder."""
        self._text_embedder = None
 
    @property
    def text_embedder(self):
        """Lazy load text embedder."""
        if self._text_embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._text_embedder = SentenceTransformer(
                    'sentence-transformers/all-mpnet-base-v2'
                )
                logger.info("Loaded text embedding model for convergence analysis")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers required: "
                    "pip install sentence-transformers"
                )
        return self._text_embedder
 
    def compute_convergence_metrics(
        self,
        historian_embeddings: Tuple[np.ndarray, np.ndarray, np.ndarray],
        individual_abstracts: List[str],
        final_abstract: str
    ) -> ConvergenceMetrics:
        """
        Compute full convergence metrics and bias scores for a triad.
 
        Args:
            historian_embeddings: Tuple of 3 historian position embeddings
            individual_abstracts: List of 3 individual abstract texts
            final_abstract: Final merged abstract text
 
        Returns:
            ConvergenceMetrics object
        """
        h1_emb, h2_emb, h3_emb = historian_embeddings
 
        # Compute historian centroid
        centroid = self._compute_centroid([h1_emb, h2_emb, h3_emb])
 
        # Embed abstracts
        logger.info("Embedding individual abstracts...")
        abstract_embeddings = self.text_embedder.encode(
            individual_abstracts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        a1_emb, a2_emb, a3_emb = [
            abstract_embeddings[i].astype(np.float32) for i in range(3)
        ]
 
        logger.info("Embedding final abstract...")
        final_emb = self.text_embedder.encode(
            [final_abstract],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].astype(np.float32)
 
        # Distances to centroid
        d_h1 = self._cosine_distance(h1_emb, centroid)
        d_h2 = self._cosine_distance(h2_emb, centroid)
        d_h3 = self._cosine_distance(h3_emb, centroid)
        mean_h_dist = float(np.mean([d_h1, d_h2, d_h3]))
 
        d_a1 = self._cosine_distance(a1_emb, centroid)
        d_a2 = self._cosine_distance(a2_emb, centroid)
        d_a3 = self._cosine_distance(a3_emb, centroid)
        mean_a_dist = float(np.mean([d_a1, d_a2, d_a3]))
 
        d_final = self._cosine_distance(final_emb, centroid)
 
        # Convergence decision
        converged = d_final < mean_a_dist * self.CONVERGENCE_THRESHOLD
 
        # --- Bias analysis ---
        # Similarity of final abstract to each individual abstract
        sim_to_1 = float(np.dot(final_emb, a1_emb))
        sim_to_2 = float(np.dot(final_emb, a2_emb))
        sim_to_3 = float(np.dot(final_emb, a3_emb))
 
        # Softmax to get normalized weights summing to 1
        raw = np.array([sim_to_1, sim_to_2, sim_to_3])
        exp_raw = np.exp(raw - raw.max())  # numerically stable softmax
        bias_weights = exp_raw / exp_raw.sum()
 
        dominant_pos = int(np.argmax(bias_weights)) + 1  # 1-indexed
        bias_score = float(bias_weights.max() - 1/3)     # 0 = balanced, ~0.67 = fully dominated
 
        logger.info(
            f"Convergence analysis: mean_abstract_dist={mean_a_dist:.4f}, "
            f"final_dist={d_final:.4f}, "
            f"threshold={mean_a_dist * self.CONVERGENCE_THRESHOLD:.4f}, "
            f"converged={converged} | "
            f"bias_weights=[{bias_weights[0]:.2f}, {bias_weights[1]:.2f}, {bias_weights[2]:.2f}], "
            f"dominant=historian_{dominant_pos}, bias_score={bias_score:.3f}"
        )
 
        return ConvergenceMetrics(
            historian_1_embedding=h1_emb,
            historian_2_embedding=h2_emb,
            historian_3_embedding=h3_emb,
            centroid_embedding=centroid,
            abstract_1_embedding=a1_emb,
            abstract_2_embedding=a2_emb,
            abstract_3_embedding=a3_emb,
            final_abstract_embedding=final_emb,
            distance_hist1_to_centroid=d_h1,
            distance_hist2_to_centroid=d_h2,
            distance_hist3_to_centroid=d_h3,
            mean_historian_distance=mean_h_dist,
            distance_abstract1_to_centroid=d_a1,
            distance_abstract2_to_centroid=d_a2,
            distance_abstract3_to_centroid=d_a3,
            mean_abstract_distance=mean_a_dist,
            distance_final_to_centroid=d_final,
            converged=converged,
            bias_weights=bias_weights,
            dominant_historian_position=dominant_pos,
            bias_score=bias_score
        )
 
    @staticmethod
    def _compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute normalized centroid of embedding vectors."""
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        return centroid.astype(np.float32)
 
    @staticmethod
    def _cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine distance (1 - cosine similarity)."""
        return float(1.0 - np.dot(v1, v2))
 
    def compute_embedding_stats(
        self, metrics: ConvergenceMetrics
    ) -> Dict[str, float]:
        """
        Compute additional statistics about embedding geometry.
 
        Args:
            metrics: ConvergenceMetrics object
 
        Returns:
            Dict with additional stats
        """
        abstract_distances = [
            metrics.distance_abstract1_to_centroid,
            metrics.distance_abstract2_to_centroid,
            metrics.distance_abstract3_to_centroid
        ]
        abstract_variance = float(np.var(abstract_distances))
        convergence_delta = metrics.mean_abstract_distance - metrics.distance_final_to_centroid
 
        sim_12 = float(np.dot(metrics.abstract_1_embedding, metrics.abstract_2_embedding))
        sim_23 = float(np.dot(metrics.abstract_2_embedding, metrics.abstract_3_embedding))
        sim_13 = float(np.dot(metrics.abstract_1_embedding, metrics.abstract_3_embedding))
        mean_pairwise_sim = float(np.mean([sim_12, sim_23, sim_13]))
 
        return {
            'abstract_distance_variance': abstract_variance,
            'convergence_delta': float(convergence_delta),
            'mean_pairwise_abstract_similarity': float(mean_pairwise_sim),
            'abstract_similarity_12': sim_12,
            'abstract_similarity_23': sim_23,
            'abstract_similarity_13': sim_13,
            'bias_weight_1': float(metrics.bias_weights[0]),
            'bias_weight_2': float(metrics.bias_weights[1]),
            'bias_weight_3': float(metrics.bias_weights[2]),
            'dominant_historian_position': int(metrics.dominant_historian_position),
            'bias_score': float(metrics.bias_score)
        }