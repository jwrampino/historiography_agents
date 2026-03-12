"""
Historian Manager: Loads real historian personas from OpenAlex data.
Based on actual scholarly work, uses embeddings for group selection.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class HistorianPersona:
    """Represents a real historian persona based on their publications."""
    historian_id: str
    name: str
    prompt: str  # Full persona prompt including paper abstracts
    papers: List[Dict]  # List of paper dicts with titles, abstracts, years
    embedding: Optional[np.ndarray] = None  # 768-d embedding vector

    def to_dict(self) -> Dict:
        """Convert persona to dictionary for JSON serialization."""
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'HistorianPersona':
        """Load persona from dictionary."""
        if 'embedding' in d and d['embedding'] is not None:
            d['embedding'] = np.array(d['embedding'], dtype=np.float32)
        return cls(**d)


class HistorianManager:
    """Manages real historian personas and group selection with geometry."""

    def __init__(
        self,
        papers_csv: str = "topic_papers.csv",
        edges_csv: str = "paper_author_edges.csv",
        ranked_csv: str = "ranked_historians.csv",
        n_historians: int = 25,
        data_dir: Optional[str] = None
    ):
        """
        Initialize HistorianManager.

        Args:
            papers_csv: Path to papers CSV from persona.ipynb
            edges_csv: Path to paper-author edges CSV
            ranked_csv: Path to ranked historians CSV
            n_historians: Number of top historians to use
            data_dir: Base directory for CSV files (defaults to parent dir)
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Default to parent directory (where CSVs are)
            self.data_dir = Path(__file__).parent.parent.parent

        self.papers_csv = self.data_dir / papers_csv
        self.edges_csv = self.data_dir / edges_csv
        self.ranked_csv = self.data_dir / ranked_csv
        self.n_historians = n_historians

        self.personas: List[HistorianPersona] = []
        self.persona_dict: Dict[str, HistorianPersona] = {}
        self._text_embedder = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load CSV files from persona.ipynb outputs."""
        logger.info("Loading historian data from CSVs...")
        papers_df = pd.read_csv(self.papers_csv)
        edges_df = pd.read_csv(self.edges_csv)
        ranked_df = pd.read_csv(self.ranked_csv)

        # Drop papers without abstracts
        papers_df = papers_df.dropna(subset=['abstract'])

        logger.info(f"Loaded {len(papers_df)} papers, {len(ranked_df)} authors")
        return papers_df, edges_df, ranked_df

    def build_persona_prompt(self, historian_name: str, papers: List[Dict]) -> str:
        """
        Build persona prompt from historian's actual published work.

        Args:
            historian_name: Name of the historian
            papers: List of paper dicts with 'title', 'abstract', 'year'

        Returns:
            Persona prompt string
        """
        # Concatenate all abstracts with titles
        corpus_parts = []
        for p in papers:
            if p.get('abstract'):
                corpus_parts.append(
                    f"Title: {p['title']} ({p['year']})\n"
                    f"Abstract: {p['abstract']}"
                )

        full_corpus = "\n\n".join(corpus_parts)

        prompt = f"""You are a historian. Your scholarly perspective is defined entirely by the following body of prior work — these are real publications that characterize your intellectual position, methodological commitments, and interpretive frameworks:

{full_corpus}

When engaging in historical research and debate:
- Reason from the methodological approach evident in this prior work
- Prioritize the kinds of sources and evidence this work relies on
- Make arguments in the style and register of this scholarship
- Do not announce or describe your perspective explicitly — embody it
- Your name in this conversation is {historian_name}"""

        return prompt

    def create_historian_personas(self) -> List[HistorianPersona]:
        """
        Create persona objects for top N historians.

        Returns:
            List of HistorianPersona objects
        """
        papers_df, edges_df, ranked_df = self.load_data()

        # Take top N historians
        top_historians = ranked_df.head(self.n_historians)

        personas = []
        for _, row in top_historians.iterrows():
            author_id = row['authorId']
            author_name = row['authorName']

            # Get all papers for this historian
            paper_ids = edges_df.loc[
                edges_df['authorId'] == author_id, 'paperId'
            ].unique()

            papers = papers_df.loc[
                papers_df['paperId'].isin(paper_ids)
            ].to_dict('records')

            # Build prompt
            prompt = self.build_persona_prompt(author_name, papers)

            # Create persona
            persona = HistorianPersona(
                historian_id=author_id,
                name=author_name,
                prompt=prompt,
                papers=papers,
                embedding=None  # Will be computed later
            )
            personas.append(persona)

        self.personas = personas
        self.persona_dict = {p.name: p for p in personas}
        logger.info(f"Created {len(personas)} historian personas")
        return personas

    @property
    def text_embedder(self):
        """Lazy load text embedder."""
        if self._text_embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._text_embedder = SentenceTransformer(
                    'sentence-transformers/all-mpnet-base-v2'
                )
                logger.info("Loaded text embedding model")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers required: "
                    "pip install sentence-transformers"
                )
        return self._text_embedder

    def compute_historian_embeddings(self):
        """
        Compute 768-d embedding for each historian.
        Strategy: Average embeddings of all their paper abstracts.
        """
        logger.info("Computing historian embeddings...")

        for persona in self.personas:
            # Collect all abstracts
            abstracts = [
                p['abstract'] for p in persona.papers
                if p.get('abstract')
            ]

            if not abstracts:
                # Fallback: embed name if no abstracts
                abstracts = [persona.name]

            # Embed all abstracts
            embeddings = self.text_embedder.encode(
                abstracts,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Average to get historian embedding
            historian_embedding = np.mean(embeddings, axis=0)

            # Normalize
            historian_embedding = historian_embedding / np.linalg.norm(
                historian_embedding
            )

            persona.embedding = historian_embedding.astype(np.float32)

        logger.info("Computed embeddings for all historians")

    def compute_triangle_geometry(
        self,
        historians: Tuple[HistorianPersona, HistorianPersona, HistorianPersona]
    ) -> Dict[str, float]:
        """
        Compute geometric features of a triangle in embedding space.

        Args:
            historians: Tuple of 3 HistorianPersona objects

        Returns:
            Dict with triangle geometry features:
            - side_1, side_2, side_3: Pairwise distances
            - perimeter: Sum of sides
            - area: Triangle area
            - min_angle, max_angle: Angle range in radians
        """
        h1, h2, h3 = historians

        # Pairwise distances (cosine distance = 1 - cosine similarity)
        side_1 = 1 - np.dot(h1.embedding, h2.embedding)  # h1-h2
        side_2 = 1 - np.dot(h2.embedding, h3.embedding)  # h2-h3
        side_3 = 1 - np.dot(h3.embedding, h1.embedding)  # h3-h1

        # Perimeter
        perimeter = side_1 + side_2 + side_3

        # Area using Heron's formula
        s = perimeter / 2  # semi-perimeter
        area_squared = s * (s - side_1) * (s - side_2) * (s - side_3)
        area = np.sqrt(max(0, area_squared))

        # Angles using law of cosines
        # angle_1 = arccos((side_2^2 + side_3^2 - side_1^2) / (2*side_2*side_3))
        def compute_angle(a, b, c):
            """Compute angle opposite to side a."""
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle)

        angle_1 = compute_angle(side_1, side_2, side_3)
        angle_2 = compute_angle(side_2, side_3, side_1)
        angle_3 = compute_angle(side_3, side_1, side_2)

        return {
            'side_1': float(side_1),
            'side_2': float(side_2),
            'side_3': float(side_3),
            'perimeter': float(perimeter),
            'area': float(area),
            'min_angle': float(min(angle_1, angle_2, angle_3)),
            'max_angle': float(max(angle_1, angle_2, angle_3)),
            'angle_variance': float(np.var([angle_1, angle_2, angle_3]))
        }

    def filter_triangular_groups(
        self,
        min_distance: float = 0.1,
        max_distance: float = 0.8,
        min_area: float = 0.001
    ) -> List[Tuple[HistorianPersona, HistorianPersona, HistorianPersona]]:
        """
        Generate groups of 3 historians with triangle geometry constraints.

        Constraints:
        - All pairwise distances within [min_distance, max_distance]
        - Triangle area >= min_area (excludes collinear/degenerate triangles)

        Args:
            min_distance: Minimum pairwise distance
            max_distance: Maximum pairwise distance
            min_area: Minimum triangle area

        Returns:
            List of valid historian triples
        """
        from itertools import combinations

        if not self.personas:
            self.create_historian_personas()

        if self.personas[0].embedding is None:
            self.compute_historian_embeddings()

        logger.info("Generating triangular groups...")

        valid_groups = []
        all_combinations = list(combinations(self.personas, 3))

        for group in all_combinations:
            geom = self.compute_triangle_geometry(group)

            # Check constraints
            sides = [geom['side_1'], geom['side_2'], geom['side_3']]

            if (all(min_distance <= s <= max_distance for s in sides) and
                geom['area'] >= min_area):
                valid_groups.append(group)

        logger.info(
            f"Generated {len(valid_groups)} valid groups from "
            f"{len(all_combinations)} total combinations"
        )
        return valid_groups

    def sample_groups(
        self,
        n_groups: int = 100,
        strategy: str = "stratified",
        **kwargs
    ) -> List[Tuple[HistorianPersona, HistorianPersona, HistorianPersona]]:
        """
        Sample groups of historians for experiments.

        Args:
            n_groups: Number of groups to sample
            strategy: Sampling strategy:
                - "stratified": Sample groups with diverse geometry
                - "filtered": Apply triangle constraints then sample
                - "random": Completely random sampling
            **kwargs: Additional arguments for strategy

        Returns:
            List of historian triples
        """
        import random

        if strategy == "filtered":
            valid_groups = self.filter_triangular_groups(**kwargs)
            if len(valid_groups) <= n_groups:
                return valid_groups
            return random.sample(valid_groups, n_groups)

        elif strategy == "random":
            from itertools import combinations
            if not self.personas:
                self.create_historian_personas()
            all_groups = list(combinations(self.personas, 3))
            return random.sample(all_groups, min(n_groups, len(all_groups)))

        elif strategy == "stratified":
            # Sample groups stratified by perimeter (proxy for diversity)
            valid_groups = self.filter_triangular_groups(**kwargs)

            # Compute perimeters
            group_perimeters = [
                (group, self.compute_triangle_geometry(group)['perimeter'])
                for group in valid_groups
            ]

            # Sort by perimeter
            group_perimeters.sort(key=lambda x: x[1])

            # Sample evenly across perimeter range
            n_available = len(group_perimeters)
            if n_available <= n_groups:
                return [g for g, _ in group_perimeters]

            indices = np.linspace(0, n_available - 1, n_groups, dtype=int)
            return [group_perimeters[i][0] for i in indices]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def save_personas(self, output_path: str = "personas/historian_personas.json"):
        """Save historian personas to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'n_historians': len(self.personas),
            'personas': [p.to_dict() for p in self.personas]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.personas)} personas to {output_path}")

    def load_personas(self, input_path: str = "personas/historian_personas.json"):
        """Load historian personas from JSON."""
        input_path = Path(input_path)

        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            return self.create_historian_personas()

        with open(input_path, 'r') as f:
            data = json.load(f)

        self.personas = [
            HistorianPersona.from_dict(p) for p in data['personas']
        ]
        self.persona_dict = {p.name: p for p in self.personas}

        logger.info(f"Loaded {len(self.personas)} personas from {input_path}")
        return self.personas

    def get_persona_by_name(self, name: str) -> Optional[HistorianPersona]:
        """Get persona by historian name."""
        return self.persona_dict.get(name)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    # Initialize manager
    manager = HistorianManager(n_historians=25)

    # Create personas
    personas = manager.create_historian_personas()
    print(f"\nCreated {len(personas)} historian personas")

    # Compute embeddings
    manager.compute_historian_embeddings()
    print("Computed embeddings for all historians")

    # Generate valid groups
    groups = manager.sample_groups(
        n_groups=50,
        strategy="filtered",
        min_distance=0.1,
        max_distance=0.7,
        min_area=0.001
    )

    print(f"\nGenerated {len(groups)} valid groups")

    # Show example group
    if groups:
        print("\nExample group:")
        example_group = groups[0]
        geom = manager.compute_triangle_geometry(example_group)
        for h in example_group:
            print(f"  - {h.name}")
        print(f"\nTriangle geometry:")
        for k, v in geom.items():
            print(f"  {k}: {v:.4f}")

    # Save personas
    manager.save_personas("personas/historian_personas.json")
    print("\nSaved personas to JSON")
