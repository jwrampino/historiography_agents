"""
Source Library: FAISS-based multimodal primary source document library.
Handles document storage, retrieval, and tracking of source usage.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import yaml

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: faiss-cpu and/or sentence-transformers not installed")


@dataclass
class PrimarySource:
    """Represents a primary source document."""
    source_id: str
    title: str
    content: str
    source_type: str  # e.g., "text", "image_caption", "audio_transcript"
    metadata: Dict  # Additional metadata (date, author, location, etc.)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding embedding."""
        data = asdict(self)
        data.pop('embedding', None)
        return data


class SourceLibrary:
    """FAISS-based library for storing and retrieving primary sources."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the source library."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        source_config = self.config['sources']
        self.embedding_model_name = source_config['embedding_model']
        self.index_path = Path(source_config['index_path'])
        self.metadata_path = Path(source_config['metadata_path'])

        # Initialize embedding model
        self.embedding_model = None
        self.dimension = None

        # FAISS index
        self.index: Optional[faiss.Index] = None

        # Source storage
        self.sources: Dict[str, PrimarySource] = {}
        self.source_id_to_index: Dict[str, int] = {}
        self.index_to_source_id: Dict[int, str] = {}

        # Usage tracking
        self.access_log: List[Dict] = []

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            # Get embedding dimension
            test_embedding = self.embedding_model.encode(["test"])
            self.dimension = test_embedding.shape[1]
            print(f"Embedding dimension: {self.dimension}")

    def create_index(self, use_gpu: bool = False):
        """Create a new FAISS index."""
        if self.dimension is None:
            self.initialize_embedding_model()

        # Use Inner Product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)

        if use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS index")
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        print(f"Created FAISS index with dimension {self.dimension}")

    def add_source(self, source: PrimarySource) -> int:
        """
        Add a source to the library and index.

        Returns:
            Index position in FAISS
        """
        if self.embedding_model is None:
            self.initialize_embedding_model()

        if self.index is None:
            self.create_index()

        # Generate embedding if not provided
        if source.embedding is None:
            embedding = self.embedding_model.encode([source.content])[0]
        else:
            embedding = source.embedding

        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)

        # Add to FAISS index
        idx = self.index.ntotal
        self.index.add(np.array([embedding], dtype=np.float32))

        # Store metadata mappings
        self.sources[source.source_id] = source
        self.source_id_to_index[source.source_id] = idx
        self.index_to_source_id[idx] = source.source_id

        return idx

    def add_sources_batch(self, sources: List[PrimarySource]):
        """Add multiple sources in batch."""
        if self.embedding_model is None:
            self.initialize_embedding_model()

        if self.index is None:
            self.create_index()

        print(f"Adding {len(sources)} sources to library...")

        # Generate embeddings for all sources
        contents = [s.content for s in sources]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        for i, source in enumerate(sources):
            idx = start_idx + i
            source.embedding = embeddings[i]
            self.sources[source.source_id] = source
            self.source_id_to_index[source.source_id] = idx
            self.index_to_source_id[idx] = source.source_id

        print(f"Total sources in library: {len(self.sources)}")

    def search(
        self,
        query: str,
        k: int = 5,
        agent_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> List[Tuple[PrimarySource, float]]:
        """
        Search for relevant sources using semantic similarity.

        Args:
            query: Search query
            k: Number of results to return
            agent_id: ID of agent performing search (for tracking)
            experiment_id: Experiment ID (for tracking)

        Returns:
            List of (source, similarity_score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            print("Warning: Index is empty")
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS index
        similarities, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), k
        )

        # Retrieve sources
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx in self.index_to_source_id:
                source_id = self.index_to_source_id[idx]
                source = self.sources[source_id]
                results.append((source, float(sim)))

        # Log access
        self._log_access(query, results, agent_id, experiment_id)

        return results

    def get_source_by_id(self, source_id: str) -> Optional[PrimarySource]:
        """Retrieve a source by its ID."""
        # Check if already loaded
        if source_id in self.sources:
            return self.sources[source_id]

        # Try lazy loading if using historian_pipeline data
        if hasattr(self, 'metadata_sources'):
            source = self._load_source_metadata(source_id)
            if source:
                self.sources[source_id] = source
                return source

        return None

    def _log_access(
        self,
        query: str,
        results: List[Tuple[PrimarySource, float]],
        agent_id: Optional[str],
        experiment_id: Optional[str]
    ):
        """Log source access for analysis."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'experiment_id': experiment_id,
            'query': query,
            'retrieved_sources': [
                {'source_id': s.source_id, 'similarity': score}
                for s, score in results
            ]
        }
        self.access_log.append(log_entry)

    def get_access_log(self, experiment_id: Optional[str] = None) -> List[Dict]:
        """Get access log, optionally filtered by experiment."""
        if experiment_id is None:
            return self.access_log
        return [log for log in self.access_log if log['experiment_id'] == experiment_id]

    def save_index(self, path: Optional[Path] = None):
        """Save FAISS index and metadata to disk."""
        if path is None:
            path = self.index_path

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))

        # Save metadata
        metadata = {
            'sources': {sid: s.to_dict() for sid, s in self.sources.items()},
            'source_id_to_index': self.source_id_to_index,
            'index_to_source_id': {str(k): v for k, v in self.index_to_source_id.items()},
            'dimension': self.dimension,
            'total_sources': len(self.sources)
        }

        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved index and metadata to {path}")

    def load_index(self, path: Optional[Path] = None):
        """Load FAISS index and metadata from disk."""
        if path is None:
            path = self.index_path

        if not path.exists():
            raise FileNotFoundError(f"Index path {path} does not exist")

        # Initialize embedding model
        self.initialize_embedding_model()

        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))

        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        self.dimension = metadata['dimension']
        self.source_id_to_index = metadata['source_id_to_index']
        self.index_to_source_id = {int(k): v for k, v in metadata['index_to_source_id'].items()}

        # Reconstruct sources (without embeddings to save memory)
        for source_id, source_data in metadata['sources'].items():
            self.sources[source_id] = PrimarySource(**source_data)

        print(f"Loaded {len(self.sources)} sources from {path}")

    def load_from_historian_pipeline(self, data_dir: Optional[Path] = None):
        """
        Load FAISS index and metadata from historian_pipeline data structure.

        Args:
            data_dir: Path to the data directory (default: ../data relative to config)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            data_dir = Path(data_dir)

        index_file = data_dir / "index" / "corpus.faiss"
        id_map_file = data_dir / "index" / "id_map.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        if not id_map_file.exists():
            raise FileNotFoundError(f"ID map not found: {id_map_file}")

        # Initialize embedding model
        self.initialize_embedding_model()

        # Load FAISS index
        print(f"Loading FAISS index from {index_file}")
        self.index = faiss.read_index(str(index_file))

        # Load ID mappings
        with open(id_map_file, 'r') as f:
            id_data = json.load(f)

        self.index_to_source_id = {int(k): v for k, v in id_data['int_to_source'].items()}
        self.source_id_to_index = {v: int(k) for k, v in id_data['int_to_source'].items()}

        # Load metadata from JSON files on-demand
        # We'll populate sources dict lazily as they're accessed
        print(f"Loaded FAISS index with {len(self.index_to_source_id)} sources")
        print(f"Metadata will be loaded on-demand from {data_dir / 'raw'}")

        # Store paths for lazy loading
        self.data_dir = data_dir
        self.metadata_sources = {
            'chronicling_america': data_dir / 'raw' / 'chronicling_america' / 'metadata',
            'internet_archive': data_dir / 'raw' / 'internet_archive' / 'metadata',
            'nara': data_dir / 'raw' / 'nara' / 'metadata'
        }

    def _load_source_metadata(self, source_id: str) -> Optional[PrimarySource]:
        """Load metadata for a source from JSON file."""
        # Try each metadata directory
        for source_name, metadata_dir in self.metadata_sources.items():
            metadata_file = metadata_dir / f"{source_id}.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Extract title and content from metadata
                    title = metadata.get('title', f"Source {source_id}")

                    # For text sources, use description; for images, use caption/description
                    content = ""
                    if 'description' in metadata:
                        if isinstance(metadata['description'], list):
                            content = " ".join(metadata['description'])
                        else:
                            content = str(metadata['description'])
                    elif 'ocr_text' in metadata:
                        content = metadata['ocr_text']
                    elif 'caption' in metadata:
                        content = metadata['caption']

                    # Determine source type
                    source_type = metadata.get('type', 'text')
                    if 'image' in metadata.get('online_format', []):
                        source_type = 'image_caption'

                    return PrimarySource(
                        source_id=source_id,
                        title=title,
                        content=content or f"No content available for {source_id}",
                        source_type=source_type,
                        metadata=metadata
                    )
                except Exception as e:
                    print(f"Error loading metadata for {source_id}: {e}")
                    continue

        # If not found, return a placeholder
        return PrimarySource(
            source_id=source_id,
            title=f"Source {source_id}",
            content=f"Metadata not found for source {source_id}",
            source_type="unknown",
            metadata={}
        )


def create_example_library(n_sources: int = 100) -> SourceLibrary:
    """Create an example source library with synthetic historical documents."""
    library = SourceLibrary()
    library.initialize_embedding_model()
    library.create_index()

    # Example historical content themes
    themes = [
        "labor movements and strikes in the industrial revolution",
        "women's suffrage and early feminist organizing",
        "colonial administration and resistance in Africa",
        "medieval trade routes and merchant guilds",
        "enlightenment philosophy and political thought",
        "agricultural practices in ancient civilizations",
        "religious reformation and theological debates",
        "military strategy in ancient warfare",
        "urban development and city planning through history",
        "technological innovation and its social impact"
    ]

    sources = []
    for i in range(n_sources):
        theme = themes[i % len(themes)]
        source = PrimarySource(
            source_id=f"source_{i:04d}",
            title=f"Historical Document {i}: {theme}",
            content=f"This is a primary source about {theme}. " * 10,
            source_type="text",
            metadata={
                'theme': theme,
                'year': 1800 + (i * 2),
                'region': ['Europe', 'Americas', 'Asia', 'Africa'][i % 4]
            }
        )
        sources.append(source)

    library.add_sources_batch(sources)
    return library


if __name__ == "__main__":
    # Example usage
    print("Creating example source library...")
    library = create_example_library(n_sources=50)

    # Test search
    results = library.search("labor movements and workers' rights", k=3)
    print("\nSearch results:")
    for source, score in results:
        print(f"  {source.source_id} (score: {score:.3f}): {source.title}")

    # Save library
    library.save_index()
