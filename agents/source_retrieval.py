"""
Source Retrieval: Retrieves multimodal sources from the corpus for agent use.
Derives retrieval queries from historian research areas.
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from collections import Counter
import re
import requests
from PIL import Image
from io import BytesIO

from sources.storage.corpus_store import CorpusStore
from sources.embeddings.faiss_index import CorpusIndex
from sources.embeddings.embedder import CorpusEmbedder

logger = logging.getLogger(__name__)


class SourceRetriever:
    """Retrieves multimodal sources for historian agents."""

    def __init__(
        self,
        corpus_store: Optional[CorpusStore] = None,
        corpus_index: Optional[CorpusIndex] = None,
        image_dir: str = "data/downloaded_images"
    ):
        """
        Initialize source retriever.

        Args:
            corpus_store: CorpusStore instance (creates new if None)
            corpus_index: CorpusIndex instance (creates new if None)
            image_dir: Directory to save downloaded images
        """
        self.store = corpus_store or CorpusStore()
        self.index = corpus_index or CorpusIndex()
        self.embedder = CorpusEmbedder()
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Load index
        try:
            self.index.load()
            logger.info(f"Loaded FAISS index with {self.index.size} vectors")
        except FileNotFoundError:
            logger.warning("FAISS index not found - retrieval may fail")

    def extract_keywords_from_papers(
        self, papers: List[Dict], top_k: int = 10
    ) -> List[str]:
        """
        Extract keywords from historian's papers to derive retrieval queries.

        Args:
            papers: List of paper dicts with 'title' and 'abstract'
            top_k: Number of top keywords to extract

        Returns:
            List of keywords
        """
        # Combine all titles and abstracts
        text = " ".join([
            f"{p.get('title', '')} {p.get('abstract', '')}"
            for p in papers
        ])

        # Simple keyword extraction: most common meaningful words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Filter stopwords
        stopwords = {
            'this', 'that', 'with', 'from', 'have', 'been', 'will',
            'their', 'there', 'which', 'these', 'those', 'would',
            'could', 'should', 'about', 'after', 'before', 'between',
            'through', 'during', 'also', 'other', 'some', 'such',
            'more', 'most', 'into', 'than', 'over', 'under', 'upon'
        }

        filtered_words = [w for w in words if w not in stopwords]

        # Count frequency
        word_counts = Counter(filtered_words)

        # Get top keywords
        keywords = [word for word, _ in word_counts.most_common(top_k)]

        return keywords

    def generate_retrieval_query(
        self, papers: List[Dict], n_keywords: int = 3
    ) -> str:
        """
        Generate a retrieval query from historian's papers.

        Args:
            papers: List of paper dicts
            n_keywords: Number of keywords to use

        Returns:
            Query string
        """
        keywords = self.extract_keywords_from_papers(papers, top_k=20)

        # Prioritize historical terms
        historical_terms = {
            'history', 'historical', 'historiography', 'historian',
            'century', 'period', 'era', 'colonial', 'empire',
            'revolution', 'movement', 'social', 'political',
            'cultural', 'economic', 'class', 'gender', 'race',
            'nation', 'state', 'power', 'memory', 'archive'
        }

        # Separate historical and other keywords
        hist_kw = [k for k in keywords if k in historical_terms]
        other_kw = [k for k in keywords if k not in historical_terms]

        # Take mix of both
        selected = hist_kw[:n_keywords//2] + other_kw[:n_keywords - n_keywords//2]

        if not selected:
            selected = keywords[:n_keywords]

        # Fallback if still no keywords found
        if not selected:
            logger.warning("No keywords extracted from papers, using default query")
            query = "history historical"
        else:
            query = " ".join(selected[:n_keywords])

        logger.info(f"Generated retrieval query: '{query}'")
        return query

    def retrieve_text_sources(
        self, query: str, n_sources: int = 3, min_quality: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve text sources from corpus using semantic search.

        Args:
            query: Search query
            n_sources: Number of sources to retrieve
            min_quality: Minimum quality score threshold

        Returns:
            List of source dicts with metadata and text
        """
        # Safety check for empty query
        if not query or not query.strip():
            logger.warning("Empty query provided, using default")
            query = "history historical"

        # Embed query
        query_vec = self.embedder.text_embedder.embed_one(query)

        # Search FAISS index
        results = self.index.search(query_vec, top_k=n_sources * 3)

        sources = []
        for result in results:
            if len(sources) >= n_sources:
                break

            # Get item from store
            item = self.store.get(result['source_id'])
            if not item:
                continue

            # Filter by modality and quality
            if item.modality not in ['text', 'mixed']:
                continue
            if item.quality_score < min_quality:
                continue

            # Get text content
            text = self._get_item_text(item)
            if not text or len(text) < 100:
                continue

            sources.append({
                'source_id': item.source_id,
                'title': item.title,
                'institution': item.institution,
                'date_original': item.date_original,
                'text': text[:2000],  # Truncate to 2000 chars
                'url': item.url_original,
                'similarity_score': result['similarity_score'],
                'modality': 'text'
            })

        logger.info(f"Retrieved {len(sources)} text sources for query '{query}'")
        return sources

    def retrieve_image_sources(
        self, query: str, n_sources: int = 2, download: bool = True
    ) -> List[Dict]:
        """
        Retrieve image sources from corpus using semantic search.

        Args:
            query: Search query
            n_sources: Number of images to retrieve
            download: Whether to download image files

        Returns:
            List of image source dicts
        """
        # Safety check for empty query
        if not query or not query.strip():
            logger.warning("Empty query provided, using default")
            query = "history historical"

        # Embed query (text-based search for images)
        query_vec = self.embedder.text_embedder.embed_one(query)

        # Search FAISS index - search MANY results since images are rare (57/4000)
        # Need to check ~350 items on average to find 5 images
        results = self.index.search(query_vec, top_k=min(500, n_sources * 100))

        sources = []
        images_found = 0
        items_checked = 0

        for result in results:
            if len(sources) >= n_sources:
                break

            # Get item from store
            item = self.store.get(result['source_id'])
            if not item:
                continue

            items_checked += 1

            # Filter by modality (only 57 images in ~4000 corpus items)
            if item.modality != 'image':
                continue

            images_found += 1

            # Download image if requested
            local_path = None
            if download and item.url_original:
                local_path = self._download_image(
                    item.url_original, item.source_id
                )

            sources.append({
                'source_id': item.source_id,
                'title': item.title,
                'institution': item.institution,
                'date_original': item.date_original,
                'url': item.url_original,
                'local_path': str(local_path) if local_path else None,
                'similarity_score': result['similarity_score'],
                'modality': 'image'
            })

        if len(sources) < n_sources:
            logger.warning(
                f"Retrieved only {len(sources)}/{n_sources} image sources for query '{query}' "
                f"(checked {items_checked} items, found {images_found} images)"
            )
        else:
            logger.info(f"Retrieved {len(sources)} image sources for query '{query}'")

        return sources

    def retrieve_random_text_sources(self, n_sources: int = 3) -> List[Dict]:
        """
        Randomly sample text sources from corpus (NO query-based retrieval).
        Uses DuckDB's ORDER BY RANDOM() for efficient sampling.

        Args:
            n_sources: Number of text sources to retrieve

        Returns:
            List of randomly sampled text source dicts
        """
        # Use DuckDB to randomly sample text items efficiently
        sql = f"""
            SELECT * FROM corpus
            WHERE modality = 'text'
            ORDER BY RANDOM()
            LIMIT {n_sources}
        """

        try:
            rows = self.store.con.execute(sql).fetchall()
            if not rows:
                logger.warning("No text sources found in corpus")
                return []

            cols = [desc[0] for desc in self.store.con.description]
            sampled_items = [self.store.get(dict(zip(cols, row))['source_id']) for row in rows]

            sources = []
            for item in sampled_items:
                if item is None:
                    continue
                text_content = self._get_item_text(item)
                sources.append({
                    'source_id': item.source_id,
                    'title': item.title,
                    'institution': item.institution,
                    'date_original': item.date_original,
                    'text': text_content,
                    'url': item.url_original,
                    'similarity_score': None,  # No semantic similarity in random sampling
                    'modality': 'text'
                })

            logger.info(f"Randomly sampled {len(sources)} text sources")
            return sources

        except Exception as e:
            logger.error(f"Failed to retrieve random text sources: {e}")
            return []

    def retrieve_random_image_sources(self, n_sources: int = 2, download: bool = True) -> List[Dict]:
        """
        Randomly sample image sources from corpus (NO query-based retrieval).
        Uses DuckDB's ORDER BY RANDOM() for efficient sampling.

        Args:
            n_sources: Number of image sources to retrieve
            download: Whether to download images

        Returns:
            List of randomly sampled image source dicts
        """
        # Use DuckDB to randomly sample image items efficiently
        sql = f"""
            SELECT * FROM corpus
            WHERE modality = 'image'
            ORDER BY RANDOM()
            LIMIT {n_sources}
        """

        try:
            rows = self.store.con.execute(sql).fetchall()
            if not rows:
                logger.warning("No image sources found in corpus")
                return []

            cols = [desc[0] for desc in self.store.con.description]
            sampled_items = [self.store.get(dict(zip(cols, row))['source_id']) for row in rows]

            sources = []
            for item in sampled_items:
                if item is None:
                    continue

                # Download image if requested
                local_path = None
                if download and item.url_original:
                    local_path = self._download_image(item.url_original, item.source_id)

                sources.append({
                    'source_id': item.source_id,
                    'title': item.title,
                    'institution': item.institution,
                    'date_original': item.date_original,
                    'url': item.url_original,
                    'local_path': str(local_path) if local_path else None,
                    'similarity_score': None,  # No semantic similarity in random sampling
                    'modality': 'image'
                })

            logger.info(f"Randomly sampled {len(sources)} image sources")
            return sources

        except Exception as e:
            logger.error(f"Failed to retrieve random image sources: {e}")
            return []

    def retrieve_source_packet(
        self, papers: List[Dict], n_text: int = 3, n_images: int = 2, random_sampling: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve a full source packet for a historian.

        Args:
            papers: Historian's papers (for deriving query if random_sampling=False)
            n_text: Number of text sources
            n_images: Number of image sources
            random_sampling: If True, randomly sample sources; if False, use query-based retrieval

        Returns:
            Dict with 'text_sources' and 'image_sources'
        """
        if random_sampling:
            # NEW: Random sampling (no query needed, each historian gets different sources)
            text_sources = self.retrieve_random_text_sources(n_sources=n_text)
            image_sources = self.retrieve_random_image_sources(n_sources=n_images)
            query = "random_sampling"
        else:
            # OLD: Query-based semantic search
            query = self.generate_retrieval_query(papers)
            text_sources = self.retrieve_text_sources(query, n_sources=n_text)
            image_sources = self.retrieve_image_sources(query, n_sources=n_images)

        return {
            'query': query,
            'text_sources': text_sources,
            'image_sources': image_sources
        }

    def _get_item_text(self, item) -> str:
        """Extract text from a corpus item."""
        # Priority: _raw_text > transcript file > metadata
        if hasattr(item, '_raw_text') and item._raw_text:
            return item._raw_text

        if item.transcript_path and Path(item.transcript_path).exists():
            try:
                return Path(item.transcript_path).read_text(
                    encoding='utf-8', errors='replace'
                )
            except Exception:
                pass

        # Fallback: concatenate metadata
        parts = []
        if item.title:
            parts.append(f"Title: {item.title}")
        if item.notes:
            parts.append(item.notes)
        if item.topic_tags:
            tags = (
                item.topic_tags if isinstance(item.topic_tags, list)
                else item.topic_tags.split('|')
            )
            parts.append(f"Subjects: {', '.join(tags)}")

        return " | ".join(parts) if parts else ""

    def _download_image(self, url: str, source_id: str) -> Optional[Path]:
        """
        Download an image from URL and save locally.

        Args:
            url: Image URL
            source_id: Corpus item source ID

        Returns:
            Local path to saved image, or None on failure
        """
        output_path = self.image_dir / f"{source_id}.jpg"

        # Skip if already downloaded
        if output_path.exists():
            return output_path

        try:
            # Download image
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Open and convert to RGB
            img = Image.open(BytesIO(response.content)).convert('RGB')

            # Save as JPEG
            img.save(output_path, 'JPEG', quality=85)

            logger.info(f"Downloaded image: {source_id}")
            return output_path

        except Exception as e:
            logger.warning(f"Failed to download image {source_id}: {e}")
            return None

    def format_sources_for_agent(self, source_packet: Dict) -> str:
        """
        Format sources into a text prompt for the agent.

        Args:
            source_packet: Dict with text_sources and image_sources

        Returns:
            Formatted string
        """
        lines = []
        lines.append("=== PRIMARY SOURCES ===\n")

        # Text sources
        lines.append("--- TEXTUAL SOURCES ---\n")
        for i, src in enumerate(source_packet['text_sources'], 1):
            lines.append(f"Source {i}: {src['title']}")
            lines.append(f"Institution: {src['institution']}")
            lines.append(f"Date: {src['date_original']}")
            lines.append(f"Text excerpt:\n{src['text'][:1500]}\n")

        # Image sources
        lines.append("\n--- IMAGE SOURCES ---\n")
        for i, src in enumerate(source_packet['image_sources'], 1):
            lines.append(f"Image {i}: {src['title']}")
            lines.append(f"Institution: {src['institution']}")
            lines.append(f"Date: {src['date_original']}")
            if src.get('local_path'):
                lines.append(f"[Image file available at: {src['local_path']}]")
            lines.append("")

        return "\n".join(lines)
