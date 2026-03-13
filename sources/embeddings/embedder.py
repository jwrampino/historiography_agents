"""
Embedding generation for all modalities.
All embeddings are 768-d — the native dimension of both
sentence-transformers/all-mpnet-base-v2 (text) and
openai/clip-vit-large-patch14 (images). No projection needed.

Text:  all-mpnet-base-v2 → 768-d
Image: CLIP ViT-L/14 → 768-d (only used when image files are present)
All others: fall back to text embedding on best available metadata
"""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from sources.config.settings import (
    TEXT_EMBEDDING_MODEL,
    CLIP_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Text Embedder
# ══════════════════════════════════════════════════════════════════════════════

class TextEmbedder:
    """
    Embeds text using sentence-transformers/all-mpnet-base-v2.
    Native output is 768-d — matches EMBEDDING_DIM exactly, no projection.
    """

    def __init__(self, model_name: str = TEXT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded text embedding model: {self.model_name} "
                        f"(dim={self._model.get_sentence_embedding_dimension()})")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required: pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns float32 array of shape (len(texts), 768).
        """
        self._load()
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # L2 normalise → cosine sim = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns 1D float32 array of length 768."""
        return self.embed([text])[0]


# ══════════════════════════════════════════════════════════════════════════════
# Image Embedder (CLIP)
# ══════════════════════════════════════════════════════════════════════════════

class ImageEmbedder:
    """
    Embeds images using CLIP ViT-L/14.
    Native output is 768-d — matches EMBEDDING_DIM exactly, no projection.
    Only used when local image files are present on disk.
    """

    def __init__(self, model_name: str = CLIP_MODEL):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model.eval()
            logger.info(f"Loaded CLIP model: {self.model_name}")
        except ImportError:
            raise RuntimeError("transformers is required: pip install transformers")

    def embed_images(self, image_paths: List[Path]) -> np.ndarray:
        """
        Embed a list of image files.
        Returns float32 array of shape (len(image_paths), 768).
        Failed images are returned as zero vectors.
        """
        import torch
        from PIL import Image

        self._load()
        if not image_paths:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(image_paths), EMBEDDING_BATCH_SIZE):
            batch_paths = image_paths[i: i + EMBEDDING_BATCH_SIZE]
            images, valid_indices = [], []

            for j, p in enumerate(batch_paths):
                try:
                    images.append(Image.open(p).convert("RGB"))
                    valid_indices.append(j)
                except Exception as e:
                    logger.warning(f"Could not open image {p}: {e}")

            batch_emb = np.zeros((len(batch_paths), EMBEDDING_DIM), dtype=np.float32)

            if images:
                inputs = self._processor(images=images, return_tensors="pt", padding=True)
                with torch.no_grad():
                    features = self._model.get_image_features(**inputs)
                    # Newer transformers may return a dataclass instead of a raw tensor
                    if not isinstance(features, torch.Tensor):
                        features = features.image_embeds if hasattr(features, "image_embeds") else features[0]
                    features = features / features.norm(dim=-1, keepdim=True)
                    features_np = features.numpy().astype(np.float32)
                # Ensure 2D: (batch, dim)
                if features_np.ndim == 1:
                    features_np = features_np[np.newaxis, :]
                elif features_np.ndim > 2:
                    features_np = features_np.reshape(features_np.shape[0], -1)
                # CLIP ViT-L/14 outputs 1024-dim; project to EMBEDDING_DIM by
                # truncating (if larger) or zero-padding (if smaller).
                clip_dim = features_np.shape[1]
                if clip_dim != EMBEDDING_DIM:
                    if clip_dim > EMBEDDING_DIM:
                        features_np = features_np[:, :EMBEDDING_DIM]
                    else:
                        pad = np.zeros((features_np.shape[0], EMBEDDING_DIM - clip_dim), dtype=np.float32)
                        features_np = np.concatenate([features_np, pad], axis=1)
                    norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    features_np = features_np / norms
                for k, vi in enumerate(valid_indices):
                    batch_emb[vi] = features_np[k]

            all_embeddings.append(batch_emb)

        return np.concatenate(all_embeddings, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Unified Corpus Embedder
# ══════════════════════════════════════════════════════════════════════════════

class CorpusEmbedder:
    """
    Routes each CorpusItem to the correct embedder based on modality.

    Routing logic:
      - Any item with usable text (>50 chars)  → TextEmbedder
      - Image/map item with a local file        → ImageEmbedder
      - Everything else                         → TextEmbedder on concatenated metadata

    In practice, since we don't download files, everything goes through
    TextEmbedder on whatever text the archive API returned.
    """

    def __init__(self):
        self._text_embedder: Optional[TextEmbedder] = None
        self._image_embedder: Optional[ImageEmbedder] = None

    @property
    def text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder()
        return self._text_embedder

    @property
    def image_embedder(self) -> ImageEmbedder:
        if self._image_embedder is None:
            self._image_embedder = ImageEmbedder()
        return self._image_embedder

    def embed_item(self, item) -> Tuple[np.ndarray, str]:
        """
        Embed a single CorpusItem.
        Returns (embedding: 1D float32 array of length 768, model_name: str).
        """
        # ── Image path: CLIP takes priority for image items with a local file ─
        if item.modality in ("image", "map") and item.file_path:
            fpath = Path(item.file_path)
            if fpath.exists():
                try:
                    vecs = self.image_embedder.embed_images([fpath])
                    if vecs.shape[0] > 0 and np.any(vecs[0]):
                        return vecs[0], CLIP_MODEL
                except Exception as e:
                    logger.warning(f"CLIP failed for {item.source_id}, falling back to text: {e}")

        # ── Text path ────────────────────────────────────────────────────
        text = self._get_text(item)
        if text and len(text) > 50:
            vec = self.text_embedder.embed_one(text[:8192])
            return vec, TEXT_EMBEDDING_MODEL

        # ── Fallback: metadata concatenation ─────────────────────────────
        fallback = self._get_text(item, force_metadata=True)
        vec = self.text_embedder.embed_one(fallback)
        return vec, TEXT_EMBEDDING_MODEL + " (metadata fallback)"

    def embed_batch(
        self, items: List, show_progress: bool = True
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Embed a list of CorpusItems.
        Returns list of (embedding, model_name) tuples in the same order.
        """
        results = []
        total = len(items)
        for i, item in enumerate(items):
            if show_progress and i % 50 == 0:
                logger.info(f"Embedding {i}/{total}")
            try:
                results.append(self.embed_item(item))
            except Exception as e:
                logger.warning(f"Embedding failed for {item.source_id}: {e}")
                results.append((np.zeros(EMBEDDING_DIM, dtype=np.float32), "error"))
        return results

    @staticmethod
    def _get_text(item, force_metadata: bool = False) -> str:
        """
        Pull the best available text for a CorpusItem.

        Priority:
          1. _raw_text set during ingestion (API descriptions, Chronicling Am. OCR)
          2. Saved transcript file
          3. Concatenated metadata fields (title, subjects, notes, location, era)
        """
        if not force_metadata:
            # 1. Raw text from ingestion
            if hasattr(item, "_raw_text") and item._raw_text and len(item._raw_text) > 50:
                return item._raw_text

            # 2. Saved transcript
            if item.transcript_path and Path(item.transcript_path).exists():
                try:
                    text = Path(item.transcript_path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    if len(text) > 50:
                        return text
                except Exception:
                    pass

        # 3. Metadata concatenation
        parts = []
        if item.title:
            parts.append(f"Title: {item.title}")
        if item.institution:
            parts.append(f"Institution: {item.institution}")
        if item.collection:
            parts.append(f"Collection: {item.collection}")
        if item.date_original:
            parts.append(f"Date: {item.date_original}")
        if item.geographic_scope:
            parts.append(f"Location: {item.geographic_scope}")
        if item.era_tag and item.era_tag != "unknown":
            parts.append(f"Era: {item.era_tag}")
        if item.topic_tags:
            tags = (
                item.topic_tags
                if isinstance(item.topic_tags, list)
                else item.topic_tags.split("|")
            )
            parts.append(f"Subjects: {', '.join(tags)}")
        if item.notes:
            parts.append(item.notes)

        return " | ".join(parts) if parts else "unknown historical document"
