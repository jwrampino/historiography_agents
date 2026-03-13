"""
Canonical data model for a single corpus item.
This mirrors the Master Corpus CSV schema from the spec exactly.
All ingestors return CorpusItem instances; the storage layer serialises them.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime
import uuid


@dataclass
class CorpusItem:
    """
    One item in the multimodal corpus.
    Every field maps 1-to-1 to a column in the Master Corpus CSV / DuckDB table.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    institution: str = ""             # controlled vocabulary — archive name
    collection: str = ""              # sub-collection or record group
    title: str = ""

    # ── Dates ───────────────────────────────────────────────────────────────
    date_original: str = ""           # ISO 8601 date/range of original creation
    date_digitized: str = ""          # ISO 8601 digitization date

    # ── Modality & Format ───────────────────────────────────────────────────
    modality: str = "text"            # text | image | audio | video | map | mixed
    format: str = ""                  # file extension / MIME type
    language: str = "en"              # ISO 639-1

    # ── Scope ───────────────────────────────────────────────────────────────
    geographic_scope: str = ""        # GeoNames IDs + human-readable
    temporal_scope: str = ""          # historical period described
    era_tag: str = "unknown"          # controlled era vocabulary

    # ── Subjects ────────────────────────────────────────────────────────────
    topic_tags: List[str] = field(default_factory=list)   # LCSH-aligned

    # ── File Paths ──────────────────────────────────────────────────────────
    file_path: str = ""               # relative path to raw file
    transcript_path: str = ""         # path to Whisper/OCR output
    thumbnail_path: str = ""          # path to 256px thumbnail

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_model: str = ""         # model name + version
    embedding_dim: int = 768
    # embedding_vector stored separately in FAISS; referenced by source_id

    # ── Rights & Access ─────────────────────────────────────────────────────
    rights_status: str = "unknown"    # DPLA rights vocabulary
    url_original: str = ""            # canonical URL at originating institution
    url_iiif: str = ""                # IIIF manifest URL if available

    # ── Provenance ──────────────────────────────────────────────────────────
    metadata_raw_path: str = ""       # path to original metadata XML/JSON

    # ── Quality ─────────────────────────────────────────────────────────────
    quality_score: float = 0.0        # 0–1: OCR confidence / image res / audio SNR

    # ── Pipeline Bookkeeping ────────────────────────────────────────────────
    ingestion_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    ingestion_version: str = "1.0.0"
    notes: str = ""

    # ── Internal (not persisted to CSV, used during processing) ─────────────
    _raw_text: str = field(default="", repr=False)   # raw text before normalisation
    _local_file: str = field(default="", repr=False) # absolute local path (temp)

    def to_dict(self) -> dict:
        """Return a dict suitable for CSV / DuckDB insertion (excludes private fields)."""
        d = asdict(self)
        # Serialise list fields
        d["topic_tags"] = "|".join(self.topic_tags) if self.topic_tags else ""
        # Drop private fields (prefixed with _)
        return {k: v for k, v in d.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusItem":
        """Reconstruct from a dict (e.g., a DuckDB row)."""
        d = dict(d)
        if isinstance(d.get("topic_tags"), str):
            d["topic_tags"] = [t for t in d["topic_tags"].split("|") if t]
        # Drop any unknown keys gracefully
        valid_keys = cls.__dataclass_fields__.keys()
        d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**d)


# ─── DuckDB Table DDL ──────────────────────────────────────────────────────────

CORPUS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS corpus (
    source_id           VARCHAR PRIMARY KEY,
    institution         VARCHAR,
    collection          VARCHAR,
    title               VARCHAR,
    date_original       VARCHAR,
    date_digitized      VARCHAR,
    modality            VARCHAR,
    format              VARCHAR,
    language            VARCHAR,
    geographic_scope    VARCHAR,
    temporal_scope      VARCHAR,
    era_tag             VARCHAR,
    topic_tags          VARCHAR,        -- pipe-separated list
    file_path           VARCHAR,
    transcript_path     VARCHAR,
    thumbnail_path      VARCHAR,
    embedding_model     VARCHAR,
    embedding_dim       INTEGER,
    rights_status       VARCHAR,
    url_original        VARCHAR,
    url_iiif            VARCHAR,
    metadata_raw_path   VARCHAR,
    quality_score       DOUBLE,
    ingestion_timestamp VARCHAR,
    ingestion_version   VARCHAR,
    notes               VARCHAR
);
"""

CORPUS_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_institution  ON corpus(institution);",
    "CREATE INDEX IF NOT EXISTS idx_modality     ON corpus(modality);",
    "CREATE INDEX IF NOT EXISTS idx_era_tag      ON corpus(era_tag);",
    "CREATE INDEX IF NOT EXISTS idx_language     ON corpus(language);",
]
