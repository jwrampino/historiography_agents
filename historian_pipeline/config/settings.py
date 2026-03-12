"""
Global configuration for the Historian AI Ensemble Pipeline — Phase 1.
All paths, API endpoints, model names, and tunable constants live here.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(os.getenv("HISTORIAN_BASE_DIR", "./data"))
RAW_DIR = BASE_DIR / "raw"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"
EMBEDDING_DIR = BASE_DIR / "embeddings"
INDEX_DIR = BASE_DIR / "index"
DB_DIR = BASE_DIR / "db"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist at import time
for _d in [RAW_DIR, TRANSCRIPT_DIR, THUMBNAIL_DIR, EMBEDDING_DIR, INDEX_DIR, DB_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Database ──────────────────────────────────────────────────────────────────
DUCKDB_PATH = DB_DIR / "corpus.duckdb"
SQLITE_PATH = DB_DIR / "pipeline_runs.sqlite"

# ─── FAISS Index ───────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = INDEX_DIR / "corpus.faiss"
FAISS_ID_MAP_PATH = INDEX_DIR / "id_map.json"   # maps FAISS int id → source_id UUID
EMBEDDING_DIM = 768      # native dim of all-mpnet-base-v2 and CLIP ViT-L/14 — no projection needed
FAISS_NLIST = 256        # IVF number of clusters (tune for corpus size)
FAISS_M = 32             # PQ sub-quantizers (768 / 32 = 24 dims each)
FAISS_NBITS = 8          # bits per sub-quantizer

# ─── Embedding Models ──────────────────────────────────────────────────────────
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"   # 768-d native
CLIP_MODEL = "openai/clip-vit-large-patch14"                        # 768-d native
EMBEDDING_BATCH_SIZE = 64

THUMBNAIL_SIZE = (256, 256)

# ─── Archive API Keys (set via environment variables) ─────────────────────────
LOC_API_KEY = os.getenv("LOC_API_KEY", "")           # Library of Congress
INTERNET_ARCHIVE_ACCESS = os.getenv("IA_ACCESS_KEY", "")
INTERNET_ARCHIVE_SECRET = os.getenv("IA_SECRET_KEY", "")
SMITHSONIAN_API_KEY = os.getenv("SMITHSONIAN_API_KEY", "")
# NARA uses no key for public API; authenticated endpoint optional
NARA_API_KEY = os.getenv("NARA_API_KEY", "")

# ─── Ingestion Limits (dev/prod toggle) ───────────────────────────────────────
MAX_ITEMS_PER_QUERY = int(os.getenv("MAX_ITEMS_PER_QUERY", "1000"))
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.5"))  # polite crawl
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30   # seconds

# ─── Quality Thresholds ────────────────────────────────────────────────────────
MIN_IMAGE_RESOLUTION = 100   # pixels on shortest side

# ─── Era Taxonomy (controlled vocabulary) ─────────────────────────────────────
ERA_TAGS = [
    "pre-colonial",
    "colonial",
    "early-modern",
    "revolutionary",
    "antebellum",
    "civil-war",
    "reconstruction",
    "gilded-age",
    "progressive-era",
    "wwi",
    "interwar",
    "wwii",
    "cold-war",
    "civil-rights",
    "late-20th-century",
    "contemporary",
    "unknown",
]

# ─── Modality Enum ─────────────────────────────────────────────────────────────
MODALITIES = ["text", "image", "audio", "video", "map", "mixed"]

# ─── Rights Vocabulary (DPLA-aligned) ─────────────────────────────────────────
RIGHTS_VOCAB = [
    "public domain",
    "no known copyright",
    "rights unclear",
    "in copyright",
    "educational use permitted",
    "unknown",
]
