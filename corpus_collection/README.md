# Historian AI Ensemble Pipeline — Phase 1: Corpus Construction

## Overview

Full end-to-end pipeline for constructing the multimodal historical corpus described in the research specification. Ingests from four prioritised archives, runs OCR/transcription, generates cross-modal embeddings, and builds a FAISS ANN index.

```
historian_pipeline/
├── config/
│   └── settings.py          # All paths, model names, API key loading, constants
├── ingestors/
│   ├── base.py              # Abstract base: rate limiting, retries, download, thumbnails
│   ├── loc_ingestor.py      # Library of Congress + Chronicling America
│   ├── internet_archive_ingestor.py
│   └── nara_smithsonian_ingestor.py
├── processing/
│   └── ocr_transcription.py # OCR (Tesseract/EasyOCR) + Whisper audio transcription
├── embeddings/
│   ├── embedder.py          # TextEmbedder, ImageEmbedder (CLIP), CorpusEmbedder (router)
│   └── faiss_index.py       # IVF-PQ index: build, save, load, search
├── storage/
│   ├── schema.py            # CorpusItem dataclass + DuckDB DDL
│   └── corpus_store.py      # DuckDB CRUD, bulk insert, CSV/JSONL export
├── utils/
│   └── text_utils.py        # Text cleaning, era detection, language normalisation
└── pipeline.py              # Phase1Pipeline orchestrator + CLI
```

## Installation

```bash
pip install -r requirements.txt

# System dependency for Tesseract OCR:
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS:
brew install tesseract
```

## Environment Variables

```bash
# Archive API Keys (only Smithsonian strictly required; others work anonymously)
export SMITHSONIAN_API_KEY="3m2sSgGCdhezvVJwC3dpgta1NjoWos6kC4rfch4N"    # https://api.si.edu/
export LOC_API_KEY="your_key_here"            # optional; increases rate limits
export NARA_API_KEY="your_key_here"           # optional

# Internet Archive (optional; anonymous access works for public items)
export IA_ACCESS_KEY="your_access_key"
export IA_SECRET_KEY="your_secret_key"

# Base data directory (default: ./data)
export HISTORIAN_BASE_DIR="/path/to/data"

# Ingestion limits
export MAX_ITEMS_PER_QUERY="1000"
export REQUEST_DELAY_SECONDS="0.5"
```

## Quick Start

```bash
# Run the full pipeline with a single query across all archives (50 items each)
python -m historian_pipeline.pipeline --query "reconstruction era" --max-items 50

# Run with more items
python -m historian_pipeline.pipeline -q "civil rights movement photographs" -n 500

# Run only specific stages
python -m historian_pipeline.pipeline -q "great migration" --stages ingest embed index

# Use a full JSON config (see example_config.json)
python -m historian_pipeline.pipeline --config example_config.json
```

## Programmatic Usage

```python
from historian_pipeline.pipeline import Phase1Pipeline, PipelineConfig, IngestorConfig

# Configure per-archive queries
cfg = PipelineConfig(
    loc=IngestorConfig(
        enabled=True,
        queries=["reconstruction era", "freedmen bureau"],
        max_items_per_query=200,
    ),
    chronicling_america=IngestorConfig(
        enabled=True,
        queries=["reconstruction"],
        max_items_per_query=500,
        extra_params={"date_start": "1865-01-01", "date_end": "1877-12-31"},
    ),
    internet_archive=IngestorConfig(
        enabled=True,
        queries=["reconstruction history"],
        max_items_per_query=200,
        extra_params={"mediatype": "texts"},
    ),
    nara=IngestorConfig(
        enabled=True,
        queries=["freedmen bureau records"],
        max_items_per_query=200,
    ),
    smithsonian=IngestorConfig(
        enabled=True,
        queries=["reconstruction era artifacts"],
        max_items_per_query=100,
    ),
    run_ocr=True,
    run_transcription=False,  # skip Whisper if no audio
    run_embedding=True,
    run_indexing=True,
    run_export=True,
)

with Phase1Pipeline(cfg) as pipeline:
    summary = pipeline.run()
    print(f"Corpus size: {summary['total_corpus_size']}")
```

## FAISS Index Query (after pipeline run)

```python
from historian_pipeline.embeddings.faiss_index import CorpusIndex
from historian_pipeline.embeddings.embedder import CorpusEmbedder
from historian_pipeline.storage.corpus_store import CorpusStore

index = CorpusIndex()
index.load()

embedder = CorpusEmbedder()
query_vec = embedder.embedder.text_embedder.embed_one(
    "photographs of freedmen during reconstruction"
)

results = index.search(query_vec, top_k=10)

store = CorpusStore()
for r in results:
    item = store.get(r["source_id"])
    print(f"[{r['similarity_score']:.3f}] {item.title} ({item.institution})")
```

## Data Directory Layout

```
data/
├── raw/
│   ├── loc/                    # Downloaded LOC files + metadata JSON
│   ├── chronicling_america/    # Newspaper pages + OCR text
│   ├── internet_archive/       # IA downloads
│   ├── nara/                   # NARA thumbnails + metadata
│   └── smithsonian/            # Smithsonian thumbnails + notes
├── transcripts/                # OCR + Whisper output (.txt)
├── thumbnails/                 # 256×256 JPEG thumbnails
├── embeddings/                 # Per-item .npy vectors (sharded by source_id prefix)
├── index/
│   ├── corpus.faiss            # FAISS IVF-PQ index
│   └── id_map.json             # int_id ↔ source_id mapping
├── db/
│   └── corpus.duckdb           # Master corpus database
├── corpus_export.csv           # Full corpus CSV export
└── logs/
    └── pipeline.log
```

## Key Design Decisions

**Deduplication**: items are keyed by `source_id` (UUID); DuckDB enforces uniqueness. Re-running the pipeline skips already-ingested items.

**Embedding alignment**: all modalities are projected to 1024-d. Text uses sentence-transformers (768→1024 zero-padded); images use CLIP ViT-L/14 (768→1024); once training data is available, replace zero-padding with a learned linear projection layer.

**FAISS fallback**: corpora < 9,984 vectors (nlist=256 × 39) automatically use a flat IndexFlatIP instead of IVF-PQ. Rebuild with IVF-PQ once corpus grows.

**Rate limiting**: all ingestors respect `REQUEST_DELAY_SECONDS` (default 0.5s) between API calls to avoid bans.

**US federal records rights**: NARA items default to `public domain` (17 U.S.C. § 105). Always verify individual item rights before redistribution.
