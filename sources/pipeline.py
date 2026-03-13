"""
Phase 1 Pipeline Orchestrator.

Runs the corpus construction pipeline:
  1. Ingest from all configured archives (metadata + description text via API)
  2. Embed all items directly from API-returned text (_raw_text field)
  3. Build FAISS index
  4. Export corpus CSV

No OCR or file downloads — embeddings are generated from the text
that each archive API returns (titles, descriptions, subject headings,
OCR text for Chronicling America pages, etc).

Usage:
    python -m sources.pipeline --query "reconstruction era" --max-items 500
    python -m sources.pipeline --config pipeline_config.json
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sources.config.settings import (
    BASE_DIR,
    LOC_API_KEY,
    SMITHSONIAN_API_KEY,
    NARA_API_KEY,
    EMBEDDING_DIR,
)
from sources.storage.corpus_store import CorpusStore
from sources.storage.schema import CorpusItem
from sources.ingestors.loc_ingestor import LOCIngestor, ChroniclingAmericaIngestor
from sources.ingestors.internet_archive_ingestor import InternetArchiveIngestor
from sources.ingestors.nara_smithsonian_ingestor import NARAIngestor, SmithsonianIngestor
from sources.embeddings.embedder import CorpusEmbedder
from sources.embeddings.faiss_index import CorpusIndex

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestorConfig:
    enabled: bool = True
    queries: List[str] = field(default_factory=list)
    max_items_per_query: int = 100
    extra_params: Dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    # Per-archive configs
    loc: IngestorConfig = field(default_factory=IngestorConfig)
    chronicling_america: IngestorConfig = field(default_factory=IngestorConfig)
    internet_archive: IngestorConfig = field(default_factory=IngestorConfig)
    nara: IngestorConfig = field(default_factory=IngestorConfig)
    smithsonian: IngestorConfig = field(default_factory=IngestorConfig)

    # Pipeline stages
    run_ingestion: bool = True
    run_embedding: bool = True
    run_indexing: bool = True
    run_export: bool = True

    # Output
    export_csv_path: str = str(BASE_DIR / "corpus_export.csv")

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        for archive in ["loc", "chronicling_america", "internet_archive", "nara", "smithsonian"]:
            if archive in data:
                setattr(cfg, archive, IngestorConfig(**data[archive]))
        for key in ["run_ingestion", "run_embedding", "run_indexing",
                    "run_export", "export_csv_path"]:
            if key in data:
                setattr(cfg, key, data[key])
        return cfg

    @classmethod
    def quick(cls, query: str, max_items: int = 100) -> "PipelineConfig":
        """Build a quick single-query config hitting all enabled archives."""
        return cls(
            loc=IngestorConfig(
                enabled=bool(LOC_API_KEY),
                queries=[query],
                max_items_per_query=max_items,
            ),
            chronicling_america=IngestorConfig(
                enabled=True,
                queries=[query],
                max_items_per_query=max_items,
            ),
            internet_archive=IngestorConfig(
                enabled=True,
                queries=[query],
                max_items_per_query=max_items,
            ),
            nara=IngestorConfig(
                enabled=True,
                queries=[query],
                max_items_per_query=max_items,
            ),
            smithsonian=IngestorConfig(
                enabled=bool(SMITHSONIAN_API_KEY),
                queries=[query],
                max_items_per_query=max_items,
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class Phase1Pipeline:
    """
    Orchestrates the Phase 1 corpus construction pipeline.

    Embedding strategy (no OCR):
      Each ingestor populates item._raw_text with whatever text the API returns:
        - LOC:               description fields, subject headings
        - Chronicling Am.:   full OCR text of the newspaper page (fetched by the API)
        - Internet Archive:  description text
        - NARA:              scope-and-content / description fields
        - Smithsonian:       notes and descriptive fields
      The embedder uses _raw_text first, then falls back to title + notes.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.store = CorpusStore()
        self.index = CorpusIndex()
        self._embedder: Optional[CorpusEmbedder] = None
        self._pending_embed: List[CorpusItem] = []

    @property
    def embedder(self) -> CorpusEmbedder:
        if self._embedder is None:
            self._embedder = CorpusEmbedder()
        return self._embedder

    # ── Public Entry Point ─────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the pipeline. Returns a summary dict."""
        t0 = time.monotonic()
        summary = {"stages": {}}

        if self.config.run_ingestion:
            n = self._stage_ingest()
            summary["stages"]["ingestion"] = {"items_inserted": n}
            logger.info(f"✓ Ingestion complete: {n} new items")

        if self.config.run_embedding:
            n = self._stage_embedding()
            summary["stages"]["embedding"] = {"items_embedded": n}
            logger.info(f"✓ Embedding complete: {n} items embedded")

        if self.config.run_indexing:
            n = self._stage_indexing()
            summary["stages"]["indexing"] = {"vectors_indexed": n}
            logger.info(f"✓ Indexing complete: {n} vectors in FAISS")

        if self.config.run_export:
            csv_path = self._stage_export()
            summary["stages"]["export"] = {"csv_path": str(csv_path)}
            logger.info(f"✓ Export complete: {csv_path}")

        summary["total_corpus_size"] = self.store.count()
        summary["elapsed_seconds"] = round(time.monotonic() - t0, 1)
        summary["corpus_stats"] = self.store.summary_stats()

        return summary

    # ── Stage: Ingestion ───────────────────────────────────────────────────

    def _stage_ingest(self) -> int:
        """
        Fetch items from all enabled archives and insert into DuckDB.
        Items are also kept in _pending_embed (with _raw_text alive) so the
        embedding stage can use them directly without reloading from DB.
        """
        total_inserted = 0
        self._pending_embed = []

        for name, ingestor, icfg in self._build_ingestors():
            if not icfg.enabled or not icfg.queries:
                continue
            for query in icfg.queries:
                logger.info(f"[{name}] query='{query}' max={icfg.max_items_per_query}")
                try:
                    for item in ingestor.fetch_items(
                        query,
                        max_items=icfg.max_items_per_query,
                        **icfg.extra_params,
                    ):
                        if self.store.insert(item):
                            total_inserted += 1
                            self._pending_embed.append(item)
                except Exception as e:
                    logger.error(f"[{name}] Ingestion error: {e}", exc_info=True)

        return total_inserted

    def _build_ingestors(self):
        return [
            ("LOC",                LOCIngestor(api_key=LOC_API_KEY),                self.config.loc),
            ("Chronicling America", ChroniclingAmericaIngestor(),                    self.config.chronicling_america),
            ("Internet Archive",   InternetArchiveIngestor(),                        self.config.internet_archive),
            ("NARA",               NARAIngestor(api_key=NARA_API_KEY),               self.config.nara),
            ("Smithsonian",        SmithsonianIngestor(api_key=SMITHSONIAN_API_KEY), self.config.smithsonian),
        ]

    # ── Stage: Embedding ───────────────────────────────────────────────────

    def _stage_embedding(self) -> int:
        """
        Embed all un-embedded items.

        If ingestion just ran, uses items still in memory (_raw_text populated).
        If run standalone (--stages embed), reloads items from DuckDB — the
        embedder then falls back to title + notes as the text source.
        """
        if self._pending_embed:
            items_to_embed = self._pending_embed
            logger.info(f"Embedding {len(items_to_embed)} freshly ingested items (raw text in memory)")
        else:
            items_to_embed = []
            for batch in self.store.items_without_embeddings():
                items_to_embed.extend(batch)
            logger.info(f"Embedding {len(items_to_embed)} items loaded from DB")

        if not items_to_embed:
            logger.info("Nothing to embed")
            return 0

        processed = 0
        BATCH = 128

        for i in range(0, len(items_to_embed), BATCH):
            batch = items_to_embed[i: i + BATCH]
            if i % 512 == 0:
                logger.info(f"  Embedding {i}/{len(items_to_embed)} ...")
            results = self.embedder.embed_batch(batch, show_progress=False)
            for item, (vec, model_name) in zip(batch, results):
                self._save_embedding(vec, item.source_id)
                self.store.update_field(item.source_id, "embedding_model", model_name)
                processed += 1

        self._pending_embed = []
        return processed

    @staticmethod
    def _save_embedding(vec: np.ndarray, source_id: str) -> Path:
        shard = source_id[:2]
        out_dir = EMBEDDING_DIR / shard
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{source_id}.npy"
        np.save(str(out_path), vec)
        return out_path

    # ── Stage: FAISS Indexing ──────────────────────────────────────────────

    def _stage_indexing(self) -> int:
        """Load all saved .npy embeddings and build the FAISS index."""
        source_ids, vecs = [], []

        for npy_path in sorted(EMBEDDING_DIR.rglob("*.npy")):
            try:
                vecs.append(np.load(str(npy_path)))
                source_ids.append(npy_path.stem)
            except Exception as e:
                logger.warning(f"Could not load {npy_path}: {e}")

        if not vecs:
            logger.warning("No embeddings found — skipping indexing")
            return 0

        self.index.build(np.stack(vecs, axis=0).astype(np.float32), source_ids)
        self.index.save()
        return len(source_ids)

    # ── Stage: Export ──────────────────────────────────────────────────────

    def _stage_export(self) -> Path:
        out_path = Path(self.config.export_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.export_csv(out_path)
        return out_path

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logging(level: str = "INFO"):
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(BASE_DIR / "logs" / "pipeline.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Historian AI Pipeline — Phase 1: Corpus Construction"
    )
    parser.add_argument(
        "--query", "-q",
        default="reconstruction era american history",
        help="Search query for quick mode (all archives)",
    )
    parser.add_argument(
        "--max-items", "-n",
        type=int,
        default=50,
        help="Max items per query per archive",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to a JSON pipeline config file",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["ingest", "embed", "index", "export"],
        default=None,
        help="Run only specific stages (default: all)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    cfg = PipelineConfig.from_json(args.config) if args.config else \
          PipelineConfig.quick(args.query, max_items=args.max_items)

    if args.stages:
        stage_map = {
            "ingest": "run_ingestion",
            "embed":  "run_embedding",
            "index":  "run_indexing",
            "export": "run_export",
        }
        for attr in stage_map.values():
            setattr(cfg, attr, False)
        for s in args.stages:
            setattr(cfg, stage_map[s], True)

    with Phase1Pipeline(cfg) as pipeline:
        summary = pipeline.run()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
