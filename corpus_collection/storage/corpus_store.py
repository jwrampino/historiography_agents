"""
Storage layer: DuckDB (primary) + CSV export (secondary).
Handles all corpus item persistence, deduplication, and bulk queries.
"""

from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Iterator

import duckdb

from historian_pipeline.storage.schema import CorpusItem, CORPUS_TABLE_DDL, CORPUS_INDEX_DDL
from historian_pipeline.config.settings import DUCKDB_PATH

logger = logging.getLogger(__name__)


class CorpusStore:
    """
    Thread-safe(ish) wrapper around DuckDB for corpus item storage.
    Use as a context manager or call .close() explicitly.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DUCKDB_PATH
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        self.con.execute(CORPUS_TABLE_DDL)
        for idx_sql in CORPUS_INDEX_DDL:
            self.con.execute(idx_sql)
        self.con.commit()
        logger.info(f"Schema initialised at {self.db_path}")

    # ── Write ──────────────────────────────────────────────────────────────

    def insert(self, item: CorpusItem, skip_duplicates: bool = True) -> bool:
        """
        Insert one CorpusItem. Returns True if inserted, False if skipped (duplicate).
        """
        if skip_duplicates and self.exists(item.source_id):
            logger.debug(f"Skipping duplicate source_id={item.source_id}")
            return False

        d = item.to_dict()
        cols = ", ".join(d.keys())
        placeholders = ", ".join(["?" for _ in d])
        values = list(d.values())

        self.con.execute(
            f"INSERT OR REPLACE INTO corpus ({cols}) VALUES ({placeholders})",
            values,
        )
        self.con.commit()
        return True

    def bulk_insert(self, items: List[CorpusItem], skip_duplicates: bool = True) -> int:
        """
        Bulk insert a list of CorpusItems. Returns count of actually inserted rows.
        """
        inserted = 0
        for item in items:
            if self.insert(item, skip_duplicates=skip_duplicates):
                inserted += 1
        logger.info(f"Bulk insert complete: {inserted}/{len(items)} items inserted")
        return inserted

    def update_field(self, source_id: str, field: str, value) -> None:
        """Update a single field on an existing row (e.g., after embedding is generated)."""
        self.con.execute(
            f"UPDATE corpus SET {field} = ? WHERE source_id = ?",
            [value, source_id],
        )
        self.con.commit()

    # ── Read ───────────────────────────────────────────────────────────────

    def exists(self, source_id: str) -> bool:
        result = self.con.execute(
            "SELECT 1 FROM corpus WHERE source_id = ? LIMIT 1", [source_id]
        ).fetchone()
        return result is not None

    def get(self, source_id: str) -> Optional[CorpusItem]:
        row = self.con.execute(
            "SELECT * FROM corpus WHERE source_id = ?", [source_id]
        ).fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in self.con.description]
        return CorpusItem.from_dict(dict(zip(cols, row)))

    def count(self, where: str = "") -> int:
        sql = "SELECT COUNT(*) FROM corpus"
        if where:
            sql += f" WHERE {where}"
        return self.con.execute(sql).fetchone()[0]

    def iter_items(
        self,
        batch_size: int = 500,
        where: str = "",
        order_by: str = "ingestion_timestamp",
    ) -> Iterator[List[CorpusItem]]:
        """
        Yield batches of CorpusItems (memory-efficient for large corpora).
        """
        sql = f"SELECT * FROM corpus"
        if where:
            sql += f" WHERE {where}"
        sql += f" ORDER BY {order_by}"

        offset = 0
        while True:
            batch_sql = sql + f" LIMIT {batch_size} OFFSET {offset}"
            rows = self.con.execute(batch_sql).fetchall()
            if not rows:
                break
            cols = [desc[0] for desc in self.con.description]
            yield [CorpusItem.from_dict(dict(zip(cols, r))) for r in rows]
            offset += batch_size

    def items_without_embeddings(self) -> Iterator[List[CorpusItem]]:
        """Yield items that haven't been embedded yet."""
        yield from self.iter_items(where="embedding_model = '' OR embedding_model IS NULL")

    def items_without_transcripts(self, modality: str = "audio") -> Iterator[List[CorpusItem]]:
        """Yield items of a given modality that haven't been transcribed."""
        yield from self.iter_items(
            where=f"modality = '{modality}' AND (transcript_path = '' OR transcript_path IS NULL)"
        )

    # ── Export ─────────────────────────────────────────────────────────────

    def export_csv(self, out_path: Path, where: str = "") -> int:
        """
        Export corpus to CSV. Returns row count.
        Uses DuckDB's native COPY for efficiency.
        """
        sql = "SELECT * FROM corpus"
        if where:
            sql += f" WHERE {where}"
        self.con.execute(f"COPY ({sql}) TO '{out_path}' (HEADER, DELIMITER ',')")
        count = self.count(where)
        logger.info(f"Exported {count} rows to {out_path}")
        return count

    def export_jsonl(self, out_path: Path, where: str = "") -> int:
        """Export corpus to JSONL (one JSON object per line)."""
        count = 0
        with open(out_path, "w") as f:
            for batch in self.iter_items(where=where):
                for item in batch:
                    f.write(json.dumps(item.to_dict()) + "\n")
                    count += 1
        logger.info(f"Exported {count} rows to {out_path}")
        return count

    # ── Stats ──────────────────────────────────────────────────────────────

    def summary_stats(self) -> dict:
        """Return a summary dict of corpus composition."""
        stats = {}
        stats["total"] = self.count()
        for col in ["institution", "modality", "era_tag", "language"]:
            rows = self.con.execute(
                f"SELECT {col}, COUNT(*) as n FROM corpus GROUP BY {col} ORDER BY n DESC"
            ).fetchall()
            stats[col] = {r[0]: r[1] for r in rows}
        return stats

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
