"""
NARA (National Archives) and Smithsonian Open Access ingestors.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, Optional

from sources.config.settings import (
    NARA_API_KEY,
    SMITHSONIAN_API_KEY,
    RAW_DIR,
)
from sources.ingestors.base import BaseIngestor
from sources.storage.schema import CorpusItem
from sources.utils.text_utils import clean_text, detect_era

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# NARA Ingestor
# ══════════════════════════════════════════════════════════════════════════════

NARA_TYPE_MAP = {
    "archival description": "text",
    "authority record": "text",
    "web page": "text",
    "photograph": "image",
    "film": "video",
    "sound recording": "audio",
    "map": "map",
    "object": "mixed",
    "architectural drawing": "image",
    "textual records": "text",
    "moving images": "video",
    "data files": "text",
    "microfilm": "text",
}


class NARAIngestor(BaseIngestor):
    """
    Ingestor for the National Archives Catalog API.

    catalog.archives.gov/api/v2/records/search now returns an HTML JS app.
    The working endpoint is catalog.archives.gov/proxy/records/search which
    returns proper JSON with the same structure.
    """

    institution_name = "nara"
    SEARCH_URL = "https://catalog.archives.gov/proxy/records/search"

    def fetch_items(
        self,
        query: str,
        max_items: int = 100,
        result_type: Optional[str] = None,
        level: Optional[str] = None,
        **kwargs,
    ) -> Iterator[CorpusItem]:
        fetched = 0
        offset = 0
        rows = min(20, max_items)
        seen_na_ids: set = set()
        max_pages = 5  # never fetch more than 5 pages regardless of dedup
        pages_fetched = 0

        while fetched < max_items and pages_fetched < max_pages:
            params = {
                "q": query,
                "rows": rows,
                "offset": offset,
            }
            if result_type:
                params["resultType"] = result_type
            if level:
                params["level"] = level
            if NARA_API_KEY:
                params["apiKey"] = NARA_API_KEY

            try:
                resp = self._get(self.SEARCH_URL, params=params)
                raw = resp.text
                if not raw or not raw.strip().startswith("{"):
                    logger.error(f"[NARA] Unexpected response (not JSON): {raw[:200]}")
                    break
                data = resp.json()
            except Exception as e:
                logger.error(f"[NARA] API error: {e}")
                break

            hits = data.get("body", {}).get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                if fetched >= max_items:
                    break
                # Deduplicate on naId before parsing (parsing is expensive)
                # naId lives at _source.record.naId, not _source.naId
                _src = hit.get("_source", {})
                na_id = str(_src.get("record", _src).get("naId", ""))
                if na_id and na_id in seen_na_ids:
                    logger.debug(f"[NARA] Skipping duplicate naId: {na_id}")
                    continue
                if na_id:
                    seen_na_ids.add(na_id)
                item = self._parse_hit(hit)
                if item:
                    yield item
                    fetched += 1

            pages_fetched += 1
            total = data.get("body", {}).get("hits", {}).get("total", {}).get("value", 0)
            if fetched >= total:
                break
            offset += rows

        logger.info(f"[NARA] Fetched {fetched} items for query: '{query}'")

    def _extract_pdf_text(self, pdf_url: str, max_pages: int = 3) -> str:
        """
        Fetch a PDF from NARA S3 and extract text from the first max_pages pages.
        Streams download capped at 10MB so large files don't block the pipeline.
        Returns empty string on any failure.
        """
        try:
            import io, warnings
            from pypdf import PdfReader
            # Use (connect, read) tuple: 5s to connect, 10s to read
            resp = self.session.get(pdf_url, timeout=(5, 10), stream=True)
            resp.raise_for_status()
            content = b""
            for chunk in resp.iter_content(chunk_size=65536):
                content += chunk
                if len(content) >= 3 * 1024 * 1024:  # 3MB cap, faster
                    break
            import logging as _logging
            _pypdf_log = _logging.getLogger("pypdf")
            _old_level = _pypdf_log.level
            _pypdf_log.setLevel(_logging.CRITICAL)
            try:
                reader = PdfReader(io.BytesIO(content))
            finally:
                _pypdf_log.setLevel(_old_level)
            pages_text = []
            for page in reader.pages[:1]:  # 1 page only for speed
                t = page.extract_text() or ""
                if t.strip():
                    pages_text.append(t.strip())
            return " ".join(pages_text)
        except Exception as e:
            logger.debug(f"[NARA] PDF extraction failed for {pdf_url}: {e}")
            return ""

    def _build_ancestor_context(self, record: dict) -> str:
        """
        Build a descriptive context string from the ancestors chain:
        record group → series → file unit titles + creator names.
        """
        ancestors = record.get("ancestors", [])
        parts = []
        for anc in sorted(ancestors, key=lambda a: a.get("distance", 99), reverse=True):
            t = anc.get("title", "")
            if t:
                parts.append(t)
            # Add creator heading if present
            for creator in anc.get("creators", [])[:1]:
                heading = creator.get("heading", "")
                if heading:
                    parts.append(f"Created by: {heading}")
        return ". ".join(parts)

    def _parse_hit(self, hit: dict) -> Optional[CorpusItem]:
        import uuid

        source_id = str(uuid.uuid4())
        source = hit.get("_source", {})
        record = source.get("record", source)
        meta_path = self.save_raw_metadata(record, source_id)

        na_id = record.get("naId", "")
        title = record.get("title", "Untitled NARA Record")

        # ── Try to get real text ──────────────────────────────────────────
        # 1. scopeAndContent (rare but possible in some records)
        description = ""
        scope = record.get("scopeAndContent", "")
        if isinstance(scope, dict):
            description = scope.get("scopeAndContent", "")
        elif isinstance(scope, str):
            description = scope
        if not description:
            description = record.get("description", "")

        # 2. PDF extraction — disabled by default (NARA S3 connections hang).
        #    Ancestor context (step 3) provides sufficient text for embedding.
        #    Re-enable: export NARA_PDF_ENABLED=true
        import os as _os
        if not description and _os.environ.get("NARA_PDF_ENABLED", "").lower() == "true":
            digital_objects = record.get("digitalObjects", [])
            pdf_urls = [
                d["objectUrl"] for d in digital_objects
                if d.get("objectFilename", "").lower().endswith(".pdf")
                and d.get("objectUrl")
            ]
            if pdf_urls:
                logger.debug(f"[NARA] Fetching PDF for '{title[:60]}'")
                description = self._extract_pdf_text(pdf_urls[0])

        # 3. Ancestor context as fallback
        if not description:
            description = self._build_ancestor_context(record)
            if description:
                logger.debug(f"[NARA] Using ancestor context for '{title[:60]}'")

        # ── Dates ─────────────────────────────────────────────────────────
        # Priority: item dates[] > ancestor exact dates (no "ca.") > title year extraction
        dates = record.get("dates", [])
        date_str = ""
        if dates and isinstance(dates, list):
            d = dates[0]
            date_str = d.get("dateRange", {}).get("inclusiveStartDate", d.get("year", ""))

        # Fall back to closest ancestor with an exact (non-approximate) year.
        # Skip ancestors where dateQualifier == "ca." — these are catalog batch dates,
        # not the item's historical date.
        if not date_str:
            ancestors_with_year = [
                a for a in record.get("ancestors", [])
                if isinstance(a.get("inclusiveStartDate"), dict)
                and a["inclusiveStartDate"].get("year")
                and a["inclusiveStartDate"].get("dateQualifier", "") != "ca."
            ]
            if ancestors_with_year:
                closest = min(ancestors_with_year, key=lambda a: a.get("distance", 99))
                date_str = str(closest["inclusiveStartDate"]["year"])

        # Last resort: extract a year from title or closest ancestor title.
        # Also discard implausibly old dates (record group creation dates like 1785).
        if not date_str:
            import re as _re
            years_in_title = _re.findall(r"\b(1[5-9]\d{2}|20[012]\d)\b", str(title))
            if years_in_title:
                date_str = years_in_title[0]

        # If still empty, or date is suspiciously early (pre-1800 for modern records),
        # try extracting from the closest ancestor's title string.
        if not date_str or (date_str.isdigit() and int(date_str) < 1800
                            and title and len(title) > 20):
            import re as _re
            ancestors_sorted = sorted(record.get("ancestors", []),
                                      key=lambda a: a.get("distance", 99))
            for anc in ancestors_sorted:
                anc_title = anc.get("title", "")
                years = _re.findall(r"\b(1[6-9]\d{2}|20[012]\d)\b", str(anc_title))
                if years:
                    date_str = years[0]
                    break

        # ── Modality ──────────────────────────────────────────────────────
        rec_types = record.get("generalRecordsTypes", [])
        rec_type = rec_types[0].lower() if rec_types else ""
        modality = NARA_TYPE_MAP.get(rec_type, "text")

        # ── Geography ─────────────────────────────────────────────────────
        geo_scope = ""
        locations = record.get("locationArray", [])
        if locations:
            geo_parts = []
            for loc in locations[:3]:
                loc_data = loc.get("location", loc)
                name = loc_data.get("name", "") if isinstance(loc_data, dict) else ""
                if name:
                    geo_parts.append(name)
            geo_scope = "; ".join(geo_parts)

        subjects = record.get("subject", [])
        topic_tags = []
        for s in subjects:
            if isinstance(s, str):
                topic_tags.append(s.strip())
            elif isinstance(s, dict):
                # NARA subjects are often {"term": "...", "type": "topical"}
                term = s.get("term", "") or s.get("heading", "") or s.get("name", "")
                if term:
                    topic_tags.append(term.strip())
        topic_tags = topic_tags[:10]

        era_tag = detect_era(date_str[:4] if date_str else "")
        url_original = f"https://catalog.archives.gov/id/{na_id}" if na_id else ""
        quality_score = self.text_quality_score(description) if description else 0.2

        item = CorpusItem(
            source_id=source_id,
            institution="National Archives (NARA)",
            collection="",
            title=str(title)[:500],
            date_original=str(date_str),
            modality=modality,
            format=rec_type,
            language="en",
            geographic_scope=geo_scope,
            era_tag=era_tag,
            topic_tags=topic_tags,
            rights_status="public domain",
            url_original=url_original,
            metadata_raw_path=str(meta_path),
            quality_score=quality_score,
            notes=description[:500].replace("\n", " ").replace("\r", " ") if description else "",
        )
        item._raw_text = clean_text(description)
        return item


# ══════════════════════════════════════════════════════════════════════════════
# Smithsonian Ingestor
# ══════════════════════════════════════════════════════════════════════════════

SMITHSONIAN_UNIT_MAP = {
    "NMNH": "National Museum of Natural History",
    "NMAAHC": "National Museum of African American History and Culture",
    "NMAH": "National Museum of American History",
    "NASM": "National Air and Space Museum",
    "NPG": "National Portrait Gallery",
    "SI": "Smithsonian Institution",
}


class SmithsonianIngestor(BaseIngestor):
    """
    Ingestor for Smithsonian Open Access API.
    Covers object photography, ethnographic documentation, scientific specimens,
    and artwork across all Smithsonian museums.
    """

    institution_name = "smithsonian"
    SEARCH_URL = "https://api.si.edu/openaccess/api/v1.0/search"
    CONTENT_URL = "https://api.si.edu/openaccess/api/v1.0/content/{id}"

    def fetch_items(
        self,
        query: str,
        max_items: int = 100,
        unit_code: Optional[str] = None,     # "NMNH", "NMAH", etc.
        online_media_type: Optional[str] = None,  # "Images", "Audio", "Videos"
        **kwargs,
    ) -> Iterator[CorpusItem]:
        """
        Yield CorpusItems from the Smithsonian Open Access API.

        Args:
            query:             Search query
            max_items:         Max results
            unit_code:         Filter by museum unit code
            online_media_type: Filter by media type ("Images", "Audio", "Videos")
        """
        if not SMITHSONIAN_API_KEY:
            logger.error("[Smithsonian] API key required — set SMITHSONIAN_API_KEY env var")
            return

        fetched = 0
        start = 0
        rows = min(100, max_items)

        while fetched < max_items:
            params = {
                "api_key": SMITHSONIAN_API_KEY,
                "q": query,
                "rows": rows,
                "start": start,
            }
            if unit_code:
                params["unit_code"] = unit_code
            if online_media_type:
                params["online_media_type"] = online_media_type

            try:
                resp = self._get(self.SEARCH_URL, params=params)
                data = resp.json()
            except Exception as e:
                logger.error(f"[Smithsonian] API error: {e}")
                break

            rows_data = data.get("response", {}).get("rows", [])
            if not rows_data:
                break

            for row in rows_data:
                if fetched >= max_items:
                    break
                item = self._parse_row(row)
                if item:
                    yield item
                    fetched += 1

            row_count = data.get("response", {}).get("rowCount", 0)
            if fetched >= row_count:
                break
            start += rows

        logger.info(f"[Smithsonian] Fetched {fetched} items for query: '{query}'")

    # Library/archive catalog units — no text, no images, not worth ingesting
    _SKIP_UNITS = {"SIL", "SLA_SRO", "SIA"}

    def _build_metadata_text(self, title: str, indexed: dict) -> str:
        """
        Build a short descriptive string from structured metadata when
        freeText is empty. Used as _raw_text fallback for text embedding.
        """
        parts = [title] if title and title != "Untitled Smithsonian Object" else []
        names = indexed.get("name", [])
        author_parts = []
        for n in names[:3]:
            if isinstance(n, dict):
                author_parts.append(n.get("content", ""))
            elif isinstance(n, str):
                author_parts.append(n)
        if author_parts:
            parts.append("By " + ", ".join(filter(None, author_parts)))
        obj_types = indexed.get("object_type", [])
        if obj_types:
            parts.append("Type: " + ", ".join(obj_types[:3]))
        topics = indexed.get("topic", [])
        if topics:
            parts.append("Topics: " + ", ".join(topics[:5]))
        places = indexed.get("place", [])
        if places:
            parts.append("Place: " + ", ".join(places[:3]))
        dates = indexed.get("date", [])
        if dates:
            parts.append("Date: " + str(dates[0]))
        return ". ".join(filter(None, parts))

    def _parse_row(self, row: dict) -> Optional[CorpusItem]:
        import uuid

        content = row.get("content", {})
        descriptive = content.get("descriptiveNonRepeating", {})
        indexed = content.get("indexedStructured", {})
        free_text = content.get("freeText", {})

        # Skip library catalog units — no text and no downloadable images
        unit_code = descriptive.get("unit_code", row.get("unitCode", "SI"))
        if unit_code in self._SKIP_UNITS:
            return None

        source_id = str(uuid.uuid4())
        meta_path = self.save_raw_metadata(row, source_id)

        import re as _re
        title_raw = descriptive.get("title", {})
        title = title_raw.get("content", "Untitled Smithsonian Object") if isinstance(title_raw, dict) else str(title_raw)
        title = _re.sub(r"<[^>]+>", "", title).strip()  # strip HTML tags e.g. <I>, </I>
        record_id = row.get("id", "")

        # Dates
        dates = indexed.get("date", [])
        date_str = dates[0] if dates else ""

        # Modality and media
        online_media = descriptive.get("online_media", {}).get("media", [])
        modality = "text"  # default — only upgrade to image if we actually download a file
        for m in online_media:
            mt = m.get("type", "").lower()
            if "audio" in mt:
                modality = "audio"
                break
            elif "video" in mt:
                modality = "video"
                break
            elif "image" in mt or "3d" in mt:
                modality = "image"
                break

        # Topics / subjects
        topics = indexed.get("topic", [])
        topic_tags = [t for t in topics if isinstance(t, str)][:10]

        # Geographic
        places = indexed.get("place", [])
        geo_scope = "; ".join(p for p in places[:3] if isinstance(p, str))

        # Institution
        institution_name = "Smithsonian Institution"

        # ── Image download ────────────────────────────────────────────────
        # Download thumbnail if available; file_path set → embedder uses CLIP.
        # thumbnail_path is a smaller copy for display; file_path is for CLIP.
        file_path = ""
        thumbnail_path = ""
        quality_score = 0.3

        if online_media:
            first_media = online_media[0]
            thumb_url = first_media.get("thumbnail", "") or first_media.get("content", "")
            if thumb_url:
                try:
                    local = self.download_file(thumb_url, filename=f"{source_id}.jpg")
                    file_path = str(local)        # CLIP reads from here
                    thumbnail_path = file_path    # same file serves as thumbnail
                    quality_score = self.image_quality_score(local)
                    modality = "image"
                    logger.debug(f"[Smithsonian] Downloaded image for '{title[:50]}'")
                except Exception as e:
                    logger.warning(f"[Smithsonian] Image download failed: {e}")
                    quality_score = 0.4

        # ── Text content ──────────────────────────────────────────────────
        # 1. freeText.notes (rich descriptions when present)
        notes_array = free_text.get("notes", [])
        notes_text = " ".join(
            n.get("content", "") for n in notes_array if isinstance(n, dict)
        )
        # 2. All other freeText fields (physicalDescription, creditLine, etc.)
        if not notes_text:
            for key, val in free_text.items():
                if key == "notes":
                    continue
                if isinstance(val, list):
                    for entry in val:
                        if isinstance(entry, dict) and entry.get("content"):
                            notes_text += " " + entry["content"]
                        elif isinstance(entry, str):
                            notes_text += " " + entry
            notes_text = notes_text.strip()
        # 3. Structured metadata fallback — always have something for text embedding
        raw_text = notes_text if notes_text else self._build_metadata_text(title, indexed)

        era_tag = detect_era(date_str[:4] if date_str else "")
        url_original = f"https://collections.si.edu/search/detail/{record_id}" if record_id else ""

        transcript_path = ""
        if notes_text:
            t_dir = RAW_DIR / self.institution_name / "notes"
            t_dir.mkdir(parents=True, exist_ok=True)
            t_path = t_dir / f"{source_id}.txt"
            t_path.write_text(clean_text(notes_text), encoding="utf-8")
            transcript_path = str(t_path)

        item = CorpusItem(
            source_id=source_id,
            institution=institution_name,
            collection=descriptive.get("data_source", ""),
            title=str(title)[:500],
            date_original=str(date_str),
            modality=modality,
            format="image/jpeg" if modality == "image" else "text/plain",
            language="en",
            geographic_scope=geo_scope,
            era_tag=era_tag,
            topic_tags=topic_tags,
            file_path=file_path,
            transcript_path=transcript_path,
            thumbnail_path=thumbnail_path,
            rights_status="no known copyright",
            url_original=url_original,
            metadata_raw_path=str(meta_path),
            quality_score=quality_score,
            notes=raw_text[:500],
        )
        item._raw_text = raw_text  # text embedding fallback if no image downloaded
        return item
