"""
Library of Congress ingestor.
Covers two endpoints:
  1. loc.gov general search API  — manuscripts, photographs, maps, audio, etc.
  2. Chronicling America via loc.gov newspapers API — digitized newspaper pages
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, Optional

from historian_pipeline.config.settings import LOC_API_KEY, RAW_DIR
from historian_pipeline.ingestors.base import BaseIngestor
from historian_pipeline.storage.schema import CorpusItem
from historian_pipeline.utils.text_utils import clean_text, detect_era

logger = logging.getLogger(__name__)

# ─── LOC format → our modality mapping ────────────────────────────────────────
LOC_FORMAT_MAP = {
    "Photographs": "image",
    "Maps": "map",
    "Audio": "audio",
    "Moving Image": "video",
    "Manuscripts/Mixed Material": "text",
    "Notated Music": "text",
    "Newspapers": "text",
    "Serials": "text",
    "Books": "text",
    "Web Archives": "text",
    "Legislation": "text",
    "3D Objects": "mixed",
}

LOC_RIGHTS_MAP = {
    "public domain": "public domain",
    "no known copyright": "no known copyright",
    "rights advisory": "rights unclear",
}


class LOCIngestor(BaseIngestor):
    """Ingestor for the Library of Congress loc.gov JSON API."""

    institution_name = "loc"
    BASE_URL = "https://www.loc.gov/search/"

    def fetch_items(
        self,
        query: str,
        max_items: int = 100,
        fa: Optional[str] = None,
        date_range: Optional[tuple] = None,
        **kwargs,
    ) -> Iterator[CorpusItem]:
        fetched = 0
        page = 1
        page_size = min(25, max_items)

        while fetched < max_items:
            params = {
                "q": query,
                "fo": "json",
                "c": page_size,
                "sp": page,
                "at": "results,pagination",
            }
            if fa:
                params["fa"] = fa
            if date_range:
                params["dates"] = f"{date_range[0]}/{date_range[1]}"
            if LOC_API_KEY:
                params["api_key"] = LOC_API_KEY

            try:
                resp = self._get(self.BASE_URL, params=params)
                data = resp.json()
            except Exception as e:
                logger.error(f"[LOC] API error on page {page}: {e}")
                break

            results = data.get("results", [])
            if not results:
                logger.info(f"[LOC] No more results at page {page}")
                break

            for result in results:
                if fetched >= max_items:
                    break
                item = self._parse_result(result)
                if item:
                    yield item
                    fetched += 1

            pagination = data.get("pagination", {})
            if not pagination.get("next"):
                break
            page += 1

        logger.info(f"[LOC] Fetched {fetched} items for query: '{query}'")

    def _parse_result(self, result: dict) -> Optional[CorpusItem]:
        import uuid

        source_id = str(uuid.uuid4())
        meta_path = self.save_raw_metadata(result, source_id)

        title = result.get("title", "Untitled")
        url_original = result.get("url", "")
        date_str = result.get("date", "")
        description = " ".join(result.get("description", []))

        formats = result.get("original_format", [])
        modality = "text"
        for fmt in formats:
            if fmt in LOC_FORMAT_MAP:
                modality = LOC_FORMAT_MAP[fmt]
                break

        subjects = result.get("subject", [])
        topic_tags = [s.strip() for s in subjects if isinstance(s, str)][:10]

        locations = result.get("location", [])
        geo_scope = "; ".join(str(loc) for loc in locations) if locations else ""

        rights_raw = result.get("rights_advisory", ["unknown"])
        rights_text = (rights_raw[0] if rights_raw else "unknown").lower()
        rights_status = "unknown"
        for key, val in LOC_RIGHTS_MAP.items():
            if key in rights_text:
                rights_status = val
                break

        era_tag = detect_era(date_str)

        file_path = ""
        thumbnail_path = ""
        resources = result.get("resources", [])
        primary_url = resources[0].get("url", "") if resources else ""

        if primary_url and modality == "image":
            try:
                local_file = self.download_file(primary_url, filename=f"{source_id}.jpg")
                file_path = str(local_file.relative_to(RAW_DIR.parent))
                thumb = self.make_thumbnail(local_file, source_id)
                if thumb:
                    thumbnail_path = str(thumb.relative_to(RAW_DIR.parent))
            except Exception as e:
                logger.warning(f"[LOC] Could not download image {primary_url}: {e}")

        url_iiif = ""
        image_services = result.get("image_services", {})
        if image_services.get("iiif_service"):
            url_iiif = image_services["iiif_service"]

        item = CorpusItem(
            source_id=source_id,
            institution="Library of Congress",
            collection=result.get("partof", [""])[0] if result.get("partof") else "",
            title=title,
            date_original=date_str,
            modality=modality,
            format=result.get("mime_type", [""])[0] if result.get("mime_type") else "",
            language=result.get("language", ["en"])[0] if result.get("language") else "en",
            geographic_scope=geo_scope,
            era_tag=era_tag,
            topic_tags=topic_tags,
            file_path=file_path,
            thumbnail_path=thumbnail_path,
            rights_status=rights_status,
            url_original=url_original,
            url_iiif=url_iiif,
            metadata_raw_path=str(meta_path),
            quality_score=1.0 if file_path else 0.5,
            notes=description[:500] if description else "",
        )
        item._raw_text = description
        return item


class ChroniclingAmericaIngestor(BaseIngestor):
    """
    Ingestor for Chronicling America digitized newspapers via loc.gov API.

    The old chroniclingamerica.loc.gov/search/pages/results/ endpoint now
    returns a 308 redirect to www.loc.gov which then 404s. The correct
    endpoint is https://www.loc.gov/newspapers/ which returns the same
    Chronicling America pages with description text (OCR snippets) included.
    """

    institution_name = "chronicling_america"
    SEARCH_URL = "https://www.loc.gov/newspapers/"

    def fetch_items(
        self,
        query: str,
        max_items: int = 100,
        date_start: str = "",
        date_end: str = "",
        state: str = "",
        **kwargs,
    ) -> Iterator[CorpusItem]:
        fetched = 0
        page = 1
        page_size = min(25, max_items)

        while fetched < max_items:
            params = {
                "q": query,
                "fo": "json",
                "c": page_size,
                "sp": page,
            }
            if date_start and date_end:
                params["dates"] = f"{date_start[:4]}/{date_end[:4]}"
            elif date_start:
                params["dates"] = f"{date_start[:4]}"
            if state:
                params["fa"] = f"location_state:{state.lower()}"

            try:
                resp = self._get(self.SEARCH_URL, params=params)
                data = resp.json()
            except Exception as e:
                logger.error(f"[ChronAm] API error on page {page}: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for item_data in results:
                if fetched >= max_items:
                    break
                item = self._parse_page(item_data)
                if item:
                    yield item
                    fetched += 1

            pagination = data.get("pagination", {})
            if not pagination.get("next"):
                break
            page += 1

        logger.info(f"[ChronAm] Fetched {fetched} newspaper pages for query: '{query}'")

    def _parse_page(self, data: dict) -> Optional[CorpusItem]:
        import uuid

        source_id = str(uuid.uuid4())
        meta_path = self.save_raw_metadata(data, source_id)

        title = data.get("title", "Untitled Newspaper Page")
        date_str = data.get("date", "")
        url_original = data.get("url", "")

        # Description contains OCR snippet text from the newspaper page
        desc = data.get("description", [])
        if isinstance(desc, list):
            ocr_text = clean_text(" ".join(desc))
        else:
            ocr_text = clean_text(str(desc))

        # Geographic scope
        city = data.get("location_city", [""])[0] if data.get("location_city") else ""
        state_val = data.get("location_state", [""])[0] if data.get("location_state") else ""
        geo_scope = ", ".join(filter(None, [city, state_val]))

        # Subject tags
        subjects = data.get("subject", [])
        topic_tags = [s.strip() for s in subjects if isinstance(s, str)][:10]

        era_tag = detect_era(date_str[:4] if date_str else "")
        quality = self.text_quality_score(ocr_text) if ocr_text else 0.3

        item = CorpusItem(
            source_id=source_id,
            institution="Library of Congress",
            collection="Chronicling America",
            title=title,
            date_original=date_str,
            modality="text",
            format="text/plain",
            language="en",
            geographic_scope=geo_scope,
            era_tag=era_tag,
            topic_tags=topic_tags,
            rights_status="public domain",
            url_original=url_original,
            metadata_raw_path=str(meta_path),
            quality_score=quality,
            notes=ocr_text[:500] if ocr_text else "",
        )
        item._raw_text = ocr_text
        return item
