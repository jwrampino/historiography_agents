"""
Internet Archive ingestor.
Uses the internetarchive Python SDK + search API.
Handles: texts, audio, video, images, and mixed collections.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, List, Optional

from sources.config.settings import (
    INTERNET_ARCHIVE_ACCESS,
    INTERNET_ARCHIVE_SECRET,
    RAW_DIR,
)
from sources.ingestors.base import BaseIngestor
from sources.storage.schema import CorpusItem
from sources.utils.text_utils import clean_text, detect_era

logger = logging.getLogger(__name__)

# IA mediatype → our modality
IA_MEDIATYPE_MAP = {
    "texts": "text",
    "audio": "audio",
    "movies": "video",
    "image": "image",
    "maps": "map",
    "data": "text",
    "web": "text",
    "collections": "mixed",
    "etree": "audio",
    "software": "text",
}


class InternetArchiveIngestor(BaseIngestor):
    """
    Ingestor for archive.org.
    Uses the `internetarchive` package for search and file download.
    Falls back to direct S3-like URLs if the package is unavailable.
    """

    institution_name = "internet_archive"
    SEARCH_URL = "https://archive.org/advancedsearch.php"
    METADATA_URL = "https://archive.org/metadata/{identifier}"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ia_configured = False
        try:
            import internetarchive as ia
            if INTERNET_ARCHIVE_ACCESS and INTERNET_ARCHIVE_SECRET:
                ia.configure(INTERNET_ARCHIVE_ACCESS, INTERNET_ARCHIVE_SECRET)
                self._ia_configured = True
                logger.info("[IA] internetarchive SDK configured with credentials")
            else:
                logger.info("[IA] internetarchive SDK available (anonymous access)")
            self._ia = ia
        except ImportError:
            logger.warning("[IA] internetarchive package not found; using raw API fallback")
            self._ia = None

    def fetch_items(
        self,
        query: str,
        max_items: int = 100,
        mediatype: Optional[str] = None,   # "texts" | "audio" | "movies" | "image" etc.
        date_range: Optional[tuple] = None, # ("YYYY-01-01", "YYYY-12-31")
        subject: Optional[str] = None,
        download_files: bool = False,       # Set True only when storage is available
        **kwargs,
    ) -> Iterator[CorpusItem]:
        """
        Yield CorpusItems from Internet Archive search.

        Args:
            query:          Full-text or field-specific query (Solr syntax)
            max_items:      Max items to retrieve
            mediatype:      Optional IA mediatype filter
            date_range:     Optional (start_date, end_date) ISO strings
            subject:        Optional subject filter
            download_files: Whether to download actual files (expensive!)
        """
        # Build Solr query
        full_query = query
        if mediatype:
            full_query += f" AND mediatype:{mediatype}"
        if subject:
            full_query += f' AND subject:"{subject}"'
        if date_range:
            full_query += f" AND date:[{date_range[0]} TO {date_range[1]}]"

        fetched = 0
        page = 1
        page_size = min(50, max_items)

        while fetched < max_items:
            params = {
                "q": full_query,
                "fl[]": [
                    "identifier", "title", "creator", "date", "description",
                    "subject", "language", "mediatype", "licenseurl",
                    "coverage", "source", "format",
                ],
                "rows": page_size,
                "page": page,
                "output": "json",
                "sort[]": "downloads desc",
            }

            try:
                resp = self._get(self.SEARCH_URL, params=params)
                data = resp.json()
            except Exception as e:
                logger.error(f"[IA] Search API error: {e}")
                break

            docs = data.get("response", {}).get("docs", [])
            if not docs:
                break

            for doc in docs:
                if fetched >= max_items:
                    break
                item = self._parse_doc(doc, download_files=download_files)
                if item:
                    yield item
                    fetched += 1

            num_found = data.get("response", {}).get("numFound", 0)
            if fetched >= num_found:
                break
            page += 1

        logger.info(f"[IA] Fetched {fetched} items for query: '{query}'")

    def _parse_doc(self, doc: dict, download_files: bool = False) -> Optional[CorpusItem]:
        import uuid

        source_id = str(uuid.uuid4())
        identifier = doc.get("identifier", "")
        meta_path = self.save_raw_metadata(doc, source_id)

        title = doc.get("title", identifier or "Untitled")
        if isinstance(title, list):
            title = title[0]

        date_str = doc.get("date", "")
        if isinstance(date_str, list):
            date_str = date_str[0]

        mediatype = doc.get("mediatype", "texts")
        modality = IA_MEDIATYPE_MAP.get(mediatype, "text")

        # Description
        desc = doc.get("description", "")
        if isinstance(desc, list):
            desc = " ".join(desc)

        # Subjects
        subjects = doc.get("subject", [])
        if isinstance(subjects, str):
            subjects = [subjects]
        topic_tags = subjects[:10]

        # Language
        lang = doc.get("language", "en")
        if isinstance(lang, list):
            lang = lang[0]
        lang = lang[:2].lower() if lang else "en"

        # Geographic
        coverage = doc.get("coverage", "")
        if isinstance(coverage, list):
            coverage = "; ".join(coverage)

        # Rights
        license_url = doc.get("licenseurl", "")
        if isinstance(license_url, list):
            license_url = license_url[0]
        rights_status = self._map_license(license_url)

        url_original = f"https://archive.org/details/{identifier}" if identifier else ""

        # Era
        era_tag = detect_era(date_str[:4] if date_str else "")

        file_path = ""
        transcript_path = ""
        thumbnail_path = ""
        quality_score = 0.5

        # ── Try to get description text as a lightweight "transcript" ──────
        if desc:
            t_dir = RAW_DIR / self.institution_name / "descriptions"
            t_dir.mkdir(parents=True, exist_ok=True)
            t_path = t_dir / f"{source_id}.txt"
            t_path.write_text(clean_text(desc), encoding="utf-8")
            transcript_path = str(t_path)
            quality_score = self.text_quality_score(desc)

        # ── Lightweight thumbnail fetch for image items (CLIP routing) ──────
        # IA exposes thumbnails at a stable URL — no SDK, no large download.
        if modality == "image" and identifier and not download_files:
            thumb_url = f"https://archive.org/services/img/{identifier}"
            try:
                t_dir = RAW_DIR / self.institution_name
                t_dir.mkdir(parents=True, exist_ok=True)
                local = self.download_file(thumb_url, dest_dir=t_dir, filename=f"{source_id}.jpg")
                file_path = str(local)
                thumbnail_path = file_path
                quality_score = 1.0
                logger.debug(f"[IA] Downloaded thumbnail for '{identifier}'")
            except Exception as e:
                logger.warning(f"[IA] Thumbnail fetch failed for {identifier}: {e}")

        # ── Optional: download actual primary file ─────────────────────────
        if download_files and identifier and self._ia:
            try:
                file_path, thumbnail_path, quality_score = self._download_ia_item(
                    identifier, source_id, modality
                )
            except Exception as e:
                logger.warning(f"[IA] Download failed for {identifier}: {e}")

        item = CorpusItem(
            source_id=source_id,
            institution="Internet Archive",
            collection=identifier,
            title=str(title)[:500],
            date_original=str(date_str),
            modality=modality,
            format=str(doc.get("format", [""])[0] if isinstance(doc.get("format"), list) else doc.get("format", "")),
            language=lang,
            geographic_scope=str(coverage),
            era_tag=era_tag,
            topic_tags=topic_tags,
            file_path=file_path,
            transcript_path=transcript_path,
            thumbnail_path=thumbnail_path,
            rights_status=rights_status,
            url_original=url_original,
            metadata_raw_path=str(meta_path),
            quality_score=quality_score,
            notes=desc[:500] if desc else "",
        )
        item._raw_text = clean_text(desc)
        return item

    def _download_ia_item(
        self, identifier: str, source_id: str, modality: str
    ) -> tuple:
        """Download the best primary file for an IA item. Returns (file_path, thumb_path, quality)."""
        dest_dir = RAW_DIR / self.institution_name / identifier
        dest_dir.mkdir(parents=True, exist_ok=True)

        item = self._ia.get_item(identifier)
        # Pick the best file by modality priority
        priority = {
            "text": ["pdf", "txt", "djvu"],
            "image": ["jpg", "jpeg", "png", "tif", "tiff"],
            "audio": ["mp3", "ogg", "flac", "wav"],
            "video": ["mp4", "ogv", "mpeg"],
            "map": ["jpg", "tif", "png"],
        }

        preferred_exts = priority.get(modality, ["pdf", "txt"])
        chosen_file = None
        for ext in preferred_exts:
            for f in item.get_files():
                if f.name.lower().endswith(ext):
                    chosen_file = f
                    break
            if chosen_file:
                break

        if not chosen_file:
            return "", "", 0.3

        local_path = dest_dir / chosen_file.name
        if not local_path.exists():
            chosen_file.download(destdir=str(dest_dir), ignore_existing=True)

        file_path = str(local_path)
        thumbnail_path = ""
        quality_score = 0.7

        if modality == "image" and local_path.exists():
            thumb = self.make_thumbnail(local_path, source_id)
            if thumb:
                thumbnail_path = str(thumb)
            quality_score = self.image_quality_score(local_path)

        return file_path, thumbnail_path, quality_score

    @staticmethod
    def _map_license(license_url: str) -> str:
        if not license_url:
            return "unknown"
        url = license_url.lower()
        if "publicdomain" in url or "cc0" in url:
            return "public domain"
        if "creativecommons" in url:
            return "no known copyright"
        if "noc" in url:
            return "no known copyright"
        return "rights unclear"
