"""
Abstract base class for all archive ingestors.
Provides shared retry logic, rate limiting, file download, and logging.
"""

from __future__ import annotations
import abc
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sources.config.settings import (
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT,
    RAW_DIR,
    THUMBNAIL_DIR,
    THUMBNAIL_SIZE,
)
from sources.storage.schema import CorpusItem

logger = logging.getLogger(__name__)


def _build_session(max_retries: int = MAX_RETRIES) -> requests.Session:
    """Build a requests Session with automatic retries and a polite User-Agent."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "HistorianPipeline/1.0 (academic research; "
                "contact: researcher@institution.edu)"
            )
        }
    )
    return session


class BaseIngestor(abc.ABC):
    """
    Abstract ingestor.  Subclasses implement `fetch_items()` for each archive.
    """

    institution_name: str = "unknown"   # override in subclass

    def __init__(self, api_key: str = "", delay: float = REQUEST_DELAY_SECONDS):
        self.api_key = api_key
        self.delay = delay
        self.session = _build_session()
        self._last_request_time: float = 0.0

    # ── Abstract Interface ─────────────────────────────────────────────────

    @abc.abstractmethod
    def fetch_items(
        self, query: str, max_items: int = 100, **kwargs
    ) -> Iterator[CorpusItem]:
        """
        Yield CorpusItem objects for a given search query.
        Implementations should:
          1. Page through the archive API
          2. Download raw files via self.download_file()
          3. Generate thumbnails via self.make_thumbnail()
          4. Save raw metadata JSON via self.save_raw_metadata()
          5. Populate all CorpusItem fields they can at ingest time
        """
        ...

    # ── Rate-Limited HTTP ─────────────────────────────────────────────────

    def _get(self, url: str, params: Optional[dict] = None, **kwargs) -> requests.Response:
        """GET with rate limiting, retries, and error logging."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        try:
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            self._last_request_time = time.monotonic()
            return resp
        except requests.RequestException as e:
            logger.error(f"[{self.institution_name}] GET {url} failed: {e}")
            raise

    # ── File Download ──────────────────────────────────────────────────────

    def download_file(
        self,
        url: str,
        dest_dir: Optional[Path] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Download a file to dest_dir (defaults to RAW_DIR / institution_name).
        Returns the local Path.
        """
        dest_dir = dest_dir or (RAW_DIR / self.institution_name)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            # Derive filename from URL, fall back to MD5 hash
            parsed = urlparse(url)
            filename = Path(parsed.path).name or hashlib.md5(url.encode()).hexdigest()

        dest_path = dest_dir / filename
        if dest_path.exists():
            logger.debug(f"File already exists, skipping download: {dest_path}")
            return dest_path

        logger.debug(f"Downloading {url} → {dest_path}")
        resp = self._get(url, stream=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return dest_path

    # ── Thumbnail Generation ───────────────────────────────────────────────

    def make_thumbnail(self, image_path: Path, source_id: str) -> Optional[Path]:
        """
        Generate a 256×256 thumbnail for an image file.
        Returns the thumbnail path, or None on failure.
        """
        try:
            from PIL import Image

            thumb_dir = THUMBNAIL_DIR / self.institution_name
            thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_path = thumb_dir / f"{source_id}.jpg"

            if thumb_path.exists():
                return thumb_path

            img = Image.open(image_path).convert("RGB")
            img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            img.save(thumb_path, "JPEG", quality=85)
            return thumb_path
        except Exception as e:
            logger.warning(f"Thumbnail generation failed for {image_path}: {e}")
            return None

    # ── Raw Metadata Persistence ───────────────────────────────────────────

    def save_raw_metadata(self, metadata: dict, source_id: str) -> Path:
        """
        Persist raw API metadata JSON alongside raw files.
        Returns the path where it was saved.
        """
        meta_dir = RAW_DIR / self.institution_name / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{source_id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return meta_path

    # ── Quality Score Helpers ──────────────────────────────────────────────

    @staticmethod
    def image_quality_score(image_path: Path) -> float:
        """
        Simple quality score: ratio of (min_dimension / 1000), capped at 1.0.
        Higher-resolution images score higher.
        """
        try:
            from PIL import Image
            img = Image.open(image_path)
            min_dim = min(img.size)
            return min(min_dim / 1000.0, 1.0)
        except Exception:
            return 0.0

    @staticmethod
    def text_quality_score(text: str) -> float:
        """
        Heuristic text quality: fraction of printable ASCII characters.
        Good OCR output scores near 1.0; garbled text scores lower.
        """
        if not text:
            return 0.0
        printable = sum(1 for c in text if c.isprintable())
        return printable / len(text)
