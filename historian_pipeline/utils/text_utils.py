"""
Text utility functions used across the pipeline.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Optional

from historian_pipeline.config.settings import ERA_TAGS


def clean_text(text: str) -> str:
    """
    Normalise and clean extracted text.
    - Unicode NFC normalisation
    - Remove control characters
    - Collapse excessive whitespace
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # NFC normalisation (handles accented chars, ligatures)
    text = unicodedata.normalize("NFC", text)

    # Remove control characters except newlines and tabs
    text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c in "\n\t")

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ─── Era Detection ─────────────────────────────────────────────────────────────

# Maps (start_year, end_year) → era_tag (inclusive ranges)
ERA_RANGES = [
    (0,    1492,  "pre-colonial"),
    (1493, 1607,  "colonial"),
    (1608, 1775,  "early-modern"),
    (1776, 1800,  "revolutionary"),
    (1801, 1860,  "antebellum"),
    (1861, 1865,  "civil-war"),
    (1866, 1876,  "reconstruction"),
    (1877, 1900,  "gilded-age"),
    (1901, 1920,  "progressive-era"),
    (1914, 1918,  "wwi"),           # overrides progressive-era for tight range
    (1919, 1938,  "interwar"),
    (1939, 1945,  "wwii"),
    (1946, 1989,  "cold-war"),
    (1954, 1968,  "civil-rights"),  # overlaps cold-war
    (1970, 1999,  "late-20th-century"),
    (2000, 2100,  "contemporary"),
]


def detect_era(year_str: str) -> str:
    """
    Detect an era tag from a year string.
    Returns the most specific matching era, or 'unknown'.

    Args:
        year_str: A string containing a 4-digit year (e.g. "1865", "1865-1870",
                  "ca. 1920", "18th century")
    """
    year = _extract_year(year_str)
    if year is None:
        return "unknown"

    # Walk in reverse to prefer more specific / later-defined ranges
    for start, end, tag in reversed(ERA_RANGES):
        if start <= year <= end:
            return tag

    return "unknown"


def _extract_year(text: str) -> Optional[int]:
    """Extract the first 4-digit year found in a string."""
    if not text:
        return None

    # Handle century strings like "18th century" → 1750 (midpoint)
    century_match = re.search(r"(\d{1,2})(?:st|nd|rd|th)\s+century", text, re.IGNORECASE)
    if century_match:
        century = int(century_match.group(1))
        return (century - 1) * 100 + 50

    # Find any 4-digit year
    year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", text)
    if year_match:
        return int(year_match.group(1))

    return None


# ─── Language Normalisation ────────────────────────────────────────────────────

LANG_ALIASES = {
    "english": "en",
    "french": "fr",
    "french, old": "fr",
    "spanish": "es",
    "german": "de",
    "latin": "la",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "japanese": "ja",
    "chinese": "zh",
    "arabic": "ar",
    "russian": "ru",
    "undetermined": "und",
    "multiple languages": "mul",
    "no linguistic content": "zxx",
}


def normalise_language(lang: str) -> str:
    """
    Normalise a language string to ISO 639-1 (2-letter code).
    Returns 'und' if the language cannot be determined.
    """
    if not lang:
        return "und"
    lang = lang.strip().lower()
    if lang in LANG_ALIASES:
        return LANG_ALIASES[lang]
    # Already a 2-letter code
    if re.match(r"^[a-z]{2}$", lang):
        return lang
    # 3-letter code — try to map common ones
    three_to_two = {
        "eng": "en", "fra": "fr", "spa": "es", "deu": "de",
        "lat": "la", "por": "pt", "ita": "it", "nld": "nl",
        "jpn": "ja", "zho": "zh", "ara": "ar", "rus": "ru",
        "und": "und",
    }
    if lang[:3] in three_to_two:
        return three_to_two[lang[:3]]
    # Fall back to first 2 chars
    return lang[:2] if len(lang) >= 2 else "und"


# ─── Topic Tag Cleaning ────────────────────────────────────────────────────────

def clean_topic_tags(tags: list) -> list:
    """Deduplicate, lowercase, strip, and truncate topic tags."""
    seen = set()
    result = []
    for tag in tags:
        tag = str(tag).strip().lower()
        tag = re.sub(r"\s+", " ", tag)
        if tag and tag not in seen and len(tag) < 200:
            seen.add(tag)
            result.append(tag)
    return result[:20]
