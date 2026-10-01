"""
modules/athena/ocr_language.py

OCR language auto-detection (backlog #68).

OCR used to be hard-wired to ``-l eng``. Approach, in order:

1. Probe one sample page with every *installed* candidate language at once
   (Tesseract accepts ``eng+fra+deu``).
2. Detect the language of the probe text with ``langdetect``.
3. Choose the matching Tesseract pack only if it is installed, and keep
   English alongside it (scientific text mixes English terms in).

Every failure path ends at ``"eng"``, the old behaviour. Detection runs once
per document, not per page.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_LANG = "eng"
_MIN_SAMPLE_CHARS = 40
_MAX_PROBE_LANGS = 4

ISO_TO_TESSERACT: dict[str, str] = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
    "pt": "por", "nl": "nld", "sv": "swe", "da": "dan", "no": "nor",
    "fi": "fin", "pl": "pol", "cs": "ces", "tr": "tur", "ro": "ron",
    "hu": "hun", "el": "ell", "ru": "rus", "uk": "ukr", "bg": "bul",
    "ar": "ara", "he": "heb", "fa": "fas", "hi": "hin", "bn": "ben",
    "ta": "tam", "te": "tel", "mr": "mar", "gu": "guj", "kn": "kan",
    "ml": "mal", "pa": "pan", "ur": "urd", "th": "tha", "vi": "vie",
    "id": "ind", "ja": "jpn", "ko": "kor",
    "zh-cn": "chi_sim", "zh-tw": "chi_tra",
}

_PROBE_PRIORITY = [
    "fra", "deu", "spa", "ita", "por", "rus", "hin", "ara", "chi_sim", "jpn", "kor",
]


def installed_languages() -> set[str]:
    """Tesseract packs actually installed (``osd`` excluded). {"eng"} on any error."""
    try:
        import pytesseract
        langs = {l for l in pytesseract.get_languages(config="") if l and l != "osd"}
        return langs or {DEFAULT_LANG}
    except Exception:
        logger.debug("Could not list Tesseract languages; assuming eng only.", exc_info=True)
        return {DEFAULT_LANG}


def probe_language_string(installed: Iterable[str]) -> str:
    """``eng+fra+deu``-style string for the probe pass; just ``eng`` if nothing else is installed."""
    have = set(installed)
    extras = [l for l in _PROBE_PRIORITY if l in have][: _MAX_PROBE_LANGS - 1]
    return "+".join([DEFAULT_LANG] + extras) if extras else DEFAULT_LANG


def detect_language(text: str) -> Optional[str]:
    """ISO 639-1 code of *text*, or None when it can't be told (too short, no langdetect)."""
    sample = " ".join((text or "").split())
    if len(sample) < _MIN_SAMPLE_CHARS:
        return None
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(sample)
    except Exception:
        logger.debug("Language detection unavailable or failed.", exc_info=True)
        return None


def choose_ocr_language(detected_iso: Optional[str], installed: Iterable[str]) -> str:
    """Tesseract ``-l`` value for a detected language; ``eng`` if unknown or pack missing."""
    if not detected_iso:
        return DEFAULT_LANG
    pack = ISO_TO_TESSERACT.get(detected_iso.lower())
    if not pack or pack == DEFAULT_LANG:
        return DEFAULT_LANG
    if pack not in set(installed):
        logger.info(
            "Document looks like %s but Tesseract pack %r is not installed; using English.",
            detected_iso, pack,
        )
        return DEFAULT_LANG
    return f"{pack}+{DEFAULT_LANG}"


def auto_ocr_language(probe_ocr, sample_image, installed: Optional[Iterable[str]] = None) -> str:
    """
    Full pipeline for one document. *probe_ocr(image, lang)* runs OCR and
    returns text (injected so this is testable without Tesseract). Never raises.
    """
    try:
        have = set(installed) if installed is not None else installed_languages()
        if have <= {DEFAULT_LANG}:
            return DEFAULT_LANG
        text = probe_ocr(sample_image, probe_language_string(have))
        return choose_ocr_language(detect_language(text), have)
    except Exception:
        logger.debug("OCR language auto-detection failed; using English.", exc_info=True)
        return DEFAULT_LANG
