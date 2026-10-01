"""
modules/iris/query_filters.py

Pulls EXIF-style filters out of a photo search phrase (#74): "photos taken in
March 2024", "pictures shot on my iPhone", "geotagged photos from last week".
What is left over is the caption/tag/semantic part of the query. Place names
("taken in Paris") are NOT resolved - that needs reverse geocoding, so they stay
in the text part and match captions as before.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

_CAMERAS = ("canon", "nikon", "sony", "fujifilm", "fuji", "apple", "iphone", "samsung", "galaxy",
            "google", "pixel", "panasonic", "lumix", "olympus", "gopro", "dji", "leica", "pentax",
            "oneplus", "xiaomi", "huawei", "motorola")
_CAM_RE = re.compile(
    r"\b(?:(?:taken|shot|captured|photographed|snapped)\s+(?:with|on|using|by)|(?:with|on|using|from))"
    r"\s+(?:(?:a|an|my|the)\s+)?(" + "|".join(_CAMERAS) + r")(?:\s+([a-z0-9][\w-]*))?", re.I)
_LOC_YES = re.compile(r"\b(?:geo-?tagged|with (?:a )?(?:location|gps)|that have (?:a )?(?:location|gps)|with locations)\b", re.I)
_LOC_NO = re.compile(r"\b(?:without (?:a )?(?:location|gps)|no (?:location|gps)|not geo-?tagged)\b", re.I)
_FILLER = frozenset(
    "find show search get me my the a an all any some of from in on at taken shot with using by were was that "
    "photos photo pictures picture pics pic images image videos video i have what which look for please and "
    "during between within".split())


@dataclass
class PhotoQuery:
    text: str = ""
    date_from: Optional[str] = None      # 'YYYY-MM-DD'
    date_to: Optional[str] = None        # inclusive day, as 'YYYY-MM-DDT23:59:59'
    camera: Optional[str] = None
    has_location: Optional[bool] = None
    summary: str = ""

    @property
    def active(self) -> bool:
        return bool(self.date_from or self.camera or self.has_location is not None)


def parse_photo_query(query: str, today: Optional[date] = None) -> PhotoQuery:
    from modules.mnemosyne.dates import resolve_range

    rest = query or ""
    out = PhotoQuery(text=rest.strip())
    notes: list[str] = []

    m = _CAM_RE.search(rest)
    if m:
        out.camera = " ".join(p for p in (m.group(1), m.group(2)) if p and p.lower() not in _FILLER)
        # a trailing word that is not a model ("iphone photos") must not become part of the camera
        if m.group(2) and m.group(2).lower() in _FILLER:
            out.camera = m.group(1)
        rest = rest[:m.start()] + " " + rest[m.end():]
        notes.append(f"camera {out.camera}")
    if _LOC_NO.search(rest):
        out.has_location = False
        rest = _LOC_NO.sub(" ", rest)
        notes.append("without location")
    elif _LOC_YES.search(rest):
        out.has_location = True
        rest = _LOC_YES.sub(" ", rest)
        notes.append("with location")

    rng = resolve_range(rest, today)
    if rng:
        out.date_from = rng.start.isoformat()
        out.date_to = rng.end.isoformat() + "T23:59:59"
        rest = re.sub(re.escape(rng.phrase), " ", rest, count=1, flags=re.I)
        notes.append(f"taken {rng.label}")

    if out.active:
        words = [w for w in re.findall(r"[\w'-]+", rest) if w.lower() not in _FILLER]
        out.text = " ".join(words)
        out.summary = ", ".join(notes)
    return out
