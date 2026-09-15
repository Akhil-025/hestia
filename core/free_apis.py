"""
core/free_apis.py

Centralized client library for free public APIs (from the public-apis/
public-apis directory: https://github.com/public-apis/public-apis) used
across Hestia's modules (Pluto, Apollo, Artemis, Dionysus, Ares, Chronos).

Design notes
------------
- Every function is a plain module-level function returning either a
  plain dict/list of results or raising ``FreeAPIError`` on failure. No
  function ever raises requests' own exception types — callers only ever
  need to catch ``FreeAPIError``.
- All calls use a short, explicit timeout and never retry internally;
  retry policy belongs to the caller (most modules already have a
  ``retry`` decorator — see modules/pluto/retry.py for the pattern).
- No API key is required for anything in this file. A couple of entries
  (FRED, NewsAPI) accept an *optional* key via env var and silently
  degrade to "unavailable" if it's not set, rather than raising, so
  Hestia keeps working out of the box.
- Every client caches nothing itself — callers that want caching (e.g.
  Chronos re-checking the same holiday within a session) should wrap
  these calls, e.g. via functools.lru_cache with a short TTL substitute.

Covered APIs (all free / keyless unless noted)
-----------------------------------------------
Currency / Finance
    - Frankfurter            https://www.frankfurter.app            (ECB FX rates)
    - SEC EDGAR               https://www.sec.gov/edgar               (US public company filings)
    - FRED*                   https://fred.stlouisfed.org             (*needs free FRED_API_KEY)

Health / Food
    - Open Food Facts        https://world.openfoodfacts.org
    - wger Workout Manager   https://wger.de/api/v2

Calendar / Productivity
    - Nager.Date              https://date.nager.at                   (public holidays)
    - Bored API                https://bored-api.appbrewery.com        (activity suggestions)

Entertainment
    - TheAudioDB              https://www.theaudiodb.com               (music metadata, key "2" test key)
    - TheMealDB                https://www.themealdb.com                (recipes, key "1" test key)
    - TheCocktailDB            https://www.thecocktaildb.com            (cocktails, key "1" test key)
    - Open Trivia DB           https://opentdb.com                      (trivia questions)

Geo / Misc
    - REST Countries           https://restcountries.com                (country metadata)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 8


class FreeAPIError(Exception):
    """Raised whenever any client in this module fails to produce a result."""


def _get(url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None,
          timeout: int = _DEFAULT_TIMEOUT) -> Any:
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise FreeAPIError(f"GET {url} failed: {e}") from e
    except ValueError as e:  # JSON decode error
        raise FreeAPIError(f"GET {url} returned non-JSON: {e}") from e


# ---------------------------------------------------------------------------
# Currency — Frankfurter (ECB rates, no key, no rate limit)
# ---------------------------------------------------------------------------

def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert `amount` from one ISO currency code to another via Frankfurter."""
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()
    if from_currency == to_currency:
        return amount
    data = _get(
        "https://api.frankfurter.app/latest",
        params={"amount": amount, "from": from_currency, "to": to_currency},
    )
    rates = data.get("rates", {})
    if to_currency not in rates:
        raise FreeAPIError(f"Frankfurter has no rate for {from_currency}->{to_currency}.")
    return float(rates[to_currency])


def fx_rate(from_currency: str, to_currency: str) -> float:
    """Return the current 1-unit exchange rate from_currency -> to_currency."""
    return convert_currency(1.0, from_currency, to_currency)


# ---------------------------------------------------------------------------
# Finance — SEC EDGAR (US public company filings, no key)
# ---------------------------------------------------------------------------

_EDGAR_HEADERS = {
    # SEC EDGAR requires a descriptive User-Agent identifying the caller.
    "User-Agent": "Hestia Personal AI System (contact: local-user@hestia.local)"
}


def sec_company_facts(cik: str) -> dict:
    """
    Fetch a public company's SEC 'company facts' (financial data over time)
    by CIK (Central Index Key, e.g. '0000320193' for Apple).
    """
    cik = cik.strip().lstrip("0").zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return _get(url, headers=_EDGAR_HEADERS)


def sec_company_search(company_name: str) -> list[dict]:
    """
    Look up ticker/CIK candidates for a company name using SEC's public
    company_tickers.json index (small, cached by SEC at the edge).
    """
    data = _get("https://www.sec.gov/files/company_tickers.json", headers=_EDGAR_HEADERS)
    name_lower = company_name.lower()
    matches = [
        row for row in data.values()
        if name_lower in row.get("title", "").lower()
    ]
    return matches[:10]


# ---------------------------------------------------------------------------
# Finance — FRED (optional key; degrades gracefully)
# ---------------------------------------------------------------------------

def fred_series_latest(series_id: str) -> Optional[float]:
    """
    Return the latest observed value for a FRED economic series
    (e.g. 'CPIAUCSL' for US CPI). Returns None (not an error) if
    FRED_API_KEY isn't configured, so callers can treat it as
    "macro context unavailable" rather than a hard failure.
    """
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        logger.info("fred_series_latest: FRED_API_KEY not set; skipping.")
        return None
    data = _get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        },
    )
    obs = data.get("observations") or []
    if not obs:
        return None
    try:
        return float(obs[0]["value"])
    except (KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Health — Open Food Facts (no key)
# ---------------------------------------------------------------------------

def food_lookup(query: str, limit: int = 5) -> list[dict]:
    """Search Open Food Facts by product name; returns simplified records."""
    data = _get(
        "https://world.openfoodfacts.org/cgi/search.pl",
        params={
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": limit,
        },
    )
    products = data.get("products", [])[:limit]
    results = []
    for p in products:
        results.append({
            "name": p.get("product_name") or p.get("generic_name") or "Unknown",
            "brand": p.get("brands"),
            "calories_kcal_100g": (p.get("nutriments") or {}).get("energy-kcal_100g"),
            "protein_g_100g": (p.get("nutriments") or {}).get("proteins_100g"),
            "sugar_g_100g": (p.get("nutriments") or {}).get("sugars_100g"),
            "nutriscore": p.get("nutriscore_grade"),
        })
    return results


# ---------------------------------------------------------------------------
# Health — wger Workout Manager (no key for public read endpoints)
# ---------------------------------------------------------------------------

def exercise_lookup(query: str, limit: int = 5) -> list[dict]:
    """Search wger's public exercise database by (English) name substring."""
    data = _get(
        "https://wger.de/api/v2/exercise/search/",
        params={"term": query, "language": "en", "format": "json"},
    )
    suggestions = (data.get("suggestions") or [])[:limit]
    results = []
    for s in suggestions:
        d = s.get("data", {})
        results.append({
            "name": d.get("name") or s.get("value"),
            "category": d.get("category"),
            "image": d.get("image"),
        })
    return results


# ---------------------------------------------------------------------------
# Calendar — Nager.Date (public holidays, no key)
# ---------------------------------------------------------------------------

def public_holidays(year: int, country_code: str = "IN") -> list[dict]:
    """All public holidays for a given year/country (ISO 3166-1 alpha-2)."""
    return _get(f"https://date.nager.at/api/v3/publicholidays/{year}/{country_code}")


def is_public_holiday(iso_date: str, country_code: str = "IN") -> Optional[str]:
    """
    Return the holiday's local name if `iso_date` (YYYY-MM-DD) is a public
    holiday in `country_code`, else None.
    """
    year = iso_date.split("-")[0]
    try:
        holidays = public_holidays(int(year), country_code)
    except FreeAPIError:
        return None
    for h in holidays:
        if h.get("date") == iso_date:
            return h.get("localName") or h.get("name")
    return None


# ---------------------------------------------------------------------------
# Productivity — Bored API (activity suggestions, no key)
# ---------------------------------------------------------------------------

# Static fallback list in case the Bored API endpoint is unreachable
# (public, community-run services occasionally go dark) — keeps
# `suggest_activity` useful even fully offline.
_ACTIVITY_FALLBACKS: list[dict] = [
    {"activity": "Go for a 20-minute walk outside", "type": "recreational"},
    {"activity": "Write down three things you're grateful for", "type": "relaxation"},
    {"activity": "Tidy up one drawer or shelf", "type": "busywork"},
    {"activity": "Call a friend or family member you haven't spoken to in a while", "type": "social"},
    {"activity": "Read 10 pages of a book you've been meaning to start", "type": "education"},
]


def suggest_activity(activity_type: Optional[str] = None) -> dict:
    """
    Suggest a random activity to fight stagnation/boredom. Falls back to a
    static local list if the remote service is unavailable.
    """
    params = {"type": activity_type} if activity_type else None
    try:
        data = _get("https://bored-api.appbrewery.com/random", params=params)
        return {"activity": data.get("activity"), "type": data.get("type")}
    except FreeAPIError as e:
        logger.warning("suggest_activity: remote fetch failed (%s); using fallback list.", e)
        import random
        return random.choice(_ACTIVITY_FALLBACKS)


# ---------------------------------------------------------------------------
# Entertainment — TheAudioDB (music metadata, no key / test key "2")
# ---------------------------------------------------------------------------

def audiodb_artist_lookup(artist_name: str) -> Optional[dict]:
    """Look up an artist's bio/genre/style from TheAudioDB (fallback to Spotify)."""
    data = _get(f"https://www.theaudiodb.com/api/v1/json/2/search.php", params={"s": artist_name})
    artists = data.get("artists") or []
    if not artists:
        return None
    a = artists[0]
    return {
        "name": a.get("strArtist"),
        "genre": a.get("strGenre"),
        "style": a.get("strStyle"),
        "bio": a.get("strBiographyEN"),
        "formed_year": a.get("intFormedYear"),
    }


def audiodb_top_tracks(artist_name: str, limit: int = 5) -> list[dict]:
    """Return an artist's most popular tracks (used as a Spotify fallback)."""
    data = _get(
        "https://www.theaudiodb.com/api/v1/json/2/mostpopular.php",
        params={"s": artist_name},
    )
    tracks = (data.get("track") or [])[:limit]
    return [
        {"track": t.get("strTrack"), "album": t.get("strAlbum")}
        for t in tracks
    ]


# ---------------------------------------------------------------------------
# Entertainment — TheMealDB / TheCocktailDB (test key "1", free for hobby use)
# ---------------------------------------------------------------------------

def meal_suggestion(cuisine: Optional[str] = None) -> Optional[dict]:
    """Random meal idea, optionally filtered by cuisine/area (e.g. 'Indian')."""
    if cuisine:
        data = _get(
            "https://www.themealdb.com/api/json/v1/1/filter.php",
            params={"a": cuisine},
        )
        meals = data.get("meals") or []
        if not meals:
            return None
        import random
        pick = random.choice(meals)
        return {"name": pick.get("strMeal"), "thumbnail": pick.get("strMealThumb")}
    data = _get("https://www.themealdb.com/api/json/v1/1/random.php")
    meals = data.get("meals") or []
    if not meals:
        return None
    m = meals[0]
    return {"name": m.get("strMeal"), "area": m.get("strArea"), "category": m.get("strCategory")}


def cocktail_suggestion() -> Optional[dict]:
    """Random cocktail idea for `plan_outing` / date-night suggestions."""
    data = _get("https://www.thecocktaildb.com/api/json/v1/1/random.php")
    drinks = data.get("drinks") or []
    if not drinks:
        return None
    d = drinks[0]
    return {"name": d.get("strDrink"), "glass": d.get("strGlass")}


# ---------------------------------------------------------------------------
# Entertainment — Open Trivia DB (no key)
# ---------------------------------------------------------------------------

def trivia_question(category: Optional[int] = None, difficulty: Optional[str] = None) -> Optional[dict]:
    """One random trivia question — used to spice up `plan_outing` suggestions."""
    params: dict[str, Any] = {"amount": 1, "type": "multiple"}
    if category:
        params["category"] = category
    if difficulty:
        params["difficulty"] = difficulty
    data = _get("https://opentdb.com/api.php", params=params)
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    return {
        "question": r.get("question"),
        "correct_answer": r.get("correct_answer"),
        "category": r.get("category"),
        "difficulty": r.get("difficulty"),
    }


# ---------------------------------------------------------------------------
# Geo — REST Countries (no key)
# ---------------------------------------------------------------------------

def country_info(name: str) -> Optional[dict]:
    """Basic country metadata (capital, region, currencies, languages)."""
    data = _get(f"https://restcountries.com/v3.1/name/{name}")
    if not data:
        return None
    c = data[0]
    return {
        "name": c.get("name", {}).get("common"),
        "capital": (c.get("capital") or [None])[0],
        "region": c.get("region"),
        "currencies": list((c.get("currencies") or {}).keys()),
        "languages": list((c.get("languages") or {}).values()),
        "population": c.get("population"),
    }