# modules/dionysus/engine.py

import json
import logging
import os
import base64
from pathlib import Path
from typing import Optional

import requests

from core.ollama_client import generate
from modules.base import BaseModule
from .db import DionysusDB

log = logging.getLogger(__name__)

_DB_PATH = str(Path(__file__).parent.parent.parent / "data" / "dionysus" / "dionysus.db")

OMDB_KEY          = os.getenv("OMDB_API_KEY", "")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# ── Prompts ───────────────────────────────────────────────────────────────────

_MOVIE_PROMPT = """You are Dionysus, an entertainment expert.
Recommend 3 movies for someone who wants: {mood_genre}
Avoid these already seen/dismissed: {dismissed}

Respond with ONLY valid JSON:
{{
  "recommendations": [
    {{"title": "Movie Title", "year": "2021", "reason": "one sentence why"}}
  ]
}}
JSON only."""

_MUSIC_PROMPT = """You are Dionysus, a music expert.
Recommend 5 songs or artists for this mood: {mood}
Avoid these already recommended: {dismissed}

Respond with ONLY valid JSON:
{{
  "recommendations": [
    {{"artist": "Artist Name", "track": "Track or Album", "reason": "one sentence why"}}
  ]
}}
JSON only."""

_OUTING_PROMPT = """You are Dionysus, a Mumbai lifestyle expert.
The user wants to plan an outing: {topic}
Available places found nearby:
{places}

Build a structured itinerary with morning, afternoon, and evening slots.
Respond with ONLY valid JSON:
{{
  "title": "outing title",
  "slots": [
    {{
      "time": "Morning 10am",
      "place": "place name",
      "activity": "what to do there",
      "duration": "2 hours",
      "travel_to_next": "15 min by auto"
    }}
  ],
  "tips": ["tip 1", "tip 2"]
}}
JSON only."""


class DionysusEngine(BaseModule):
    name = "dionysus"
    _INTENTS = {
        "recommend_movie",
        "find_restaurant",
        "recommend_music",
        "plan_outing",
    }

    def __init__(self, ollama_cfg: dict = None, browser_agent=None, memory=None):
        self._ollama  = ollama_cfg or {}
        self._browser = browser_agent
        self._memory  = memory
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        self.db = DionysusDB(_DB_PATH)

    def can_handle(self, intent: str) -> bool:
        return intent in self._INTENTS

    def handle(self, intent: str, entities: dict, context: dict) -> dict:
        if intent == "recommend_movie":
            return self._recommend_movie(entities)
        if intent == "find_restaurant":
            return self._find_restaurant(entities)
        if intent == "recommend_music":
            return self._recommend_music(entities)
        if intent == "plan_outing":
            return self._plan_outing(entities)
        return {"response": "Unknown Dionysus intent.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        return {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ollama_call(self, prompt: str) -> str:
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
            fmt="json",
        )

    def _ollama_text(self, prompt: str) -> str:
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
        )

    def _parse(self, raw: str, intent: str) -> Optional[dict]:
        try:
            return json.loads(raw)
        except Exception:
            log.warning("Dionysus: failed to parse JSON for %s", intent)
            return None

    # ── recommend_movie ───────────────────────────────────────────────────────

    def _recommend_movie(self, entities: dict) -> dict:
        mood_genre = (
            entities.get("mood")
            or entities.get("genre")
            or entities.get("raw_query", "something good")
        )
        dismissed = self.db.dismissed_titles("movie")

        raw    = self._ollama_call(
            _MOVIE_PROMPT.format(
                mood_genre=mood_genre,
                dismissed=", ".join(dismissed) or "none",
            )
        )
        result = self._parse(raw, "recommend_movie")

        if not result:
            return {"response": "I had trouble finding movies.", "data": {}, "confidence": 0.3}

        recs   = result.get("recommendations", [])
        lines  = [f"Movies for '{mood_genre}'\n"]
        enriched = []

        for rec in recs:
            title  = rec.get("title", "")
            year   = rec.get("year", "")
            reason = rec.get("reason", "")

            # OMDB live data
            omdb = self._fetch_omdb(title, year)
            rating   = omdb.get("imdbRating", "N/A") if omdb else "N/A"
            runtime  = omdb.get("Runtime", "")       if omdb else ""
            streaming= self._streaming_note(omdb)    if omdb else ""

            lines.append(f"  {title} ({year})  ★ {rating}")
            if runtime:
                lines.append(f"    {runtime}")
            lines.append(f"    {reason}")
            if streaming:
                lines.append(f"    {streaming}")
            lines.append("")

            self.db.log("movie", title, reason,
                        float(rating) if rating != "N/A" else None)
            enriched.append({**rec, "imdb_rating": rating, "runtime": runtime})

        return {
            "response": "\n".join(lines).strip(),
            "data": {"recommendations": enriched},
            "confidence": 0.9,
        }

    def _fetch_omdb(self, title: str, year: str) -> Optional[dict]:
        if not OMDB_KEY:
            return None
        try:
            params = {"t": title, "apikey": OMDB_KEY}
            if year:
                params["y"] = year
            r = requests.get("https://www.omdbapi.com/", params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            return data if data.get("Response") == "True" else None
        except Exception as e:
            log.warning("OMDB fetch failed for %s: %s", title, e)
            return None

    @staticmethod
    def _streaming_note(omdb: dict) -> str:
        # OMDB free tier doesn't give streaming — note availability from ratings
        rated = omdb.get("Rated", "")
        genre = omdb.get("Genre", "")
        if rated or genre:
            return f"Rated {rated} | {genre}"
        return ""

    # ── find_restaurant ───────────────────────────────────────────────────────

    def _find_restaurant(self, entities: dict) -> dict:
        cuisine = entities.get("cuisine", "")
        area    = entities.get("area") or entities.get("location", "Mumbai")
        budget  = entities.get("budget", "")

        query = f"best {cuisine} restaurants in {area} Mumbai"
        if budget:
            query += f" {budget} budget"

        if not self._browser:
            return {
                "response": "Browser agent not available for restaurant search.",
                "data": {},
                "confidence": 0.0,
            }

        raw_results = self._browser.search_web(query)

        if not raw_results or raw_results.startswith("No results"):
            return {
                "response": f"Couldn't find restaurants for {cuisine} in {area}.",
                "data": {},
                "confidence": 0.3,
            }

        # log and format
        self.db.log("restaurant", query, raw_results)

        lines = [f"Restaurants — {cuisine or 'any'} in {area}\n"]
        for i, name in enumerate(raw_results.split(" | ")[:5], 1):
            lines.append(f"  {i}. {name.strip()}")

        return {
            "response": "\n".join(lines).strip(),
            "data": {"query": query, "results": raw_results},
            "confidence": 0.85,
        }

    # ── recommend_music ───────────────────────────────────────────────────────

    def _recommend_music(self, entities: dict) -> dict:
        mood      = entities.get("mood") or entities.get("raw_query", "good vibes")
        dismissed = self.db.dismissed_titles("music")

        # Ollama recommendations
        raw    = self._ollama_call(
            _MUSIC_PROMPT.format(
                mood=mood,
                dismissed=", ".join(dismissed) or "none",
            )
        )
        result = self._parse(raw, "recommend_music")

        if not result:
            return {"response": "I had trouble finding music.", "data": {}, "confidence": 0.3}

        recs  = result.get("recommendations", [])
        lines = [f"Music for '{mood}'\n"]
        enriched = []

        for rec in recs:
            artist = rec.get("artist", "")
            track  = rec.get("track", "")
            reason = rec.get("reason", "")

            # Spotify live data
            spotify = self._fetch_spotify(artist, track)
            preview = spotify.get("preview_url") if spotify else None
            sp_link = spotify.get("external_urls", {}).get("spotify", "") if spotify else ""
            popularity = spotify.get("popularity") if spotify else None

            lines.append(f"  {artist} — {track}")
            lines.append(f"    {reason}")
            if popularity is not None:
                lines.append(f"    Spotify popularity: {popularity}/100")
            if sp_link:
                lines.append(f"    {sp_link}")
            lines.append("")

            self.db.log("music", f"{artist} — {track}", reason)
            enriched.append({
                **rec,
                "spotify_link": sp_link,
                "popularity": popularity,
                "preview_url": preview,
            })

        return {
            "response": "\n".join(lines).strip(),
            "data": {"recommendations": enriched},
            "confidence": 0.9,
        }

    def _get_spotify_token(self) -> Optional[str]:
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            return None
        try:
            creds = base64.b64encode(
                f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()
            ).decode()
            r = requests.post(
                "https://accounts.spotify.com/api/token",
                headers={"Authorization": f"Basic {creds}"},
                data={"grant_type": "client_credentials"},
                timeout=8,
            )
            r.raise_for_status()
            return r.json().get("access_token")
        except Exception as e:
            log.warning("Spotify token fetch failed: %s", e)
            return None

    def _fetch_spotify(self, artist: str, track: str) -> Optional[dict]:
        token = self._get_spotify_token()
        if not token:
            return None
        try:
            query = f"track:{track} artist:{artist}"
            r = requests.get(
                "https://api.spotify.com/v1/search",
                headers={"Authorization": f"Bearer {token}"},
                params={"q": query, "type": "track", "limit": 1},
                timeout=8,
            )
            r.raise_for_status()
            items = r.json().get("tracks", {}).get("items", [])
            return items[0] if items else None
        except Exception as e:
            log.warning("Spotify search failed for %s — %s: %s", artist, track, e)
            return None

    # ── plan_outing ───────────────────────────────────────────────────────────

    def _plan_outing(self, entities: dict) -> dict:
        topic = (
            entities.get("topic")
            or entities.get("time")
            or entities.get("raw_query", "a day out in Mumbai")
        )

        # get user location from memory, default to Mumbai coords
        lat, lon = 19.0760, 72.8777
        if self._memory:
            loc = self._memory.get_preference("location", "")
            if loc:
                topic = f"{topic} near {loc}"

        if not self._browser:
            return {
                "response": "Browser agent not available for outing planning.",
                "data": {},
                "confidence": 0.0,
            }

        # search for real places via Hephaestus
        place_types = ["restaurants", "parks", "attractions", "cafes"]
        place_results = []
        for ptype in place_types:
            result = self._browser.search_web(f"best {ptype} in Mumbai {topic}")
            if result and not result.startswith("No results"):
                place_results.append(f"{ptype.upper()}: {result}")

        places_block = "\n".join(place_results) or "No places found"

        # Ollama builds the itinerary
        raw    = self._ollama_call(
            _OUTING_PROMPT.format(topic=topic, places=places_block)
        )
        result = self._parse(raw, "plan_outing")

        if not result:
            return {
                "response": "I had trouble planning the outing.",
                "data": {},
                "confidence": 0.3,
            }

        response = self._format_outing(result)
        self.db.log("outing", result.get("title", topic), response)

        return {
            "response": response,
            "data": result,
            "confidence": 0.9,
        }

    @staticmethod
    def _format_outing(result: dict) -> str:
        lines = [f"Outing Plan: {result.get('title', '')}", ""]

        for slot in result.get("slots", []):
            lines.append(f"[{slot.get('time', '')}]")
            lines.append(f"  Place    : {slot.get('place', '')}")
            lines.append(f"  Activity : {slot.get('activity', '')}")
            lines.append(f"  Duration : {slot.get('duration', '')}")
            travel = slot.get("travel_to_next", "")
            if travel:
                lines.append(f"  Travel   : {travel}")
            lines.append("")

        tips = result.get("tips", [])
        if tips:
            lines.append("TIPS")
            for tip in tips:
                lines.append(f"  • {tip}")

        return "\n".join(lines).strip()