# modules/dionysus/engine.py

import json
import logging
import os
import re
import base64
from pathlib import Path
from typing import Optional

import requests

from core.ollama_client import generate
from core.free_apis import (
    FreeAPIError,
    audiodb_artist_lookup as _fa_audiodb_artist,
    audiodb_top_tracks as _fa_audiodb_top_tracks,
    meal_suggestion as _fa_meal_suggestion,
    cocktail_suggestion as _fa_cocktail_suggestion,
    trivia_question as _fa_trivia_question,
)
from modules.base import BaseModule
from .db import DionysusDB

log = logging.getLogger(__name__)

_DB_PATH = str(Path(__file__).parent.parent.parent / "data" / "dionysus" / "dionysus.db")

OMDB_KEY          = os.getenv("OMDB_API_KEY", "")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# Backlog #147: a dismissal stops counting after this many days, so a title can
# come back after a long gap. 0 / None = dismissals never expire.
_DEFAULT_DISMISS_EXPIRE_DAYS = 180.0

# Backlog #152: used when the user doesn't say when. Chronos parses the phrase.
_DEFAULT_RECHARGE_SCHEDULE = "every Sunday at 4pm"
_DEFAULT_RECHARGE_DURATION = "2 hours"

# Backlog #151: rough per-person ranges (rupees) for words like "cheap" or
# "fine dining". These are ballpark figures, not live prices; edit freely.
_BUDGET_TIERS = {
    "cheap":   (0, 500),
    "mid":     (500, 1500),
    "premium": (1500, None),
}
_TIER_WORDS = {
    "cheap": "cheap", "budget": "cheap", "inexpensive": "cheap",
    "affordable": "cheap", "low": "cheap",
    "mid": "mid", "moderate": "mid", "mid-range": "mid", "midrange": "mid",
    "premium": "premium", "expensive": "premium", "luxury": "premium",
    "splurge": "premium", "fine": "premium", "upscale": "premium",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

_MOVIE_PROMPT = """You are Dionysus, an entertainment expert.
Recommend 3 movies for someone who wants: {mood_genre}
Avoid these already seen/dismissed: {dismissed}
{taste}
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
{taste}
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
{budget_rule}
Build a structured itinerary with morning, afternoon, and evening slots.
For each slot give "cost_per_person": a rough estimate in rupees as a plain
number (0 if free). These are estimates, not quoted prices.
Respond with ONLY valid JSON:
{{
  "title": "outing title",
  "slots": [
    {{
      "time": "Morning 10am",
      "place": "place name",
      "activity": "what to do there",
      "duration": "2 hours",
      "cost_per_person": 400,
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
        "dismiss_recommendation",
        "mark_seen",
        "recommend_recipe",
        # Backlog #146, #150, #152, #269
        "find_events",
        "surprise_me",
        "schedule_recharge",
        "more_like_this",
        "less_like_this",
    }

    def __init__(self, ollama_cfg: dict = None, browser_agent=None, memory=None, llm=None,
                 mood_aware: bool = True,
                 dismiss_expire_days: Optional[float] = _DEFAULT_DISMISS_EXPIRE_DAYS):
        self._ollama  = ollama_cfg or {}
        # Backlog #147. ``dionysus.dismiss_expire_days``; 0 or None = never expire.
        self._expire_days = float(dismiss_expire_days) if dismiss_expire_days else None
        self._chronos = None
        # Backlog #149. Opt-out via ``dionysus.mood_aware: false``.
        self._mood_aware = bool(mood_aware)
        self._apollo = None
        self._browser = browser_agent
        self._memory  = memory
        self._llm_instance = llm  # HestiaLLM | None — preferred path
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        self.db = DionysusDB(_DB_PATH)

    def attach_apollo(self, apollo) -> None:
        """Read-only link to Apollo so recommendations can reflect a mood the
        user has *logged* (backlog #149). Never used when they gave a mood."""
        self._apollo = apollo

    def attach_chronos(self, chronos) -> None:
        """Link to Chronos so recharge routines (backlog #152) can be created as
        repeating reminders. Without it, ``schedule_recharge`` says it can't."""
        self._chronos = chronos

    def _dismissed(self, type_: str) -> list[str]:
        return self.db.dismissed_titles(type_, expire_days=self._expire_days)

    def _mood_hint(self, entities: dict, kind: str):
        """Recent logged mood to use when the request itself gave none.

        Returns None (use the request as-is) if mood-awareness is off,
        Apollo isn't attached, the user said a mood/genre, the query names
        something specific, or no mood was logged in the last 48 hours.
        Otherwise ``{"mood", "prompt", "header", "low_trend"}``.
        """
        if not self._mood_aware or self._apollo is None:
            return None
        if entities.get("mood") or entities.get("genre"):
            return None
        if not _is_generic_request(entities.get("raw_query")):
            return None
        try:
            ctx = self._apollo.recent_mood_context()
        except Exception:
            log.warning("Apollo mood lookup failed; ignoring.", exc_info=True)
            return None
        if not ctx or not ctx.get("mood"):
            return None
        mood = ctx["mood"]
        if ctx.get("low_trend"):
            prompt = (
                f"gentle, comforting and warm, for someone who has felt low "
                f"lately (most recently: {mood}); avoid forced cheerfulness "
                f"and avoid anything bleak"
            )
            note = " Your recent moods have been low, so I've leaned toward comforting picks."
        else:
            prompt = f"suited to someone who is feeling {mood}"
            note = ""
        return {
            "mood": mood, "prompt": prompt, "low_trend": bool(ctx.get("low_trend")),
            "header": f"{kind} based on your logged mood: {mood}.{note}",
        }

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
        if intent == "dismiss_recommendation":
            return self._dismiss_recommendation(entities)
        if intent == "mark_seen":
            return self._mark_seen(entities)
        if intent == "recommend_recipe":
            return self._recommend_recipe(entities)
        if intent == "find_events":
            return self._find_events(entities)
        if intent == "surprise_me":
            return self._surprise_me(entities)
        if intent == "schedule_recharge":
            return self._schedule_recharge(entities)
        if intent == "more_like_this":
            return self._give_feedback(entities, +1)
        if intent == "less_like_this":
            return self._give_feedback(entities, -1)
        return {"response": "Unknown Dionysus intent.", "data": {}, "confidence": 0.0}

    def get_context(self) -> dict:
        return {}

    # ── dismiss_recommendation / mark_seen ──────────────────────────────────────
    #
    # DionysusDB has always supported dismiss()/mark_seen()/seen_titles() —
    # used to keep repeat recommendations out of _recommend_movie's and
    # _recommend_music's prompts — but nothing ever called them, so users
    # had no way to say "not that one" or "already watched it".

    def _dismiss_recommendation(self, entities: dict) -> dict:
        title = (entities.get("title") or entities.get("raw_query", "")).strip()
        if not title:
            return {
                "response": "Which recommendation should I dismiss?",
                "data": {},
                "confidence": 0.5,
            }
        self.db.dismiss(title)
        return {
            "response": f"Got it, I won't suggest '{title}' again.",
            "data": {"title": title},
            "confidence": 0.9,
        }

    def _mark_seen(self, entities: dict) -> dict:
        title = (entities.get("title") or entities.get("raw_query", "")).strip()
        if not title:
            return {
                "response": "Which movie should I mark as watched?",
                "data": {},
                "confidence": 0.5,
            }
        type_ = entities.get("type", "movie")
        updated = self.db.mark_seen(title, type_=type_)
        if not updated:
            return {
                "response": f"I don't have '{title}' in your recommendation history.",
                "data": {"title": title},
                "confidence": 0.4,
            }
        return {
            "response": f"Marked '{title}' as watched — I won't recommend it again.",
            "data": {"title": title},
            "confidence": 0.9,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ollama_call(self, prompt: str) -> str:
        if self._llm_instance is not None:
            return self._llm_instance.generate(prompt, fmt="json")
        return generate(
            prompt,
            model=self._ollama.get("model", "mistral"),
            host=self._ollama.get("host", "127.0.0.1"),
            port=self._ollama.get("port", 11434),
            fmt="json",
        )

    def _ollama_text(self, prompt: str) -> str:
        if self._llm_instance is not None:
            return self._llm_instance.generate(prompt)
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

    def _recommend_movie(self, entities: dict, surprise: bool = False) -> dict:
        mood_genre = (
            entities.get("mood")
            or entities.get("genre")
            or entities.get("raw_query", "something good")
        )
        hint = None if surprise else self._mood_hint(entities, "Movies")
        if hint:
            mood_genre = hint["prompt"]
        if surprise:
            mood_genre = "something different from my usual taste"
        # Exclude both explicitly dismissed titles and ones the user has
        # already marked as watched — previously only `dismissed_titles`
        # was consulted here, so a movie logged via mark_seen() could still
        # be recommended again.
        dismissed = self._dismissed("movie")
        seen = self.db.seen_titles("movie")
        # "More like this" titles are things the user already knows; "less like
        # this" ones are never wanted. Neither should be recommended back.
        taste_titles = (self.db.feedback_titles("movie", 1)
                        + self.db.feedback_titles("movie", -1))
        exclude = sorted(set(dismissed) | set(seen) | set(taste_titles))

        raw    = self._ollama_call(
            _MOVIE_PROMPT.format(
                mood_genre=mood_genre,
                dismissed=", ".join(exclude) or "none",
                taste=self._taste_block("movie", surprise),
            )
        )
        result = self._parse(raw, "recommend_movie")

        if not result:
            return {"response": "I had trouble finding movies.", "data": {}, "confidence": 0.3}

        recs   = result.get("recommendations", [])
        if surprise:
            lines = ["Surprise movies, outside your usual taste\n"]
        else:
            lines = [hint["header"] + "\n"] if hint else [f"Movies for '{mood_genre}'\n"]
        enriched = []

        for rec in recs:
            title  = rec.get("title", "")
            year   = rec.get("year", "")
            reason = rec.get("reason", "")

            # OMDB live data
            omdb = self._fetch_omdb(title, year)
            rating   = omdb.get("imdbRating", "N/A") if omdb else "N/A"
            runtime  = omdb.get("Runtime", "")       if omdb else ""
            rating_note = self._rating_note(omdb) if omdb else ""

            lines.append(f"  {title} ({year})  ★ {rating}")
            if runtime:
                lines.append(f"    {runtime}")
            lines.append(f"    {reason}")
            if rating_note:
                lines.append(f"    {rating_note}")
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
    def _rating_note(omdb: dict) -> str:
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

        # Backlog #151: respect a budget. Search results are just titles, so we
        # never invent prices: we flag results that state a price over budget
        # and explain what the budget words usually mean.
        budget_info = _parse_budget(budget)
        lines = [f"Restaurants — {cuisine or 'any'} in {area}\n"]
        for i, name in enumerate(raw_results.split(" | ")[:5], 1):
            flag = ""
            if budget_info and budget_info["amount"] is not None:
                price = _price_in_text(name)
                if price is not None and price > budget_info["amount"]:
                    flag = "  (price shown is over your budget)"
            lines.append(f"  {i}. {name.strip()}{flag}")
        note = _budget_note(budget_info)
        if note:
            lines.extend(["", note])

        return {
            "response": "\n".join(lines).strip(),
            "data": {"query": query, "results": raw_results,
                     "budget": budget_info},
            "confidence": 0.85,
        }

    # ── recommend_music ───────────────────────────────────────────────────────

    def _recommend_music(self, entities: dict, surprise: bool = False) -> dict:
        mood      = entities.get("mood") or entities.get("raw_query", "good vibes")
        hint      = None if surprise else self._mood_hint(entities, "Music")
        if hint:
            mood = hint["prompt"]
        if surprise:
            mood = "something different from my usual taste"
        dismissed = sorted(
            set(self._dismissed("music"))
            | set(self.db.feedback_titles("music", 1))
            | set(self.db.feedback_titles("music", -1))
        )

        # Ollama recommendations
        raw    = self._ollama_call(
            _MUSIC_PROMPT.format(
                mood=mood,
                dismissed=", ".join(dismissed) or "none",
                taste=self._taste_block("music", surprise),
            )
        )
        result = self._parse(raw, "recommend_music")

        if not result:
            return {"response": "I had trouble finding music.", "data": {}, "confidence": 0.3}

        recs  = result.get("recommendations", [])
        if surprise:
            lines = ["Surprise music, outside your usual taste\n"]
        else:
            lines = [hint["header"] + "\n"] if hint else [f"Music for '{mood}'\n"]
        enriched = []

        for rec in recs:
            artist = rec.get("artist", "")
            track  = rec.get("track", "")
            reason = rec.get("reason", "")

            # Spotify live data (needs SPOTIFY_CLIENT_ID/SECRET). When
            # those aren't configured — or the lookup otherwise fails —
            # fall back to TheAudioDB (free, keyless) for at least a
            # bio/genre note instead of silently dropping the metadata.
            spotify = self._fetch_spotify(artist, track)
            preview = spotify.get("preview_url") if spotify else None
            sp_link = spotify.get("external_urls", {}).get("spotify", "") if spotify else ""
            popularity = spotify.get("popularity") if spotify else None

            audiodb_genre = None
            if not spotify:
                try:
                    artist_info = _fa_audiodb_artist(artist)
                except FreeAPIError:
                    artist_info = None
                if artist_info:
                    audiodb_genre = artist_info.get("genre")

            lines.append(f"  {artist} — {track}")
            lines.append(f"    {reason}")
            if popularity is not None:
                lines.append(f"    Spotify popularity: {popularity}/100")
            elif audiodb_genre:
                lines.append(f"    Genre (TheAudioDB): {audiodb_genre}")
            if sp_link:
                lines.append(f"    {sp_link}")
            lines.append("")

            self.db.log("music", f"{artist} — {track}", reason)
            enriched.append({
                **rec,
                "spotify_link": sp_link,
                "popularity": popularity,
                "preview_url": preview,
                "audiodb_genre": audiodb_genre,
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

        # Use the user's saved location preference, if any, to narrow the search.
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

        # Backlog #151: a stated budget steers the plan, and the total is
        # added up in code from the per-slot estimates (not taken on trust).
        budget_info = _parse_budget(entities.get("budget"))
        budget_rule = ""
        if budget_info and budget_info["amount"] is not None:
            budget_rule = (
                f"\nThe user's budget is about \u20b9{budget_info['amount']:,.0f} per "
                f"person for the whole outing, so choose places and activities "
                f"that fit within it.\n"
            )
        elif budget_info and budget_info["tier"]:
            lo, hi = _BUDGET_TIERS[budget_info["tier"]]
            budget_rule = (
                f"\nThe user wants {budget_info['tier']} options "
                f"({_range_text(lo, hi)} per person overall).\n"
            )

        # Ollama builds the itinerary
        raw    = self._ollama_call(
            _OUTING_PROMPT.format(topic=topic, places=places_block,
                                  budget_rule=budget_rule)
        )
        result = self._parse(raw, "plan_outing")

        if not result:
            return {
                "response": "I had trouble planning the outing.",
                "data": {},
                "confidence": 0.3,
            }

        # Sprinkle in a couple of free, keyless extras so the plan feels
        # less generic: a food idea (TheMealDB) and, roughly half the
        # time, a trivia icebreaker (Open Trivia DB). Both are
        # best-effort — an outing plan must never fail because a bonus
        # API call did.
        extra_tips: list[str] = []
        try:
            meal = _fa_meal_suggestion()
            if meal:
                extra_tips.append(f"Food idea: {meal.get('name')} ({meal.get('area', 'various')} cuisine)")
        except FreeAPIError:
            log.debug("plan_outing: meal suggestion unavailable.", exc_info=True)
        try:
            trivia = _fa_trivia_question()
            if trivia:
                extra_tips.append(f"Icebreaker: {trivia.get('question')}")
        except FreeAPIError:
            log.debug("plan_outing: trivia question unavailable.", exc_info=True)

        if extra_tips:
            result.setdefault("tips", [])
            result["tips"].extend(extra_tips)

        total = _outing_total(result)
        if total is not None:
            result["estimated_cost_per_person"] = total
            limit = budget_info["amount"] if budget_info else None
            if limit is None and budget_info and budget_info["tier"]:
                limit = _BUDGET_TIERS[budget_info["tier"]][1]
            if limit is not None:
                result["budget_per_person"] = limit
                if total > limit:
                    result["over_budget_by"] = total - limit

        response = self._format_outing(result)
        self.db.log("outing", result.get("title", topic), response)

        return {
            "response": response,
            "data": result,
            "confidence": 0.9,
        }

    def _recommend_recipe(self, entities: dict) -> dict:
        """
        Suggest a meal or cocktail via TheMealDB / TheCocktailDB (both
        free, community test key "1" — fine for hobby/personal-scale
        use). New intent: `recommend_recipe`.
        """
        cuisine = (entities.get("cuisine") or entities.get("area") or "").strip() or None
        want_cocktail = (entities.get("category") or "").strip().lower() == "drink"

        try:
            if want_cocktail:
                pick = _fa_cocktail_suggestion()
                if not pick:
                    return {"response": "Couldn't find a cocktail suggestion right now.", "data": {}, "confidence": 0.3}
                response = f"Try: {pick['name']} (served in a {pick.get('glass', 'glass')})"
                data = {"type": "cocktail", **pick}
            else:
                pick = _fa_meal_suggestion(cuisine)
                if not pick:
                    return {"response": "Couldn't find a meal suggestion right now.", "data": {}, "confidence": 0.3}
                response = f"Try cooking: {pick['name']}"
                if pick.get("area"):
                    response += f" ({pick['area']} cuisine)"
                data = {"type": "meal", **pick}
        except FreeAPIError:
            log.exception("_recommend_recipe: upstream recipe API failed.")
            return {"response": "Recipe service is unavailable right now — try again shortly.", "data": {}, "confidence": 0.2}

        self.db.log("recipe", pick.get("name", ""), response)
        return {"response": response, "data": data, "confidence": 0.85}

    # ── find_events (backlog #146) ────────────────────────────────────────────

    def _find_events(self, entities: dict) -> dict:
        """Events in the user's city via web search (BookMyShow / Insider style
        listings). Results are search hits, not verified listings, and the reply
        says so. Each hit is logged as type "event" so it can be dismissed,
        marked seen, or given more/less-like-this feedback like anything else."""
        if not self._browser:
            return {
                "response": "Browser agent not available for event search.",
                "data": {},
                "confidence": 0.0,
            }

        area = (entities.get("area") or entities.get("location") or "").strip()
        if not area or area.lower() in _HERE_WORDS:
            loc = self._memory.get_preference("location", "") if self._memory else ""
            area = loc or "Mumbai"
        when = (entities.get("date") or entities.get("time") or "").strip()
        category = (entities.get("category") or entities.get("genre")
                    or entities.get("topic") or "").strip()

        parts = [category, "events in", area]
        if "mumbai" not in area.lower():
            parts.append("Mumbai")
        if when:
            parts.append(when)
        base = " ".join(p for p in parts if p)

        hidden = {t.strip().lower() for t in self._dismissed("event")}
        hidden |= {t.strip().lower() for t in self.db.seen_titles("event")}

        found: list[str] = []
        for query in (base, f"{base} BookMyShow Insider"):
            raw = self._browser.search_web(query)
            if not raw or str(raw).startswith("No results"):
                continue
            for item in str(raw).split(" | "):
                item = item.strip()[:200]
                if (item and item.lower() not in hidden
                        and item.lower() not in {f.lower() for f in found}):
                    found.append(item)

        if not found:
            return {
                "response": f"Couldn't find new events for {category or 'anything'} in {area}.",
                "data": {"query": base, "events": []},
                "confidence": 0.3,
            }

        found = found[:5]
        for title in found:
            self.db.log("event", title, base)

        header = f"Events — {category or 'anything'} in {area}"
        if when:
            header += f" ({when})"
        lines = [header, ""]
        for i, title in enumerate(found, 1):
            lines.append(f"  {i}. {title}")
        lines += ["", "These are search results, not confirmed listings. Check each "
                      "page for dates, prices and availability."]
        return {
            "response": "\n".join(lines).strip(),
            "data": {"query": base, "events": found},
            "confidence": 0.8,
        }

    # ── surprise_me (backlog #150) ────────────────────────────────────────────

    def _surprise_me(self, entities: dict) -> dict:
        """Movies (default) or music picked on purpose from outside the user's
        recent taste, to counter recommendation staleness."""
        text = " ".join(
            str(entities.get(k) or "") for k in ("type", "category", "topic", "raw_query")
        ).lower()
        wants_music = bool(re.search(r"\b(music|songs?|tracks?|artists?|albums?|listen)\b", text))
        if wants_music:
            result = self._recommend_music(entities, surprise=True)
        else:
            result = self._recommend_movie(entities, surprise=True)
        result.setdefault("data", {})["surprise"] = True
        return result

    def _taste_block(self, type_: str, surprise: bool = False) -> str:
        """Extra prompt lines built from the user's own feedback (#269) and,
        for "surprise me" (#150), a request to step outside their recent picks."""
        liked = self.db.feedback_titles(type_, 1)
        disliked = self.db.feedback_titles(type_, -1)
        parts: list[str] = []
        if surprise:
            recent = self.db.recent_titles(type_, 10)
            line = ("SURPRISE REQUEST: deliberately choose outside the user's usual "
                    "taste. ")
            if recent:
                line += f"Their recent picks were: {', '.join(recent)}. "
            line += ("Pick a different genre, era or country from those, but still "
                     "something a curious person would enjoy.")
            parts.append(line)
        elif liked:
            parts.append(f"The user liked these (lean toward similar): {', '.join(liked)}.")
        if disliked:
            parts.append(f"The user did not like these (avoid similar): {', '.join(disliked)}.")
        return ("\n".join(parts) + "\n") if parts else ""

    # ── more_like_this / less_like_this (backlog #269) ────────────────────────

    def _give_feedback(self, entities: dict, value: int) -> dict:
        title = str(entities.get("title") or "").strip()
        type_ = _norm_type(entities.get("type") or entities.get("category"))
        if title.lower() in _DEICTIC:
            title = ""

        known = True
        if not title:
            row = self.db.latest(type_)
            if row is None:
                return {
                    "response": "Which recommendation do you mean?",
                    "data": {},
                    "confidence": 0.5,
                }
            title, type_ = row["title"], row["type"]
            self.db.set_feedback(title, value, type_)
        else:
            hit = self.db.set_feedback(title, value, type_)
            if hit:
                title, type_ = hit["title"], hit["type"]
            else:
                # Not something we recommended, but the opinion is still worth
                # keeping, so remember it as a taste signal.
                known = False
                type_ = type_ or "movie"
                self.db.log(type_, title, "")
                self.db.set_feedback(title, value, type_)

        data = {"title": title, "type": type_, "feedback": value, "known": known}
        steers = type_ in ("movie", "music")

        if value > 0:
            if not steers:
                return {
                    "response": (f"Noted, you liked '{title}'. I use this to shape "
                                 f"movie and music picks."),
                    "data": data,
                    "confidence": 0.85,
                }
            seed = {"mood": f"similar in feel to {title}", "raw_query": f"more like {title}"}
            result = (self._recommend_movie(seed) if type_ == "movie"
                      else self._recommend_music(seed))
            result["response"] = f"Noted, more like '{title}'.\n\n" + result["response"]
            result.setdefault("data", {}).update(data)
            return result

        # less like this: also dismiss, so it works for every type (and expires
        # with other dismissals if the user has that turned on).
        self.db.dismiss(title)
        tail = " and I'll steer away from similar picks" if steers else ""
        return {
            "response": f"Got it, less like '{title}'. I won't suggest it again{tail}.",
            "data": data,
            "confidence": 0.9,
        }

    # ── schedule_recharge (backlog #152) ──────────────────────────────────────

    def _schedule_recharge(self, entities: dict) -> dict:
        """A repeating "recharge" downtime block, created as a Chronos reminder."""
        if self._chronos is None:
            return {
                "response": "I need Chronos to set repeating reminders, and it isn't connected right now.",
                "data": {},
                "confidence": 0.2,
            }

        raw = str(entities.get("raw_query") or "")
        schedule = str(entities.get("schedule") or "").strip()
        if not schedule:
            m = re.search(r"\bevery\s+[^,.;]+", raw, re.I)
            if m:
                schedule = re.split(r"\s+for\s+\d", m.group(0), maxsplit=1)[0].strip()
        if not schedule and re.search(r"\bweekends?\b", raw, re.I):
            schedule = "every weekend at 4pm"
        if not schedule:
            schedule = _DEFAULT_RECHARGE_SCHEDULE

        duration = str(entities.get("duration") or "").strip()
        if not duration:
            m = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)\b", raw, re.I)
            duration = f"{m.group(1)} {m.group(2)}" if m else _DEFAULT_RECHARGE_DURATION

        for routine in self.db.list_routines():
            if routine["schedule"].strip().lower() == schedule.lower():
                return {
                    "response": f"You already have a recharge routine {schedule}.",
                    "data": {"schedule": schedule, "duplicate": True},
                    "confidence": 0.85,
                }

        task = f"take your recharge break ({duration}): switch off work and do something just for you"
        try:
            res = self._chronos.handle(
                "set_reminder",
                {"raw_query": f"remind me to {task} {schedule}", "task": task},
                {},
            )
        except Exception:
            log.exception("schedule_recharge: Chronos failed.")
            return {
                "response": "I couldn't set the recharge reminder. Please try again.",
                "data": {},
                "confidence": 0.2,
            }

        data = (res or {}).get("data") or {}
        if not data.get("recurring"):
            # Chronos asked a question or refused; pass that on as it is.
            return {
                "response": (res or {}).get("response", "I couldn't set that up."),
                "data": data,
                "confidence": float((res or {}).get("confidence") or 0.3),
            }

        self.db.add_routine(task, schedule, data.get("id"))
        return {
            "response": (f"Recharge routine set: {duration} of downtime, {schedule}. "
                         f"{res.get('response', '')} To drop it, say 'cancel reminder recharge'.").strip(),
            "data": {**data, "schedule": schedule, "duration": duration},
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
            cost = _to_rupees(slot.get("cost_per_person"))
            if cost is not None:
                lines.append("  Cost     : " + ("free" if cost == 0 else f"~\u20b9{cost:,.0f} per person"))
            travel = slot.get("travel_to_next", "")
            if travel:
                lines.append(f"  Travel   : {travel}")
            lines.append("")

        total = result.get("estimated_cost_per_person")
        if total is not None:
            lines.append(f"Estimated total: ~\u20b9{total:,.0f} per person "
                         f"(a rough estimate, not live prices)")
            over = result.get("over_budget_by")
            if over:
                lines.append(f"That is about \u20b9{over:,.0f} over your budget of "
                             f"\u20b9{result.get('budget_per_person', 0):,.0f}. "
                             f"Ask me for a cheaper plan if you like.")
            lines.append("")

        tips = result.get("tips", [])
        if tips:
            lines.append("TIPS")
            for tip in tips:
                lines.append(f"  • {tip}")

        return "\n".join(lines).strip()


_GENERIC_WORDS = frozenset(
    """recommend suggest me a an some the movie movies film films song songs
    music tune tunes track tracks watch watching listen listening play
    something anything good nice great new tonight today now please what
    should i can you could give find want need like to for of on any
    something's whats what's""".split()
)


def _is_generic_request(raw_query) -> bool:
    """True when the query names nothing specific (no genre, title, artist or
    mood word), so a logged mood is a fair thing to go on."""
    tokens = [t for t in re.findall(r"[a-z']+", str(raw_query or "").lower())]
    return all(t in _GENERIC_WORDS for t in tokens)


# ── helpers for #151 (budget), #269 (feedback targets), #146 (events) ─────────

_HERE_WORDS = frozenset({"near me", "nearby", "here", "around me", "around here", "close by"})

_DEICTIC = frozenset({
    "that", "this", "it", "them", "those", "these", "that one", "this one",
    "the last one", "last one", "the last", "the last recommendation",
})

_TYPE_WORDS = {
    "movie": "movie", "movies": "movie", "film": "movie", "films": "movie",
    "music": "music", "song": "music", "songs": "music", "track": "music",
    "artist": "music", "album": "music",
    "restaurant": "restaurant", "food": "restaurant", "place": "restaurant",
    "outing": "outing", "plan": "outing",
    "recipe": "recipe", "meal": "recipe", "cocktail": "recipe", "drink": "recipe",
    "event": "event", "events": "event",
}


def _norm_type(value) -> Optional[str]:
    return _TYPE_WORDS.get(str(value or "").strip().lower())


_NUM_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand|lakhs?|lacs?)?", re.I)
_MULT = {"k": 1_000, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
         "lac": 100_000, "lacs": 100_000}


def _number(match) -> float:
    n = float(match.group(1).replace(",", ""))
    return n * _MULT.get((match.group(2) or "").lower(), 1)


def _to_rupees(value) -> Optional[float]:
    """A rupee amount from whatever the model wrote ("400", "\u20b9300-500", "free").
    Ranges use the upper end so estimates err on the high side."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    text = str(value).strip().lower()
    if re.search(r"\bfree\b", text):
        return 0.0
    nums = [_number(m) for m in _NUM_RE.finditer(text)]
    return max(nums) if nums else None


def _outing_total(result: dict) -> Optional[float]:
    """Sum of the per-slot cost estimates, or None if no slot gave one."""
    costs = [
        _to_rupees(slot.get("cost_per_person"))
        for slot in (result.get("slots") or []) if isinstance(slot, dict)
    ]
    costs = [c for c in costs if c is not None]
    return sum(costs) if costs else None


def _parse_budget(raw) -> Optional[dict]:
    """``{"amount": per-person rupees | None, "tier": cheap/mid/premium | None}``
    from text like "under 1500", "2k for two", or "cheap"; None if neither."""
    text = str(raw or "").strip().lower()
    if not text or text in {"none", "null", "n/a"}:
        return None
    amount = None
    m = _NUM_RE.search(text)
    if m:
        amount = _number(m)
        if re.search(r"for\s+(?:two|2)\b|\bcouple\b", text):
            amount /= 2
    tier = next((_TIER_WORDS[w] for w in re.findall(r"[a-z-]+", text) if w in _TIER_WORDS), None)
    if amount is None and tier is None:
        return None
    return {"raw": text, "amount": amount, "tier": tier}


def _range_text(lo, hi) -> str:
    if hi is None:
        return f"\u20b9{lo:,.0f}+"
    return f"\u20b9{lo:,.0f}\u2013{hi:,.0f}"


def _budget_note(info: Optional[dict]) -> str:
    if not info:
        return ""
    if info["amount"] is not None:
        return (f"Budget: about \u20b9{info['amount']:,.0f} per person. Search results "
                f"rarely show prices, so check the menu before you go.")
    lo, hi = _BUDGET_TIERS[info["tier"]]
    return (f"Budget: {info['tier']}, which is typically {_range_text(lo, hi)} per "
            f"person in Mumbai (a rough range, not live prices).")


_PRICE_RE = re.compile(r"(?:\u20b9|rs\.?|inr)\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I)


def _price_in_text(text: str) -> Optional[float]:
    """A per-person price stated in a search-result title, if there is one."""
    m = _PRICE_RE.search(text or "")
    if not m:
        return None
    price = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
    if re.search(r"for\s+(?:two|2)\b", text, re.I):
        price /= 2
    return price
