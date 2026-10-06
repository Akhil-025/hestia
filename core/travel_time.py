"""
core/travel_time.py

Driving-time estimates between two place names, for backlog #95 (flag
back-to-back calendar events that don't leave enough time to get between
them).

Providers (``hermes.travel.provider``):

  flat    (default) never calls out; ``estimate()`` returns ``None`` so the
          caller falls back to its flat allowance. Nothing leaves the machine.
  osrm    Nominatim (OpenStreetMap) to geocode each place, then the public
          OSRM demo server to route between them. No API key. Both are shared
          community services: results are cached and geocoding is throttled
          to one request a second, as Nominatim's usage policy asks. Driving
          only.
  google  Google Distance Matrix (needs ``hermes.travel.google_api_key``).

Choosing osrm or google sends your event locations to that service. That is
why the default is ``flat``.

Every failure returns ``None`` (never raises) so a flaky lookup degrades to
the flat allowance instead of breaking the schedule check.

Status: tested against a fake transport only; the real services were not
reachable from the build sandbox.
"""
from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PROVIDERS = ("flat", "osrm", "google")
_NOMINATIM = "https://nominatim.openstreetmap.org/search"
_OSRM = "https://router.project-osrm.org/route/v1/driving"
_GOOGLE = "https://maps.googleapis.com/maps/api/distancematrix/json"
_MIN_GEOCODE_INTERVAL = 1.0
_CACHE_MAX = 512
_TIMEOUT = 8.0

# (url, params, headers, timeout) -> parsed JSON (raises on failure)
GetJson = Callable[[str, dict, dict, float], Any]


def _urllib_get_json(url: str, params: dict, headers: dict, timeout: float) -> Any:
    full = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    req = urllib.request.Request(full, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https hosts)
        return json.loads(resp.read().decode("utf-8"))


@dataclass(frozen=True)
class TravelEstimate:
    minutes: int
    source: str          # "osrm" | "google"

    def to_dict(self) -> dict[str, Any]:
        return {"minutes": self.minutes, "source": self.source}


def _norm(place: str) -> str:
    return " ".join((place or "").lower().split())


class TravelTimeEstimator:
    def __init__(
        self,
        provider: str = "flat",
        *,
        google_api_key: str = "",
        user_agent: str = "Hestia-assistant/1.0 (personal use)",
        get_json: Optional[GetJson] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        provider = (provider or "flat").strip().lower()
        if provider not in PROVIDERS:
            logger.warning("Unknown travel provider %r; using 'flat'.", provider)
            provider = "flat"
        if provider == "google" and not google_api_key:
            logger.warning("travel.provider is 'google' but no google_api_key; using 'flat'.")
            provider = "flat"
        self.provider = provider
        self._key = google_api_key
        self._headers = {"User-Agent": user_agent, "Accept": "application/json"}
        self._get = get_json or _urllib_get_json
        self._sleep = sleep
        self._clock = clock
        self._last_geocode = float("-inf")
        self._geo_cache: dict[str, Optional[tuple[float, float]]] = {}
        self._route_cache: dict[tuple[str, str], Optional[TravelEstimate]] = {}

    @property
    def enabled(self) -> bool:
        return self.provider != "flat"

    def estimate(self, origin: str, destination: str) -> Optional[TravelEstimate]:
        """Driving minutes between two places, or ``None`` if unknown."""
        o, d = _norm(origin), _norm(destination)
        if not self.enabled or not o or not d:
            return None
        if o == d:
            return TravelEstimate(0, self.provider)
        key = (o, d)
        if key in self._route_cache:
            return self._route_cache[key]
        try:
            result = self._google(o, d) if self.provider == "google" else self._osrm(o, d)
        except Exception:
            logger.exception("Travel lookup %r -> %r failed.", o, d)
            return None          # not cached: a later try may work
        if len(self._route_cache) >= _CACHE_MAX:
            self._route_cache.pop(next(iter(self._route_cache)))
        self._route_cache[key] = result
        return result

    def _geocode(self, place: str) -> Optional[tuple[float, float]]:
        if place in self._geo_cache:
            return self._geo_cache[place]
        wait = _MIN_GEOCODE_INTERVAL - (self._clock() - self._last_geocode)
        if wait > 0:
            self._sleep(wait)
        self._last_geocode = self._clock()
        data = self._get(
            _NOMINATIM, {"q": place, "format": "json", "limit": 1}, self._headers, _TIMEOUT
        )
        point: Optional[tuple[float, float]] = None
        if isinstance(data, list) and data:
            try:
                point = (float(data[0]["lat"]), float(data[0]["lon"]))
            except (KeyError, TypeError, ValueError):
                point = None
        self._geo_cache[place] = point
        return point

    def _osrm(self, o: str, d: str) -> Optional[TravelEstimate]:
        a, b = self._geocode(o), self._geocode(d)
        if a is None or b is None:
            return None
        coords = f"{a[1]:.6f},{a[0]:.6f};{b[1]:.6f},{b[0]:.6f}"   # OSRM wants lon,lat
        data = self._get(f"{_OSRM}/{coords}", {"overview": "false"}, self._headers, _TIMEOUT)
        if not isinstance(data, dict) or data.get("code") != "Ok":
            return None
        routes = data.get("routes") or []
        if not routes:
            return None
        seconds = routes[0].get("duration")
        if not isinstance(seconds, (int, float)) or seconds < 0:
            return None
        return TravelEstimate(int(math.ceil(seconds / 60)), "osrm")

    def _google(self, o: str, d: str) -> Optional[TravelEstimate]:
        data = self._get(
            _GOOGLE,
            {"origins": o, "destinations": d, "mode": "driving", "key": self._key},
            self._headers, _TIMEOUT,
        )
        if not isinstance(data, dict) or data.get("status") != "OK":
            return None
        try:
            element = data["rows"][0]["elements"][0]
            if element.get("status") != "OK":
                return None
            seconds = element["duration"]["value"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(seconds, (int, float)) or seconds < 0:
            return None
        return TravelEstimate(int(math.ceil(seconds / 60)), "google")
