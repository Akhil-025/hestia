"""
web_ui.py — Hestia local Flask dashboard (Mnemosyne-compatible).
"""

import collections
import datetime
import logging
import threading
import time
from typing import Callable, Optional

from flask import Flask, jsonify, request, render_template

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 2_000
STATS_TTL = 30

# Rate limiting for /api/chat — simple fixed-window-per-IP counter.
_RATE_LIMIT = 10
_RATE_WINDOW = 10.0


class HestiaWebUI:
    def __init__(
        self,
        memory,  # MnemosyneEngine
        skill_loader=None,
        process_fn: Optional[Callable[[str], str]] = None,
        host: str = "127.0.0.1",
        port: int = 5000,
        api_key: Optional[str] = None,
        apollo=None,  # ApolloEngine | None — powers /api/moods
    ) -> None:
        self.memory = memory
        self.skill_loader = skill_loader
        self.process_fn = process_fn
        self.host = host
        self.port = port
        self.apollo = apollo
        # Optional shared-secret auth for /api/*. If unset, the API is
        # unauthenticated (fine for strictly-localhost, single-user use —
        # but anything reachable beyond localhost should set this).
        self.api_key = api_key

        self._thread: Optional[threading.Thread] = None
        self._stats_cache: dict = {}
        self._rate_buckets: dict = collections.defaultdict(list)

        self.app = Flask(__name__, template_folder="templates")

        self._warn_missing_deps()
        self._register_auth_guard()
        self._register_ui_routes()
        self._register_memory_routes()
        self._register_chat_routes()
        self._register_admin_routes()

    # ── Startup ─────────────────────────────────────────

    def _warn_missing_deps(self) -> None:
        # skill_loader is an optional, not-yet-implemented feature — its
        # absence is expected in most deployments, so this stays at debug
        # level rather than warning on every single startup.
        if self.skill_loader is None:
            logger.debug("[WebUI] skill_loader not provided (optional).")
        if self.process_fn is None:
            logger.warning("[WebUI] process_fn not provided")

    def _register_auth_guard(self) -> None:
        """If api_key is configured, require it (via X-API-Key header or
        ?api_key= query param) on every /api/* request. The UI page itself
        (/) and static assets remain open so the dashboard can load."""
        if not self.api_key:
            logger.warning(
                "[WebUI] No api_key configured — /api/* endpoints are "
                "unauthenticated. Set api_key if this is reachable beyond localhost."
            )
            return

        @self.app.before_request
        def _check_api_key():
            if not request.path.startswith("/api/"):
                return None
            supplied = request.headers.get("X-API-Key") or request.args.get("api_key")
            if supplied != self.api_key:
                return jsonify({"error": "Unauthorized"}), 401
            return None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="HestiaWebUI"
        )
        self._thread.start()
        logger.info("[WebUI] Running at http://%s:%s", self.host, self.port)

    def _run(self) -> None:
        try:
            from waitress import serve
            serve(self.app, host=self.host, port=self.port, threads=4)
        except ImportError:
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    # ── UI ──────────────────────────────────────────────

    def _register_ui_routes(self) -> None:
        @self.app.route("/")
        def index():
            return render_template("index.html")

    # ── MEMORY ROUTES (FULLY FIXED) ─────────────────────

    def _register_memory_routes(self) -> None:
        app = self.app

        @app.route("/api/history")
        def api_history():
            try:
                limit = max(1, min(int(request.args.get("limit", 50)), 500))
            except ValueError:
                return jsonify({"error": "invalid limit"}), 400

            exclude = [e for e in request.args.get("exclude", "").split(",") if e]

            try:
                rows = self.memory.db.get_recent_interactions(limit * 2)

                if exclude:
                    rows = [r for r in rows if r["intent"] not in exclude]

                rows = rows[:limit]

                data = [
                    {
                        "query": r["query"],
                        "response": r["response"],
                        "intent": r["intent"],
                        "timestamp": r.get("pushed_at"),
                    }
                    for r in rows
                ]
                return jsonify(data)

            except Exception:
                logger.exception("[WebUI] history error")
                return jsonify({"error": "Failed"}), 500

        @app.route("/api/notes")
        def api_notes():
            try:
                limit = max(1, min(int(request.args.get("limit", 50)), 500))
            except ValueError:
                limit = 50

            try:
                rows = self.memory.db.get_by_intent("take_note", limit)

                data = [
                    {
                        "query": r["query"],
                        "response": r["response"],
                        "intent": r["intent"],
                        "timestamp": r.get("pushed_at"),
                    }
                    for r in rows
                ]

                return jsonify(data)

            except Exception:
                logger.exception("[WebUI] notes error")
                return jsonify({"error": "Failed"}), 500

        @app.route("/api/facts")
        def api_facts():
            try:
                return jsonify(self.memory.db.get_all_facts())
            except Exception:
                logger.exception("[WebUI] facts error")
                return jsonify([])

        @app.route("/api/location", methods=["POST"])
        def api_location():
            """
            Accept {"lat": ..., "lon": ...} from the browser's
            navigator.geolocation (see templates/index.html) and persist it
            via Mnemosyne so every god can read it back through
            MnemosyneEngine.get_context()["device_location"].
            """
            try:
                payload = request.get_json(silent=True) or {}
                lat = payload.get("lat")
                lon = payload.get("lon")

                if lat is None or lon is None:
                    return jsonify({"ok": False, "error": "lat and lon are required"}), 400

                try:
                    lat = float(lat)
                    lon = float(lon)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": "lat/lon must be numbers"}), 400

                if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
                    return jsonify({"ok": False, "error": "lat/lon out of range"}), 400

                self.memory.set_device_location(lat, lon, source="browser_gps")
                return jsonify({"ok": True})
            except Exception:
                logger.exception("[WebUI] location error")
                return jsonify({"ok": False, "error": "internal error"}), 500

        @app.route("/api/preferences")
        def api_preferences():
            try:
                facts = self.memory.db.get_all_facts()
                prefs = {f["key"]: f["value"] for f in facts}
                return jsonify(prefs)
            except Exception:
                logger.exception("[WebUI] prefs error")
                return jsonify({})

        @app.route("/api/moods")
        def api_moods():
            if not self.apollo:
                return jsonify([])
            try:
                days = max(1, min(int(request.args.get("days", 7)), 90))
                return jsonify(self.apollo.db.get_mood(days))
            except Exception:
                logger.exception("[WebUI] moods error")
                return jsonify([])

        @app.route("/api/stats")
        def api_stats():
            cache = self._stats_cache

            if cache.get("data") and time.monotonic() < cache.get("expires_at", 0):
                return jsonify(cache["data"])

            try:
                data = self._get_stats()

                self._stats_cache = {
                    "data": data,
                    "expires_at": time.monotonic() + STATS_TTL,
                }

                return jsonify(data)

            except Exception:
                logger.exception("[WebUI] stats error")
                return jsonify({})

        @app.route("/api/skills")
        def api_skills():
            if not self.skill_loader:
                return jsonify([])

            try:
                registry = self.skill_loader.SKILL_REGISTRY

                return jsonify([
                    {
                        "intent": intent,
                        "description": meta.get("description", ""),
                        "source": meta.get("source", ""),
                        "examples": meta.get("examples", []),
                    }
                    for intent, meta in registry.items()
                ])

            except Exception:
                logger.exception("[WebUI] skills error")
                return jsonify([])

    # ── CHAT ────────────────────────────────────────────

    def _register_chat_routes(self) -> None:
        @self.app.route("/api/chat", methods=["POST"])
        def api_chat():
            if not self.process_fn:
                return jsonify({"error": "Chat disabled"}), 503

            ip = request.remote_addr or "unknown"
            now = time.monotonic()
            bucket = self._rate_buckets[ip]
            bucket[:] = [t for t in bucket if now - t < _RATE_WINDOW]
            if len(bucket) >= _RATE_LIMIT:
                return jsonify({"error": "Rate limit exceeded"}), 429
            bucket.append(now)

            if not request.is_json:
                return jsonify({"error": "JSON required"}), 415

            body = request.get_json(silent=True) or {}
            text = body.get("text", "").strip()

            if not text:
                return jsonify({"error": "Empty"}), 400

            if len(text) > MAX_TEXT_LENGTH:
                return jsonify({"error": "Too long"}), 413

            try:
                response = self.process_fn(text)
                return jsonify({"response": response or "..."})
            except Exception:
                logger.exception("[WebUI] chat error")
                return jsonify({"error": "Failed"}), 500

    # ── ADMIN ───────────────────────────────────────────

    def _register_admin_routes(self) -> None:
        @self.app.route("/api/reload", methods=["POST"])
        def api_reload():
            if not self.skill_loader:
                return jsonify({"error": "No loader"}), 503

            try:
                self.skill_loader.reload()
                return jsonify({"status": "ok"})
            except Exception:
                logger.exception("[WebUI] reload error")
                return jsonify({"error": "Failed"}), 500

    # ── STATS ───────────────────────────────────────────

    def _get_stats(self) -> dict:
        try:
            data = self.memory.get_stats()
        except Exception:
            data = {}

        data["uptime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return data