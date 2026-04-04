"""
web_ui.py — Hestia local Flask dashboard.

"""

import datetime
import logging
import threading
import time
from typing import Callable, Optional

from flask import Flask, jsonify, request, render_template

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_TEXT_LENGTH = 2_000   # characters; guards against runaway LLM calls
STATS_TTL       = 30      # seconds before stats cache is considered stale


class HestiaWebUI:
    """Local Flask dashboard served at localhost:5000.

    Provides history, notes, facts, skills, moods, preferences,
    and a live webchat interface backed by Hestia's process_fn.
    """

    def __init__(
        self,
        memory,
        skill_loader=None,
        process_fn: Optional[Callable[[str], str]] = None,
        host: str = "127.0.0.1",
        port: int = 5000,
    ) -> None:
        """
        Args:
            memory:       HestiaMemory instance.
            skill_loader: SkillLoader instance (optional).
            process_fn:   Hestia.process_text callable for webchat (optional).
            host:         Bind address — keep as 127.0.0.1 for local-only.
            port:         TCP port.
        """
        self.memory       = memory
        self.skill_loader = skill_loader
        self.process_fn   = process_fn
        self.host         = host
        self.port         = port

        self._thread: Optional[threading.Thread] = None

        # Stats TTL cache  {data: dict, expires_at: float}
        self._stats_cache: dict = {}

        self.app = Flask(__name__, template_folder="templates")

        self._warn_missing_deps()
        self._register_ui_routes()
        self._register_memory_routes()
        self._register_chat_routes()
        self._register_admin_routes()

    # ── Startup ──────────────────────────────────────────────────────────────

    def _warn_missing_deps(self) -> None:
        if self.skill_loader is None:
            logger.warning("[WebUI] skill_loader not provided — /api/skills will return []")
        if self.process_fn is None:
            logger.warning("[WebUI] process_fn not provided — /api/chat will return 503")

    def start(self) -> None:
        """Start the server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="HestiaWebUI"
        )
        self._thread.start()
        logger.info("[WebUI] Dashboard running at http://%s:%s", self.host, self.port)

    def _run(self) -> None:
        """Attempt waitress first; fall back to Flask dev server."""
        try:
            from waitress import serve
            logger.info("[WebUI] Using waitress WSGI server")
            serve(self.app, host=self.host, port=self.port, threads=4)
        except ImportError:
            logger.warning(
                "[WebUI] waitress not installed — falling back to Flask dev server. "
                "Run `pip install waitress` for a more robust server."
            )
            # Silence Werkzeug request logs; keep error logs
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    # ── Route groups ─────────────────────────────────────────────────────────

    def _register_ui_routes(self) -> None:
        @self.app.route("/")
        def index():
            return render_template("index.html")

    def _register_memory_routes(self) -> None:
        app = self.app

        @app.route("/api/history")
        def api_history():
            try:
                limit = max(1, min(int(request.args.get("limit", 50)), 500))
            except ValueError:
                return jsonify({"error": "limit must be an integer"}), 400

            exclude = [e for e in request.args.get("exclude", "").split(",") if e]

            try:
                if exclude:
                    data = self.memory.get_recent_filtered(limit=limit, exclude_intents=exclude)
                else:
                    data = self.memory.get_recent(limit=limit)
                return jsonify(data)
            except Exception:
                logger.exception("[WebUI] /api/history error")
                return jsonify({"error": "Failed to fetch history"}), 500

        @app.route("/api/notes")
        def api_notes():
            try:
                limit = max(1, min(int(request.args.get("limit", 50)), 500))
            except ValueError:
                limit = 50
            try:
                notes = self.memory.get_by_intent("take_note", limit=limit)
                return jsonify(notes)
            except Exception:
                logger.exception("[WebUI] /api/notes error")
                return jsonify({"error": "Failed to fetch notes"}), 500

        @app.route("/api/facts")
        def api_facts():
            try:
                facts = self.memory.get_all_facts()
            except AttributeError:
                logger.warning("[WebUI] memory.get_all_facts() not implemented")
                facts = []
            except Exception:
                logger.exception("[WebUI] /api/facts error")
                return jsonify({"error": "Failed to fetch facts"}), 500
            return jsonify(facts)

        @app.route("/api/moods")
        def api_moods():
            try:
                days = max(1, min(int(request.args.get("days", 30)), 365))
            except ValueError:
                days = 30
            try:
                moods = self.memory.get_recent_moods(days=days)
            except AttributeError:
                logger.warning("[WebUI] memory.get_recent_moods() not implemented")
                moods = []
            except Exception:
                logger.exception("[WebUI] /api/moods error")
                return jsonify({"error": "Failed to fetch moods"}), 500
            return jsonify(moods)

        @app.route("/api/preferences")
        def api_preferences():
            try:
                prefs = self.memory.get_all_preferences()
                return jsonify(prefs)
            except Exception:
                logger.exception("[WebUI] /api/preferences error")
                return jsonify({"error": "Failed to fetch preferences"}), 500

        @app.route("/api/skills")
        def api_skills():
            if not self.skill_loader:
                return jsonify([])
            try:
                registry = self.skill_loader.SKILL_REGISTRY
                skills = [
                    {
                        "intent":      intent,
                        "description": meta.get("description", ""),
                        "source":      meta.get("source", ""),
                        "examples":    meta.get("examples", []),
                    }
                    for intent, meta in registry.items()
                ]
                return jsonify(skills)
            except Exception:
                logger.exception("[WebUI] /api/skills error")
                return jsonify({"error": "Failed to fetch skills"}), 500

        @app.route("/api/stats")
        def api_stats():
            # Serve from cache if still fresh
            cache = self._stats_cache
            if cache.get("data") and time.monotonic() < cache.get("expires_at", 0):
                return jsonify(cache["data"])
            try:
                data = self._get_stats()
                self._stats_cache = {
                    "data":       data,
                    "expires_at": time.monotonic() + STATS_TTL,
                }
                return jsonify(data)
            except Exception:
                logger.exception("[WebUI] /api/stats error")
                return jsonify({"error": "Failed to fetch stats"}), 500

    def _register_chat_routes(self) -> None:
        @self.app.route("/api/chat", methods=["POST"])
        def api_chat():
            if not self.process_fn:
                return jsonify({"error": "Chat not available — process_fn not configured"}), 503

            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 415

            body = request.get_json(silent=True) or {}
            text = body.get("text", "")

            if not isinstance(text, str):
                return jsonify({"error": "text must be a string"}), 400

            text = text.strip()
            if not text:
                return jsonify({"error": "Empty message"}), 400

            if len(text) > MAX_TEXT_LENGTH:
                return jsonify({
                    "error": f"Message too long (max {MAX_TEXT_LENGTH} characters)"
                }), 413

            try:
                t0 = time.monotonic()
                response = self.process_fn(text)
                elapsed = time.monotonic() - t0
                logger.debug("[WebUI] chat processed in %.2fs", elapsed)
                return jsonify({"response": response or "..."})
            except Exception:
                logger.exception("[WebUI] /api/chat process_fn error")
                return jsonify({"error": "Hestia encountered an error processing your message"}), 500

    def _register_admin_routes(self) -> None:
        @self.app.route("/api/reload", methods=["POST"])
        def api_reload():
            """Hot-reload the skill registry without restarting the process.
            Only available if skill_loader is configured.
            """
            if not self.skill_loader:
                return jsonify({"error": "No skill_loader configured"}), 503
            try:
                if hasattr(self.skill_loader, "reload"):
                    self.skill_loader.reload()
                    logger.info("[WebUI] Skills reloaded via /api/reload")
                    return jsonify({"status": "ok", "skills": len(self.skill_loader.SKILL_REGISTRY)})
                else:
                    return jsonify({"error": "skill_loader does not support reload()"}), 501
            except Exception:
                logger.exception("[WebUI] /api/reload error")
                return jsonify({"error": "Reload failed"}), 500

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_stats(self) -> dict:
        """
        Fetch stats via memory API (no direct SQL access).
        """
        data = self.memory.get_stats()

        data["uptime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        return data