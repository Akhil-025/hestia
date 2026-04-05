"""
web_ui.py — Hestia local Flask dashboard (Mnemosyne-compatible).
"""

import datetime
import logging
import threading
import time
from typing import Callable, Optional

from flask import Flask, jsonify, request, render_template

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 2_000
STATS_TTL = 30


class HestiaWebUI:
    def __init__(
        self,
        memory,  # MnemosyneEngine
        skill_loader=None,
        process_fn: Optional[Callable[[str], str]] = None,
        host: str = "127.0.0.1",
        port: int = 5000,
    ) -> None:
        self.memory = memory
        self.skill_loader = skill_loader
        self.process_fn = process_fn
        self.host = host
        self.port = port

        self._thread: Optional[threading.Thread] = None
        self._stats_cache: dict = {}

        self.app = Flask(__name__, template_folder="templates")

        self._warn_missing_deps()
        self._register_ui_routes()
        self._register_memory_routes()
        self._register_chat_routes()
        self._register_admin_routes()

    # ── Startup ─────────────────────────────────────────

    def _warn_missing_deps(self) -> None:
        if self.skill_loader is None:
            logger.warning("[WebUI] skill_loader not provided")
        if self.process_fn is None:
            logger.warning("[WebUI] process_fn not provided")

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
                        "pushed_at": r.get("pushed_at"),  # NEW
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
                        "pushed_at": r.get("pushed_at"),  # NEW
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
            # Not implemented in Mnemosyne yet
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