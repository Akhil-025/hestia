"""
web_ui.py — Hestia local Flask dashboard (Mnemosyne-compatible).
"""

import collections
import datetime
import hmac
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote, urlparse

from flask import (
    Flask, Response, jsonify, redirect, render_template, request, session,
    stream_with_context,
)

from core.web_auth import PasswordAuth, resolve_secret_key, safe_next
from core.web_live import LiveHub

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
        pluto=None,  # PlutoEngine | None — powers /api/pluto/*
        artemis=None,  # ArtemisEngine | None — powers /api/artemis/*
        chronos=None,  # ChronosEngine | None — powers /api/chronos/*
        athena=None,  # AthenaEngine | None — powers /api/athena/*
        stt=None,  # HestiaSTT | None — powers /api/stt
        tts=None,  # HestiaTTS | None — powers /api/tts
        voice_state=None,  # core.voice_state.VoiceState | None — powers /api/voice/*
        iris=None,  # IrisEngine | None — photos in the cross-module search
        live_hub: Optional[LiveHub] = None,  # live updates (#187); one is made if omitted
        process_traced_fn: Optional[Callable] = None,  # text -> (reply, routing record) (#189)
        config_path=None,  # laptop_config.yaml, for the settings page (#188)
        password: Optional[str] = None,  # login (#186) — plain text; prefer password_hash
        password_hash: Optional[str] = None,
        secret_key: Optional[str] = None,  # signs session cookies
        session_hours: float = 12.0,
        cookie_secure: bool = False,
        allowed_origins=None,  # extra Origins allowed to POST (reverse proxies)
        session_key_file=None,  # where a generated cookie-signing key is kept
    ) -> None:
        self.memory = memory
        self.skill_loader = skill_loader
        self.process_fn = process_fn
        self.host = host
        self.port = port
        self.apollo = apollo
        self.pluto = pluto
        self.artemis = artemis
        self.chronos = chronos
        self.athena = athena
        self.stt = stt
        self.tts = tts
        self.voice_state = voice_state
        self.iris = iris
        self.live_hub = live_hub or LiveHub()
        self.process_traced_fn = process_traced_fn
        self.config_path = Path(config_path) if config_path else None
        self.auth = PasswordAuth(password=password, password_hash=password_hash)
        self._secret_key = secret_key
        self._session_hours = max(0.1, float(session_hours))
        self._cookie_secure = bool(cookie_secure)
        self._allowed_origins = {str(o).rstrip("/") for o in (allowed_origins or [])}
        self._session_key_file = session_key_file
        self._settings_lock = threading.Lock()
        self._dash_cache: tuple = (0.0, None)
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
        self._register_voice_routes()
        self._register_admin_routes()
        self._register_pluto_routes()
        self._register_artemis_routes()
        self._register_apollo_routes()
        self._register_chronos_routes()
        self._register_athena_routes()
        self._register_mnemosyne_routes()
        self._register_login_routes()
        self._register_dashboard_routes()
        self._register_live_routes()
        self._register_search_routes()
        self._register_settings_routes()
        self._register_export_routes()

    # ── Startup ─────────────────────────────────────────

    def _warn_missing_deps(self) -> None:
        # skill_loader is an optional, not-yet-implemented feature — its
        # absence is expected in most deployments, so this stays at debug
        # level rather than warning on every single startup.
        if self.skill_loader is None:
            logger.debug("[WebUI] skill_loader not provided (optional).")
        if self.process_fn is None:
            logger.warning("[WebUI] process_fn not provided")

    # Hosts that are only reachable from this machine. Anything else
    # (0.0.0.0, a LAN IP, a hostname, etc.) means /api/* is potentially
    # reachable by other devices/users.
    _LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

    _SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

    def _auth_mode(self) -> str:
        if self.auth.enabled:
            return "password"
        return "api_key" if self.api_key else "none"

    def _api_key_ok(self) -> bool:
        if not self.api_key:
            return False
        supplied = request.headers.get("X-API-Key") or request.args.get("api_key") or ""
        return hmac.compare_digest(supplied.encode("utf-8"), self.api_key.encode("utf-8"))

    def _is_authenticated(self) -> bool:
        if self._api_key_ok():
            return True
        return bool(self.auth.enabled and session.get("auth") is True)

    def _origin_ok(self) -> bool:
        """Browsers send Origin on cross-site POSTs. Reject one that is not us.

        A cookie login makes cross-site request forgery possible (another
        page makes your browser POST to this one); SameSite=Lax already stops
        most of it and this closes the rest. No Origin header (curl, scripts)
        is fine: a script has no ambient cookie to abuse.
        """
        origin = request.headers.get("Origin")
        if not origin or origin == "null":
            return origin is None
        origin = origin.rstrip("/")
        if origin in self._allowed_origins:
            return True
        return urlparse(origin).netloc == request.host

    def _register_auth_guard(self) -> None:
        """Gate /api/* (and the page itself, when a password is set).

        Three modes, chosen by what is configured:

        * password (#186)  a login page and a signed session cookie. The page
          and every /api/* call need a session; the API key, if one is also
          set, still works for scripts.
        * api_key          /api/* needs X-API-Key (or ?api_key=); the page is
          open so it can show its key prompt (the original behaviour).
        * none             unauthenticated, allowed only on a loopback host.
          Binding to a LAN address with neither configured is refused at
          start-up rather than run open by accident.
        """
        password_on = self.auth.enabled
        if not self.api_key and not password_on:
            if self.host not in self._LOOPBACK_HOSTS:
                raise ValueError(
                    f"[WebUI] Refusing to start: host={self.host!r} is not "
                    "loopback-only and no password or api_key is configured. "
                    "Set webui.password_hash (python scripts/hash_web_password.py) "
                    "or webui.api_key, or bind host to 127.0.0.1 for "
                    "strictly-local use."
                )
            logger.warning(
                "[WebUI] No password or api_key configured — /api/* endpoints are "
                "unauthenticated. This is only safe because host=%r is "
                "loopback-only.", self.host,
            )
            return

        if password_on:
            from datetime import timedelta
            app = self.app
            app.config.update(
                SECRET_KEY=resolve_secret_key(
                    self._secret_key,
                    self._session_key_file or "data/web_session.key",
                ),
                SESSION_COOKIE_NAME="hestia_session",
                SESSION_COOKIE_HTTPONLY=True,
                SESSION_COOKIE_SAMESITE="Lax",
                SESSION_COOKIE_SECURE=self._cookie_secure,
                PERMANENT_SESSION_LIFETIME=timedelta(hours=self._session_hours),
            )

        @self.app.before_request
        def _guard():
            path = request.path
            if path.startswith("/static/") or path == "/api/auth/status":
                return None
            if password_on and request.method not in self._SAFE_METHODS \
                    and not self._origin_ok():
                return jsonify({"error": "Cross-origin request refused"}), 403
            if path in ("/login", "/logout"):
                return None
            if self._is_authenticated():
                return None
            if path.startswith("/api/"):
                return jsonify({"error": "Unauthorized", "login": password_on}), 401
            if password_on:
                nxt = request.full_path.rstrip("?")
                return redirect("/login?next=" + quote(nxt))
            return None

    def start(self) -> None:
        try:
            from core.event_bus import bus
            self.live_hub.attach_bus(bus)
        except Exception:
            logger.exception("[WebUI] live updates unavailable; pages will poll.")
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="HestiaWebUI"
        )
        self._thread.start()
        logger.info("[WebUI] Running at http://%s:%s", self.host, self.port)

    def _run(self) -> None:
        try:
            from waitress import serve
            # Each open live-update stream holds one thread, so leave room for
            # ordinary requests beside the (capped) streams.
            serve(self.app, host=self.host, port=self.port, threads=16)
        except ImportError:
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    # ── UI ──────────────────────────────────────────────

    def _register_ui_routes(self) -> None:
        @self.app.route("/")
        def index():
            return render_template("index.html", auth_mode=self._auth_mode())

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
                # mood_entries() adds the score/valence/timestamp fields the
                # Moods tab reads; it used to read fields the table never had.
                if hasattr(self.apollo, "mood_entries"):
                    return jsonify(self.apollo.mood_entries(days))
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

        @app.route("/api/export")
        def api_export():
            try:
                return jsonify({
                    "facts": self.memory.db.get_all_facts(),
                    "notes": self.memory.db.get_by_intent("take_note", 1000),
                    "history": self.memory.db.get_recent_interactions(1000),
                    "exported_at": datetime.datetime.now().isoformat(),
                })
            except Exception:
                logger.exception("[WebUI] export error")
                return jsonify({"error": "Failed"}), 500

    # ── CHAT ────────────────────────────────────────────

    def _check_rate_limit(self, ip: str) -> bool:
        """Returns True if the request is allowed, False if the caller
        should be rejected with 429. Shared by /api/chat and
        /api/chat/stream so the limit can't be bypassed by switching
        endpoints."""
        now = time.monotonic()
        bucket = self._rate_buckets[ip]
        bucket[:] = [t for t in bucket if now - t < _RATE_WINDOW]
        if len(bucket) >= _RATE_LIMIT:
            return False
        bucket.append(now)
        return True

    def _run_chat(self, text: str) -> tuple:
        """Run one chat turn. Returns ``(reply, routing_record_or_None)``.

        The ``origin("web")`` tag lets the live feed tell this tab's own
        messages from ones that arrived by voice or Telegram.
        """
        with self.live_hub.origin("web"):
            if self.process_traced_fn is not None:
                return self.process_traced_fn(text)
            return self.process_fn(text), None

    @staticmethod
    def _public_trace(trace: Optional[dict]) -> Optional[dict]:
        """The part of a routing record the "Why this answer?" panel shows."""
        if not trace:
            return None
        out = {k: trace.get(k) for k in (
            "module", "intent", "confidence", "source", "reason",
            "latency_ms", "request_id")}
        out["checked"] = [str(c)[:120] for c in list(trace.get("checked") or [])[:8]]
        return out

    @staticmethod
    def _validate_chat_text(body: dict) -> tuple[Optional[str], Optional[tuple]]:
        """Returns (text, None) on success or (None, (error_body, status))
        on failure."""
        text = (body.get("text") or "").strip()
        if not text:
            return None, ({"error": "Empty"}, 400)
        if len(text) > MAX_TEXT_LENGTH:
            return None, ({"error": "Too long"}, 413)
        return text, None

    def _register_chat_routes(self) -> None:
        @self.app.route("/api/chat", methods=["POST"])
        def api_chat():
            if not self.process_fn:
                return jsonify({"error": "Chat disabled"}), 503

            ip = request.remote_addr or "unknown"
            if not self._check_rate_limit(ip):
                return jsonify({"error": "Rate limit exceeded"}), 429

            if not request.is_json:
                return jsonify({"error": "JSON required"}), 415

            body = request.get_json(silent=True) or {}
            text, error = self._validate_chat_text(body)
            if error:
                return jsonify(error[0]), error[1]

            try:
                response, trace = self._run_chat(text)
                return jsonify({"response": response or "...",
                                "trace": self._public_trace(trace)})
            except Exception:
                logger.exception("[WebUI] chat error")
                return jsonify({"error": "Failed"}), 500

        @self.app.route("/api/chat/stream", methods=["POST"])
        def api_chat_stream():
            if not self.process_fn:
                return jsonify({"error": "Chat disabled"}), 503

            ip = request.remote_addr or "unknown"
            if not self._check_rate_limit(ip):
                return jsonify({"error": "Rate limit exceeded"}), 429

            if not request.is_json:
                return jsonify({"error": "JSON required"}), 415

            body = request.get_json(silent=True) or {}
            text, error = self._validate_chat_text(body)
            if error:
                return jsonify(error[0]), error[1]

            # The reply is produced before the response starts so the routing
            # record (#189) can travel in a header; the words are still sent
            # one at a time as before.
            try:
                response, trace = self._run_chat(text)
                response = response or "..."
            except Exception:
                logger.exception("[WebUI] chat stream error")
                return jsonify({"error": "Failed"}), 500

            def generate():
                for word in response.split(" "):
                    yield word + " "

            headers = {}
            public = self._public_trace(trace)
            if public:
                headers["X-Hestia-Trace"] = json.dumps(
                    public, ensure_ascii=True, separators=(",", ":"))
            return Response(stream_with_context(generate()), mimetype="text/plain",
                            headers=headers)

    # ── VOICE (STT / TTS) ────────────────────────────────
    # Wraps core/stt.py + core/tts.py for the web UI's mic button and
    # per-reply speaker icon. Both engines are normally built for the
    # local voice loop (mic → speakers); nothing here touches that loop
    # — self.stt.transcribe_audio() just runs the model over audio we
    # decoded ourselves, and self.tts.synthesize_wav_bytes() bypasses the
    # speak()/queue path entirely, so a browser request can't interfere
    # with (or get cancelled by) local barge-in.

    def _register_voice_routes(self) -> None:
        @self.app.route("/api/stt", methods=["POST"])
        def api_stt():
            if not self.stt:
                return jsonify({"error": "Voice input not available"}), 503

            audio_file = request.files.get("audio")
            if not audio_file:
                return jsonify({"error": "No audio uploaded"}), 400

            raw = audio_file.read()
            if not raw:
                return jsonify({"error": "Empty audio"}), 400
            if len(raw) > 15 * 1024 * 1024:  # 15MB — a few minutes of speech, plenty
                return jsonify({"error": "Audio too large"}), 413

            try:
                import ffmpeg
                import numpy as np

                # Decode whatever the browser recorded (webm/opus, ogg, wav...)
                # into raw float32 PCM at 16kHz mono — the exact format
                # HestiaSTT.transcribe_audio() expects.
                proc = (
                    ffmpeg
                    .input("pipe:0")
                    .output("pipe:1", format="f32le", acodec="pcm_f32le", ac=1, ar=16000)
                    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                )
                out, err = proc.communicate(input=raw)
                if proc.returncode != 0:
                    logger.error(
                        "[WebUI] stt decode failed: %s",
                        (err or b"").decode("utf-8", "ignore")[-500:],
                    )
                    return jsonify({"error": "Could not decode audio"}), 400

                audio = np.frombuffer(out, dtype=np.float32)
                text = self.stt.transcribe_audio(audio)
                return jsonify({"text": text})
            except Exception:
                logger.exception("[WebUI] stt error")
                return jsonify({"error": "Transcription failed"}), 500

        @self.app.route("/api/tts", methods=["POST"])
        def api_tts():
            if not self.tts:
                return jsonify({"error": "Voice output not available"}), 503

            if not request.is_json:
                return jsonify({"error": "JSON required"}), 415

            body = request.get_json(silent=True) or {}
            text = (body.get("text") or "").strip()
            if not text:
                return jsonify({"error": "Empty text"}), 400
            text = text[:MAX_TEXT_LENGTH]

            try:
                wav_bytes = self.tts.synthesize_wav_bytes(text)
                if not wav_bytes:
                    return jsonify({"error": "Synthesis failed"}), 500
                return Response(wav_bytes, mimetype="audio/wav")
            except Exception:
                logger.exception("[WebUI] tts error")
                return jsonify({"error": "Synthesis failed"}), 500

        # Live voice-pipeline state for the "listening" indicator and the
        # do-not-disturb toggle (backlog #178, #173). The state comes from
        # the voice loop itself (core/voice_state.py), so the dot reflects
        # whether the *server's* microphone is actually open — which matters
        # when voice mode runs headless and there is no terminal to look at.
        @self.app.route("/api/voice/state")
        def api_voice_state():
            if self.voice_state is None:
                return jsonify({
                    "active": False, "state": "inactive", "detail": None,
                    "mic_open": False, "dnd": False,
                    "dnd_remaining_minutes": None, "held_notifications": 0,
                })
            return jsonify(self.voice_state.snapshot())

        @self.app.route("/api/voice/dnd", methods=["POST"])
        def api_voice_dnd():
            if self.voice_state is None:
                return jsonify({"error": "Voice state not available"}), 503
            body = request.get_json(silent=True) or {}
            on = body.get("on")
            if not isinstance(on, bool):
                return jsonify({"error": "'on' (true/false) required"}), 400
            minutes = body.get("minutes")
            if minutes is not None:
                if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) \
                        or not (0 < minutes <= 24 * 60):
                    return jsonify({"error": "'minutes' must be between 0 and 1440"}), 400
            self.voice_state.set_dnd(on, minutes)
            return jsonify(self.voice_state.snapshot())

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

    # ── PLUTO (finance) ─────────────────────────────────

    def _register_pluto_routes(self) -> None:
        app = self.app

        @app.route("/api/pluto/expenses")
        def api_pluto_expenses():
            if not self.pluto:
                return jsonify([])
            try:
                limit = max(1, min(int(request.args.get("limit", 100)), 500))
                return jsonify(self.pluto.pf_manager.db.get_expenses(limit))
            except Exception:
                logger.exception("[WebUI] pluto expenses error")
                return jsonify([])

        @app.route("/api/pluto/totals")
        def api_pluto_totals():
            if not self.pluto:
                return jsonify([])
            try:
                return jsonify({
                    "by_category": self.pluto.pf_manager.db.get_totals_by_category(),
                    "grand_total": self.pluto.pf_manager.db.get_grand_total(),
                })
            except Exception:
                logger.exception("[WebUI] pluto totals error")
                return jsonify({"by_category": [], "grand_total": 0})

        @app.route("/api/pluto/investments")
        def api_pluto_investments():
            if not self.pluto:
                return jsonify([])
            try:
                return jsonify(self.pluto.pf_manager.db.get_investments())
            except Exception:
                logger.exception("[WebUI] pluto investments error")
                return jsonify([])

        @app.route("/api/pluto/portfolio")
        def api_pluto_portfolio():
            if not self.pluto:
                return jsonify({"error": "Pluto disabled"}), 503
            try:
                return jsonify(self.pluto.portfolio_optimizer.optimize())
            except Exception:
                logger.exception("[WebUI] pluto portfolio error")
                return jsonify({"error": "Failed"}), 500

        # --- backlog #139-#145 -------------------------------------------
        # Each returns the engine's {response, data, confidence} dict, 503 when
        # Pluto is off, and a generic 500 (details go to the log, not the page).

        def _pluto_call(label, fn):
            if not self.pluto:
                return jsonify({"error": "Pluto disabled"}), 503
            try:
                return jsonify(fn())
            except Exception:
                logger.exception("[WebUI] pluto %s error", label)
                return jsonify({"error": "Failed"}), 500

        @app.route("/api/pluto/forecast")
        def api_pluto_forecast():
            try:
                days = max(1, min(int(request.args.get("days", 7)), 60))
            except ValueError:
                return jsonify({"error": "days must be a whole number"}), 400
            return _pluto_call("forecast", lambda: self.pluto.forecaster.forecast(horizon_days=days))

        @app.route("/api/pluto/backtest/sweep")
        def api_pluto_backtest_sweep():
            ticker = (request.args.get("ticker") or "").strip()[:30]
            if not ticker:
                return jsonify({"error": "ticker is required"}), 400
            return _pluto_call("sweep", lambda: self.pluto.backtester.sweep_sma_crossover(
                ticker,
                (request.args.get("fast") or "5 10 20")[:60],
                (request.args.get("slow") or "30 50 100")[:60],
                range_=(request.args.get("range") or "1y")[:4],
            ))

        @app.route("/api/pluto/data-sources")
        def api_pluto_data_sources():
            from modules.pluto.throttle import MONITOR
            from modules.pluto.market_data import retry_settings
            return jsonify({"sources": MONITOR.snapshot(), "retry": retry_settings()})

        @app.route("/api/pluto/rebalance")
        def api_pluto_rebalance():
            entities = {k: request.args[k][:30] for k in ("threshold", "contribution") if request.args.get(k)}
            return _pluto_call("rebalance", lambda: self.pluto.rebalancer.suggest(entities))

        @app.route("/api/pluto/quant-score")
        def api_pluto_quant_score():
            ticker = (request.args.get("ticker") or "").strip()[:20]
            if not ticker:
                return jsonify({"error": "ticker is required"}), 400
            return _pluto_call("quant-score", lambda: self.pluto.mi_manager.explain_quant_score(ticker))

        @app.route("/api/pluto/explain")
        def api_pluto_explain():
            name = (request.args.get("name") or "").strip()[:60]
            if not name:
                return jsonify({"error": "name is required"}), 400
            return _pluto_call("explain", lambda: self.pluto.explainer.explain({"name": name}))

        @app.route("/api/pluto/networth")
        def api_pluto_networth():
            return _pluto_call("networth", lambda: self.pluto.networth.net_worth({}))

        @app.route("/api/pluto/receipt", methods=["POST"])
        def api_pluto_receipt():
            """Upload a receipt photo (backlog #141). The browser's file name is never used on disk."""
            import os
            import tempfile
            from modules.pluto.receipts import IMAGE_EXTENSIONS, MAX_IMAGE_BYTES
            if not self.pluto:
                return jsonify({"error": "Pluto disabled"}), 503
            if (request.content_length or 0) > MAX_IMAGE_BYTES + 100_000:
                return jsonify({"error": "That image is larger than 10 MB."}), 413
            f = request.files.get("image")
            if f is None or not f.filename:
                return jsonify({"error": "Choose a photo first."}), 400
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                return jsonify({"error": "Use a jpg, png, webp, bmp or tiff photo."}), 400
            fd, tmp = tempfile.mkstemp(suffix=ext, prefix="receipt_")
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(f.read(MAX_IMAGE_BYTES + 1))
                entities = {"image_path": tmp}
                for k in ("amount", "merchant", "confirm"):
                    if request.form.get(k):
                        entities[k] = request.form[k][:60]
                return jsonify(self.pluto.receipts.ingest(entities))
            except Exception:
                logger.exception("[WebUI] pluto receipt error")
                return jsonify({"error": "Failed"}), 500
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # ── ARTEMIS (habits/goals) ───────────────────────────

    def _register_artemis_routes(self) -> None:
        app = self.app

        @app.route("/api/artemis/habits")
        def api_artemis_habits():
            if not self.artemis:
                return jsonify([])
            try:
                habits = self.artemis.tracker.get_habits()
                return jsonify([h.to_dict() | {"name": name} for name, h in habits.items()])
            except Exception:
                logger.exception("[WebUI] artemis habits error")
                return jsonify([])

        @app.route("/api/artemis/goals")
        def api_artemis_goals():
            if not self.artemis:
                return jsonify([])
            try:
                goals = self.artemis.tracker.get_goals()
                return jsonify([g.to_dict() | {"name": name} for name, g in goals.items()])
            except Exception:
                logger.exception("[WebUI] artemis goals error")
                return jsonify([])

        @app.route("/api/artemis/summary")
        def api_artemis_summary():
            if not self.artemis:
                return jsonify({})
            try:
                return jsonify(self.artemis.tracker.summary())
            except Exception:
                logger.exception("[WebUI] artemis summary error")
                return jsonify({})

        @app.route("/api/artemis/habits/<name>/complete", methods=["POST"])
        def api_artemis_complete_habit(name):
            if not self.artemis:
                return jsonify({"error": "Artemis disabled"}), 503
            try:
                return jsonify(self.artemis.tracker.complete_habit(name))
            except Exception:
                logger.exception("[WebUI] artemis complete_habit error")
                return jsonify({"error": "Failed"}), 500

        @app.route("/api/artemis/heatmap")
        def api_artemis_heatmap():
            """Per-habit completion grid for the Heatmap tab (#183).

            ``since`` is the first day completion dates were recorded for
            that habit: earlier days are *unknown*, not missed, so the UI
            greys them out rather than painting them as failures.
            """
            if not self.artemis:
                return jsonify({"days": [], "habits": []})
            try:
                from datetime import date, timedelta, timezone, datetime

                days = max(7, min(int(request.args.get("days", 84)), 180))
                today = datetime.now(timezone.utc).date()
                span = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
                first = span[0]
                habits = []
                for name, info in sorted(self.artemis.tracker.habit_history().items()):
                    habits.append({
                        "name": name,
                        "since": info["since"],
                        "streak": info["streak"],
                        "done": [d for d in info["dates"] if d >= first],
                    })
                return jsonify({"days": span, "habits": habits})
            except Exception:
                logger.exception("[WebUI] artemis heatmap error")
                return jsonify({"days": [], "habits": []})

    # ── APOLLO (health dashboards) ───────────────────────

    def _register_apollo_routes(self) -> None:
        app = self.app

        def _dashboard():
            days = max(7, min(int(request.args.get("days", 30)), 180))
            return self.apollo.dashboard_data(days)

        @app.route("/api/apollo/dashboard")
        def api_apollo_dashboard():
            if not self.apollo or not hasattr(self.apollo, "dashboard_data"):
                return jsonify({})
            try:
                return jsonify(_dashboard())
            except Exception:
                logger.exception("[WebUI] apollo dashboard error")
                return jsonify({})

        def _section(key):
            def view():
                if not self.apollo or not hasattr(self.apollo, "dashboard_data"):
                    return jsonify([])
                try:
                    return jsonify(_dashboard().get(key, []))
                except Exception:
                    logger.exception("[WebUI] apollo %s error", key)
                    return jsonify([])
            return view

        for key in ("sleep", "weight", "water", "mood", "streaks", "steps"):
            app.add_url_rule(
                f"/api/apollo/{key}", endpoint=f"api_apollo_{key}", view_func=_section(key)
            )

    # ── CHRONOS (time/calendar) ──────────────────────────

    def _register_chronos_routes(self) -> None:
        app = self.app

        @app.route("/api/chronos/now")
        def api_chronos_now():
            if not self.chronos:
                return jsonify({})
            try:
                return jsonify(self.chronos.get_context())
            except Exception:
                logger.exception("[WebUI] chronos now error")
                return jsonify({})

        @app.route("/api/chronos/weather")
        def api_chronos_weather():
            if not self.chronos:
                return jsonify({"error": "Chronos disabled"}), 503
            try:
                context = {}
                try:
                    context = self.memory.get_context() or {}
                except Exception:
                    pass
                return jsonify(self.chronos.handle("get_weather", {}, context))
            except Exception:
                logger.exception("[WebUI] chronos weather error")
                return jsonify({"error": "Failed"}), 500

    # ── ATHENA (document RAG) ────────────────────────────

    def _register_athena_routes(self) -> None:
        app = self.app

        @app.route("/api/athena/status")
        def api_athena_status():
            if not self.athena:
                return jsonify({})
            try:
                return jsonify(self.athena.stats())
            except Exception:
                logger.exception("[WebUI] athena status error")
                return jsonify({})

        @app.route("/api/athena/query", methods=["POST"])
        def api_athena_query():
            if not self.athena:
                return jsonify({"error": "Athena disabled"}), 503

            body = request.get_json(silent=True) or {}
            query = (body.get("query") or "").strip()
            if not query:
                return jsonify({"error": "Empty"}), 400
            if len(query) > MAX_TEXT_LENGTH:
                return jsonify({"error": "Too long"}), 413

            try:
                # "debug": true adds the semantic/BM25 score breakdown (#64).
                entities = {"query": query}
                if body.get("debug"):
                    entities["debug"] = True
                return jsonify(self.athena.handle("search", entities, {}))
            except Exception:
                logger.exception("[WebUI] athena query error")
                return jsonify({"error": "Failed"}), 500

        @app.route("/api/athena/feedback", methods=["POST"])
        def api_athena_feedback():
            """Mark a source of the last search relevant / not relevant (#63).
            Body: {"index": 2, "relevant": false} or {"file_name": "x.pdf", ...}."""
            if not self.athena:
                return jsonify({"error": "Athena disabled"}), 503
            body = request.get_json(silent=True) or {}
            allowed = ("index", "file_name", "relevant", "subject", "module",
                       "page", "page_number", "chunk_number")
            entities = {k: body[k] for k in allowed if k in body}
            if not entities:
                return jsonify({"error": "Empty"}), 400
            try:
                return jsonify(self.athena.handle("mark_feedback", entities, {}))
            except Exception:
                logger.exception("[WebUI] athena feedback error")
                return jsonify({"error": "Failed"}), 500

        @app.route("/api/athena/ingest", methods=["POST"])
        def api_athena_ingest():
            """Start indexing in the background (backlog #70); poll /ingest-status."""
            starter = getattr(self.athena, "start_ingest_background", None)
            if not callable(starter):
                return jsonify({"error": "Athena disabled"}), 503
            try:
                if starter():
                    return jsonify({"started": True}), 202
                return jsonify({"started": False, "error": "Already running"}), 409
            except Exception:
                logger.exception("[WebUI] athena ingest start error")
                return jsonify({"error": "Failed"}), 500

        @app.route("/api/athena/ingest-status")
        def api_athena_ingest_status():
            getter = getattr(self.athena, "ingest_status", None)
            if not callable(getter):
                return jsonify({"running": False, "total": 0, "done": 0, "percent": 0})
            try:
                return jsonify(getter())
            except Exception:
                logger.exception("[WebUI] athena ingest status error")
                return jsonify({"running": False, "total": 0, "done": 0, "percent": 0})

        @app.route("/api/athena/citation-graph")
        def api_athena_citation_graph():
            """Which indexed papers cite which, as nodes/links JSON (backlog #60)."""
            getter = getattr(self.athena, "citation_graph", None)
            if not callable(getter):
                return jsonify({"error": "Athena disabled"}), 503
            try:
                return jsonify(getter(request.args.get("subject") or None))
            except Exception:
                logger.exception("[WebUI] athena citation graph error")
                return jsonify({"error": "Failed"}), 500

        @app.route("/api/athena/citation-graph/view")
        def api_athena_citation_graph_view():
            """The same graph as a self-contained interactive page (shown in an iframe)."""
            renderer = getattr(self.athena, "citation_graph_html", None)
            if not callable(renderer):
                return "Athena disabled", 503
            try:
                html = renderer(request.args.get("subject") or None, request.args.get("focus") or None)
                return html, 200, {"Content-Type": "text/html; charset=utf-8",
                                   "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"}
            except Exception:
                logger.exception("[WebUI] athena citation graph view error")
                return "Could not build the citation graph.", 500

    # ── MNEMOSYNE (knowledge graph, study, quizzes) ──────

    def _register_mnemosyne_routes(self) -> None:
        app = self.app

        @app.route("/api/mnemosyne/graph")
        def api_mnemosyne_graph():
            """Nodes and links for the force-directed graph view (backlog #32)."""
            getter = getattr(self.memory, "get_graph_data", None)
            if not callable(getter):
                return jsonify({"nodes": [], "links": []})
            try:
                max_nodes = max(1, min(int(request.args.get("max_nodes", 150)), 500))
                return jsonify(getter(max_nodes))
            except Exception:
                logger.exception("[WebUI] mnemosyne graph error")
                return jsonify({"nodes": [], "links": []})

        @app.route("/api/mnemosyne/dashboard")
        def api_mnemosyne_dashboard():
            """Facts / summaries / DB size / embeddings (#49)."""
            getter = getattr(self.memory, "get_memory_dashboard", None)
            if not callable(getter):
                return jsonify({})
            try:
                return jsonify(getter())
            except Exception:
                logger.exception("[WebUI] mnemosyne dashboard error")
                return jsonify({})

        @app.route("/api/mnemosyne/export")
        def api_mnemosyne_export():
            """Full memory backup as a download (#44). ?format=json (default) or markdown."""
            exporter = getattr(self.memory, "export_memory", None)
            if not callable(exporter):
                return jsonify({"error": "Export unavailable"}), 503
            fmt = "markdown" if request.args.get("format", "json").lower() in ("md", "markdown") else "json"
            try:
                body = exporter(fmt)
            except Exception:
                logger.exception("[WebUI] mnemosyne export error")
                return jsonify({"error": "Failed"}), 500
            ext, mime = ("md", "text/markdown") if fmt == "markdown" else ("json", "application/json")
            return Response(body, mimetype=mime, headers={
                "Content-Disposition": f"attachment; filename=hestia_memory.{ext}"})

        @app.route("/api/mnemosyne/learning")
        def api_mnemosyne_learning():
            """Study-card counts plus the per-subject quiz strength/weakness map."""
            out: dict = {"study": {"total": 0, "due": 0, "new": 0}, "quiz": {}}
            try:
                store = getattr(self.memory, "study_store", None)
                if store is not None:
                    out["study"] = store.stats()
                    out["study"]["upcoming"] = [
                        {"key": c["fact_key"], "due": c["due_date"]}
                        for c in store.list_cards(limit=10)
                    ]
                quiz = getattr(self.memory, "quiz_engine", None)
                if quiz is not None:
                    out["quiz"] = quiz.get_strength_weakness_map()
            except Exception:
                logger.exception("[WebUI] mnemosyne learning error")
            return jsonify(out)

    # ── LOGIN (#186) ────────────────────────────────────

    def _register_login_routes(self) -> None:
        app = self.app

        @app.route("/api/auth/status")
        def api_auth_status():
            return jsonify({"mode": self._auth_mode(),
                            "authenticated": self._is_authenticated()
                            if self._auth_mode() != "none" else True})

        @app.route("/login", methods=["GET", "POST"])
        def login():
            if not self.auth.enabled:
                return redirect("/")
            nxt = safe_next(request.values.get("next"))
            if request.method == "GET":
                if session.get("auth") is True:
                    return redirect(nxt)
                return render_template("login.html", error=None, next=nxt)

            ip = request.remote_addr or "unknown"
            wait = self.auth.locked_for(ip)
            if wait:
                resp = render_template(
                    "login.html", next=nxt,
                    error=f"Too many attempts. Try again in {wait} seconds.")
                return Response(resp, status=429, headers={"Retry-After": str(wait)})

            supplied = request.form.get("password")
            if supplied is None and request.is_json:
                supplied = (request.get_json(silent=True) or {}).get("password")
            if not self.auth.verify(supplied or ""):
                locked = self.auth.record_failure(ip)
                logger.warning("[WebUI] failed login from %s", ip)
                msg = ("Too many attempts. Try again in a minute." if locked
                       else "That password is not right.")
                return Response(render_template("login.html", error=msg, next=nxt),
                                status=401)
            self.auth.record_success(ip)
            session.clear()           # fresh session: no fixation
            session["auth"] = True
            session.permanent = True
            return redirect(nxt, code=303)

        @app.route("/logout", methods=["POST"])
        def logout():
            session.clear()
            return redirect("/login", code=303)

    # ── TODAY + ACTIVITY (#180, #182) ───────────────────

    _DASH_TTL = 20.0

    @staticmethod
    def _iso_utc(ts) -> Optional[str]:
        """SQLite stores 'YYYY-MM-DD HH:MM:SS' in UTC; give browsers an
        unambiguous ISO string (a bare one is read as *local* time)."""
        if not ts:
            return None
        ts = str(ts)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", ts):
            return ts.replace(" ", "T") + "Z"
        return ts

    @staticmethod
    def _module_of(intent: str) -> str:
        from core.web_live import module_for
        return module_for(intent)

    def _register_dashboard_routes(self) -> None:
        app = self.app

        @app.route("/api/dashboard/today")
        def api_dashboard_today():
            from core.web_dashboard import build_today
            now = time.monotonic()
            stamp, cached = self._dash_cache
            if cached is not None and now - stamp < self._DASH_TTL \
                    and not request.args.get("refresh"):
                return jsonify(cached)
            try:
                data = build_today(self)
            except Exception:
                logger.exception("[WebUI] dashboard error")
                return jsonify({"error": "Failed"}), 500
            self._dash_cache = (now, data)
            return jsonify(data)

        @app.route("/api/activity")
        def api_activity():
            try:
                limit = max(1, min(int(request.args.get("limit", 50)), 200))
            except ValueError:
                return jsonify({"error": "invalid limit"}), 400
            module = (request.args.get("module") or "").strip().lower()
            try:
                rows = self.memory.db.get_recent_interactions(limit * 4 if module else limit)
            except Exception:
                logger.exception("[WebUI] activity error")
                return jsonify({"error": "Failed"}), 500

            items = []
            for r in reversed(rows):
                intent = r.get("intent") or "chat"
                items.append({
                    "kind": "message", "ts": self._iso_utc(r.get("pushed_at")),
                    "query": r.get("query"), "response": (r.get("response") or "")[:800],
                    "intent": intent, "module": self._module_of(intent),
                })
            for ev in self.live_hub.recent(100, ("reminder", "notification")):
                default_module = "chronos" if ev["type"] == "reminder" else "core"
                items.append({"kind": ev["type"], "ts": ev["ts"], "text": ev.get("text"),
                              "module": ev.get("module") or default_module})
            if module:
                items = [i for i in items if i["module"] == module]
            items.sort(key=lambda i: i.get("ts") or "", reverse=True)
            return jsonify({
                "items": items[:limit],
                "modules": sorted({i["module"] for i in items}),
            })

    # ── LIVE UPDATES (#187) ─────────────────────────────

    def _register_live_routes(self) -> None:
        @self.app.route("/api/live/stream")
        def api_live_stream():
            sub = self.live_hub.subscribe(
                request.headers.get("Last-Event-ID") or request.args.get("last_id"))
            if sub is None:
                return jsonify({"error": "Too many live connections"}), 429
            return Response(
                self.live_hub.stream(sub), mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform",
                         "X-Accel-Buffering": "no"})

    # ── CROSS-MODULE SEARCH (#184) ──────────────────────

    def _register_search_routes(self) -> None:
        @self.app.route("/api/search")
        def api_search():
            from core.web_search import search_all, MAX_QUERY_LENGTH
            q = (request.args.get("q") or "").strip()
            if len(q) < 2:
                return jsonify({"error": "Type at least 2 characters"}), 400
            if len(q) > MAX_QUERY_LENGTH:
                return jsonify({"error": "Query too long"}), 400
            if not self._check_rate_limit("search:" + (request.remote_addr or "unknown")):
                return jsonify({"error": "Rate limit exceeded"}), 429
            try:
                limit = int(request.args.get("limit", 8))
            except ValueError:
                limit = 8
            return jsonify(search_all(self, q, limit))

    # ── SETTINGS (#188) ─────────────────────────────────

    def _register_settings_routes(self) -> None:
        from core import web_settings

        @self.app.route("/api/settings", methods=["GET", "POST"])
        def api_settings():
            if self.config_path is None or not self.config_path.exists():
                return jsonify({"error": "Settings file not available"}), 503
            if request.method == "GET":
                try:
                    return jsonify({"settings": web_settings.read_settings(self.config_path),
                                    "file": self.config_path.name,
                                    "restart_required": True})
                except Exception:
                    logger.exception("[WebUI] settings read error")
                    return jsonify({"error": "Failed"}), 500
            if not request.is_json:
                return jsonify({"error": "JSON required"}), 415
            changes = (request.get_json(silent=True) or {}).get("changes")
            try:
                with self._settings_lock:
                    saved = web_settings.save_settings(self.config_path, changes)
            except web_settings.SettingsError as exc:
                return jsonify({"error": str(exc)}), 400
            except Exception:
                logger.exception("[WebUI] settings save error")
                return jsonify({"error": "Failed"}), 500
            logger.info("[WebUI] settings changed: %s", ", ".join(s["key"] for s in saved))
            return jsonify({"saved": saved, "restart_required": True,
                            "backup": self.config_path.name + ".bak"})

    # ── DATA EXPORT (#190) ──────────────────────────────

    def _register_export_routes(self) -> None:
        from core import web_export

        @self.app.route("/api/export/datasets")
        def api_export_datasets():
            return jsonify(web_export.available(self))

        @self.app.route("/api/export/<module>/<dataset>")
        def api_export_dataset(module: str, dataset: str):
            fmt = (request.args.get("format") or "csv").lower()
            if fmt not in ("csv", "json"):
                return jsonify({"error": "format must be csv or json"}), 400
            try:
                rows = web_export.get_rows(self, module, dataset)
            except Exception:
                logger.exception("[WebUI] export error (%s/%s)", module, dataset)
                return jsonify({"error": "Failed"}), 500
            if rows is None:
                return jsonify({"error": "Unknown or unavailable dataset"}), 404
            body = (web_export.to_csv(rows) if fmt == "csv"
                    else web_export.to_json(module, dataset, rows))
            stamp = datetime.datetime.now().strftime("%Y%m%d")
            return Response(
                body,
                mimetype="text/csv; charset=utf-8" if fmt == "csv" else "application/json",
                headers={
                    "Content-Disposition":
                        f"attachment; filename=hestia_{module}_{dataset}_{stamp}.{fmt}",
                    "Cache-Control": "no-store",
                })

    # ── STATS ───────────────────────────────────────────

    def _get_stats(self) -> dict:
        try:
            data = self.memory.get_stats()
        except Exception:
            data = {}

        data["uptime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return data