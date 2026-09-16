# modules/hecate/intent_registry.py
"""
Single source of truth mapping every canonical, NLU-facing intent name to
the module that owns it.

Why this file exists
---------------------
Before this file existed, "which intents exist and which module handles
them" was encoded independently in at least three places:

  1. core/nlu.py             — _CANONICAL_VALID_INTENTS (fallback whitelist)
  2. config/nlu_prompt.txt   — the "Valid intents:" block fed to the LLM
  3. modules/hecate/engine.py — ~15 separate `if intent in {...}` /
                                 `.startswith("x_")` tiers, one per module
  4. modules/hestia/orchestrator.py — _MODULE_PREFIXES (yet another copy of
                                 the prefix list, used for stripping)

These four copies had already drifted from each other and from what the
modules themselves actually declare in their own `_INTENTS` sets:
  - modules/pluto/engine.py exposes working, tested intents
    ("optimize_portfolio", "backtest_strategy", "forecast_spending",
    "financial_advisor_chat") that were never added to the NLU whitelist —
    meaning a user could never actually reach them by voice or chat.
  - modules/iris/iris_engine.py exposes "iris_analyse" / "iris_query" the
    same way — real, working, but unreachable via classified intent.
  - modules/athena/engine.py exposes "athena_ingest" / "athena_status" the
    same way.

All four are now registered below (see INTENT_MODULE_MAP), closing that
gap. Going forward, adding a new intent to a module means adding ONE line
here — core/nlu.py's whitelist, Hecate's routing table, and the
orchestrator's prefix-stripping all read from this file instead of keeping
their own copy.

config/nlu_prompt.txt is a plain-text prompt fed to the LLM and can't
import Python, so it still needs its own "Valid intents:" listing — but it
should be kept in sync with ALL_INTENTS below by hand (or regenerated from
it) whenever this file changes. core/nlu.py's prompt-parsing is defensive:
if that block and this registry ever disagree, ALL_INTENTS here wins.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Module prefixes
# ---------------------------------------------------------------------------
# Used by HestiaOrchestrator to strip a module's prefix off an intent before
# calling that module's can_handle()/handle() (which operate on the
# stripped form, e.g. "search" not "athena_search"), and by Hecate as a
# best-effort fallback for intents that carry a real module's prefix but
# aren't (yet) registered in INTENT_MODULE_MAP below.
#
# Note: hermes_/chronos_/artemis_ intents are NOT actually prefixed in the
# canonical intent set (e.g. the real intent is "read_email", not
# "hermes_read_email") — these three are kept in the tuple purely so a
# hallucinated/prefixed variant (which LLMs do occasionally emit, e.g.
# "hermes_read_email") still strips down to something can_handle() may
# recognise, and so Hecate's prefix-fallback tier can still route it.
MODULE_PREFIXES: tuple[str, ...] = (
    "apollo_",
    "ares_",
    "orpheus_",
    "dionysus_",
    "pluto_",
    "hermes_",
    "chronos_",
    "athena_",
    "iris_",
    "artemis_",
    "hephaestus_",
    "mnemosyne_",
    "metis_",
)

# prefix -> module name, derived from MODULE_PREFIXES. Used by Hecate's
# generic prefix-fallback tier (one loop instead of one hand-written
# `if intent.startswith(...)` block per module).
PREFIX_TO_MODULE: dict[str, str] = {p: p.rstrip("_") for p in MODULE_PREFIXES}


# ---------------------------------------------------------------------------
# Canonical intent -> module map
# ---------------------------------------------------------------------------
# This is the routing table. Every entry here must correspond to a name the
# target module's own `_INTENTS` (or equivalent can_handle() set) actually
# accepts once HestiaOrchestrator strips the module's prefix off it — see
# each module's engine.py for the authoritative per-module set this was
# cross-checked against.
INTENT_MODULE_MAP: dict[str, str] = {
    # --- Core (unprefixed, general assistant capabilities) ---
    "chat": "core",
    "take_note": "core",
    "get_notes": "core",
    "delete_notes": "core",
    "get_history": "core",
    "save_name": "core",
    "get_user_info": "core",
    "set_preference": "core",
    "get_system_info": "core",
    # Diagnostics (backlog #3, #8, #259). Owned by core because they
    # report on the *assistant itself*, not on any one module's domain —
    # and because CoreModule is the only module guaranteed to be
    # registered, so "are your modules up?" is answerable even when the
    # module being asked about failed to load.
    "modules_status": "core",
    "explain_routing": "core",
    "report_mistake": "core",

    # --- Mnemosyne (long-term memory) ---
    # "recall"/"remember"/"get_facts" are internal intent names Hecate
    # assigns via its text-trigger tier (see _MNEMOSYNE_TRIGGERS in
    # engine.py) rather than names the NLU ever classifies directly, so
    # they're intentionally not listed here. learn_fact/forget_fact ARE
    # real NLU-classified intents.
    "learn_fact": "mnemosyne",
    "forget_fact": "mnemosyne",

    # --- Chronos (time / date / weather / reminders) ---
    "get_time": "chronos",
    "get_date": "chronos",
    "get_weather": "chronos",
    "set_reminder": "chronos",
    "get_holiday": "chronos",

    # --- Hermes (Gmail + Google Calendar) ---
    "read_email": "hermes",
    "send_email": "hermes",
    "list_events": "hermes",
    "create_event": "hermes",
    "delete_events": "hermes",

    # --- Athena (document / RAG search) ---
    "athena_search": "athena",
    # Previously unreachable via classified intent — see module docstring.
    "athena_ingest": "athena",
    "athena_status": "athena",

    # --- Iris (media search & ingestion) ---
    "iris_search": "iris",
    "iris_ingest": "iris",
    "iris_status": "iris",
    # Previously unreachable via classified intent — see module docstring.
    "iris_analyse": "iris",
    "iris_query": "iris",

    # --- Artemis (habits & goals) ---
    "add_goal": "artemis",
    "get_goals": "artemis",
    "update_goal": "artemis",
    "remove_goal": "artemis",
    "abandon_goal": "artemis",
    "add_habit": "artemis",
    "complete_habit": "artemis",
    "list_habits": "artemis",
    "remove_habit": "artemis",
    "productivity_summary": "artemis",
    "get_at_risk_goals": "artemis",
    "get_motivation": "artemis",
    "suggest_activity": "artemis",

    # --- Ares (strategic/analytical) ---
    "ares_analyse_risk": "ares",
    "ares_swot_analysis": "ares",
    "ares_strategic_plan": "ares",
    "ares_decision_support": "ares",
    "ares_premortem_analysis": "ares",
    "ares_competitive_analysis": "ares",
    "ares_contingency_plan": "ares",
    "ares_war_room_briefing": "ares",

    # --- Apollo (health tracking) ---
    "apollo_log_workout": "apollo",
    "apollo_track_sleep": "apollo",
    "apollo_log_mood": "apollo",
    "apollo_log_health": "apollo",
    "apollo_get_health_summary": "apollo",
    "apollo_log_weight": "apollo",
    "apollo_log_water": "apollo",
    "apollo_set_health_goal": "apollo",
    "apollo_get_goal_progress": "apollo",
    "apollo_lookup_food": "apollo",
    "apollo_suggest_exercise": "apollo",

    # --- Orpheus (creative writing) ---
    "orpheus_write_poem": "orpheus",
    "orpheus_brainstorm": "orpheus",
    "orpheus_creative_prompt": "orpheus",
    "orpheus_generate_lyrics": "orpheus",
    "orpheus_write_story": "orpheus",
    "orpheus_continue_writing": "orpheus",
    "orpheus_critique_writing": "orpheus",
    "orpheus_rewrite_style": "orpheus",
    "orpheus_generate_names": "orpheus",
    "orpheus_get_creations": "orpheus",

    # --- Metis (writing assistance & editing) ---
    "metis_correct_text": "metis",
    "metis_improve_clarity": "metis",
    "metis_suggest_style": "metis",
    "metis_detect_tone": "metis",
    "metis_rewrite_text": "metis",
    "metis_draft_content": "metis",
    "metis_summarize_text": "metis",
    "metis_expand_text": "metis",
    "metis_shorten_text": "metis",
    "metis_generate_outline": "metis",
    "metis_check_plagiarism": "metis",
    "metis_generate_citation": "metis",
    "metis_check_consistency": "metis",
    "metis_readability_report": "metis",
    "metis_writing_stats": "metis",

    # --- Dionysus (entertainment/leisure) ---
    "dionysus_recommend_movie": "dionysus",
    "dionysus_find_restaurant": "dionysus",
    "dionysus_recommend_music": "dionysus",
    "dionysus_plan_outing": "dionysus",
    "dionysus_dismiss_recommendation": "dionysus",
    "dionysus_mark_seen": "dionysus",
    "dionysus_recommend_recipe": "dionysus",

    # --- Pluto (personal finance) ---
    "pluto_log_expense": "pluto",
    "pluto_get_budget_summary": "pluto",
    "pluto_track_investment": "pluto",
    "pluto_spending_report": "pluto",
    "pluto_convert_currency": "pluto",
    "pluto_company_lookup": "pluto",
    "pluto_analyze_asset": "pluto",
    # Previously unreachable via classified intent (real, tested features —
    # see modules/pluto/engine.py's _QUANT_INTENTS and tests/test_pluto.py —
    # but absent from the old NLU whitelist).
    "pluto_optimize_portfolio": "pluto",
    "pluto_backtest_strategy": "pluto",
    "pluto_forecast_spending": "pluto",
    "pluto_financial_advisor_chat": "pluto",

    # --- Hephaestus (browser automation) ---
    "hephaestus_browser_action": "hephaestus",
    "hephaestus_search_web": "hephaestus",
    "hephaestus_check_flight": "hephaestus",
    "hephaestus_scrape_page": "hephaestus",
    "hephaestus_open_app": "hephaestus",
}

# Every canonical intent name. This is what core/nlu.py uses both as the
# JSON-schema enum constraint sent to Ollama (see HestiaNLU._build_schema())
# and as the fallback whitelist if config/nlu_prompt.txt can't be parsed.
ALL_INTENTS: frozenset[str] = frozenset(INTENT_MODULE_MAP)


# ---------------------------------------------------------------------------
# Versioning (backlog #11)
# ---------------------------------------------------------------------------
# Clients that integrate against the intent set — the Telegram bot, the web
# UI's command palette, any future mobile/PWA client, the sync API — cached
# their own copy of "what intents exist" with no way to tell that the
# server's copy had changed underneath them. The symptom is a client
# offering a button for an intent the server no longer routes, or missing
# one it now does, with no error anywhere.
#
# Two values, deliberately separate:
#
#   REGISTRY_VERSION   Hand-bumped, semantic. MAJOR when an intent is
#                      REMOVED or REASSIGNED to a different module (a
#                      breaking change for any client that hardcoded it);
#                      MINOR when intents are added (backwards compatible).
#                      Clients compare major versions to decide whether to
#                      refuse to start or merely refresh.
#
#   registry_fingerprint()  Derived, exact. Changes on ANY edit to the map,
#                      including ones a human forgot to bump the version
#                      for. Clients cache it and re-fetch the intent list
#                      when it differs — no judgement call required.
#
# Bump REGISTRY_VERSION in the same commit that edits INTENT_MODULE_MAP.
# tests/test_registry_contract.py asserts the version is well-formed and
# that the fingerprint is stable across imports.
REGISTRY_VERSION: str = "2.1.0"


def registry_fingerprint() -> str:
    """
    Short, stable hash of the full intent->module mapping.

    Order-independent (the map is sorted before hashing) so reordering
    entries for readability doesn't look like a breaking change, while any
    real addition, removal or reassignment does.
    """
    import hashlib

    payload = ";".join(f"{k}={v}" for k, v in sorted(INTENT_MODULE_MAP.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def registry_info() -> dict:
    """
    Version metadata for clients and the /health endpoints.

    Kept as a plain dict (no dataclass) so it can be returned straight
    from FastAPI and the sync API without a serialiser.
    """
    return {
        "version": REGISTRY_VERSION,
        "fingerprint": registry_fingerprint(),
        "intent_count": len(INTENT_MODULE_MAP),
        "module_count": len(set(INTENT_MODULE_MAP.values())),
    }


def module_for_intent(intent: str) -> str | None:
    """Return the module name that owns *intent*, or None if unregistered."""
    return INTENT_MODULE_MAP.get(intent)


def strip_module_prefix(intent: str) -> str:
    """Remove a known module prefix from *intent*, if present."""
    for prefix in MODULE_PREFIXES:
        if intent.startswith(prefix):
            return intent[len(prefix):]
    return intent
