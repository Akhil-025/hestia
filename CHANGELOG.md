# Changelog

All notable changes to Hestia. Newest first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Backlog numbers in brackets refer to `hestia_improvement_backlog.md`.

Hestia is a single-developer personal project, so "releases" are just
dated batches of work rather than shipped versions. The point of keeping
this file is to know what actually landed when — with 280 backlog items,
"did I do that one?" stops being answerable from memory.

---

## [Unreleased]

### Added

- **Observability layer** (`core/observability.py`)
  - Per-query request IDs on every log record, via a `contextvars`
    ContextVar and a `logging.Filter`. One query's full trace across
    `core/`, `modules/` and `api.py` is now `grep 'req=<id>'`. [#17]
  - Routing log: every classification and routing decision appended to
    `logs/routing.jsonl` (rotating, 5 MB × 3) as one JSON object per line —
    query, intent, confidence, module, Hecate's reason, latency, and
    whether the intent came from the LLM, the cache or an alias. [#5]
  - In-memory ring buffer of the last 50 decisions, backing a
    conversational "why did you route this there?". [#3]
  - Feedback log: `logs/feedback.jsonl`, an explicit user correction
    attached to the previous turn's decision. [#259]
  - `Diagnostics.module_status()`, which probes whichever of
    `ready()`/`available()` each module exposes and normalises the result
    to `ready`/`degraded`/`unknown`/`error`. [#8]
- **Diagnostic intents** on `CoreModule`: `modules_status`,
  `explain_routing`, `report_mistake`. Registered in
  `intent_registry.py`, with few-shot examples and disambiguation notes in
  `config/nlu_prompt.txt`. [#3, #8, #259]
- **Startup config validation** (`core/config_validation.py`). Fails fast
  naming every offending key path at once, instead of failing deep inside
  a module later with a message that never mentioned config. Catches the
  quoted-`"11434"` port and the truthy-string-`"false"` flag. [#14]
- **CLI flags** in `main.py` [#1, #14, #275]
  - `--dry-run "query"`: real classification, real routing decision, no
    handler executed and nothing written. Flags a registry/Hecate
    disagreement and a `can_handle()` rejection explicitly.
  - `--check-config`: validate and exit 0/1 without booting anything.
  - `--verbose` / `--quiet`.
- **Intent aliases** (`core/intent_aliases.py`,
  `config/intent_aliases.yaml`): config-driven phrase→intent mappings
  resolved *before* the LLM call, with exact/prefix/contains match modes.
  Every target is validated against `ALL_INTENTS` at load time, so the
  alias file can't become a second, diverging definition of what intents
  exist. [#22]
- **NLU classification cache** (`core/nlu_cache.py`): short-TTL (90 s),
  context-fingerprinted, LRU-bounded. Never caches the parse-failure or
  backend-unreachable shapes, and hands out copies since callers mutate
  results. [#28]
- **Registry versioning** [#11]
  - `REGISTRY_VERSION` (hand-bumped, semantic) and
    `registry_fingerprint()` (derived, order-independent, changes on any
    real edit), both surfaced on `/health`.
- **`GET /health/modules`**: one JSON blob aggregating every module's
  state plus registry info and NLU cache counters, for the web UI to poll.
  Never 500s — a missing `Diagnostics` reports `status: "unknown"`. [#19]
- **Graceful shutdown** on SIGTERM/SIGINT, so a `kill` runs `_shutdown()`
  (mic released, heartbeat stopped, event bus drained) instead of skipping
  it entirely. [#18]
- **`CONTRIBUTING.md`**, documenting the add-one-line-to-the-registry
  pattern and the architecture invariants. [#245]
- **This file.** [#249]
- `HestiaOrchestrator.last_decision`, so the routing decision is
  observable rather than only appearing in a debug log line.
- Tests: `test_observability.py`, `test_config_validation.py`,
  `test_intent_aliases.py`, `test_nlu_cache.py`,
  `test_registry_contract.py`, `test_core_diagnostics.py`,
  `test_api_health.py`, `test_main_cli.py` — 197 cases.

### Fixed

- **The sync API served none of its routes.** Every route in `api.py` was
  declared with `@app.get(...)` against the module-level
  `app = create_app()`, while `main.start_sync_api()` serves a *different*
  instance from its own `create_app(api_key=...)` call. A decorator only
  attaches a route to the object it names, so the app actually being
  served had no `/health`, no `/sync/pull` and no `/sync/push` — just the
  FastAPI shell and `/docs`. Callers got 404s, nothing was logged, and the
  code read as though the routes existed. Routes now live on an
  `APIRouter` included by `create_app()`; the request-logging middleware
  and the catch-all exception handler had the same problem and are now
  applied by `_install_middleware()`.
- **Test-isolation bug between `test_main.py` and `test_pluto.py`.**
  `test_main.py` stubbed `modules.pluto` as a plain module with no
  `__path__`, so after it ran, every `modules.pluto.<submodule>` import in
  the same session failed with "'modules.pluto' is not a package" —
  `test_pluto.py` errored at collection or not depending purely on
  pytest's file ordering. `conftest.py` now installs a conditional
  *package* stub (only when the real import fails) with a real `__path__`.
- **`sounddevice` stubbed in `conftest.py`.** It raises
  `OSError("PortAudio library not found")` at import time on any machine
  without PortAudio, which made `test_main.py`, `test_stt.py`,
  `test_tts.py`, `test_barge_in.py` and `test_wake_word.py` fail at
  *collection* — reporting as errors rather than running, on exactly the
  machines most likely to run the suite unattended. Collectible tests went
  from 931 to 1302 as a result.
- **Stale rotating log handler.** `_JsonlLog` reused whatever handler was
  already attached to its logger, which after a config change or module
  reload meant silently writing to the previously-configured path. It now
  closes and rebuilds.

### Changed

- `config/nlu_prompt.txt`: three new intents in the `Valid intents:`
  block, with few-shot examples and disambiguation notes (`modules_status`
  vs. small-talk "how are you", `explain_routing` vs. a general question
  about routing).
- `intent_registry.py` bumped to 2.1.0 (additive: three new core intents).
- The prompt/registry drift invariant is now a **CI failure**
  (`tests/test_registry_contract.py`) rather than a runtime warning. It
  was previously only logged, which is how the original drift survived
  long enough to leave eight working intents unreachable. [#26]

### Notes

- Pre-existing failures not touched by this batch: 12 cases in
  `test_iris_embeddings.py` (ChromaDB / sentence-transformers stubs) and
  `test_mnemosyne.py::test_learn_and_recall_end_to_end`.
- `tests/test_pluto.py` needs `pypfopt`, `qdrant_client`, `xgboost` and
  `langgraph` installed to collect.
