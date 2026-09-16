# Contributing to Hestia

Notes-to-future-self for working on this codebase. The point of this file
is to stop the drift bugs that have already happened here from happening
again — each rule below exists because something silently broke when it
wasn't followed.

---

## The one rule: `intent_registry.py` is the source of truth

**Adding a new intent is one line in `modules/hecate/intent_registry.py`,
plus one line in the module that owns it, plus one entry in the NLU
prompt.** Nothing else.

```python
# modules/hecate/intent_registry.py
INTENT_MODULE_MAP: dict[str, str] = {
    ...
    "apollo_log_stretching": "apollo",   # <- the one line
}
```

### Why this matters

"Which intents exist and which module handles them" used to be encoded
independently in four places: `core/nlu.py`'s whitelist,
`config/nlu_prompt.txt`'s valid-intent list, a stack of ~15 hand-written
`if intent in {...}` tiers in `modules/hecate/engine.py`, and the
orchestrator's own copy of the module-prefix list. They drifted. The
result was that **four working, tested modules had intents no user could
ever reach** — Pluto's `optimize_portfolio`, `backtest_strategy`,
`forecast_spending` and `financial_advisor_chat`, Iris's `iris_analyse`
and `iris_query`, and Athena's `athena_ingest` and `athena_status` — real
features, with tests, unreachable by voice or chat because the NLU was
never told they existed.

Nothing errored. There was no failing test, no log line. The features
simply didn't exist from the user's side.

### The full checklist for a new intent

1. **Register it.** One entry in `INTENT_MODULE_MAP`.
2. **Declare it in the owning module.** Add the *unprefixed* name to that
   module's `_INTENTS` set — `"log_stretching"`, not
   `"apollo_log_stretching"`. The orchestrator strips the module prefix
   before calling `can_handle()`/`handle()`, so a prefixed entry in
   `_INTENTS` can never match. `modules/base.py` enforces the naming
   convention at class-definition time.
3. **Handle it.** A branch in that module's `handle()` returning the
   standard `{"response": str, "data": dict, "confidence": float}` dict.
4. **Add it to `config/nlu_prompt.txt`.** Both the `Valid intents:` block
   *and* at least one few-shot example. The prompt is plain text fed to
   the LLM and can't import Python, so this is the one copy that has to be
   maintained by hand.
5. **Bump `REGISTRY_VERSION`.** MINOR for an addition, MAJOR if you
   removed or reassigned an intent (that's breaking for any client that
   cached the intent list).
6. **Run the tests.** `tests/test_registry_contract.py` fails if the
   prompt and the registry disagree in either direction, and if a module
   doesn't declare an intent the registry assigned to it.

### What not to do

- Don't add an `if intent == ...` branch to `modules/hecate/engine.py`.
  Tier 1 routes every registered intent in O(1) already. The text-trigger
  tiers exist only for phrasings the NLU reliably *mis*classifies — that's
  a model-accuracy workaround, not a routing table.
- Don't add a phrasing to `config/nlu_prompt.txt` when what you actually
  want is a deterministic mapping. Use `config/intent_aliases.yaml`
  instead: it resolves before the LLM call, costs nothing, and doesn't
  grow the prompt. The prompt file is already 40 KB and is sent on every
  single query.
- Don't put an intent name in `config/intent_aliases.yaml` that isn't in
  the registry. It'll be dropped at load time with a warning, and
  `tests/test_intent_aliases.py` will fail.

---

## Architecture invariants

These are deliberate and load-bearing. Changing any of them is a
conscious decision, not a refactor.

- **Single process.** No Docker, no microservices. Modules are Python
  objects in one process, wired in `main.HestiaBuilder`.
- **SQLite + ChromaDB.** Structured data in SQLite, vectors in Chroma. No
  third store.
- **No module calls another module.** Cross-module work goes through
  Hecate's routing decision (`secondary` + `synthesize`) or the event bus
  in `core/event_bus.py`. A module importing another module's engine is a
  bug.
- **Hecate is the only component that routes.** `HecateEngine.decide()`
  is called once per query, by the orchestrator. Nothing else decides
  where a query goes.
- **Modules never raise to the caller.** Every `handle()` returns a
  response dict; failures become `_err`/`_miss`-style responses, not
  stack traces. `main.process_text` is the last line of defence and never
  raises either.
- **Construction and wiring are separate.** `HestiaBuilder` builds
  subsystems; `Hestia.__init__` decides dependency order and connects
  them. New subsystems get a `build_*` factory, so they stay
  independently testable.

---

## Observability

Added for the backlog's #3/#5/#8/#17/#259 cluster; all of it lives in
`core/observability.py`.

- **Request IDs.** Every query gets one, stamped onto every log record via
  a `contextvars` ContextVar and `RequestIdFilter`. To trace one query end
  to end: `grep 'req=a1b2c3d4' <logfile>`.
- **Routing log.** `logs/routing.jsonl`, one JSON object per
  classification: query, intent, confidence, module, Hecate's reason,
  latency, and whether the intent came from the LLM, the cache, or an
  alias. JSONL because it exists to be analysed by a script.
- **Feedback log.** `logs/feedback.jsonl`. Written by the
  `report_mistake` intent ("that was wrong"). Treat these as labelled
  data points for the eval set, not as a complaints file.
- **Diagnostic intents.** `modules_status` ("are your modules up?"),
  `explain_routing` ("why did that go to Pluto?"), `report_mistake`. All
  three read from the injected `Diagnostics` object and touch no module
  state, so they still work when the module being asked about is the
  broken one.

If you add a subsystem worth monitoring, give it a `ready()` or
`available()` method. `Diagnostics.module_status()` probes whichever one
it finds and normalises the result; a module with neither is reported as
`unknown` rather than being assumed healthy.

---

## Configuration

- `config/laptop_config.yaml` is validated at startup by
  `core/config_validation.py`, which fails fast with every problem listed
  at once and names the offending key path.
- `python main.py --check-config` validates and exits without booting
  anything. Use it in CI and after hand-editing the YAML.
- Adding a config key: add it to `_SCHEMA` in `core/config_validation.py`
  (required only if Hestia genuinely can't start without it), to
  `config/laptop_config.example.yaml` with an explanatory comment, and to
  `_KNOWN_TOP_LEVEL` if it's a new top-level section — otherwise it's
  reported as a probable typo.
- **Secrets never go in the YAML.** `yaml.safe_load` doesn't interpolate
  `${VAR}`, so a placeholder would never resolve anyway. Read them from
  the environment (`TELEGRAM_BOT_TOKEN`, `HESTIA_SYNC_API_KEY`) and
  document the variable in the example config.

---

## Testing

Run everything: `python run_tests.py` (or `python -m pytest tests/`).
`run_tests.py` just guarantees the repo root is on `sys.path` so
`import modules.x` resolves regardless of the directory you're in.

- **`tests/conftest.py` fakes imports, not behaviour.** It stubs packages
  that are hardware-bound (`sounddevice`, `vosk`, `webrtcvad`,
  `faster_whisper`, `pyttsx3`), network-bound at import time, or heavy and
  optional (`modules.pluto`'s backend chain). Nothing in there asserts
  anything; per-test customisation belongs in the test file.
- **If you stub a package, stub the leaf dependency, and keep packages as
  packages.** `test_main.py` used to stub `modules.pluto` as a plain
  module with no `__path__`. After it ran, every
  `modules.pluto.<submodule>` import in the same session failed with
  "'modules.pluto' is not a package", so `test_pluto.py` errored at
  collection depending purely on file ordering. The `conftest.py` stub is
  now conditional (only installs if the real import fails) and sets a real
  `__path__`.
- **Routes belong to a router, not to an app instance.** Every route in
  `api.py` used to be declared with `@app.get(...)` against the
  module-level `app`, while `main.start_sync_api()` serves its own
  instance from `create_app(api_key=...)`. Decorators only attach to the
  object they name, so the app actually being served had no `/health` and
  no `/sync/*` at all — 404s with nothing in the logs, and code that read
  as though the routes existed. Declare routes on `router`;
  `create_app()` includes it. `tests/test_api_health.py` guards this.
- **Test the contract, not the implementation.** For a module, that means
  `can_handle()` agrees with the registry, `handle()` returns the standard
  dict on *every* path including failures, and nothing raises.

---

## Debugging a misroute

1. `python main.py --dry-run "the query that went wrong"` — runs real
   classification and real routing, executes no handler, writes nothing.
   It flags a registry/Hecate disagreement and a `can_handle()` rejection
   explicitly, because those are the two usual causes.
2. Ask her: "why did you route that there?" (`explain_routing`) — same
   information, mid-conversation, and it works in voice mode.
3. `python main.py --verbose` for Hecate's tier-by-tier debug trace.
4. Check `logs/routing.jsonl` for the pattern across many queries rather
   than one.
5. If the intent was right but the module was wrong, it's a Hecate tier
   ordering problem. If the intent itself was wrong, it's the NLU — and if
   the correct mapping is unambiguous, `config/intent_aliases.yaml` is the
   cheap fix.
