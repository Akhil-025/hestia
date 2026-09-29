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

## Resilience and extensibility patterns

Added alongside observability; same rationale — make failure visible and
recoverable instead of silent.

- **Circuit breakers** (`core/circuit_breaker.py`). Every module dispatch
  goes through a per-module breaker: three consecutive failures opens it
  for 60s, during which queries to that module get an immediate, honest
  "taking a short break" response instead of a slow failing call. A
  `NotImplementedError` (a module deliberately not supporting a call
  shape) never counts toward the breaker — only real, unexpected failures
  do. Check `orch.circuit_breaker_status` or ask "module status" to see
  current state.
- **Confidence-weighted clarification.** Hecate's Tier 1 dispatches any
  *registered* intent unconditionally — that's what guarantees every
  module's intents are reachable (see the drift-bug story above). Tier 0.5
  runs first and catches the case that guarantee doesn't cover: a
  registered intent the NLU itself wasn't confident about. Below 0.45
  confidence, she asks you to rephrase instead of guessing. Don't lower
  this threshold to "fix" a module that keeps getting misrouted — that's
  an NLU accuracy problem; fix it with `config/intent_aliases.yaml` or a
  better few-shot example instead.
- **Drop-in skills** (`core/module_loader.py`, `skills/`). For something
  small enough not to justify editing `main.py` — a single-file
  `BaseModule` with at most `ollama_cfg`/`memory` as dependencies — drop a
  file in `skills/` defining a top-level `Skill` class. It's discovered
  and registered automatically on the next restart. This is NOT where the
  14 built-in modules live or where a new *full* module should go; it's
  for the smaller tier below that. A skill whose `name` collides with an
  existing module is skipped with a warning, not silently allowed to
  shadow it.
- **Session TTL.** `OrchestratorContext` tracks wall-clock time between
  turns; a 30-minute gap (`hecate.session_ttl_seconds`) clears
  conversational context (`recent_intents`, `entities`, `time_context`,
  `memory_context`) so returning after a long break starts clean instead
  of carrying stale entities from an unrelated earlier topic.
  `active_modules` is registration state, not conversation state, and is
  never touched by this.
- **Multi-intent splitting** (`core/query_splitter.py`). Splitting a
  compound query on string shape alone is unreliable — "mac and cheese"
  looks exactly like "log my workout and check the weather" at that
  level. So the splitter only proposes *candidates*; the real decision
  (commit to the split, or treat it as one query) is made by actually
  classifying both halves and requiring two different, concrete,
  registered intents. If you're debugging why a compound query didn't
  split, check the two halves individually with `--dry-run` first — if
  either one alone doesn't resolve to a clean intent, that's why.
- **Hot reload** (`core/hot_reload.py`). `config/nlu_prompt.txt` reloads
  live — no restart needed to tune a few-shot example. `laptop_config.yaml`
  changes are detected and validated live but NOT hot-applied (see the
  module's docstring for why most keys can't be); you still restart to
  actually pick up a config change, but you find out immediately whether
  the edit was even valid.

---

## NLU, entities, and multi-language

- **Golden-dataset eval.** `hestia_test_prompts.md` isn't just documentation
  — `scripts/eval_intents.py` parses it as a real dataset. If you add a new
  test prompt there with an explicit expected answer, follow the existing
  format: a module-level bullet under a `**ModuleName (...)**` header in
  Section 1, or an intent-level `- "prompt" → \`intent_name\`` (optionally
  `must be \`X\`, never \`Y\`\`) in Section 2. Write the intent name
  unprefixed (`rewrite_style`, not `orpheus_rewrite_style`) — the eval
  script resolves that automatically against the registry, and fails
  loudly (`tests/test_eval_parser.py`) if resolution is ever ambiguous.
  Run `python scripts/eval_intents.py --dataset-only` after editing to
  confirm the parser still picks it up before assuming it's wired in.
- **Entity confidence is a signal, not a decision.**
  `core/entity_confidence.py` scores individual extracted values; it
  never decides what to do about a low score. If you want a module to act
  on it (re-prompt, downgrade to a clarifying question), read
  `entity_confidence` from the NLU result yourself and use the existing
  slot-filling mechanism — don't build a second confidence-driven
  reprompt path.
- **Slot-filling only where the entity shape is verified.** Extending
  `_clarify(slot=, entities=)` to a new intent means checking that
  intent's real handler for the exact entity key it reads — same
  discipline as `core/intent_chains.py`'s `CHAINABLE_TARGETS` comment.
  Guessing the key name means the orchestrator faithfully re-dispatches
  with a value the handler never looks at, and the conversation silently
  goes nowhere.
- **Intent chaining is regex-detected, not semantic.**
  `core/intent_chains.py` only fires on an anaphoric reference anchored at
  the START of the second segment ("add it", "save that"). If you add a
  new chainable target, verify its entity shape the same way `take_note`
  was verified (read the real handler), and keep the detection
  conservative — a false negative just means no chaining happens (the
  segment dispatches with whatever the NLU itself extracted, same as
  before this feature existed); a false positive means good content gets
  overwritten with the wrong thing.
- **Hindi/Hinglish phrases go in the existing intent's alias block, not a
  new top-level YAML key.** `config/intent_aliases.yaml` is a flat mapping
  of intent name -> phrase list; a repeated top-level key silently
  overwrites the earlier one when the YAML loads (this happened once
  already — see CHANGELOG.md). If `apollo_track_sleep:` already has
  English phrases, add the Hinglish ones as more list items under that
  SAME key, not a second `apollo_track_sleep:` block further down the
  file. `tests/test_intent_aliases.py`'s shipped-file tests would have
  caught this except duplicate-key detection has to run on the raw text,
  not the parsed dict — the dict has already silently lost the duplicate
  by the time a test could see it. If you're ever unsure, grep the file
  for the intent name first.
- **Script detection ≠ language identification.**
  `core/language_detect.py` reliably tells you what Unicode script a
  query is written in; it deliberately does NOT try to guess whether Roman
  script text is English or Hinglish — that's unsolvable from characters
  alone. Don't build logic that assumes `detect_script() == "latin"` means
  "definitely English."

---

## Mnemosyne (memory & knowledge)

- **Decay flags, never deletes.** `run_decay_check` marks a fact `stale`
  after months of no access; nothing ever auto-deletes a fact. If you add
  a new decay-adjacent feature, keep that boundary — deletion stays an
  explicit `forget()`/`forget_matching()` call the user asked for.
- **Contradiction (#37) vs. deduplication (#42) are mirror images, on
  purpose.** Contradiction detection compares KEYS (fuzzy string match,
  different value = possible conflict). Deduplication compares VALUES
  (embedding similarity, different key = possible duplicate). Don't merge
  these into one "similarity check" — they catch different mistakes
  ("you're describing the same slot differently" vs. "you already told me
  this under a different label") and conflating them raises the false-
  positive rate of both.
- **Dedup only applies to genuinely new keys.** `learn()` checks for a
  duplicate value ONLY when the key doesn't already exist. An update to
  an existing key is always a legitimate upsert. Getting this boundary
  wrong is exactly the embedding-drift failure
  `test_mnemosyne_embedding_drift.py` exists to catch — read that file
  before touching `learn()`'s dedup branch.
- **Every `vector_store.add()` call must go through `upsert`, never a
  plain insert.** Chroma's `upsert` replaces an existing doc_id's
  embedding; an insert-only path can leave a stale vector next to a new
  one, or silently duplicate. `MnemosyneVectorStore.add()` already does
  this correctly — if you add a second write path to the vector store,
  make it upsert too.
- **Schema changes need a migration, not just a new `CREATE TABLE`
  column.** `CREATE TABLE IF NOT EXISTS` is a no-op against an existing
  table — see `modules/mnemosyne/schema.py`'s `_migrate_facts_columns`
  for the idempotent `ALTER TABLE ... ADD COLUMN` pattern (catch
  "duplicate column", don't try to detect it in advance) and backfill any
  column whose value should default to another existing column's value
  rather than NULL.
- **The quiz engine (`core/quiz_engine.py`, `core/quiz_store.py`) is
  intentionally its own schema**, not bolted onto Mnemosyne's facts
  table — multiple-choice questions and per-subject scoring are a
  different shape from key/value facts. It's also not yet wired to a
  user-facing intent (see CHANGELOG.md) — if you pick that up, follow the
  standard intent checklist above, and feed `generate_quiz` from Athena's
  search/query results for the "ingested notes/documents" source the
  backlog asks for, rather than Mnemosyne facts.

---

## Athena (research & documents)

- **Change detection is mtime+size, not a content hash, deliberately.** A
  full hash would mean reading every file's entire content on every
  ingestion run just to check whether it changed. If you need stronger
  guarantees (e.g. a file edited with its mtime deliberately preserved),
  that's a known, accepted trade-off — don't silently "fix" it to a full
  hash without discussing the performance cost first.
- **`_chunk_id` and `_chunk_id_from_metadata` must always agree.** The
  first builds a chunk's id at ingestion time (from `file_info` + a fresh
  chunk dict); the second reconstructs the SAME id later from a search
  result's stored metadata (for feedback marking, dedup, etc.). If you
  change what one of them reads, check the other still produces an
  identical string for the same chunk — `test_athena_feedback.py`'s
  `test_chunk_id_from_metadata_matches_ingestion_format` guards this, but
  only for the fields it currently knows to check.
- **Per-document-type config lives in `AthenaConfig.chunk_config_by_type`,
  resolved once, in `MergedLocalRAG._resolve_chunk_config`.** Don't add a
  second per-extension lookup elsewhere — extend that one dict and that
  one resolver.
- **Dataclasses in `local_rag.py`/`models.py`: check `@dataclass` vs.
  `@dataclass(frozen=True)` before mutating an instance in place.**
  `SearchResult` is frozen; a down-weighting or scoring adjustment builds
  a new instance via `dataclasses.replace(...)`, not `result.score = ...`.
- **Feedback demotes, never excludes.** `_apply_feedback_weighting` floors
  at 10% of the original score no matter how much negative feedback a
  chunk has. If you're tempted to let enough negative feedback zero a
  chunk out entirely, don't — a chunk irrelevant to one question can
  still be exactly right for a different one.
- **`SynthesisService` (literature review, research gaps, document
  comparison, translation) gathers content via `list_files`/
  `get_chunks_for_file`, never via `search()`.** Those two return a
  document's content in original order for REPRESENTATIVE coverage;
  `search()` returns a relevance-ranked subset for one query. Don't swap
  one for the other — a literature review built from search results
  would be biased toward whatever the search query happened to match.
- **Citations are file-based, not academic.** `CitationRegistry` only
  knows file_name/subject/page — no author, year, or journal, because
  Athena doesn't extract that from ingested PDFs. Never have a citation
  formatter fabricate a plausible-looking author or year; an empty field
  is honest, a guessed one isn't. If you add real bibliographic
  extraction later, that's a distinctly bigger feature, not an extension
  of this one.
- **New supported document type → add a `test_athena_file_type_coverage.py`
  fixture too.** That file's `test_every_declared_supported_extension_has_coverage_above`
  fails on purpose if `document_processor.py`'s `SUPPORTED_EXTENSIONS`
  grows without matching real-fixture coverage. Generate the fixture with
  the format's own writer library at test time (see the existing
  `.docx`/`.pptx`/`.epub` tests) rather than checking in a binary file.
- **`pytest.importorskip` isn't enough to detect a genuinely-missing
  library in this test suite specifically** — `tests/conftest.py`/
  `test_athena.py`'s stub setup also writes fake, import-succeeding
  modules (`fitz`, `chromadb`, `torch`, ...) for OTHER tests' sake. A
  test that needs the REAL library (not just something importable under
  that name) has to additionally check for a real attribute the stub
  doesn't have — see `test_pdf_extraction_with_a_real_generated_file`'s
  `hasattr(fitz, "Document")` check for the pattern.

---

## Iris (vision & media)

- **Whole-library scans are a different tool from at-ingest checks, not a
  bigger version of them.** `DuplicateDetector.find_all_duplicate_groups`
  exists specifically because at-ingest duplicate checking has a real
  blind spot: two near-identical files ingested in the same concurrent
  batch never see each other in the DB. If you add another "check
  everything already ingested" feature, don't try to make the at-ingest
  path do double duty — write it as its own pass over `db.get_all_*`,
  same as this one.
- **Schema migrations for `files`: never put a new index in the same
  `executescript` as `CREATE TABLE IF NOT EXISTS`.** That statement is a
  no-op against a pre-existing table, so an index referencing a NEW
  column has to be created separately, after the `ALTER TABLE` migration
  loop that actually adds the column to old databases — this bit a real
  edit while adding the EXIF columns (see CHANGELOG.md's "Fixed" note),
  caught only because the test run happened to hit the real persisted
  `data/iris/iris.db`, not a fresh one.
- **Album clustering re-runs from scratch, never incrementally.**
  `organize_into_albums` calls `clear_all_events()` before reclustering —
  don't change this to "only cluster new photos" without thinking through
  that adding one photo can legitimately change which cluster several
  EXISTING photos best belong to.
- **Vision-LLM calls that compare multiple images send them together, in
  one call.** `_send_to_ollama` takes a single image or a list; passing
  two images together (see `describe_change`) lets the model make a
  direct comparison. Two separate single-image calls diffed afterward is
  a different, weaker feature — don't conflate them.
- **A guard warns; it does not silently proceed, and it does not silently
  block either.** `check_storage_quota` returns a warning dict instead of
  ingesting; the caller decides whether that's shown to the user or
  overridden with `force=True`. Don't make ingestion fail outright on a
  quota being exceeded — that turns a heads-up into an outage.

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
6. If Hestia asked you to rephrase instead of doing anything
   (`clarify_intent`), the NLU recognised something but wasn't confident —
   check `logs/routing.jsonl` for that query's actual confidence score.
7. If a module suddenly answers "taking a short break" for every query,
   its circuit breaker is open — check `orch.circuit_breaker_status` or
   ask "module status"; it'll self-heal after the cooldown, or check the
   module's own logs for what's actually failing.
