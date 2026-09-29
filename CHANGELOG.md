# Changelog

All notable changes to Hestia. Newest first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Backlog numbers in brackets refer to `hestia_improvement_backlog.md`.

Hestia is a single-developer personal project, so "releases" are just
dated batches of work rather than shipped versions. The point of keeping
this file is to know what actually landed when — with 280 backlog items,
"did I do that one?" stops being answerable from memory.

---

## [Unreleased] — Metis & Orpheus (Writing)

Completes all six items in section 15 (#164–#169) plus #270. New tests:
`tests/test_writing_workflow.py`.

### Added

- **Writing session (#164).** `metis_writing_session` — one command that has
  Orpheus draft a poem / story / song, then Metis critique it once and
  polish it. The draft is kept as version 1 of the creation; the polished
  text is version 2. `polish: false` gives critique only.
- **Optional polish pass (#270).** Orpheus's `write_poem`, `write_story` and
  `generate_lyrics` accept `polish: true` (or default it on with
  `writing.polish_pass: true`). One critique, at most one revision, never
  iterative. Creative text is edited as art (voice, imagery and line breaks
  protected, personal style profile not applied). A revision that gutted or
  ballooned the piece is rejected and the draft is kept. `metis_polish_text`
  runs the same pass on any text.
- **Style profile (#165).** `metis_learn_style` / `show_style_profile` /
  `clear_style_profile`. Builds a profile from pasted samples (measured
  habits such as sentence length, contractions and punctuation, plus a short
  LLM voice summary). Correct, clarity, style, tone-shift, rewrite, draft,
  expand and shorten respect it; `use_style: false` opts out. Samples are
  stored in full locally and can be wiped.
- **Length and readability targets (#166).** shorten / expand / summarise /
  draft take `target_words`, `min_words`, `max_words` and `target_grade`. The
  result is measured, retried once with the exact miss described, and
  reported honestly if still off. Text is never truncated to hit a number.
  `readability_report` now includes measured Flesch scores.
- **Version history (#167).** Append-only `creation_versions` table. New
  Orpheus intents `revise_creation`, `get_versions`, `restore_version`.
  Restoring appends a copy, so nothing is ever lost. Existing databases are
  upgraded on open, with each old creation's text recorded as version 1.
- **Export (#168).** `orpheus_export_creation` and `metis_export_session`
  write `.md` / `.txt` into `writing.export_dir` (default `data/exports`).
  Filenames are reduced to a safe basename, existing files are never
  overwritten, and poem line breaks survive in Markdown
  (`core/text_export.py`).
- **Plagiarism check surfaces sources (#169).** `check_plagiarism` now
  searches distinctive passages as exact phrases, opens each returned page to
  confirm the passage is really there, and lists title, URL and
  confirmed/unverified status. It still gives no originality score. With no
  browser it says so and returns the quoted passages for a manual check.
- **Ten new intents** (four Orpheus, six Metis). Registry version bumped to
  2.9.0 (minor: intents added); `config/nlu_prompt.txt` and
  `config/intent_aliases.yaml` updated to match. New optional `writing:`
  config block (`polish_pass`, `export_dir`, `plagiarism_web_check`),
  validated in `core/config_validation.py`.

### Fixed

- `distinctive_phrases()` could return an all-stopword window as a search
  query; such windows are now skipped.
- Metis `_safe_db` discarded the new row id; it now returns it.
- Metis / Orpheus listings order same-second rows by id, so "latest" is
  deterministic.

### Known gaps

- Plagiarism spot-checks depend on the browser agent (DuckDuckGo via
  Playwright). They sample a few passages, so "no matches" never proves a
  text is original.
- The style profile is measured plus LLM-summarised; it steers the model but
  does not guarantee the output matches the user's voice.

---

## [Unreleased] — Chronos (Time, Scheduling, Reminders)

Completes nine of ten items (#81–#87, #89, #90); **#88 is partial** (see Known gaps). Chronos already contained most of the
logic; this batch wires it in, fixes what testing turned up, and covers it
with tests (`tests/test_chronos_recurrence.py`, `test_chronos_engine.py`,
`test_chronos_wiring.py`, `test_chronos_weather.py`,
`test_chronos_main_wiring.py`, and standalone `test_chronos_ics.py` and
`test_chronos_reminders.py` for the iCalendar codec and `ReminderService`).

### Added

- **Recurring reminders (#81, #84).** "every weekday at 7am", "every 2 weeks
  on monday", "every month on the 31st", "every year on march 5", raw cron
  literals, `until` / `for N times` limits. Stored as a JSON rule and
  advanced with a compare-and-swap so it can never double-advance.
- **Snooze (#83)**, **per-reminder time zones (#85)**, **location reminders
  (#82)** ("remind me to buy milk when I get home"), **holiday-aware
  scheduling (#87)**, **missed-reminder catch-up on startup (#89)**, the
  **"what's on my plate" agenda (#86)**, **weather-triggered suggestions
  (#88)** and **ICS export/import (#90)**.
- **Ten new intents**, all owned by `chronos`: `list_reminders`,
  `cancel_reminder`, `snooze_reminder`, `get_agenda`, `mark_holiday`,
  `unmark_holiday`, `save_place`, `export_calendar`, `import_calendar`,
  `weather_plan`. Registry version bumped to 2.8.0 (minor: intents added);
  `config/nlu_prompt.txt` and `config/intent_aliases.yaml` updated to match.
- **Schema migration** (`modules/mnemosyne/schema.py`): reminder columns,
  `user_holidays` and `places` tables, added to existing databases one column
  at a time. Safe to run on every startup; tested against an old-format DB.
- **`MnemosyneEngine.add_location_listener`**, called on every GPS or
  Telegram location update. IP-derived fixes are deliberately not forwarded:
  they can be kilometres off and would arm or fire location reminders
  spuriously.
- **Config options** under `chronos:` (`scheduler_enabled`,
  `scheduler_interval_seconds`, `default_snooze_minutes`,
  `skip_public_holidays`, `holiday_country`, `proactive_weather`,
  `exports_dir`), validated in `core/config_validation.py`. All optional.
- "nth weekday of the month" repeats ("first monday of every month") are
  now refused with an explanation instead of silently becoming a different
  schedule.

### Changed

- `main.py` passes the Chronos options through, attaches Hermes / Artemis /
  Dionysus to Chronos, and starts and stops its scheduler.
- When Chronos's scheduler is running, `HestiaHeartbeat.handle_reminders` is
  set to `False`, so a one-shot reminder is never announced by both.
- Location reminders read "when you get home" rather than "when you get to
  home".

### Fixed

- `modules.chronos.recurrence` failed to import (`strip_duration` missing).
- `parse_duration` read "and" as "an" + "d" (days): "2 hours and 15 minutes"
  came out as more than a day.
- `Recurrence.describe()` rejected a time-zone argument, which broke
  recurring reminders, listing and the agenda.
- "every month on the 31st" was parsed as "every Monday" ("mon" matched
  inside "month"); "every year on march 5" ignored the date.
- Tomorrow's agenda listed today's occurrence of a recurring reminder as
  "Overdue".

### Known gaps

- **#88 (weather-triggered suggestions) is partial.** Rain assessment, the
  `weather_plan` intent, the Dionysus indoor idea and the once-a-day
  proactive warning are tested against a stubbed forecast, but the live
  Open-Meteo request was never run (blocked in the build sandbox).
- 13 tests in Athena / Iris / Mnemosyne semantic recall fail in a sandbox
  with a stubbed ChromaDB (`Collection` has no `upsert` / `delete`); they
  are unrelated to Chronos and fail identically without these changes.
  `tests/test_pluto.py` needs `pypfopt`, which was not installed.

---

## [Unreleased] — Iris (Vision & Media)

Completes 7 of 8 `[Q]`/`[M]` items. `[L]` items out of scope as usual
(#71 CLIP-based semantic image search beyond what already exists, #77
real-time webcam object detection). **#72** (face clustering) is also
deferred — it needs a face-detection library (`face_recognition`, `dlib`,
or similar) not currently in `requirements.txt`; flagged rather than
guessed at, the same treatment given to Athena's `fitz`-dependent items.

### Added

- **EXIF-based search** (`modules/iris/analyser.py::_extract_exif`,
  `IrisDB.search_files_by_exif`). Date taken, GPS coordinates, and camera
  make/model, extracted via Pillow (already a dependency, no new one
  needed) and stored per file. GPS hemisphere signs (N/S, E/W) handled
  explicitly rather than assumed. Never raises on an image with no EXIF
  block at all (screenshots, re-saved images, most PNGs) or a malformed
  one — returns an empty dict rather than guessing. [#74]
- **Whole-library duplicate scan** (`DuplicateDetector.find_all_duplicate_groups`,
  `iris_find_duplicates` intent). The existing `find_duplicates` only ever
  checked one incoming file against what's already in the DB at ingest
  time — this is a genuine gap the new scan closes: two near-identical
  files ingested in the SAME batch are processed concurrently (see
  `FileIngestor._process_batch`'s `asyncio.gather`), so neither has been
  written to the DB yet when the other's at-ingest check runs, and both
  get indexed as if unique. A full scan afterward is the only way to
  catch that pair. Reports exact (file_hash) and near (perceptual hash
  within threshold) groups separately; a file appears in at most one
  group. [#73]
- **Manual caption/tag correction** (`IrisDB.correct_caption`,
  `iris_correct_caption` intent). Marks `caption_source='user'` so a
  future re-analysis pass (should one ever be added) knows not to
  silently overwrite a person's correction. [#78]
- **Album/collection auto-organization** (`IrisEngine.organize_into_albums`,
  `_cluster_by_distance`, `iris_organize_albums` intent). Greedy cosine-
  distance clustering over CLIP embeddings (no new clustering-library
  dependency added for this), writing into the `events`/`event_files`
  tables that already existed in the schema but were unused until now.
  Re-clusters from scratch on every call rather than incrementally —
  adding one new photo can legitimately change which cluster several
  existing photos best belong to. Clusters below a minimum size (3) are
  dropped rather than surfaced as one-or-two-photo "albums". [#79]
- **"Describe what changed" photo comparison**
  (`IrisEngine.describe_change`, `iris_compare_photos` intent).
  Generalized `IrisAnalyser._send_to_ollama` to accept multiple images in
  one call (llava's ollama API already supports this) so both photos go
  to the vision model TOGETHER for a direct comparison, rather than two
  independent captions diffed after the fact — which would lose whatever
  direct before/after comparison the model itself could make. [#76]
- **Storage-budget guard** (`IrisConfig.storage_quota_bytes`,
  `IrisEngine.check_storage_quota`). Estimates the incoming directory's
  total size before ingesting and warns (doesn't silently proceed) if
  ingesting would push total ingested media past the configured quota.
  Disabled by default (`None`) — most people don't want a quota until
  they've hit a real storage problem once. The estimate sums every file
  under the source directory regardless of whether it's already ingested
  (duplicates get skipped at ingest time and don't consume additional
  space), erring toward warning a little early rather than ever silently
  blowing past the quota. [#80]
- Extracted `_load_image_base64` as a shared helper (`analyse_file` and
  `describe_change` both need to load/downscale/encode an image the same
  way; previously only inlined in `analyse_file`).
- Tests: `test_iris_exif_and_quota.py`, `test_iris_duplicates_and_captions.py`,
  `test_iris_albums_and_compare.py` — 101 new cases (3 honestly skipped
  where `imagehash` isn't installed in this sandbox — see #73's tests),
  1808 total passing (same 12 pre-existing failures as every batch before
  this one, all pre-dating this work).
- `iris_find_duplicates`, `iris_correct_caption`, `iris_organize_albums`,
  `iris_compare_photos` registered; `intent_registry.py` bumped to 2.7.0.

### Fixed

- A real migration-ordering bug caught by running against the actual
  persisted `data/iris/iris.db` (not a fresh test database): the new
  `idx_files_date_taken` index was created in the same `executescript`
  block as `CREATE TABLE IF NOT EXISTS`, which is a no-op against a
  table that already exists — so on any pre-existing database, the index
  tried to reference `date_taken` before the migration's `ALTER TABLE`
  (which runs afterward) had added it. Moved the index creation to after
  the migration loop.

---

## [Unreleased] — Athena (Research & Documents), part 1

Partial completion of the Athena backlog section — the items with the
best verified-feasibility ratio given this environment (see Notes). `[L]`
items out of scope as usual (#51 LaTeX scaffolding, #52 PowerPoint
generation, #60 citation graph visualization, #66 methodology generator).

### Added

- **Change-aware re-ingestion** (`MergedLocalRAG.ingest_file`). Previously
  `_is_ingested` checked only (file_name, subject) EXISTENCE — a modified
  file with the same name was silently treated as already-indexed
  forever, with no error or warning anywhere; its content update was
  invisible to the index. Now tracked via a cheap mtime+size signature
  (not a full content hash, which would mean reading every large PDF on
  every ingestion run just to check whether it changed) stored in each
  chunk's metadata; a changed file has its stale chunks deleted before
  being re-ingested fresh. `IngestionStats` now distinguishes new/
  updated/unchanged/failed rather than one undifferentiated chunk count. [#61]
- **"What's new since I last checked" digest**
  (`MergedLocalRAG.get_changes_since_last_check`, `athena_check_updates`
  intent). A dry preview — nothing is ingested — built directly on #61's
  signature tracking. [#57]
- **Configurable chunk size/overlap per document type**
  (`AthenaConfig.chunk_config_by_type`). A dense academic PDF and a one-
  page markdown note no longer chunk identically. Also fixed a real dead-
  config bug found while implementing this: `chunk_overlap` was stored on
  `PDFProcessor` and never once read inside `semantic_chunking` — the
  carry-forward between chunks was hardcoded to "the whole previous
  paragraph" regardless of the configured overlap number. It's now an
  actual character-count carry-forward, tunable and unit-tested. [#62]
- **Feedback loop / down-weighting** (`core/chunk_feedback_store.py`,
  `MergedLocalRAG.mark_feedback`, `athena_mark_feedback` intent). Marking
  a source irrelevant demotes it (score × up to 0.8 per negative mark,
  floored at 10% of original — never fully excluded, since a chunk
  irrelevant to one question can still be exactly right for another) and
  re-sorts results; marking it relevant is recorded but doesn't boost —
  the asymmetry is deliberate, since "still useful" shouldn't need an
  active vote the way "actively unhelpful" does. [#63]
- **Retrieval score breakdown in debug mode**
  (`SourceDocument.semantic_score`/`bm25_score`, `entities.debug`/
  `show_scores` on the `search`/`athena_search` intent). `SearchResults`
  already computed separate semantic and BM25 scores internally; they
  simply weren't threaded through to the response. Opt-in — off by
  default, since it roughly doubles the field count of what's usually a
  short source list and most callers just want the answer. [#64]
- Tests: `test_athena_reingestion.py`, `test_athena_chunking.py`,
  `test_athena_debug_scores.py`, `test_athena_feedback.py` — 70 new
  cases, 1696 total passing (same 12 pre-existing failures — see Notes).
- `athena_check_updates` and `athena_mark_feedback` registered;
  `intent_registry.py` bumped to 2.4.0.

### Fixed

- A self-inflicted regression caught by the full-suite run before it
  shipped: adding `_handle_mark_feedback` accidentally deleted the `def
  _handle_check_updates(...)` line during editing, leaving its body
  orphaned as dead code after a `return` (syntactically valid — a bare
  string literal — so `ast.parse` didn't catch it; only actually calling
  the intent did). Fixed; both handlers verified present and dispatching
  correctly.

### Deferred (not implemented this batch)

- **#58, #59, #68** (table extraction, figure/caption extraction, OCR
  language auto-detection) — all need `fitz` (PyMuPDF)'s table/image APIs
  and/or real multi-language PDF fixtures to verify quality against, and
  `fitz` isn't installed in this sandbox (confirmed via `import fitz`
  failing) even though it's clearly a real dependency in the target
  environment (imported unconditionally in `pdf_processor.py`). Writing
  this code against documented APIs I can't exercise felt like the wrong
  trade-off given how much of this section WAS independently verifiable.
- **#53** PDF export of generated reports — needs a PDF-writing dependency
  (e.g. `reportlab`) not currently in `requirements.txt`.
- **#70** ingestion progress reporting to the web UI — not reached.
- **#34/#35's intent wiring** (quiz engine built in the Mnemosyne batch,
  not yet connected to a voice/chat intent) — still outstanding.

## [Unreleased] — Athena (Research & Documents), part 2

Completes everything from part 1's deferred list except #53/#58/#59/#68/#70
(see that section's Deferred note for why those specifically stayed out).

### Added

- **Literature review generator** (`SynthesisService.generate_literature_review`,
  `athena_literature_review` intent). Synthesizes across every document
  ingested under a subject — recurring themes, agreement/disagreement
  between sources — via the same local LLM used everywhere else, no new
  dependency. [#54]
- **Citation management** (`modules/athena/services/citation_service.py`,
  `athena_get_citations` intent, BibTeX or APA-style output). Explicitly
  FILE-based, not full academic citations — Athena tracks file_name/
  subject/page per chunk, not author/year/journal, and the module's own
  docstring says so plainly rather than fabricating plausible-looking
  metadata the underlying data doesn't have. [#55]
- **Research-gap detection** (`SynthesisService.detect_research_gaps`,
  `athena_research_gaps` intent). Explicitly scoped to gaps the SOURCES
  THEMSELVES mention (limitations, future-work language) — the prompt
  instructs the model not to invent gaps beyond what's actually
  discussed, and to say so plainly when none are. [#56]
- **Multi-document comparative queries** (`SynthesisService.compare_documents`,
  `athena_compare_documents` intent). Compares named documents
  specifically (not a subject-wide synthesis); reports which requested
  file names didn't match anything ingested, since a typo'd file name is
  a likely mistake worth surfacing rather than silently ignoring. [#65]
- **"Translate this document" pipeline** (`SynthesisService.translate_document`,
  `athena_translate_document` intent). Translates via the same local LLM,
  no separate translation API/library. Long documents are regrouped into
  fewer, larger pieces before translation (not one LLM call per small
  ingestion chunk) for both efficiency and better per-call context. [#69]
- **File-type coverage tests in CI** (`tests/test_athena_file_type_coverage.py`).
  A real, minimal fixture generated at test time for every format
  `document_processor.py` claims to support, using that format's OWN
  writer library (python-docx, python-pptx, EbookLib) rather than a
  checked-in binary fixture that can rot silently. Includes an explicit
  invariant test that fails if a new supported extension is added without
  matching fixture coverage. `.pdf`/`.epub` skip gracefully via
  `pytest.importorskip` when their library isn't installed in a given
  environment — with a note on why `importorskip` alone wasn't enough for
  `fitz` specifically (this test suite's own chromadb/torch stub setup
  also writes a fake, import-succeeding `fitz` module for other tests'
  sake, so the real-vs-stub distinction needed an extra check). [#67]
- Two new supporting `MergedLocalRAG` methods: `list_files` (distinct
  ingested files, optionally by subject) and `get_chunks_for_file` (raw
  chunk text for one file, in original order — representative document
  coverage, not relevance-ranked search results). Both needed by every
  feature in this batch and by `athena_search`.
- Tests: `test_athena_citations.py`, `test_athena_synthesis.py`,
  `test_athena_file_type_coverage.py` — 78 new cases, 1752 total passing
  (same 12 pre-existing failures as every batch before this one).
- `athena_literature_review`, `athena_research_gaps`,
  `athena_compare_documents`, `athena_get_citations`,
  `athena_translate_document` registered; `intent_registry.py` bumped to
  2.6.0.

---

## [Unreleased] — Mnemosyne (Memory & Knowledge)

Completes the `[Q]`/`[M]` items in the Mnemosyne backlog section. `[L]`
items out of scope: knowledge-graph extraction (#31), spaced-repetition
scheduling (#33), Obsidian vault sync (#39), episodic memory clustering
(#43), and arXiv/IEEE monitoring (#47) — each an independent multi-week
project. **#32** (D3 graph visualization) is also deferred: it's meant to
visualize the knowledge graph #31 would build, and #31 is out of scope, so
there's no real graph data to visualize yet.

**#34/#35** (quiz generation, strength/weakness map) are implemented as a
complete, tested, standalone engine (`core/quiz_engine.py`,
`core/quiz_store.py`) but NOT yet wired into the orchestrator as
voice/chat intents ("quiz me on X") — that needs a new registered module
plus the usual intent_registry.py/nlu_prompt.txt checklist, which is
flagged as clear follow-up work rather than rushed alongside the rest of
this already-large batch.

### Added

- **Fact schema extended** (`modules/mnemosyne/schema.py`): `last_accessed`,
  `access_count`, `importance`, `stale` columns on `facts`, added via an
  idempotent migration (`ALTER TABLE ... ADD COLUMN`, guarded against
  "duplicate column" so it's safe to run on every startup against both a
  fresh and a pre-existing database) so existing installs don't need a
  manual migration step.
- **Fact expiry/decay** (`MnemosyneEngine.run_decay_check`). Facts not
  accessed in 6 months (configurable) are FLAGGED for review, never
  auto-deleted — being recalled (`remember()`, `get_user_info`) clears the
  flag, since that's direct evidence the fact is still relevant. [#36]
- **Per-fact confidence/source tracking** — the `source`/`confidence`
  columns already existed; this batch actually surfaces them: recall
  responses now carry a provenance clause (see #50 below) built from them. [#38]
- **Contradiction detection** (`MnemosyneEngine.check_for_contradiction`).
  Keyed off KEY similarity (fuzzy string match), not value-embedding
  similarity — two facts about unrelated topics can have similar
  embeddings just for being ordinary sentences, which would make
  embedding-based contradiction detection fire constantly on unrelated
  pairs. A near-identical key ("favorite_color" vs "favourite_colour")
  with a different value is a much stronger, lower-false-positive signal.
  Non-blocking: `learn_fact` still writes the new fact and appends an
  informational note rather than gating on confirmation. [#37]
- **Weekly/monthly digest** (`MnemosyneEngine.generate_periodic_digest`).
  Distinct from the existing count-based Summariser (every N raw
  interactions): this is a TIME-based rollup over already-generated
  summaries, wired into `core/heartbeat.py` on the same 7-day/30-day
  rolling-gap pattern as the #6/#30 heartbeat jobs. [#40]
- **Bulk "forget everything about X"** (`MnemosyneEngine.find_matching_facts`,
  `forget_matching`). Extends the existing single-fact `forget_fact`
  intent — entities carrying a `pattern` (instead of an exact `key`)
  triggers substring-matched bulk deletion, gated by the same
  confirmation mechanism as a single fact (doubly important for a bulk,
  irreversible delete): lists what will be forgotten before doing it. [#41]
- **Semantic deduplication on ingest** (`MnemosyneEngine._find_duplicate_value`,
  wired into `learn()`). Mirror case of contradiction detection: same
  VALUE (high embedding similarity), different key. Only applies to
  genuinely NEW keys — an update to an EXISTING key is a legitimate
  upsert and is never intercepted (see `test_mnemosyne_embedding_drift.py`
  for why that boundary matters). A duplicate bumps the existing fact's
  reference count instead of writing a near-identical second copy. [#42]
- **Memory export** (`MnemosyneEngine.export_memory`, JSON or Markdown).
  Facts + goals + summaries, independent of the sync API (that's for
  device-to-device delta sync; this is a point-in-time backup/portability
  snapshot). [#44]
- **Importance-weighted fact ranking** (`MnemosyneEngine.get_top_facts_scored`,
  `set_fact_importance`). Replaces `get_top_facts_for_context`'s old pure-
  recency ordering with a blend of recency (exponential half-life),
  access frequency (log-scaled), an explicit 0-1 importance weight, and
  confidence — weights are visible class-level constants, not buried in a
  SQL expression, specifically so they're tunable and unit-testable. [#45]
- **Embedding drift tests** (`tests/test_mnemosyne_embedding_drift.py`).
  Guards the invariants that prevent a fact's stored vector going stale
  relative to its current text: updating a fact re-embeds (never reuses
  the old vector), `MnemosyneVectorStore.add()` always calls Chroma's
  `upsert` (replace) rather than a plain insert, and the new
  deduplication logic (#42) never accidentally intercepts a real update
  to an existing key. [#46]
- **Dated recall** (`MnemosyneEngine.recall_on_date`). "What did I say on
  Tuesday" is a genuinely different query shape from semantic search
  (`remember()`, which has no notion of "on this specific day" at all) —
  a first-class dated lookup over both facts created and interactions
  logged that day. [#48]
- **Memory dashboard** (`MnemosyneEngine.get_memory_dashboard`). Extends
  the existing `get_memory_stats` (facts/goals/summaries counts) with
  on-disk DB size (including WAL/shm sidecar files, since this connection
  runs in WAL mode) and embedding count. [#49]
- **Memory provenance in responses** (`MnemosyneEngine._provenance_phrase`).
  Recall responses now carry a short clause — "(mentioned yesterday)",
  "(inferred)" — built from a fact's `created_at`/`source`. Wired into
  `_format_result` (semantic recall), `_handle_get_user_info` (direct key
  lookup), and `recall_on_date`. [#50]
- **Quiz generation + strength/weakness tracking** (`core/quiz_engine.py`,
  `core/quiz_store.py`, own SQLite schema — kept independent of
  Mnemosyne's, since multiple-choice questions and per-subject scoring
  are a genuinely different data shape from key/value facts). Generated
  questions are validated before storage (exact choice count, in-range
  correct index, non-empty question) rather than trusting the model's
  JSON shape blindly. `get_weakest_subjects` requires a minimum attempt
  count before calling a subject "weak" — a single wrong answer is noise,
  not a rate. **Engine only; not yet wired to a user-facing intent** — see
  this section's intro. [#34, #35]
- Tests: `test_mnemosyne_extended.py` (62 cases), `test_quiz_engine.py`
  (28 cases), `test_mnemosyne_embedding_drift.py` (8 cases) — 98 new
  cases, 1650 total passing (same 12 pre-existing failures as every batch
  before this one — all in `test_iris_embeddings.py` and one
  `test_mnemosyne.py` case, both pre-dating this work and caused by a
  ChromaDB version mismatch in this environment, not by anything in this
  changeset).

### Changed

- `MnemosyneEngine.learn()` now returns `{"deduplicated": bool,
  "matched_key": Optional[str]}` instead of `None`. Every existing caller
  (Ares, Metis, Orpheus, CoreModule) calls it as a fire-and-forget
  statement and never inspected the return value, so this is additive in
  practice — confirmed by grepping every call site before making the
  change.
- `get_user_info` responses now include a provenance clause by default
  (see #50) — `tests/test_mnemosyne.py`'s
  `test_get_user_info_with_real_key_still_works` updated to check the
  core content rather than an exact string match.

---

## [Unreleased] — NLU & Intent Classification

Completes the remaining `[Q]`/`[M]` items in the NLU & Intent
Classification backlog section (#22, #26, #28 landed in the Core
Architecture batch above and aren't repeated here). The one `[L]` item in
this section — fine-tuning a small local classifier on logged queries so
NLU stops depending on Ollama's JSON-mode reliability (#25) — is out of
scope for the reason every other `[L]` item has been: a genuinely
independent, multi-week project.

### Added

- **Confusion-matrix eval script** (`scripts/eval_intents.py`). Parses
  `hestia_test_prompts.md` — a file that already existed with hand-written
  ground truth embedded in its prose — as a real golden dataset: 60
  module-level cases from its "Per-Module Sanity Checks" section, 10
  exact-intent cases from its "Boundary Cases" section's explicit
  `→ \`intent\`` annotations, and 36 deliberately-ambiguous cases correctly
  left unscored rather than assigned a fabricated single answer. Found and
  fixed a real mismatch along the way: the markdown's ground-truth intents
  are written unprefixed (`rewrite_style`) while the registry stores them
  module-prefixed (`orpheus_rewrite_style`); unresolved, every one of
  those cases would have been a false failure. `--dataset-only` runs the
  parser with no Ollama connection (this is what CI gates on); the full
  live-model run is opt-in via `--require-ollama`. [#21]
- **Per-entity confidence scores** (`core/entity_confidence.py`). A second
  question from the NLU's existing top-level confidence — not "is this
  the right intent" but "does this extracted VALUE look trustworthy".
  Heuristic and deterministic (no second LLM call): dates/times scored via
  `dateparser`, emails via structural validation, amounts via presence in
  the source text, free text via a length floor. Every `HestiaNLU.understand()`
  result now carries an `entity_confidence` dict alongside `entities`.
  Scoring only — deciding what to DO about a low score is left to callers
  like the slot-filling mechanism below. [#23]
- **Intent chaining** (`core/intent_chains.py`). "Summarize this paper and
  add it to my reading list" now pipes the first segment's dispatched
  response into the second segment's entities, rather than the second
  segment (multi-intent-split from #13) being dispatched with nothing but
  the literal words "add it to my reading list". Detection is a regex over
  the second segment's own text (an anaphoric reference — "add it", "save
  that", "note it down" — anchored at the start), not a semantic guess; a
  wrong detection costs one entity substitution, never a wrong action,
  since the substituted content is real dispatched output from earlier in
  the SAME query. Only `take_note` is wired as a verified chain target —
  its entity shape was checked directly against
  `modules/hestia/core_module.py._take_note` rather than assumed; other
  plausible targets are explicitly left for when their shapes are
  similarly verified. [#24]
- **Multi-language support at the NLU layer** (Hindi/Hinglish) [#27]
  - `core/language_detect.py`: deterministic Unicode-script classification
    (devanagari/latin/mixed/other) — NOT language identification for
    Romanized Hinglish, which script detection alone cannot do reliably;
    see the module's docstring for that distinction. Wired into the
    routing log as a new `script` field, so a future accuracy breakdown
    can ask "is classification worse for Devanagari input" from the log
    alone.
  - `config/intent_aliases.yaml`: ~20 common Hinglish phrasings merged
    into the relevant intents' existing alias blocks, reusing #22's
    deterministic pre-LLM matching rather than depending on the model's
    Hindi/Hinglish reliability for phrases used often. (Caught and fixed
    a real bug while writing this: a first draft added these as duplicate
    top-level YAML keys, which silently overwrote the earlier English
    blocks sharing the same intent name — merged into the originals
    instead.)
  - `config/nlu_prompt.txt`: explicit multi-language instructions (intent
    names always stay English snake_case regardless of input
    language/script; entity values stay in the user's original language)
    plus five Hinglish few-shot examples.
- **Conversational slot-filling** (`PendingSlotFill`,
  `HestiaOrchestrator._resolve_pending_slot`). `modules/hermes/engine.py`'s
  `_clarify()` already asked "Who should I send it to?" when `send_email`
  was missing a recipient, but nothing captured the reply — it vanished
  into a fresh, unrelated classification. `_clarify()` now optionally
  carries a `slot`/`entities` pair; the orchestrator holds it as a
  `PendingSlotFill` and feeds the next query's text VERBATIM into that
  entity key before re-dispatching straight back to the same handler —
  deliberately not re-running it through the NLU, since "raj@example.com"
  is an answer, not a new command. Wired to the three genuinely
  single-slot cases (`send_email`'s `to`/`body`, `create_event`'s
  `title`); chains correctly into the existing yes/no confirmation gate
  and into a second missing slot. [#29]
- **Per-intent accuracy tracking + weekly report**
  (`Diagnostics.per_intent_accuracy`, `worst_performing_intents`,
  `weekly_accuracy_summary`). Built on the routing log (#5) and feedback
  log (#259) that already existed: accuracy_estimate = 1 -
  (explicitly-flagged-wrong / total-classified), excluding any intent
  below a minimum sample count so a single outcome is never reported as a
  rate. Explicitly labelled an ESTIMATE, and honestly so — it only
  reflects mistakes the user bothered to report via `report_mistake`, so
  it's a lower bound on the true error rate, not a measurement of it.
  Wired into `core/heartbeat.py` as a rolling 7-day job (day-count gap,
  not a calendar-week flag, to avoid firing twice across a week
  boundary). Refactored the JSONL-log-reading logic shared with #6's
  daily review into `_read_jsonl_since()`. [#30]
- Tests: `test_eval_parser.py`, `test_entity_confidence.py`,
  `test_intent_chains.py`, `test_language_detect.py`,
  `test_slot_filling.py`, plus extensions to `test_multi_intent.py`,
  `test_observability.py`, `test_heartbeat.py`, and `test_intent_aliases.py`
  — 178 new cases, 1552 total passing (same 12 pre-existing failures as
  every batch before this one).

---

## [Unreleased] — Core Architecture & Orchestration, part 2

Completes the remaining `[Q]`/`[M]` items in the Core Architecture &
Orchestration backlog section. The four `[L]` items in that section — a
trained classifier replacing the fallback tiers (#4), a real message-queue
event bus (#10), shadow-mode dual routing (#16), and splitting Hestia into
2-3 processes (#20) — are each independently a multi-week project and are
out of scope here.

### Added

- **Confidence-weighted clarification** (`modules/hecate/engine.py`).
  Registered-but-uncertain intents (confidence below 0.45) no longer
  execute blind — Hecate's new Tier 0.5 routes them to a `clarify_intent`
  response asking the user to rephrase, before Tier 1's "any registered
  intent dispatches unconditionally" rule would otherwise act on them. A
  low-confidence `chat` guess is unaffected — it already has a correct
  home via the existing force-chat fallback. [#2]
- **Per-module circuit breakers** (`core/circuit_breaker.py`). Three
  consecutive failures opens a module's breaker for 60s; further queries
  to that module get a fast, honest "taking a short break" response
  instead of a slow failing call. Half-open probing after cooldown, reset
  on success. Status exposed via `orchestrator.circuit_breaker_status` and
  folded into `/health/modules`. [#7]
- **Drop-in skills** (`core/module_loader.py`, `skills/`). A single-file
  `BaseModule` subclass under `skills/` (convention: a top-level class
  named `Skill`) is auto-discovered and registered at startup, without
  editing `main.py`. Scoped deliberately smaller than a full module: at
  most `ollama_cfg`/`memory` as constructor dependencies, one file per
  skill, no recursion into subdirectories. The 14 built-in modules are
  unaffected and still registered explicitly. [#9]
- **Conversation session TTL** (`OrchestratorContext.maybe_expire_session`,
  `modules/hestia/orchestrator.py`). A 30-minute gap (configurable via
  `hecate.session_ttl_seconds`) since the last turn clears
  `recent_intents`/`entities`/`time_context`/`memory_context` at the top
  of `dispatch()`, so "add it to the list" three hours after an unrelated
  conversation doesn't inherit that conversation's entities.
  `active_modules` (registration state, not conversation state) is never
  touched by expiry. [#12]
- **Multi-intent query splitting** (`core/query_splitter.py`,
  `Hestia._try_multi_intent`). "Log my workout and tell me the weather" is
  now split and dispatched as two requests. The splitter only finds
  candidate split points (conservatively — connector priority, minimum
  segment length, a verb-shaped-start heuristic to avoid "mac and
  cheese"); the actual commit decision requires both candidate segments to
  independently classify, via a real NLU call each, to two DIFFERENT,
  concrete, registered intents. A bad split candidate costs one discarded
  NLU call, never a wrong action. [#13]
- **Nightly low-confidence review** (`Diagnostics.write_review_queue`,
  `core/heartbeat.py`). Once a day (00:00-05:59, alongside the existing
  nightly summary), the last day's sub-0.6-confidence classifications are
  appended to `logs/review_queue.jsonl`, deduplicated by request id, for
  manual labelling — the same routing log from [#5] now feeds a standing
  job instead of only being useful when someone remembers to grep it. [#6]
- **Hot reload** (`core/hot_reload.py`, mtime-polling, no new
  dependencies). `config/nlu_prompt.txt` is fully hot-reloaded —
  `HestiaNLU.reload_prompt()` re-reads it, rebuilds the JSON schema,
  re-checks registry drift, and invalidates the classification cache — so
  tuning the prompt no longer needs a restart.
  `config/laptop_config.yaml` changes are detected and re-validated with
  the same `core.config_validation` used at startup (typos caught in
  seconds, not at the next restart) and logged with a diff of which
  top-level sections changed; most keys still require a restart to take
  effect, and this is stated honestly rather than half-implemented as a
  silent no-op. [#15]
- `clarify_intent` registered to `core`; `intent_registry.py` bumped to
  2.2.0.
- `hecate.session_ttl_seconds` and `hecate` added to
  `core/config_validation.py`'s schema.
- Tests: `test_circuit_breaker.py`, `test_orchestrator_resilience.py`,
  `test_module_loader.py`, `test_session_ttl.py`, `test_query_splitter.py`,
  `test_multi_intent.py`, `test_hot_reload.py`, `test_nlu_reload.py`, plus
  extensions to `test_observability.py`, `test_heartbeat.py` and
  `test_main_cli.py` — 172 new cases, 1429 total passing (same 12
  pre-existing failures as before this batch — see Notes).

---

## [Unreleased] — Core Architecture & Orchestration, part 1

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
