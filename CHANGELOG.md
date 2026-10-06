# Changelog

All notable changes to Hestia. Newest first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Backlog numbers in brackets refer to `hestia_improvement_backlog.md`.

Hestia is a single-developer personal project, so "releases" are just
dated batches of work rather than shipped versions. The point of keeping
this file is to know what actually landed when — with 280 backlog items,
"did I do that one?" stops being answerable from memory.

---

## [Unreleased] — Fine-tunable intent classifier: augmentation, embedding backend, gated distilbert (#25)

Partial: everything is built and tested except the actual fine-tune, which needs `torch`, `transformers` and a
model download that the build sandbox did not have. The training loop in `core/transformer_classifier.py` is
therefore unrun; `scripts/finetune_classifier.py` gates what it keeps because of that.

- **Backends** (`classifier.backend`): `tfidf` (default, unchanged), `embedding` (all-MiniLM-L6-v2 nearest
  neighbours), `ensemble` (both averaged) and `transformer` (a fine-tuned model, loaded only). All decline through
  one shared rule, `decide()`. Intent only, like before: no entity extraction.
- **Thresholds** for the embedding and ensemble backends are chosen by k-fold cross-validation on the training
  data (variants stay in their parent's fold) to reach 95% precision, never from the golden prompts.
  Cross-validation is optimistic on this data, so expect real precision a little lower.
- **Augmentation** (`classifier.augment`, off by default): typo / polite-opener / Hinglish-tail variants for intents
  with fewer than 6 examples, lower weight, golden prompts forbidden. Measured on the golden prompts: answered
  58% -> 61%, right 94.7% -> 95.0% (54/57 -> 57/60). Three more cases answered, so within noise on a 99-case set.
- **`scripts/classifier_bench.py`**: scores backends with a 95% interval; `--save-baseline` records the TF-IDF
  numbers. **`scripts/finetune_classifier.py`**: fine-tunes, calibrates on a validation split, and installs the
  model only if it answers 5+ points more of the golden prompts without losing 2+ points of precision
  (precision floor 90%). Otherwise it is left as `<out>.candidate` (exit code 3). Run it with
  `--device cuda` on the RTX 4050 after installing a CUDA build of `torch`.
- **`--train-classifier`** (existing command) now honours `classifier.backend` / `augment` and prints the
  calibrated thresholds. `--label` is unchanged. Weekly retraining skips the transformer backend.
- Not measured: the embedding, ensemble and fine-tuned backends on the real models. Run
  `python scripts/classifier_bench.py --backends tfidf,embedding,ensemble --augment both --show-wrong`.
- New tests: `test_classifier_backends.py` (47, fakes only: no network, torch or model download).

---

## [Unreleased] — Trained classifier, module events, shadow mode, process split (#4, #10, #16, #20)

These were the four `[L]` items left in #1–20. New tests: `test_classifier_wiring.py`, `test_module_events.py`,
`test_shadow.py`, `test_event_queue.py`, `test_process_split.py`, `test_main_roles.py` (130 tests, all passing here).
Existing suites touched by the edits (Hecate, orchestrator, heartbeat, main, NLU, config validation) fail exactly the
same tests as the original zip, in this sandbox, which lacks pytest (a small shim was used), `dateparser` and PortAudio.
Not exercised here: a real `Hestia(role=...)` boot, the voice process with real audio, and the supervisor with real
subprocesses, because there is no Ollama or audio hardware in the sandbox.

### Added

- **Trained intent classifier wired in (#4).** `core/intent_classifier.py` existed but nothing used it and its
  `core/classifier_data.py` was missing. Now: training data from the NLU prompt examples, aliases, your labels
  (`--label QUERY INTENT`) and confident, un-flagged routing-log lines, with the golden prompts held out. Hecate
  consults it in `primary` mode (Tier 0.8, before registry dispatch, when the NLU said chat or was unsure) or `assist`
  mode (Tier 4.9, last resort); the NLU uses it when Ollama is unreachable or fails. `--train-classifier` trains,
  saves and scores it; the heartbeat retrains weekly. Off by default (`classifier.mode`). Measured on held-out golden
  prompts: answers ~58%, right on ~95% of those. The module docstring's earlier figures (~90% / 98%) did not
  reproduce and were corrected.
- **Module-to-module events (#10).** `core/module_events.py`; see the backlog note for the finding that modules
  previously never used the bus. `intent.handled` is published for every dispatch.
- **Shadow mode (#16).** `core/shadow.py`, `shadow:` config, `--shadow-report`.
- **Process split (#20).** `core/event_queue.py` (durable SQLite queue, request/reply, bus bridge),
  `core/process_split.py` (role profiles, voice frontend, supervisor), `--role all|core|voice|jobs|supervisor`.

### Changed

- `Hestia.__init__` takes `role`; `build_io` takes `only=` (components to build); `HestiaHeartbeat` takes `classifier`.
- Non-voice local commands (do-not-disturb etc.) now speak through `_speak()` instead of `tts.speak()` directly, so
  split mode can route them; behaviour in `--role all` is identical.
- Config validation knows the `classifier`, `shadow` and `processes` blocks.

## [Unreleased] — Hecate decision engine: conference, what-if, routing audit, weekly focus (#158, #160, #162, #163)

Registry 2.22.0 → 2.23.0: new intents `audit_routing`, `conference`, `what_if` (core) and `weekly_focus` (Chronos).
New tests: 138 in `tests/test_decision_engine.py`. Full suite compared with the original zip: 3320 → 3464 passed
and the same 102 failures and 15 collection errors before and after (missing optional packages in this sandbox,
and pytest itself, which was unavailable here, so the suite was run through a small pytest-compatible shim rather
than the real thing), so nothing regressed.

### Added

- **Routing audit trail (#162).** `HecateEngine.decide()` now returns `checked`: an ordered list of everything it
  examined, including the checks that did not decide (inactive owning module, text triggers that missed, the
  confidence gate). `Diagnostics.record_classification` stores it and `audit_last()` reads it back, so "what did you
  check before answering that?" (`audit_routing`) lists the steps in order. `explain_routing` stays the short
  one-line version. Both skip over each other and look at the last real query. Records written before this change
  have no trail and say so.
- **Conference (#158).** `conference` intent. Hecate picks 2–3 modules whose data bears on the topic (or uses
  `entities.modules`) and marks the decision `conference=[...]`; `core/conference.py` asks each for one read-only
  view through `HestiaOrchestrator.call_module()` (same circuit breakers as a normal dispatch), lists the views
  verbatim and adds a short summary. Needs at least two modules with something recorded; otherwise it says so.
  It surfaces Apollo-vs-Artemis disagreement from `core/consensus.py` and never picks a winner.
  Config: `conference.enabled`, `conference.llm_summary`.
- **What-if (#160).** `what_if` intent, `core/whatif.py`. Plain arithmetic on existing data, with the working
  shown: cutting a recurring charge (Pluto: money freed over 3/6/12 months, share of budget, effect on today's
  safe-to-spend), dropping a habit (Artemis: streak lost, 30-day rate, nearby Apollo workouts or goal), changing
  sleep (Apollo: new 7-night average against the burnout check's 6h line). Anything else gets "I can't project
  that" instead of an invented number. Read-only. Config: `whatif.enabled`.
- **Weekly focus (#163).** `weekly_focus` intent in Chronos. Ranks overdue and due goals, calendar deadlines,
  overdue reminders and streaks about to break into the few that need attention this week. Goals are also scored
  by how many open (non-busy) calendar days remain before they are due. Rules are plain and documented in
  `modules/chronos/agenda.py`; a missing source adds a note and the rest still answer.
- `HestiaOrchestrator.attach_conference`, `attach_whatif`, `call_module`; `CoreModule` answers `audit_routing`
  and says so honestly when the conference or what-if layer is not attached.
- NLU prompt entries and aliases for all four intents.

### Changed

- Hecate's decision dict has two new keys (`conference`, `checked`); the two tests that pin its exact shape were
  updated.
- Ares's `decision_support` is deliberately not a conference lens because it saves the decision and schedules a
  reminder; the conference reads `outcome_stats` instead.

### Not done / not verified

- **Not run against a live Ollama model.** The conference summary is tested with a fake; with `llm_summary: true`
  and the model down, it falls back to the plain summary.
- **`main.py` wiring is untested here.** `tests/test_telegram_main_hooks.py` and the other `main.py` tests cannot
  import in this sandbox. The changes compile and the objects they construct are covered by their own tests.
- Conference topic matching is a small word list in `HecateEngine._CONFERENCE_TOPICS`; topics it does not cover
  need `entities.modules` from the NLU or the user naming the areas.
- What-if does not cover quitting a job, buying something, or moving; those are decisions for Ares, not
  projections from logged data.

---

## [Unreleased] — Telegram bot: inline buttons, photo/PDF filing, generated /help, per-chat roles, typing indicator, voice notes (#191–#196)

No registry change. New tests: 104 in `tests/test_telegram_bot.py` (133 in the file) and 32 in
`tests/test_telegram_main_hooks.py`. Full suite compared with the original zip: 3270 → 3406 passed, and the same
16 failures and 15 collection errors in both (missing optional packages in this sandbox: chromadb, psycopg2,
vectorbt and others), so nothing regressed. `python-telegram-bot` is faked in `tests/conftest.py`, as before.

### Changed

- **Replies run in a worker thread** instead of on the bot's event loop, so one slow request (ingestion, a
  backtest) no longer blocks every other chat. Calls into `process_text` are serialised with a lock.
- Long replies are split at line breaks to fit Telegram's 4096-character limit instead of failing.
- **Fixed:** a failed download in `_handle_voice` raised `UnboundLocalError` (the `subprocess` import sat inside
  the `try`). The test that documented the bug now asserts the fix. Temp audio files are always removed.
- `/start` mentions `/help`.

### Added

- **Inline buttons (#191, partial).** Confirm / Cancel when the orchestrator is waiting on a yes/no; Snooze (10m,
  1h, configurable) / Done on pushed reminders. `telegram.push_notifications` (default off) mirrors `speak`
  events to owner chats, skipping them during do-not-disturb. Snooze calls Chronos directly.
- **Photo and PDF handling (#192, partial).** Photo → Iris (own folder under `data/telegram_inbox/`), PDF →
  Athena (subject "Telegram"), image-as-file → Iris, anything else declined. 20 MB limit.
- **`/help` (#193).** Generated from `INTENT_MODULE_MAP`, filtered by role; `/help <section>` for one module.
- **Per-chat roles (#194).** `telegram.roles` and `telegram.role_policies`. Fail-closed; compound messages are
  checked part by part; restricted chats can't share location, answer others' confirmations, or run
  voice-control commands; pushes go only to unrestricted chats.
- **Typing indicator (#195).** Refreshed while a request runs.
- **Voice notes (#196).** Transcript now goes through the same role check and buttons as typed text.
- `main.py`: `build_telegram_bot` takes optional hooks; `Hestia._classify_for_telegram`,
  `_has_pending_confirmation`, `_telegram_snooze`, `_telegram_ingest_photo`, `_telegram_ingest_document`.
- Config keys `telegram.roles`, `role_policies`, `push_notifications`, `snooze_minutes` (validated; documented in
  `laptop_config.example.yaml`).

### Not done / not verified

- **Not run against live Telegram.** The handlers, the Bot API push and the real `python-telegram-bot` 
  `CallbackQueryHandler`, `filters.Document.ALL` and `send_action` calls are tested with the library faked.
- The Iris and Athena hooks are tested against mocks, not real instances (chromadb and CLIP aren't installed here).
- Role scoping depends on the NLU classifying a message the same way before and after; a message whose
  classification changes between the check and the run is not re-checked.
- The pending-confirmation owner is tracked in the bot and resets if the bot restarts mid-confirmation.
- `config/laptop_config.yaml` (your real config) was not changed; copy the new keys from the example if wanted.
- #223 (alert on repeated module errors) and #256 (streak celebrations) could reuse `push()` but are not done.

---

## [Unreleased] — Pluto: forecast ranges, rebalancing, receipts, explain-holding, backtest sweeps, throttle visibility, score breakdown (#139–#145)

Registry version 2.22.0 (seven new `pluto_` intents, also in `config/nlu_prompt.txt` with examples and a
disambiguation rule, and a few phrases in `config/intent_aliases.yaml`). New tests: 201 in
`tests/test_pluto_extras.py`. Mutation testing was started on the new modules but not finished (it hit the
command time limit), so there is no mutation score for this batch.

### Changed

- **`rebalance_portfolio` is no longer an alias of `optimize_portfolio`.** It now compares holdings with
  targets you set. `optimize_portfolio` (max-Sharpe from past prices) is unchanged.
- A 429 from Yahoo Finance or CoinGecko is no longer retried; it starts a cooldown. Other failures retry as before.
- `analyze_asset` data and `generate_quant_score` now carry a `breakdown`.

### Added

- **Forecast ranges (#139).** An 80% range per day and for the total, from a walk-forward backtest. On
  synthetic data it covered about 90% of outcomes. Falls back to recent spending spread when history is short
  and says so. A failure computing the range never loses the forecast itself.
- **Rebalancing (#142).** `set_target_allocation`, `rebalance_portfolio`; by asset class or by holding; sell/buy
  amounts or a buy-only split of new money; refuses targets that don't add up or leave a held class uncovered.
- **Receipts (#141, partial).** `log_receipt` from a photo path via pytesseract. Saves only a labelled total.
- **Explain holding (#140, partial).** Position, price behaviour, headlines, fundamentals, optional model summary
  that is told to treat headlines as data. Each source fails independently.
- **Backtest sweep (#143, partial).** `backtest_sweep`, `/api/pluto/backtest/sweep`, and a Portfolio-tab panel
  with equity curves. Own simulator, not vectorbt.
- **Throttle visibility (#144).** `modules/pluto/throttle.py`, `data_source_status`, `/api/pluto/data-sources`,
  a Portfolio-tab line, per-session retry tuning via `market_data.configure_retry`. `retry()` gained
  `give_up_on` and `on_retry`. The portfolio optimiser stops asking once throttled.
- **Score breakdown (#145, partial).** `explain_quant_score`, `/api/pluto/quant-score`. The heuristic's parts add
  up exactly to the score; XGBoost contributions were checked to reproduce the model's output.
- Web routes: `/api/pluto/forecast`, `/backtest/sweep`, `/data-sources`, `/rebalance`, `/quant-score`, `/explain`.
- `PlutoDB.delete_setting`.

### Not done / not verified

- Not run against the live Yahoo news and quote-summary endpoints, SEC EDGAR, or real Tesseract; those parsers
  are tested on fixtures shaped like the documented responses.
- The sweep simulator has not been compared with a vectorbt run (vectorbt does not install on Python 3.12 here).
- The Portfolio-tab additions are syntax-checked, not viewed in a browser. No receipt upload button, no
  quant-score web view.
- `tests/test_pluto.py` still has the same 5 failures (vectorbt/pypfopt/Postgres) as before; they fail
  identically without these changes. `tests/test_backlog_gaps.py` has 6 failures from missing optional packages.

---

## [Unreleased] — Pluto: budgets, subscriptions, safe-to-spend, health score, scenarios, tax export (#133–#135, #137, #138, #266)

Registry version 2.21.0 (eight new `pluto_` intents, also added to `config/nlu_prompt.txt` with examples and
disambiguation rules). New tests: 252 in `tests/test_pluto_planning.py` and 15 Hypothesis tests in
`tests/test_pluto_planning_property.py` (267 total; whole suite 3,906 passing). Mutation testing
(`scripts/mutation_check.py`) on `planning.py` took the first score from 79% to 95% of 160 sampled mutants;
the 8 survivors are equivalent changes (ordering constants, a date bound that cannot matter, a last-day
calculation with the same result).

### Fixed

- **`log_expense` saved "nan" and "inf" as expenses.** `float("nan") <= 0` is False, so the old check let
  them through. Amounts now go through `parse_amount`, which rejects non-finite, zero, negative and
  absurd values and reads "₹1,500", "5k", "2 lakh".
- `track_investment` accepted negative or non-finite quantity and price; `convert_currency` accepted nan.
- `"1e999"` is refused rather than being read as 1.
- Found while testing the new code: a scenario return of "-5" was read as +5 and "-3" years as 3 (the
  number pattern ignored the sign); both now keep it, so a loss scenario shows a loss.

### Added

- **Budgets (#134).** "Set a food budget of 8000", "am I over budget?", "remove my food budget". The
  heartbeat speaks one alert per category per month at 80% and when over, held during quiet hours.
- **Subscriptions (#133).** Finds regular weekly, fortnightly, monthly, quarterly and yearly charges
  (three or more at a steady gap), flags price rises and ones that have stopped.
- **Safe to spend (#266).** Budget left per remaining day, after recurring bills still due this month.
- **Financial health (#135).** Savings rate, spending steadiness and diversification, each shown, with
  missing parts named. Needs "my monthly income is ...".
- **Scenarios (#138).** "What if I invest 10000 a month for 15 years at 12 percent" with a range.
- **Tax export (#137, partial).** Expenses and investments for a financial year to CSV, with a few hints.
- `modules/pluto/README.md`.
- Database: `budgets`, `settings` and `alerts_sent` tables, created in place (existing data untouched);
  `PlutoDB.log_expense` takes an optional timestamp; range queries by date.

### Not done

#131 broker sync, #132 price/news alerts, #136 multi-currency net worth, #139 forecast ranges, #140, #141,
#142 rebalancing, #143, #144, #145. See `modules/pluto/README.md` for known limits. The 5 failures in
`tests/test_pluto.py` that need vectorbt/pypfopt/a Postgres connection fail identically without these changes.

---

## [Unreleased] — Testing & QA: coverage, property tests, pipeline tests, contract test, mutation testing, smoke test, latency budget, replay (#207–#215)

Test suite 3317 -> 3558 passing (241 new). The 11 failures in this sandbox
(`test_iris_embeddings`, one Athena and one Mnemosyne test) are identical with
and without these changes: they come from stubbed `chromadb`. Not done: nothing
in section 20 is skipped, but #211 is partial (see below).

### Fixed (found by the new tests)

- **Negative numbers were logged as positive (Apollo).** "-30 min" became a
  30-minute workout, "-3 hours" a sleep of 3, "-5" a rating of 5 and a pain
  score of 5, because `_extract_number` dropped the sign. It now keeps the sign
  where it matters (`signed=True`) and still reads "7-8 hours" as a range.
- **A 400-digit number crashed `_parse_duration`** with `OverflowError` (it
  parsed to infinity). `_extract_number` now returns `None` for anything
  non-finite.
- **A `None` or non-numeric NLU confidence crashed `HecateEngine.decide()`.**
  It now means "unknown", the same 0.5 a missing key always meant.
- **Streaming TTS could stall on runs of punctuation (`core/tts.py`).** The
  sentence-boundary regex was quadratic: 20,000 characters of `?!` took 6.5 s
  to split. A lookbehind makes it linear; a Hypothesis test proves it matches
  exactly what the old pattern did.

### Added

- **Coverage with history (#207).** `python run_tests.py --coverage` prints
  total line coverage, the change since the last recorded run (a drop of more
  than one point is called out) and the least-covered files, and appends the run
  to `tests/coverage_history.csv` (commit it to keep the trend).
  `--cov-fail-under N` fails below N%; `--no-record` skips the log. Files no
  test imports count as 0% instead of vanishing. Needs `pip install coverage`.
  `scripts/coverage_history.py`.
- **Property-based tests (#208).** `tests/test_parsers_property.py`: Hypothesis
  tests over Apollo's duration, sleep, weight, water, goal, rating, date and step
  parsers (never raise, always in range, kg/lb limits agree, negatives rejected,
  every supported date format round-trips). Needs `hypothesis`; skipped without it.
- **Full-path integration tests (#209).** `tests/test_pipeline_integration.py`
  runs the real NLU, orchestrator and Hecate with only the model call scripted,
  and the real Apollo engine on a temporary database: stored values, retry on an
  invalid intent, parse failures, low-confidence clarification, a crashing
  module, circuit breaker, concurrent turns.
- **Registry/module contract test (#210).** `tests/test_intent_contract.py`
  reads every module's `*_INTENTS` set with `ast`, fails if one declares an
  intent the registry doesn't route to it, if the registry sends a module an
  intent its `can_handle()` rejects, or if the registry names a module that
  doesn't exist. Nine deliberate exceptions (legacy spellings, Mnemosyne's
  internal intents, two intents registered to core by design) are listed with
  reasons in `DOCUMENTED_ALIASES`; an entry that stops being true fails the test.
- **Mutation testing (#214).** `scripts/mutation_check.py FILE --tests ...`
  breaks a file one change at a time and reports which breakages no test
  noticed. Run on `modules/hecate/engine.py`, the existing tests caught 40% of
  93 mutants; the new `tests/test_hecate_routing_tiers.py` (71 tests, one per
  tier and boundary) brings that to 99%. The one survivor is provably
  unreachable. The file is always restored (backup, `finally`, and recovery
  after a killed run), and stale bytecode is cleared per mutant.
- **Smoke test (#212).** `python scripts/smoke_test.py` boots the real Hestia,
  sends ten read-only canonical queries and exits 1 if any reply is empty, an
  error message, a leaked traceback or too slow (2 if Hestia won't start).
  `--routing-only` resolves routing without running any handler; `--queries
  FILE` and `--json` are available.
- **Latency budget (#211, partial).** `tests/test_voice_latency.py` pins
  Hestia's own per-turn overhead (p95 budget, no slowdown over a long session,
  concurrent turns, a slow module not blocking others, sentence-splitting
  speed). `python scripts/voice_latency.py` times STT (with `--wav`), NLU,
  dispatch, TTS and the whole turn on a real install against
  `config/latency_budget.json` (created with starting values) and a recorded
  baseline (`--update-baseline`). Not done: it has never been run against real
  models, and the starting budgets are guesses.
- **NLU replay harness (#215).** `python scripts/replay_queries.py extract`
  builds an anonymised corpus from `logs/routing.jsonl` (emails, URLs, phone
  and long numbers, IPs, user folders and any `--name`); `run --save` /
  `run --against` compares two runs and exits 1 on any query that was right and
  is now wrong. Rows from the log are marked unverified: they hold the model's
  own past answers, so they detect change until you correct and verify them.
- **Athena format fixtures (#213).** Already satisfied by #67; recorded only.
- `requirements-dev.txt` (pytest, hypothesis, coverage); `.gitignore` entries
  for mutation backups and the machine-specific latency baseline.

---

## [Unreleased] — Hephaestus: page watching, form filling, site scrapers, repo summary, browser session (#101–#103, #105–#110)

Registry version 2.20.0 (minor: 6 intents added). New tests:
`tests/test_hephaestus_backlog.py`. Not done: #104 (breadboard photo check,
which belongs with Iris's vision stack). Everything below was tested against
fake browsers only; none of it has run against a real Playwright/Chromium or a
real website, so try the first watch and the first form on a page you control.

### Added

- **Page watching (#101, #109).** "Watch https://… and tell me if the price
  drops", "keep an eye on … for admit card, check every hour"
  (`watch_page`), plus `list_watches`, `stop_watching` and `check_watches`
  (check now). The page is read once when you ask, and a watch is only created
  if that works (and, for prices, if a price is found), so a typo can't become
  a watch that never fires. The heartbeat then re-checks each watch on its own
  interval (default daily, minimum 30 minutes, so effectively the 30-minute
  tick) and speaks only about meaningful changes: a keyword newly appearing, a
  price falling or reaching `target_price`, or, with neither set, at least
  three alphabetic words added or removed. Whitespace, digit-only churn and
  re-ordering never alert. Three failed checks in a row produce one "I
  couldn't read X" alert, not one per retry. Alerts raised between
  `hephaestus.monitors.quiet_hours` (default 22–7) are held and spoken
  afterwards. Watches and alerts live in `data/hephaestus.db`, so they survive
  restarts. Up to 20 watches; at most 10 pages are fetched per tick.
- **URL guard.** Watch and form URLs must be http(s) to a public-looking host:
  localhost, private/link-local addresses, `.local`/`.lan`/`.internal` names and
  URLs with a login embedded are refused. It checks the address as written and
  does not resolve DNS.
- **Form filling (#102, partial).** "Fill in my scholarship form"
  (`fill_form`) completes a form saved under `hephaestus.forms`. Selectors and
  values live in config; `{placeholders}` are filled from what you say or
  asked for. It always asks "say yes" first, naming the site and field count
  but never the values, and only submits if the saved form has a
  `submit_selector`. The browser agent no longer submits if any field failed to
  fill. You can't create a form by voice.
- **Site scrapers (#105).** `hephaestus.scrapers` (a URL pattern plus a CSS
  selector) and an optional `hephaestus.scraper_dir` of Python files that
  define `SCRAPERS`. `scrape_page` and watches use a matching scraper first and
  fall back to generic scraping. None are bundled. The scraper directory is
  executed as code, like `skills/`.
- **Politeness delay (#106).** `hephaestus.min_request_interval_seconds`
  (default 1.0 from config) spaces requests to the same site, capped at 10 s
  per request.
- **Failure screenshots (#103).** `browser.screenshot_dir` saves a full-page
  screenshot when a page action fails; newest 20 kept. Off by default.
- **Browser session (#107, partial).** One browser, one shared context for all
  pages (previously a new context per page was created and never closed), and
  `browser.idle_timeout_seconds` to close an idle browser. Cookies and consent
  choices now persist between tasks until it closes. Not a pool.
- **`--headed` (#108).** Shows the browser and adds 250 ms between actions.
- **Repo summary (#110, partial).** "Summarize the code in C:\projects\hestia"
  (`summarize_repo`): size by language, where the code lives, entry points,
  manifests, test-file ratio, very long functions and files, TODO/FIXME
  counts, Python files that don't parse, bare `except`. Deterministic, no LLM,
  names and counts only, never file contents; skips dependency and build
  folders, doesn't follow symlinks, and caps what it reads.
  `hephaestus.repo_roots` can restrict which folders it may scan.

### Changed

- **`browser.enabled` and `browser.headless` are now honoured.** They were in
  the example config but never read, so the browser was always headless and
  always on. A config with `browser.enabled: false` now really disables it.
- `HestiaBrowserAgent` gained keyword-only options (`screenshot_dir`,
  `slow_mo_ms`, `idle_timeout_seconds`) and `fetch_text`, `fetch_elements`
  (return `None` on failure so a monitor can't mistake an error message for the
  page changing) and `close_if_idle`. Existing calls are unchanged.
- `HephaestusEngine` takes optional keyword arguments (`monitor_store`,
  `monitor_browser`, `scrapers`, `forms`, `repo_roots`, `min_host_interval`,
  `quiet_hours`); the positional signature is unchanged, and an engine built
  without them behaves as before. `list_watches`, `stop_watching` and
  `summarize_repo` work even if the browser is unavailable.
- Monitors use a **separate headless browser** and one dedicated worker thread.
  Playwright's sync API is tied to the thread that started it, and the
  heartbeat runs on its own thread, so sharing the chat browser would have
  failed on whichever thread didn't launch it. The monitor browser is closed
  after each batch. This follows from how Playwright is documented to work and
  is untested against real Playwright.
- `HestiaHeartbeat` takes an optional `hephaestus` and calls its
  `check_web_monitors()` every tick. `Hestia._shutdown` now closes the browsers.
- Config validation knows the new `browser.*` and `hephaestus.*` keys. See
  `config/laptop_config.example.yaml`; every key is optional.
- `god_function.md`'s Hephaestus line was out of date (it said it couldn't
  launch desktop apps) and now lists every intent.

---

## [Unreleased] — Hermes: email digest, drafting, search, schedule gaps, meeting slots, conflicts, recurring events, inbox-zero plan (#92–#100)

Registry version 2.19.0 (minor: 6 intents added). New tests:
`tests/test_hermes_backlog.py`. All new settings are optional (`hermes:` in
the config); an existing config behaves as before. Not done: #91 (Todoist).
Everything below was tested against a fake Google agent only, not the live
Gmail/Calendar APIs.

### Added

- **Email digest (#92, partial).** "How urgent is my inbox", "email digest"
  (`email_digest`). Unread mail is scored from sender, subject and snippet
  (urgent wording, questions, newsletters/no-reply senders, promotions,
  receipts, plus an optional `hermes.vip_senders` list) and summarised as
  "N need attention, M routine, K newsletters", leading with the most urgent.
  It's a keyword heuristic and it runs on request; it isn't scheduled into the
  heartbeat.
- **Draft from an instruction (#93).** "Draft an email to Priya saying I can't
  make it, suggest Thursday instead" (`draft_email`). Wording comes from the
  LLM when one is available (now passed to Hermes from `main.py`), otherwise a
  template that handles decline, running late, thanks and follow-up and
  otherwise restates your instruction. The draft goes through the normal
  send confirmation, so nothing is sent without a "yes". With no recipient it
  shows the draft and asks who to send it to, keeping the same draft.
- **Email search (#94).** "Find emails from Raj about the invoice last week"
  (`search_email`): sender, subject, date (`yesterday`, `last week`, a date) or
  keywords, over the whole mailbox rather than just unread. Backed by a new
  `HestiaGoogleAgent.search_emails`.
- **Schedule gaps (#95, partial).** "Any back-to-back meetings tomorrow?"
  (`check_schedule_gaps`) flags overlaps and gaps under `hermes.buffer_minutes`
  (default 10). When consecutive events have different locations it adds a flat
  `hermes.travel_minutes` (default 30) — an estimate, not a maps lookup, and
  the reply says so. All-day events are ignored.
- **Meeting slots (#96, partial).** "Find a time for a 45 minute meeting with
  sam@example.com tomorrow" (`find_meeting_slot`) offers up to three
  non-overlapping slots inside `hermes.work_hours` (default 9–18, weekdays
  unless you name a day) using Google free/busy. If an attendee's calendar
  isn't visible, or a name has no configured address, it says so rather than
  treating them as free. It proposes times; it doesn't create the event or
  send invites. New `HestiaGoogleAgent.free_busy` and `list_events_between`.
- **Conflict check on create (#97).** Creating an event that overlaps a timed
  event now says what it clashes with and asks before adding it (same
  confirm-then-execute path as `send_email`). A failed lookup never blocks the
  create. Adjacent events and all-day events aren't conflicts.
- **Recurring events (#98).** `create_event` accepts a repeat phrase —
  "every weekday", "weekly", "every Monday and Wednesday", "every other week",
  monthly, yearly — with optional `count` or end date, and reads it from the
  raw query if the NLU doesn't pass it. A phrase it can't read is a question,
  not a silently-created one-off. Also accepts `duration`. Only the first
  occurrence is checked for conflicts.
- **Inbox-zero plan (#99, partial).** "Help me get to inbox zero"
  (`inbox_zero`) suggests read / reply / snooze / archive for each unread
  message. It suggests only: archiving would need Gmail's modify scope, which
  isn't requested, so nothing in the mailbox changes.
- **Recipient check before send (#100).** The confirm-before-send step already
  existed. `send_email` now also refuses a recipient that is neither an address
  nor a name in `hermes.contacts` (e.g. a misheard "John") and asks for the
  address, which slot-fills back into the send.

### Changed

- `HermesEngine` takes optional keyword arguments (`llm`, `contacts`,
  `vip_senders`, `work_hours`, `buffer_minutes`, `travel_minutes`); the
  positional signature is unchanged.
- `HestiaGoogleAgent.create_event` takes an optional `recurrence` list.
  `read_emails` shares its fetch code with the new search; behaviour is unchanged.
- `config/nlu_prompt.txt` lists the six new intents with examples and routing
  rules. `config/laptop_config.example.yaml` documents the `hermes:` section;
  your live `config/laptop_config.yaml` was not changed.

---

## [Unreleased] — Voice pipeline: voices per module, repeat that, do-not-disturb, mic calibration, wake-word sensitivity, listening indicator, graceful fallback (#170, #171, #172, #173, #174, #175 partial, #176, #178, #179)

New tests: `tests/test_voice_pipeline.py`. All new settings are optional; an
existing config behaves exactly as before. Not done: #177 (speaker
identification).

### Added

- **Per-module voices (#170).** `tts.voices` defines named profiles (any of
  `voice_name`, `piper_model_path`, `rate`, `volume`); `tts.voice_by_module`
  says which module speaks in which, with an optional `default`. Spoken
  replies use the module that handled the query; proactive announcements use
  the module that raised them (or an explicit `voice` in the `speak` event).
  A profile naming a voice or Piper model that doesn't exist keeps the rest of
  its settings and falls back to the base voice with a warning.
- **"Repeat that" (#171).** "Repeat that", "say it again", "what did you say"
  replay the last reply or announcement. The whole utterance must be the
  command, so "can you repeat that recipe for pancakes" goes to the normal
  pipeline. Acknowledgements ("Yes?", "I didn't catch that") and the repeat
  itself don't overwrite what gets repeated.
- **Do-not-disturb (#173).** "Do not disturb [for 30 minutes / an hour]",
  "mute notifications", "resume notifications", "is do not disturb on".
  Proactive announcements (the `speak` bus event: reminders, nudges, the
  morning brief) are held and read out together when DND ends; replies to
  your own questions are never held. Timed DND expires without a timer
  thread. At most 50 notifications are held (oldest dropped). A duration
  that can't be parsed turns DND on open-ended and says so rather than
  guessing. Also `POST /api/voice/dnd`.
- **Mic calibration (#174).** `python main.py --calibrate-mic` measures room
  noise, your voice and (if TTS starts) Hestia's own echo, then saves
  recommended `barge_in.min_rms`, VAD aggressiveness and wake-word
  sensitivity to `data/mic_calibration.json`. Used at startup only for
  settings your config doesn't set, and ignored if it was made on a
  different input device. `barge_in.use_calibration: false` turns it off.
  `stt.vad_aggressiveness` is a new setting.
- **Echo cancellation (#175, partial).** `core/echo_cancel.py`: an NLMS
  adaptive filter that subtracts Hestia's playback from the mic before
  barge-in's VAD and RMS checks, freezing adaptation during double-talk.
  Opt-in via `barge_in.echo_cancel.enabled`.
- **Wake-word sensitivity (#176).** `wake_word.sensitivity`: `quiet` (also
  accepts near-miss spellings such as "hesta"/"hestiya", compared word by
  word), `normal` (the previous exact-match behaviour, default) and `noisy`
  (ignores long utterances and low-confidence matches). Switch live with "I'm
  in a noisy room" / "I'm in a quiet room".
- **Listening indicator (#178).** `GET /api/voice/state` and a status line
  above the chat box showing whether the server's mic is open (waiting for
  the wake word, listening, thinking, speaking, or typed-only) plus a DND
  badge. `core/voice_state.py` holds the shared state.
- **Typed fallback (#179).** Each voice component now builds independently:
  a missing Vosk model, STT load failure or absent audio stack disables only
  that part (`NullTTS` stands in for speech) and `--voice` carries on with
  typed input, saying why. Three consecutive mic errors during a session
  also fall back to typing instead of looping.

### Changed

- **Streaming TTS splitter (#172).** `speak_stream` already existed; its
  sentence splitting is now abbreviation-aware ("Dr.", "e.g.", initials,
  decimals), speaks line-separated lists as they arrive, and flushes a long
  unpunctuated first sentence at a clause break so speech starts sooner.
- A failure while speaking one utterance no longer kills the TTS worker
  thread (it previously left `wait_until_done()` able to hang).

### Not done / limits

- **#177 speaker identification** not started.
- **Echo cancellation is unproven on real hardware.** Tests use synthetic
  echo only. It needs `tts.engine: piper` (pyttsx3 plays through the OS, so
  there's nothing to cancel against; it's left off with a warning). If it
  doesn't help, raise `delay_ms`, or use a headset.
- "Repeat that" after a barge-in replays the whole reply, not just the part
  that was cut off.
- The web UI indicator and DND badge were checked through the JSON endpoints
  and for JS syntax only, not in a browser.
- `config/laptop_config.yaml` (your live config) was not changed; the new
  options are documented in `config/laptop_config.example.yaml`.

---

## [Unreleased] — Dionysus: events, surprise me, budgets, recharge routines, more/less like this, expiring dismissals (#146, #147, #150, #151, #152, #269)

Registry version 2.18.0 (minor: 5 intents added). New tests:
`tests/test_dionysus_backlog.py`. Dionysus's SQLite file gains two columns on
`recommendations` (`dismissed_at`, `feedback`) and a `recharge_routines` table;
existing databases are upgraded in place on first start and keep their rows.

### Added

- **Event finder (#146).** "Any live music events this weekend?" / "what's
  happening in Bandra on Saturday" (`dionysus_find_events`). Runs two web
  searches (plain and BookMyShow/Insider-flavoured), drops duplicates and
  anything you dismissed or marked seen, and shows up to 5. "Near me" uses
  your saved location. The results are search hits, not verified listings, and
  the reply says so. Each one is logged as type `event`, so dismiss, "seen" and
  more/less-like-this work on events too.
- **Dismissals expire (#147).** A dismissed title can come back after
  `dionysus.dismiss_expire_days` (default 180; 0 turns expiry off). Dismissals
  already in the database are dated the day of the upgrade, so none expire at
  once. Titles you marked as watched and titles you gave "less like this" never
  expire. Dismissals already survived restarts; that part needed no change.
- **Surprise me (#150).** "Surprise me with a movie" / "...with some music"
  (`dionysus_surprise_me`; movies unless you say music). The prompt lists your
  recent picks and asks for a different genre, era or country. It does not use
  your logged mood or your "liked" titles, and still excludes dismissed, seen
  and disliked ones.
- **Budgets and cost (#151).** Give a `budget` ("under 1500", "2k for two",
  "cheap", "mid-range", "fine dining") to `plan_outing` or `find_restaurant`.
  Outings: the model adds a rough `cost_per_person` to each slot, and the total
  is added up in code (ranges use the upper end). The reply shows the estimate
  and, if it is over budget, by how much. Restaurants: results that state a
  price over your budget are flagged, and a line explains what the budget word
  usually means. The word-to-range table (`_BUDGET_TIERS`: cheap up to 500,
  mid 500 to 1500, premium 1500 and up, per person in rupees) is a rough
  assumption, not live data.
- **Recharge routine (#152).** "Schedule a weekend recharge, every Saturday at
  5pm for 2 hours" (`dionysus_schedule_recharge`). Creates a repeating Chronos
  reminder ("take your recharge break (2 hours): switch off work and do
  something just for you"). With no day given it uses "every weekend at 4pm"
  if you said weekend, otherwise `every Sunday at 4pm`; default length 2 hours.
  Asking for the same schedule twice does not create a second reminder. Cancel
  it with "cancel reminder recharge". Needs Chronos with its database; without
  it the reply says so.
- **More / less like this (#269).** "More like that" / "I loved Interstellar,
  more like it" (`dionysus_more_like_this`) records a +1 and, for movies and
  music, immediately fetches similar picks. "Less like this"
  (`dionysus_less_like_this`) records a -1 and dismisses the title. With no
  title, or "that"/"this one", it refers to the most recent recommendation.
  Your liked and disliked titles are added to every later movie and music
  prompt (lean toward / avoid similar), and neither kind is recommended back.
  A title Dionysus never suggested is accepted and remembered as a taste signal.
- Phrase aliases (no LLM call) for "more/less like this|that", "surprise me" and
  "schedule a recharge". NLU prompt examples and routing notes for all five
  intents.
- Config: `dionysus.dismiss_expire_days` (validated, in
  `laptop_config.example.yaml`). `main.py` hands Chronos to Dionysus
  (`attach_chronos`) next to the existing Chronos/Dionysus wiring.

### Changed

- `plan_outing` and `find_restaurant` replies are unchanged unless a budget or
  cost estimate applies. `DionysusDB.dismissed_titles()` takes an optional
  `expire_days`.

### Not done

- #148 (group outing coordination) is conditional on multi-user support, which
  doesn't exist.
- #269 is chat commands only. There are no buttons in the web UI, which has no
  Dionysus view to put them in.
- #151 restaurants: search results are titles only, so there is no
  per-restaurant price estimate. Outing costs are the model's estimates.
- #152 is a repeating reminder, not a blocked-out calendar slot; Chronos has no
  event-with-duration concept, so the length lives in the reminder text.
- #243 (focus mode) and #272 (unified inbox) are cross-module and were left
  alone.

---

## [Unreleased] — Ares: career ranking, review reminders, outcome tracking, Monte Carlo, playbooks (#153-#157)

Registry version 2.17.0 (minor: 9 intents added). New tests:
`tests/test_ares_backlog.py`. New files: `modules/ares/db.py`,
`modules/ares/simulate.py`. Ares now keeps a small SQLite file
(`data/ares/ares.db`, or `ares.db_path`) for tracked decisions and playbooks.

### Added

- **Career / GATE ranking (#153).** "Rank my GATE options: M.Tech, PSU, private
  job" (`ares_career_ranking`). The model scores each option 1 to 10 on each
  criterion; the weighted total and the ranking are computed in code, not by
  the model. GATE mode (detected from the wording, or `mode: gate`) swaps in
  GATE-specific criteria. Name your own criteria with optional weights
  ("salary:5, location:2"). The prompt forbids inventing cutoffs, salaries or
  seat counts: anything it would need to look up is listed under "verify
  before deciding".
- **Revisit reminders (#154).** "Remind me to revisit my job decision in 2
  weeks" (`ares_schedule_review`; "tomorrow", "next month", "in three weeks"
  also work, default 14 days, max 2 years) creates a Mnemosyne reminder and
  links it to the saved analysis. Without a saved analysis the reminder is
  still set and the reply says so. Optional `ares.auto_review_days` does this
  automatically for every plan, decision and career ranking (off by default).
- **Outcome tracking (#155).** Every strategic plan, decision and career
  ranking is saved automatically. "The Pune decision worked out well" /
  "the startup plan failed" (`ares_record_outcome`: worked, mixed or failed)
  logs how it went; "how have my past decisions turned out"
  (`ares_outcome_stats`) shows counts, success rate, decisions due for review
  and recent entries. Once 3 or more resolved decisions had a confidence
  score, new decisions and career rankings get a TRACK RECORD line with a
  calibrated confidence next to the raw one. The adjustment shrinks toward
  zero for small samples and is clamped to 0.05 to 0.95. Outcomes logged for
  something Ares never analysed are kept but excluded from calibration.
- **Monte Carlo simulator (#156).** "Simulate: 60% chance of gaining 10 lakh,
  40% chance of losing 2 lakh" or "between 50k and 200k, most likely 100k"
  (`ares_simulate_outcomes`). Pure Python, no LLM, so no invented numbers.
  Reports expected value, median, 5th to 95th percentile, worst/best and
  chance of loss; several options (structured entities) are ranked with a
  head-to-head win rate. Understands k, m, lakh and crore, and a fixed cost.
  Probabilities that don't add up are normalised or padded with a "nothing
  happens" outcome, and the reply says which.
- **Playbooks (#157).** "Save a playbook called job offer that runs a
  premortem and always considers salary, growth and commute"
  (`ares_save_playbook`), then "run my job offer playbook on the Bangalore
  role" (`ares_run_playbook`), plus `ares_list_playbooks` and
  `ares_delete_playbook`. A playbook stores the analysis type, standing
  criteria and optional default topic/options; its criteria are injected into
  the analysis prompt for that run only.
- Config: `ares.auto_review_days`, `ares.db_path` (both optional; see
  `laptop_config.example.yaml`).

### Changed

- `decision_support` and `strategic_plan` now save what they produce (the
  reply text is unchanged unless a review reminder or track-record note
  applies).
- Option extraction for decisions moved into a shared helper; it also accepts
  a list for `options`.

---

## [Unreleased] — Artemis: habit controls, focus timer, milestones, badges, nudges (#122-#125, #127-#130)

Registry version 2.16.0 (minor: 12 intents added). New tests:
`tests/test_artemis_habit_controls.py`, `tests/test_artemis_extras.py`.

### Added

- **Grace periods (#123).** A habit's streak can survive missed days.
  "Give my reading habit a grace period of 2 days" (`set_habit_grace`, 0 to 7,
  "turn off the grace period" sets it back to strict). The default for habits
  with no setting of their own is `artemis.habit_grace_days` (default 0, so
  nothing changes until you opt in). When a grace period saves a streak, the
  completion reply says how many days were missed.
- **Pause / resume (#128).** "Pause my running habit for two weeks" or "pause
  meditation" (`pause_habit`, 1 to 365 days, open-ended if no length is
  given); "resume running" (`resume_habit`). Paused days never count as missed,
  so the streak survives a holiday; completing a paused habit resumes it.
  Missed days after the pause ends still break the streak. Habit names are
  matched loosely ("running" finds "morning run").
- **Weekly habit review (#125).** "How consistent have I been this week?"
  (`weekly_habit_review`): consistency over the last 7 days per habit and
  overall, the weakest habit, and large moves against the week before. Paused
  days, days before a habit's history begins, and today (until it's done) are
  left out of the denominator rather than counted as misses. Habits that were
  completed before history recording began are reported as "not enough
  history" instead of guessed at.
- `list_habits` now shows `paused` and the grace period; `habit_history()`
  includes `paused`.

- **Focus sessions / Pomodoro (#122).** "Start a 25 minute focus session on my
  thesis" (`start_focus`, 1 to 180 minutes, default 25), "stop the pomodoro"
  (`stop_focus`), "how much have I focused this week" (`focus_stats`). A
  timer in the engine announces the end and suggests a 5-minute break, or a
  15-minute one after every fourth finished session in a day. Stopping early
  logs the minutes that actually elapsed; minutes never exceed the planned
  length. Focus time appears in the productivity summary. A session that
  ran out while the app was closed is logged as finished when you next start
  or check one; the spoken alert is lost if Hestia restarts mid-session.
- **Goal decomposition (#124).** "Break down my goal to publish the paper into
  steps" (`decompose_goal`) asks the LLM for 4 to 7 milestones, creates the
  goal if it doesn't exist, and stores the steps on it. "I finished step 2 of
  <goal>" or "tick off the next step" (`complete_milestone`) ticks one; goal
  progress follows the steps and the goal completes with the last. If the LLM
  returns nothing usable, nothing is created and you're told so (no invented
  steps).
- **Goal templates (#130).** "What goal templates do you have"
  (`list_goal_templates`) and "start the run a 5k template"
  (`add_goal_from_template`): 8 templates (5K, read a book, learn a language,
  emergency fund, research paper, lose weight, side project, declutter), each
  with milestones, a due date and a priority.
- **Badges (#127).** 11 badges earned from streaks (7/30/100/365 days),
  completions (50/250), tracking 3 habits, completed goals (1/5) and focus
  time (10/50 hours). A new one is announced in the reply that earns it, once;
  "show my badges" (`list_badges`) lists them with dates. Badges are awarded
  from existing data the first time they're checked, so an existing streak
  earns its badge at your next completion.
- **Smart nudges (#129).** Each live habit completion records its time of day.
  Once a habit has 5 recorded times, the heartbeat nudges you if it's an hour
  past your usual (median) time and it isn't logged: one nudge per habit per
  day, the longest streak first, none after 22:00, none for paused habits.
  Config: `artemis.nudges.{enabled, lateness_minutes}` and `artemis.timezone`
  (falls back to `chronos.timezone`). The heartbeat ticks every 30 minutes, so
  a nudge can arrive up to that much after the threshold. Back-dated
  completions don't record a time.

### Changed

- **Consensus (#159).** An Artemis "push" signal now fires only when skipping
  today would actually break the streak, so a paused habit or one with grace
  days left no longer counts as at risk.
- State file: habits gain optional `grace_days` and `pauses` keys, written only
  when set, so untouched files keep their exact old shape.

### Not done

- #121 (heatmap) was already built under #183 and is still only partly
  checked: nothing here was looked at in a browser.
- Streak-milestone notifications (#256) are separate (the badge announcement
  fires on the completing reply, not as a push). The weekly review is
  on-request, not a scheduled Heartbeat prompt.

---

## Earlier — Athena: citation graph (#60)

Closes Athena's last open item: #60. Registry version 2.14.0 (minor: 1 intent
added). New tests: `tests/test_athena_citation_graph.py` (33).

### Added

- **Citation graph (#60).** "Show the citation graph", "who cites <paper>",
  "what does <paper> cite" (`athena_citation_graph`) draws which of your
  indexed papers cite which of your other indexed papers. Arrows run from the
  citing paper to the cited one; bigger nodes are cited more often; dashed
  arrows are less certain matches; click an arrow to see the reference entry
  that produced it. Force layout or by-year timeline. Writes a self-contained
  `.html` (no CDN, works offline) and `.json` (add "as dot" for Graphviz) to
  `data/athena/exports/`.
- `modules/athena/bibliography.py` (reference-list parsing, document identity,
  matching), `modules/athena/services/citation_graph_service.py` (builds the
  graph, caches each file's parse in `data/athena/cache/citation_parse.json`),
  `modules/athena/citation_graph_view.py` (HTML / DOT / JSON).
- **Web.** `GET /api/athena/citation-graph[?subject=]` (JSON) and
  `/api/athena/citation-graph/view` (the page, served with a locked-down CSP);
  "Show citation graph" on the Athena page.
- `MergedLocalRAG.list_document_sources()`: indexed files with their paths.

### How links are decided

- A link needs evidence: an exact DOI, an exact arXiv id, or the other
  document's own title inside the reference entry. Author + year alone never
  makes a link.
- A reference that two of your documents match equally well makes no link and
  is counted in the notes. A document with a title under three words can only be
  matched by DOI or arXiv id.
- A link whose cited paper is dated after the citing paper is kept but flagged.

### Known limits

- Reference lists are read from the original files, so a moved or deleted file
  shows as "missing". Scanned PDFs and files without a "References" heading
  can be cited but cannot cite anything.
- Only PDF, Word, Markdown, text and ePub text are read; slides are not.
- Tables of references drawn without a recognisable heading, and heavily
  two-column PDFs whose lines interleave, may parse poorly. The first build over
  a large library reads every PDF once; later builds only re-read changed files.
- Citations to other subjects are missing from a subject-limited graph.
- Not verified against a real chromadb or against real-world PDF collections:
  tests use generated PDFs. Try it on your own library and report parse failures.

---

## [Unreleased] — Wiring the "done but unreachable" gaps

An audit of the ✅ marks found ten items whose engine method existed and had
tests, but that no person could reach. All ten are now wired: #36, #38, #40,
#44, #45, #48, #49, #63, #64, #74. Registry version 2.13.0 (minor: 5 intents
added). New tests: `tests/test_backlog_gaps.py` (52).

### Added

- **Intents** `recall_on_date`, `get_memory_stats`, `export_memory`,
  `set_fact_importance`, `review_stale_facts` (registry, NLU prompt examples,
  pre-LLM aliases). `modules/mnemosyne/dates.py` resolves "last Tuesday",
  "3 March", "last week" to a local calendar range; `export.py` is the shared
  export builder; `scripts/export_memory.py` backs up without starting Hestia.
- **Heartbeat jobs.** `run_background_jobs` now runs the daily decay check and
  the weekly/monthly digest; the morning brief speaks the stale-fact count.
- **Web.** `GET /api/mnemosyne/export`, `GET /api/mnemosyne/dashboard`,
  `POST /api/athena/feedback`, and a `debug` flag on `/api/athena/query`.
- **Iris search** understands "from March 2024", "shot on my iPhone",
  "geotagged".

### Fixed

- `export_memory` was capped at 1,000 facts.
- A kept stale fact was re-flagged on the next decay run.
- `get_user_info` cleared a fact's stale flag before reading it, so the
  "may be out of date" warning could never appear.
- Athena `mark_feedback` required `page_number` while search results say
  `page`, so a source dict could not be passed straight back.
- Athena `get_citations` said it used the latest search but never stored one.
- The periodic digest took the last N summaries (which could include earlier
  digests) instead of a time window.

### Not changed

- Searching photos by place name ("taken in Paris") still needs reverse
  geocoding.
- The 13 tests that need a real chromadb/sentence-transformers still fail in a
  stubbed environment; run them on your machine.

---

## [Unreleased] — Athena: ingestion, PDF structure, generated files

Eight of Athena's nine open items: #51, #52, #53, #58, #59, #66, #68, #70.
#60 (citation graph) is not built. Registry version 2.12.0 (minor: 2 intents
added). New tests: `tests/test_athena_progress_ocr.py`,
`tests/test_athena_pdf_structures.py`, `tests/test_athena_generation.py`.

### Added

- **Ingestion progress (#70).** `progress.py`; `POST /api/athena/ingest`
  starts indexing on a worker thread, `GET /api/athena/ingest-status` reports
  files done, current file, ETA and failures. The Athena page has an
  "Index documents" button and a progress bar.
- **OCR language detection (#68).** `ocr_language.py` detects each scanned
  document's language once and uses the matching Tesseract pack only if it is
  installed; otherwise English. Switch: `ocr_auto_language`.
- **Tables and figure captions (#58, #59).** PDF tables become labelled-row
  chunks and "Figure N:" captions become their own chunks, both with page
  numbers. "Find the graph that shows X" and "which table lists Y" are
  answered from them. Image pixels are not extracted. PDFs indexed earlier
  are re-processed once to gain these chunks.
- **Generated files (#53, #51, #52, #66).** `generation.py` builds one report
  model and renders it as PDF (reportlab), LaTeX (`.tex` + `.bib`) or
  PowerPoint (python-pptx). New intents `athena_generate_report` and
  `athena_methodology` (registry entry, NLU prompt entry and examples,
  alias phrases, tests). Files go to `data/athena/exports/`. When the model is
  down or returns junk, no file is written and the reply says why.

### Known limits

- References are file-based, as with the existing citation manager: no author
  or year is invented.
- The LaTeX output is a scaffold to edit.

---

## [Unreleased] — Mnemosyne: study, graph, episodes, vault, papers

Finishes section 3 apart from the IEEE half of #47: #31-#35, #39, #43 and
the arXiv half of #47. Registry version 2.11.0 (minor: 9 intents added).
New tests: `tests/test_mnemosyne_study_graph.py`,
`tests/test_mnemosyne_quiz_flow.py`,
`tests/test_mnemosyne_episodes_obsidian_papers.py`,
`tests/test_mnemosyne_web_heartbeat.py`,
`tests/test_mnemosyne_orchestrated_flows.py`.

### Added

- **Quizzes and the weak-spot map (#34, #35).** `start_quiz`, `answer_quiz`
  and `quiz_performance` drive the existing quiz engine as a multi-turn
  conversation on the slot-fill mechanism (#29). Spoken answers are parsed
  ("B", "bee", "the second one", or the choice text); an unclear answer is
  asked again rather than guessed, so a misheard reply never corrupts the
  stats. Questions come from Athena documents, then Obsidian notes, then
  facts. The map ranks subjects weakest-first and reports an improving or
  declining trend once there are 10 attempts.
- **Spaced repetition (#33).** SM-2 scheduling for facts tagged as study
  material (`add_study_fact`, `review_study`). Recall is auto-graded from
  your answer and the correct answer is always shown. The morning brief
  says how many cards are due. Forgetting a fact removes its card.
- **Knowledge graph (#31, #32).** Facts become relations as you learn them
  ("sister name" links You to Priya). `graph_connections` answers "what
  connects to X" and "how is X related to Y". Every edge remembers its
  source, so forgetting a fact removes exactly what only it contributed. A
  new Graph page in the dashboard draws it with D3.
- **Episodes (#43).** Related interactions are grouped into episodes
  (hourly, incremental). `recall_episode` answers "what were we doing about
  X".
- **Obsidian sync (#39).** Off by default. Wikilink-aware, heading-aware
  chunks; incremental by content hash; deleted notes are removed. Write-back
  is a second switch and only creates new notes in a `Hestia/` subfolder.
- **arXiv monitoring (#47, arXiv only).** `watch_papers` saves topics; a
  daily check summarises new papers with the local model and writes them
  into Athena's documents folder. IEEE is not implemented (needs an API key).

### Changed

- Quiz choices are shuffled after validation (small models put the answer
  at "A" most of the time) and `correct_index` is remapped.
- The heartbeat calls `run_background_jobs` each tick; each job keeps its own
  cadence and failure isolation.

## [Unreleased] — Apollo (health) and its cross-module items

Completes section 9 (#111–#120) and the six items that depend on Apollo's
data (#126, #149, #159, #161, #183, #233). Registry version 2.10.0 (minor: 14
intents added). New tests: `tests/test_apollo_insights.py`,
`tests/test_apollo_features.py`, `tests/test_apollo_crossmodule.py`.

### Added

- **Schema migrations.** `apollo.db` now upgrades itself via
  `PRAGMA user_version`. Each migration runs once, in a transaction, and is
  idempotent, so old databases keep their rows and a half-applied run is
  harmless.
- **Sleep quality score (#111).** 0-100 from duration, bed/wake consistency
  over the last 7 nights (correct across midnight) and your 1-5 rating or
  quality word. Say "slept 11pm to 6:30am" to add times. Missing pieces are
  named in the reply rather than scored around silently. New intent
  `get_sleep_quality`.
- **Correlations (#112).** `get_correlations`: mood after short (<6h) vs long
  (7h+) sleep, and mood on workout vs non-workout days. Pure statistics, no
  LLM. It reports n, refuses below `apollo.min_sample` (default 4) per
  group, and says "in your logs", not "causes".
- **Units (#113).** `set_units` (kg/lb, ml/oz, or "metric"/"imperial").
  A unit typed in a message wins. Storage stays kg/ml. Weight and water goals
  are read in your unit.
- **Meals (#114).** `log_meal` and `get_meal_summary`. Calories come from
  your number ("as you entered it") or an Open Food Facts lookup labelled as
  a rough estimate (an assumed 100 g is disclosed). Daily kcal and protein
  only. No target is shown unless you set a `calories` goal, and Apollo
  refuses to set one below 1,200 kcal.
- **Hydration pacing (#115).** `hydration_status` on demand, plus heartbeat
  nudges only when you are behind pace by `threshold_ml` inside
  `wake_start`..`wake_end`, with a daily cap and cooldown. Only for people who
  use water tracking. Nudge state lives in the DB.
- **Workout streaks (#116).** `get_workout_streaks`: consecutive days and
  consecutive weeks with at least N sessions, per exercise type.
- **Pain / injury (#117).** `log_pain` and `get_pain_trend`: severity 0-10 by
  body area, a 7-day vs previous-7-day trend once there is enough data, and
  template-only wording. Red-flag phrases, pain at 9+/10, pain logged for
  three weeks or more, or a worsening trend at 6+/10 produce a "see a
  clinician" message. It never diagnoses.
- **Weekly summary (#118).** `get_weekly_summary`, and sent once per ISO week
  through the heartbeat (on/after Sunday 18:00 by default, so a machine that
  was off on Sunday still sends). Deterministic text, and the sent marker is
  stored in the DB.
- **Step import (#119, partial).** `import_steps` reads CSV/JSON from
  `apollo.import_dir` only (basename resolved inside that folder, symlinks out
  of it ignored). Layout: one record per row with a date column (`date`,
  `day`, `start_date`, `timestamp`, ...) and a steps column (`steps`,
  `step_count`, `count`, `value`, ...), matched case- and punctuation-
  insensitively. Several records on one day are summed, and re-importing is
  idempotent (`upserted` by date). Untested against real exports.
- **Goal pace (#120).** `get_goal_pace` for a weight goal: distance, trend
  (least-squares over 28 days, needs a week of spread), ETA, and on/behind pace
  for an optional deadline (`by 2026-12-31`, `in 8 weeks`). A change faster than
  about 1 kg/week is flagged, not cheered, and so is a deadline that would
  require it. Check-in reminders are off unless `apollo.goal_reminders` is
  enabled.
- **Habit vs mood (#126).** Artemis habits now record a capped (400 days)
  list of completion dates, and `habit_mood_correlation` compares habits
  kept on high- vs low-mood days. Old state files load unchanged; days before
  recording began are unknown, not missed, so correlations only cover days
  logged after this ships.
- **Mood-aware Dionysus (#149).** Movie and music picks use your logged mood
  (last 48h) only when you gave none and the request is generic, and say so.
  Persistently low moods lean comforting. `dionysus.mood_aware: false`
  turns it off.
- **Burnout signals (#161).** `burnout_check` combines sleep and mood
  (Apollo), habit consistency (Artemis) and a weekly-spend spike (Pluto, a weak
  signal that can't raise a flag alone). It needs two sources with data and
  reports low / watch / elevated with each signal listed, framed as things worth
  a look. Sent weekly via the heartbeat only when watch/elevated. Three
  elevated weeks in a row suggests talking to someone.
- **Tension surfacing (#159).** `core/consensus.py`: when Apollo says rest
  (short sleep, low mood) and Artemis says push (an at-risk streak), a note is
  appended to the reply on `complete_habit`, `get_motivation`,
  `suggest_exercise` and `suggest_activity` explaining both sides. It never
  rewrites the module's answer. Kill switch: `consensus.enabled: false`.
- **Dashboards (#183, partial).** `/api/apollo/{dashboard,sleep,weight,water,
  mood,streaks,steps}` and `/api/artemis/heatmap`. New Health and Heatmap
  tabs, plus spending and investment charts on the Portfolio tab. The heatmap
  hatches days before history began.
- **DB maintenance (#233).** `core/db_maintenance.py`: in the heartbeat's
  0-5am window, at most weekly per file, checkpoints and `VACUUM`s a SQLite
  database only when at least 20% of it is free pages. Skips (and retries next
  tick) a locked database, checks free disk first, never deletes data, and only
  touches real SQLite files under `data/`. Artemis is JSON, so its part is the
  history cap above.

### Changed

- **Moods tab fixed.** It read `valence`/`timestamp`, which the table never
  had. `/api/moods` now returns those fields plus a score, and the tab shows the
  mood text.
- New optional config blocks `apollo:`, `consensus:`, `maintenance:` and
  `dionysus.mood_aware`, validated in `core/config_validation.py` and shown in
  `config/laptop_config.example.yaml`.

### Notes

- `HEARTBEAT.md` may still contain a fixed "drink some water" reminder;
  delete it if you want only the adaptive nudge.
- Apollo's day boundaries use `apollo.timezone` (default `chronos.timezone`).
  The habit/mood pairing uses UTC days on both sides to match Artemis.

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

Completes all 10 Iris items. The first batch (#73, #74, #76, #78, #79, #80)
left #71, #72, #75 and #77 open; they are now done, with the limits stated
under each entry.

### Added (second batch: #71, #72, #75, #77)

- **Video support** (`modules/iris/video.py`, `IrisAnalyser._analyse_video`).
  Evenly spaced frames per video (`iris.video.frames`, default 4; near-black
  frames skipped), each captioned by the vision model and merged into one
  caption/tag/mood record; the frames' CLIP embeddings are averaged into one
  embedding under the video's own id, so semantic search, find-similar,
  albums and re-index treat videos like photos. Duration stored
  (`files.duration_seconds`, migrated) and search results show
  `(video, 0:42)`. Perceptual hashing is skipped for non-images. Needs
  OpenCV; without it a video is recorded as a retryable analysis error.
  No audio, motion or on-screen text. [#75]
- **Face grouping, local-only** (`modules/iris/faces.py`, `faces`/`people`/
  `face_scans` tables, intents `iris_scan_faces`, `iris_list_people`,
  `iris_name_person`, `iris_find_person`, `iris_forget_faces`). OpenCV
  YuNet + SFace; the two ONNX files must be supplied. **Off by default**
  (face embeddings are biometric): enable with `iris.faces.enabled`. Stored
  only in `iris.db`; no face images kept; groups are never named
  automatically ("Person 3" until you say otherwise); naming a group after
  an existing person merges them; `iris_forget_faces` deletes all of it.
  Grouping is deliberately strict (wrong merges are worse than a split you
  can merge). Incremental: new faces join existing people first. "Photos of
  Mom at the beach" is answered from the face groups once Mom is named. [#72]
- **Camera object detection** (`modules/iris/detection.py`, intent
  `iris_detect_objects`, `python -m modules.iris.detection`). YOLO over
  everyday objects, from the webcam (one frame, or a watch of up to 60s that
  reports what appears/disappears, flicker-filtered) or a saved photo, whose
  result is stored in `files.objects` and searched alongside captions and
  tags. **Camera off by default** (`iris.camera.enabled`), opened only for the
  request and always released; frames are not saved. `ultralytics` is
  optional and not in the active requirements. Not PCB-fault or gesture
  recognition. [#77]
- **Semantic search extras** (`embeddings.filter_hits`,
  `ImageVectorIndex.indexed_ids/get_embedding`, intents `iris_reindex`,
  `iris_find_similar`). CLIP search itself already existed; this adds
  re-indexing of anything analysed before CLIP was available, look-alike
  search, and optional relevance cut-offs (`iris.semantic.max_distance`,
  `relative_margin`; off by default as they need tuning per library). [#71]
- Config: `iris.video`, `iris.faces`, `iris.camera`, `iris.semantic` added to
  `IrisConfig` and `laptop_config.example.yaml`. `opencv-python-headless`
  added to `requirements.txt`.
- Intents registered in `intent_registry.py` and `nlu_prompt.txt`.
- Tests: `test_iris_video_faces_detection.py`, 47 cases using fake
  embedders, face backends, detectors and cameras, plus real synthetic videos
  through OpenCV. Not exercised: real CLIP/Chroma, the YuNet/SFace models,
  YOLO, a physical webcam. The 17 pre-existing Iris/Chroma-stub failures in
  this sandbox are unchanged.

### Added (first batch: #73, #74, #76, #78, #79, #80)

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
