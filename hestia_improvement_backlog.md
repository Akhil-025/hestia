# 🔥 HESTIA IMPROVEMENT BACKLOG

280+ concrete ideas, grouped by area, pulled from the actual codebase's gaps (see the integration audit), patterns from other assistant/RAG/home-automation projects, and general software-engineering practice. Not a roadmap — a menu. Pick what serves you; ignore the rest.

Each item is a single sentence so you can copy rows straight into an issue tracker. `[Q]` = quick win (hours), `[M]` = medium (a weekend), `[L]` = large (multi-week project).

---

## 1. Core Architecture & Orchestration (Hestia / Hecate)

1. `[Q]` Add a `--dry-run` flag to `main.py` that prints the resolved routing decision without executing the handler.
2. `[M]` Give Hecate a confidence-weighted fallback: if the top intent's confidence is below a threshold, ask a clarifying question instead of guessing.
3. `[M]` Add a "why did you route this here" debug command that dumps the registry lookup → trigger-tier → fallback-tier path Hecate actually took for the last query.
4. `[L]` Replace the hand-tuned fallback tiers with a small trained classifier (even a logistic regression over embeddings) that can be retrained from logged misroutes.
5. `[Q]` Log every intent classification (query, chosen intent, confidence, module, latency) to a rotating file for later analysis.
6. `[M]` Build a nightly job that reviews the last day's low-confidence classifications and surfaces them for you to manually label.
7. `[M]` Add per-module circuit breakers so a crashing module (e.g. Pluto's Postgres pool) degrades gracefully instead of taking down dispatch.
8. `[Q]` Add a `modules_status` command that reports each module's `available`/`ready` state in one place (several modules already expose this individually).
9. `[M]` Introduce a plugin/skill-loading system so new modules can be dropped into `modules/` and auto-registered without editing `main.py`.
10. `[L]` Add a real event bus (you have `core/event_bus.py` — check if modules actually publish/subscribe or just import it) so modules can react to each other's events without direct calls, without breaking the "no module calls another" contract.
11. `[Q]` Version the intent registry (`INTENT_MODULE_MAP`) so old client integrations (Telegram bot, web UI) can detect a breaking change.
12. `[M]` Add a "conversation session" concept with a TTL so multi-turn context doesn't leak across unrelated topics hours apart.
13. `[M]` Let Hecate support multi-intent queries ("log my workout and tell me the weather") by splitting compound requests before routing.
14. `[Q]` Add a config validation step at startup that fails fast with a clear error if `laptop_config.yaml` is missing required keys, instead of failing deep in a module.
15. `[M]` Add hot-reload for `config/nlu_prompt.txt` and `laptop_config.yaml` so you don't need a full restart to tune prompts.
16. `[L]` Add a "shadow mode" for new intents: route to the new handler but also run the old fallback, log both outputs, and diff them before switching over for real.
17. `[Q]` Add a global request ID that flows through logs across `core/`, `modules/`, and `api.py` so one query's full trace can be grepped in one shot.
18. `[M]` Add graceful shutdown handling (SIGTERM) that finishes in-flight requests and flushes the heartbeat/scheduler state before exiting.
19. `[M]` Add a health-check endpoint in `api.py` that aggregates every module's `available()`/`ready()` into one JSON blob for the web UI to poll.
20. `[L]` Consider splitting the single Python process into 2–3 processes (voice pipeline, core assistant, background jobs) communicating over a local queue, without going all the way to Docker microservices — a middle ground between the monolith and HEARTH.txt's 11-service vision.

## 2. NLU & Intent Classification

21. `[M]` Add a confusion-matrix eval script using `hestia_test_prompts.md` as a golden dataset, run in CI.
22. `[Q]` Add synonyms/aliases config so "log my sleep" and "I slept for" both map cleanly without prompt-engineering every phrasing.
23. `[M]` Add entity extraction confidence scores, not just intent confidence, so "remind me tomorrow" with a garbled date can trigger a clarification instead of a wrong reminder.
24. `[M]` Support intent chaining/pipelines ("summarize this paper and add it to my reading list" → athena_search + summarize + artemis-style tracking).
25. `[L]` Fine-tune a small local classifier (distilbert-sized) on your own logged queries so NLU stops depending on Ollama's JSON-mode reliability.
26. `[Q]` Add a regression test that fails CI if `config/nlu_prompt.txt`'s intent list and `intent_registry.py`'s `ALL_INTENTS` ever drift (you already have the invariant — codify it).
27. `[M]` Add multi-language support (start with Hindi/Hinglish given your context) at the NLU layer.
28. `[Q]` Cache repeated identical queries' NLU classification for a short TTL to cut Ollama round-trips during rapid testing.
29. `[M]` Add slot-filling for missing required entities (ask "which app?" instead of failing `open_app` silently).
30. `[M]` Track per-intent accuracy over time (via user corrections / thumbs-down) and surface a weekly "worst-performing intents" report.

## 3. Mnemosyne — Memory & Knowledge

31. `[L]` Build the knowledge graph HEARTH.txt describes: extract entities/concepts from facts and notes, store relationships, expose "what connects to X."
32. `[M]` Add a lightweight graph visualization in the web UI (even a force-directed D3 graph of facts/concepts) rather than the full 3D vision.
33. `[L]` Add spaced-repetition scheduling (SM-2 algorithm) for facts tagged as "study material," with due-today surfacing in the morning brief.
34. `[M]` Add quiz generation from ingested notes/documents (pull N facts, generate multiple-choice via Ollama, track score history).
35. `[M]` Add "strength/weakness map" per subject by tracking quiz performance over time.
36. `[Q]` Add fact expiry/decay — facts not reinforced or referenced in N months get flagged for review instead of living forever.
37. `[M]` Add contradiction detection: when learning a new fact, check semantic similarity against existing facts and flag conflicts ("you said X before, now Y — update?").
38. `[M]` Add per-fact confidence/source tracking (user-stated vs. inferred vs. imported) so recall can express uncertainty appropriately.
39. `[L]` Add Obsidian vault sync: watch a configured vault folder, ingest markdown notes with wikilink-aware chunking, and write back structured notes Hestia generates.
40. `[M]` Add weekly/monthly auto-summarization of interactions (you have `add_summary`/`get_recent_summaries` — wire this into an actual heartbeat-triggered digest).
41. `[Q]` Add a "forget everything about X" bulk-delete flow beyond single-fact `forget_fact`.
42. `[M]` Add semantic deduplication on ingest — don't store near-identical facts twice, merge them with a reference count.
43. `[L]` Add episodic memory clustering: group related interactions into "episodes" (e.g. all messages about a specific project) for better long-range recall.
44. `[M]` Expose a "memory export" (JSON/Markdown dump of all facts+summaries) for backup and portability, independent of the sync API.
45. `[M]` Add importance scoring so `get_top_facts_for_context` weighs recency, frequency of reference, and explicit user emphasis, not just recency.
46. `[Q]` Add unit tests around embedding drift — if you ever change embedding models, verify old vectors are re-embedded, not silently stale.
47. `[L]` Add arXiv/IEEE monitoring: scheduled fetch of new papers matching saved interest queries, auto-summarized and queued in Athena.
48. `[M]` Add "recall what I said on X date" as a first-class dated query, not just semantic search (you may already partially have this — confirm date-range filtering works on ChromaDB metadata).
49. `[Q]` Add a memory size/cost dashboard (# facts, DB size, embedding count) surfaced via `get_memory_stats`.
50. `[M]` Add memory provenance in responses — when Hestia recalls something, cite roughly when/how it learned it ("you mentioned this on Tuesday").

## 4. Athena — Research & Documents

51. `[L]` Add document *generation*, not just ingestion: LaTeX report scaffolding from a set of notes/citations.
52. `[L]` Add PowerPoint generation via python-pptx from a document/summary (you already parse pptx — mirror it for output).
53. `[M]` Add PDF export of any generated report/summary.
54. `[M]` Add a literature-review generator that synthesizes across multiple ingested papers into one structured draft.
55. `[M]` Add citation management — track sources per fact/claim and auto-generate a bibliography (BibTeX or APA) on request.
56. `[M]` Add research-gap detection: compare a set of papers' stated future-work sections and surface recurring unaddressed gaps.
57. `[Q]` Add a "what's new since I last checked" digest per ingested folder (diff against last ingestion timestamp).
58. `[M]` Add table extraction from PDFs (not just text/OCR) so quantitative data in papers is queryable.
59. `[M]` Add figure/chart extraction with captions indexed separately for "find the graph that shows X" queries.
60. `[L]` Add cross-document citation graphs — which papers cite which, visualized.
61. `[Q]` Add a re-ingestion command that only processes changed/new files instead of a full rebuild every time.
62. `[M]` Add configurable chunk size/overlap per document type (a textbook chapter vs. a two-page abstract shouldn't chunk the same way).
63. `[M]` Add a feedback loop: let the user mark a retrieved chunk as irrelevant, and down-weight that chunk/source in future hybrid search.
64. `[Q]` Surface retrieval scores (semantic + BM25 breakdown) in responses when debug mode is on, to help tune the hybrid weighting.
65. `[M]` Add multi-document comparative queries ("compare how these three papers define X").
66. `[L]` Add methodology-generator: given a research question, draft a study design skeleton (variables, controls, expected analysis).
67. `[Q]` Add file-type coverage checks in CI — a test per supported format (pdf/docx/pptx/epub/txt) that ingests a fixture file and asserts non-empty extraction.
68. `[M]` Add OCR language auto-detection instead of assuming English-only documents.
69. `[M]` Add a "translate this document" pipeline (useful for non-English papers).
70. `[Q]` Add ingestion progress reporting (X of Y files processed) to the web UI instead of a silent batch job.

## 5. Iris — Vision & Media

71. `[L]` Add CLIP-based semantic image search (already on your own roadmap in README) to replace caption-only matching.
72. `[M]` Add face clustering (privacy-respecting, local-only) so "photos of person X" works without external APIs.
73. `[M]` Add duplicate/near-duplicate detection across the whole library, not just the perceptual-hash function that already exists — surface it as a cleanup tool.
74. `[Q]` Add EXIF-based search (date taken, location, camera) alongside caption search.
75. `[M]` Add video support (frame sampling + captioning), not just static images, if your library has video.
76. `[M]` Add a "describe what changed" mode comparing two photos of the same subject over time (useful for progress photos, plant growth, etc.).
77. `[L]` Add real-time object detection over a webcam/phone-camera feed for the "hardware debugging via image" and "gesture recognition" use cases HEARTH.txt describes, scoped down to something achievable (e.g. YOLO for common objects, not full PCB fault detection).
78. `[Q]` Add a manual re-tag/correct-caption flow so wrong AI captions can be fixed and the correction feeds back into search relevance.
79. `[M]` Add album/collection auto-organization by clustering embeddings (event detection: "these 40 photos are probably one trip").
80. `[Q]` Add a storage-budget guard — warn before ingesting a folder that would blow past a configured disk quota.

## 6. Chronos — Time, Scheduling, Reminders

81. `[M]` Add recurring reminders (daily/weekly/custom cron-like), not just one-shot.
82. `[M]` Add location-aware reminders ("remind me when I get home") if you ever add device location (Mnemosyne already has `get_device_location` — wire it in).
83. `[Q]` Add snooze support for reminders instead of only fire-once-and-forget.
84. `[M]` Add natural-language recurring rule parsing ("every weekday at 7am").
85. `[Q]` Add timezone override per reminder (useful if you travel) instead of one global config timezone.
86. `[M]` Add a "what's on my plate today" aggregator that merges Chronos reminders + Hermes calendar events + Artemis due goals into one timeline.
87. `[Q]` Add holiday-aware scheduling (don't fire "study" reminders on days you've marked as holidays).
88. `[M]` Add weather-triggered suggestions ("rain expected — move your outdoor plan?") by combining `_get_weather` with Dionysus's outing planner.
89. `[Q]` Add a missed-reminder catch-up on startup (if Hestia was offline when a reminder fired, surface it once on next launch instead of silently dropping it).
90. `[M]` Add ICS export/import so Chronos reminders can round-trip with any standard calendar app.

## 7. Hermes — Communication & Scheduling

91. `[L]` Add Todoist integration (explicitly in `HEARTH.txt`, currently absent) for task prioritization and sorting.
92. `[M]` Add email prioritization/triage — classify inbox by urgency using the same NLU/LLM stack, surface a daily digest.
93. `[M]` Add email draft generation from a short instruction ("reply saying I can't make it, suggest Thursday instead").
94. `[Q]` Add email search by sender/subject/date range as a distinct intent from generic reading.
95. `[M]` Add travel-time estimation between calendar events (flag back-to-back meetings with no buffer).
96. `[M]` Add smart meeting scheduling — given a set of attendees/constraints, propose slots (needs free/busy lookups you may already get from Google Calendar API).
97. `[Q]` Add calendar conflict detection when creating a new event.
98. `[M]` Add recurring event support if not already covered by `_create_event`.
99. `[L]` Add a unified "inbox zero" mode: batch-process unread emails with suggested actions (archive/reply/snooze/delegate).
100. `[Q]` Add a dry-run/confirmation step before `send_email` actually sends, surfaced clearly in voice mode where a misfire is costly.

## 8. Hephaestus — Automation & (Currently) Browser

101. `[M]` Add scheduled/recurring browser tasks (check a site daily and alert on change — price drop, application portal update).
102. `[M]` Add form-filling automation for repetitive tasks (application forms, recurring submissions).
103. `[Q]` Add a screenshot-on-failure debug mode for browser automation so failed scrapes are diagnosable.
104. `[L]` If you ever revisit HEARTH.txt's original hardware-debugging vision, scope it small: a "photo of a breadboard, tell me if anything looks obviously wrong" using a vision-LLM prompt rather than full PCB fault classification.
105. `[M]` Add site-specific scrapers as pluggable modules (job boards, GATE result pages, application portals) rather than generic scraping only.
106. `[Q]` Add rate-limiting/politeness delays to `_scrape_page` to avoid hammering sites and getting blocked.
107. `[M]` Add a headless-browser session pool so repeated automation tasks don't pay full browser-launch cost each time.
108. `[Q]` Add a `--headed` debug override for browser automation so you can watch what it's doing when something breaks.
109. `[M]` Add change-detection diffing for monitored pages (store last snapshot, alert only on meaningful diffs, not every whitespace change).
110. `[L]` Add a code-analysis/automation-engine feature (mentioned in HEARTH.txt under Hephaestus): point it at a repo, get a summary of structure/issues — you could reuse the same LLM stack that just did this audit.

## 9. Apollo — Health & Wellness

111. `[M]` Add sleep-quality scoring beyond duration (consistency of bed/wake time, using a rolling window).
112. `[M]` Add correlation surfacing ("your mood tends to dip on days you sleep under 6h") from your own logged data — genuinely useful and low-risk since it's reflecting the user's own patterns back, not diagnosing.
113. `[Q]` Add configurable units at the profile level (kg/lb, ml/oz) instead of per-log-entry parsing only.
114. `[M]` Add meal logging beyond `_lookup_food`, with a running daily macro/calorie summary.
115. `[M]` Add a hydration reminder that adapts to logged water intake pace across the day, not a fixed schedule.
116. `[Q]` Add workout streak tracking parallel to Artemis's habit streaks, specific to exercise types.
117. `[M]` Add injury/pain logging distinct from general mood, with simple trend surfacing over weeks.
118. `[Q]` Add a weekly health summary auto-sent via the heartbeat (sleep avg, workouts, weight trend, water compliance).
119. `[M]` Add integration with phone step-count/health data if you ever add a mobile companion (Health Connect / Google Fit export import).
120. `[Q]` Add configurable goal reminders ("you're 2kg from your target — still 3 weeks out, on pace").

## 10. Artemis — Habits & Goals

121. `[M]` Add visual habit graphs (calendar heatmap style) in the web UI — the data already exists via `workout_dates`-style tracking patterns.
122. `[M]` Add a Pomodoro/focus-session timer with start/stop voice commands feeding into productivity stats.
123. `[Q]` Add habit "grace periods" (missing one day doesn't reset a streak if within an allowed buffer) as a configurable option.
124. `[M]` Add goal decomposition — break a large goal into sub-tasks/milestones automatically via LLM, then track each.
125. `[Q]` Add a weekly habit review prompt (already listed under Mnemosyne's academic memory in HEARTH.txt, but fits naturally here) summarizing consistency %.
126. `[M]` Add habit correlation with Apollo's mood/sleep logs ("habits you kept on high-mood days").
127. `[L]` Add an achievement/badge system (distinct from full XP/leveling) tied to milestones like 30-day streaks.
128. `[Q]` Add "pause a habit" (vacation mode) so streaks don't break during planned breaks.
129. `[M]` Add smart nudges — if a habit is usually done by a certain time and hasn't been logged, send a gentle reminder instead of waiting for end-of-day.
130. `[M]` Add goal templates (common goal types pre-filled with sensible milestones) to reduce setup friction.

## 11. Pluto — Finance

131. `[L]` Add Zerodha/Groww/broker API integration for live portfolio sync (explicitly missing per the audit).
132. `[M]` Add price-movement and news-triggered alerts (HEARTH.txt spec'd, not yet built) using your existing market-data fetchers.
133. `[Q]` Add recurring-expense detection (subscriptions) from logged expenses.
134. `[M]` Add budget-vs-actual variance alerts per category, not just a summary report.
135. `[M]` Add a "financial health score" combining savings rate, spending volatility, and investment diversification.
136. `[Q]` Add multi-currency net worth aggregation if you hold assets in more than one currency.
137. `[M]` Add tax-relevant categorization/export (useful come filing season) for logged expenses/investments.
138. `[L]` Add scenario planning ("what if I invest ₹X/month for Y years at Z% return") as a distinct forecasting mode from the existing forecast_spending.
139. `[Q]` Add confidence intervals / uncertainty ranges on forecasts instead of point estimates only.
140. `[M]` Add a "explain this holding" mode that pulls recent news + fundamentals for a specific stock/asset via the existing news/market-intelligence pipeline.
141. `[Q]` Add expense receipt photo ingestion (pairs naturally with Iris's OCR pipeline) instead of manual text entry only.
142. `[M]` Add investment rebalancing suggestions based on drift from target allocation.
143. `[L]` Add a backtesting UI (you already have `backtest_sma_crossover` — expose parameter sweeps and visualize equity curves in the web UI).
144. `[Q]` Add rate-limit/retry backoff tuning visibility — surface when a market-data API is being throttled instead of failing silently.
145. `[M]` Add a "quant score explainability" view — when `generate_quant_score` returns a number, show the feature breakdown that produced it.

## 12. Dionysus — Social & Leisure

146. `[M]` Add an event finder for your city using a free events API or scraped local listings (HEARTH.txt spec'd, currently absent).
147. `[Q]` Add "seen it, don't recommend again" persistence across restarts (confirm `dismiss`/`mark_seen` already does this — extend to auto-expire dismissals after a long enough time).
148. `[M]` Add group/friends outing coordination (shared availability + preferences) if you ever add multi-user support.
149. `[Q]` Add mood-based recommendations — feed Apollo's current mood log into movie/music suggestions.
150. `[M]` Add a "surprise me" mode that deliberately picks outside your recent taste cluster to counter recommendation staleness.
151. `[Q]` Add cost estimation alongside restaurant/outing suggestions so recommendations respect a budget.
152. `[M]` Add recurring "recharge routine" scheduling (weekend downtime blocks) integrated with Chronos.

## 13. Ares — Strategy

153. `[M]` Add a GATE/career-ranking specialization mode using the existing decision-support toolkit but scoped prompts.
154. `[Q]` Add a "revisit this decision" reminder — schedule a follow-up check on a past strategic_plan/decision_support output.
155. `[M]` Add outcome tracking — let the user record what actually happened after a plan/decision, and use it to calibrate future confidence.
156. `[M]` Add a lightweight Monte Carlo simulator for numeric decisions (expected value under uncertainty) rather than the full Hecate-level system.
157. `[Q]` Add named "playbooks" (saved SWOT/premortem templates for recurring decision types) instead of starting from scratch each time.

## 14. Hecate — Decision Engine

158. `[L]` Build the multi-agent "conference" feature: route one query through 2–3 relevant modules' perspectives (e.g. Pluto + Ares on a financial risk) and synthesize a combined answer.
159. `[M]` Add weighted-voting consensus when modules disagree (e.g. Apollo says rest, Artemis says push through a habit streak) — surface the tension explicitly rather than silently picking one.
160. `[L]` Add a lightweight what-if simulator: given a proposed change (quit a habit, cut a subscription), project its downstream effect using existing module data instead of a bespoke Monte Carlo engine.
161. `[M]` Add burnout-signal fusion from Apollo (sleep/mood) + Artemis (habit consistency) + Pluto (spending stress proxies) into one weekly risk flag.
162. `[Q]` Add a routing-decision audit log surfaced to the user on request ("what did you check before answering that?").
163. `[M]` Add critical-path extraction across active goals/deadlines (Artemis + Hermes + Chronos) into one "what actually needs attention this week" list.

## 15. Metis & Orpheus — Writing

164. `[Q]` Add a "writing session" wrapper that chains Orpheus draft → Metis critique/polish in one command instead of two separate intents.
165. `[M]` Add style-profile learning — build a lightweight profile of the user's own writing voice from samples, and have Metis's rewrite/correct respect it instead of a generic tone.
166. `[Q]` Add word-count/readability targets as parameters to Metis's shorten/expand instead of fixed heuristics.
167. `[M]` Add version history for Orpheus creations so edits don't overwrite the original draft.
168. `[Q]` Add export of a creation/writing session to a plain text or markdown file directly from the CLI.
169. `[M]` Add plagiarism-check source citation (currently `_check_plagiarism` exists — confirm it surfaces sources, not just a similarity score).

## 16. Voice Pipeline (STT / TTS / Wake Word / Barge-in)

170. `[M]` Add per-entity TTS voices (HEARTH.txt's "11 voices" idea, scoped down) — even 2–3 distinct Piper voices for different response types (health vs. finance vs. casual) adds real personality.
171. `[Q]` Add a "repeat that" / "say it again" voice command using the last TTS output buffer.
172. `[M]` Add streaming TTS (start speaking the first sentence while the rest is still generating) to cut perceived latency.
173. `[Q]` Add a mute/do-not-disturb voice toggle that suppresses proactive notifications without killing the whole assistant.
174. `[M]` Tune `min_rms`/`vad_aggressiveness` automatically per-device via a short calibration routine instead of manual config tuning (the config file already flags this as finicky).
175. `[M]` Add acoustic echo cancellation (even a basic adaptive filter) to fix the self-interruption issue the config comments call out.
176. `[Q]` Add wake-word sensitivity levels (quiet room vs. noisy room presets).
177. `[M]` Add speaker identification (differentiate you from a housemate/guest) if privacy-sensitive modules (Pluto, Mnemosyne) should behave differently by speaker.
178. `[Q]` Add a visual "listening" indicator in the web UI synced to actual mic state, for when voice mode runs headless.
179. `[M]` Add graceful STT fallback to typed input when the mic/model fails to load, instead of crashing voice mode.

## 17. Web UI & Dashboard

180. `[M]` Build the "Total Launcher"-style single-screen dashboard from `Project_Hestia.txt` as a web UI view instead of an Android launcher — today's stats, top priority, deadlines, in one glance.
181. `[Q]` Add dark/light theme toggle if not already present.
182. `[M]` Add a live activity feed (last N interactions across all modules) for at-a-glance oversight.
183. `[M]` Add per-module dashboards (Pluto portfolio chart, Artemis habit heatmap, Apollo sleep trend) as separate tabs.
184. `[Q]` Add a search bar that queries across Mnemosyne + Athena + Iris at once ("find everything related to X").
185. `[M]` Add mobile-responsive layout if the web UI is currently desktop-only (likely, given Flask + basic templates).
186. `[Q]` Add a simple password/session-based login (HEARTH.txt spec'd this explicitly, and it's cheap given Flask already exists).
187. `[M]` Add WebSocket-based live updates (new reminder fired, new message) instead of polling.
188. `[Q]` Add a settings page in the UI for the config toggles currently only editable via YAML.
189. `[M]` Add a "explain this response" expandable panel per chat message showing which module/intent/confidence produced it.
190. `[M]` Add data export buttons (CSV/JSON) per module directly from the UI instead of DB access.

## 18. Telegram Bot

191. `[Q]` Add inline buttons for common actions (confirm/cancel, snooze reminder) instead of text-only replies.
192. `[M]` Add photo-message handling routed straight to Iris ingestion, and PDF-message handling routed to Athena — natural extensions of the existing `process_paper` pattern.
193. `[Q]` Add a `/help` command that's auto-generated from the intent registry instead of hand-maintained.
194. `[M]` Add per-chat-id role scoping (you vs. a family member gets a different allowed intent set) building on `allowed_chat_ids`.
195. `[Q]` Add typing indicators while a long-running intent (document ingestion, backtest) executes.
196. `[M]` Add voice-note support in Telegram (send audio → STT → same NLU pipeline as the desktop voice mode).

## 19. Security & Privacy

197. `[M]` Wire `audit_secrets.py` into CI/pre-commit so secret-pattern regressions are caught before a commit, not just on manual runs.
198. `[L]` Add encryption at rest for the most sensitive data classes explicitly called out in HEARTH.txt: journals, finance, health logs — even simple SQLCipher/Fernet-at-the-field-level beats plaintext SQLite.
199. `[M]` Add a secrets-rotation checklist/script (Google OAuth tokens, Telegram token, API keys) with expiry warnings.
200. `[Q]` Add `.env.example` alongside `.gitignore`'d real `.env`, mirroring the pattern you already use for `laptop_config.yaml`.
201. `[M]` Add per-module data-access scoping — Dionysus shouldn't be able to touch Pluto's DB even in-process; enforce via separate DB files/connections, not just convention.
202. `[Q]` Add a "what does Hestia know about me" export/review command for transparency (ties in nicely with Mnemosyne's memory export idea).
203. `[M]` Add rate-limiting on `api.py`'s sync endpoints to prevent abuse if ever exposed beyond localhost.
204. `[Q]` Add input length/sanitization checks consistently across all modules' entity parsing (Pluto already has `_sanitize_input` — audit whether others need the same).
205. `[M]` Add an audit log of destructive actions (delete_notes, forget_fact, remove_goal, delete_events) with a short undo window.
206. `[L]` Add a content-safety filter pass on LLM outputs before they're spoken/sent (HEARTH.txt's ShieldGemma idea, scoped to any local safety classifier or even simple rule-based checks).

## 20. Testing & QA

207. `[Q]` Add a coverage report to `run_tests.py`'s output and track it over time.
208. `[M]` Add property-based tests (Hypothesis) for the parsing-heavy functions (`_parse_weight`, `_parse_duration`, `_parse_water`, date parsing) — these are exactly the kind of code that breaks on edge-case input.
209. `[M]` Add integration tests that exercise the full NLU → Hecate → module path, not just unit tests per module.
210. `[Q]` Add a test that fails if any module's declared `_INTENTS` set contains an intent missing from `intent_registry.py` (codify the invariant you already benefit from).
211. `[M]` Add load/latency tests for the voice pipeline (STT + NLU + TTS round-trip time) with a regression budget.
212. `[Q]` Add a smoke-test script that boots `main.py`, sends 10 canonical queries, and checks for non-error responses — good as a pre-deploy gate.
213. `[M]` Add fixture-based tests for each ingestible file format in Athena (one small real pdf/docx/pptx/epub per format).
214. `[Q]` Add mutation testing on at least the critical routing logic to catch tests that pass but don't actually assert anything meaningful.
215. `[M]` Add a test harness that replays real (anonymized) past queries against the NLU to catch classification regressions when you tweak the prompt.

## 21. Observability & Ops

216. `[M]` Add structured (JSON) logging option so logs are machine-parseable for later analysis, alongside the current human-readable format.
217. `[M]` Add basic metrics (Prometheus-style counters/histograms) for intent volume, latency per module, and error rate — Pluto already has a metrics decorator pattern; generalize it.
218. `[Q]` Add a `/mnt`-style rotating backup of the SQLite DB + ChromaDB directories on a schedule, verified with a restore-test script (not just "hope the copy worked").
219. `[M]` Add disk-space monitoring with a warning threshold, especially given Iris/Athena can grow large media/embedding stores.
220. `[Q]` Add a single `hestia doctor` command that checks Ollama is running, models are pulled, config is valid, and DBs are reachable — a pre-flight check before `main.py` starts.
221. `[M]` Add auto-restart-on-crash for the main process (systemd unit or a simple supervisor loop) — this is the one piece of HEARTH.txt's "self-healing" vision that's cheap to actually build.
222. `[Q]` Add log rotation/retention policy so `Hestia/logs` doesn't grow unbounded (mirrors the doc's own "clean old logs" maintenance note).
223. `[M]` Add alerting (Telegram push) on repeated errors from the same module within a short window, rather than silent log-only failures.

## 22. Performance

224. `[M]` Add response streaming for chat replies in the web UI and CLI (Hestia already has `stream_chat`/`try_stream_chat` — confirm every surface actually uses it instead of blocking on full generation).
225. `[Q]` Add embedding batching for Athena/Iris/Mnemosyne ingestion instead of one-at-a-time calls, if not already batched.
226. `[M]` Add a warm model pool (keep Ollama's model loaded, avoid cold-start latency on the first query after idle).
227. `[Q]` Add caching for repeated identical RAG queries within a short window (partial infra likely already exists in Athena's cache dir — confirm it's actually hit in practice).
228. `[M]` Profile the NLU round-trip specifically — it's on the critical path for every single query, and a smaller/faster classification-only model (already stubbed in config comments) could meaningfully cut latency.
229. `[Q]` Add lazy-loading for heavy modules (Iris's embedding models, Pluto's forecasting libs) so `main.py` starts fast even if you're not using finance/vision today.
230. `[M]` Add async I/O for the modules that make external network calls (weather, market data, Gmail) so one slow API doesn't block the whole dispatch loop.

## 23. Data & Storage

231. `[M]` Add a schema-migration tool (Alembic or hand-rolled) instead of ad-hoc `_init_schema` calls per module, so DB changes are versioned and reversible.
232. `[Q]` Add a data-integrity check command (foreign keys, orphaned rows) runnable on demand.
233. `[M]` Add compaction/vacuum scheduling for SQLite DBs that grow with heavy logging (Apollo, Artemis, Pluto).
234. `[M]` Add a unified "export everything" command (all module DBs → one portable archive) for real backup/migration, not per-module ad hoc export.
235. `[Q]` Add checksums on backup archives so a silently-corrupted backup is caught before you need it.
236. `[M]` Consider moving from SQLite to SQLite-with-WAL-mode (if not already) to reduce lock contention across the many modules writing concurrently.

## 24. Personalization & Adaptive Behavior

237. `[M]` Add a personality/tone config per entity (already spec'd in HEARTH.txt's "Week 69" personality pass) — small system-prompt tweaks per module's LLM calls, cheap to add.
238. `[M]` Add adaptive response length — shorter answers on voice/mobile, longer on web UI, without maintaining two separate prompt sets.
239. `[Q]` Add a user-set "energy level" or "mode" (focused/relaxed/tired) that shifts how proactive suggestions behave across modules.
240. `[M]` Add preference learning from implicit signals (which recommendations get accepted vs. dismissed) instead of only explicit `set_preference` calls.
241. `[L]` Add the predictive-scheduling idea from `Project_Hestia.txt` in a scoped form: learn your actual study/code time patterns from logged sessions and suggest a next-day schedule, with the documented manual fallback if the model's off.

## 25. Smart Home / Physical World (scoped-down HEARTH.txt vision)

242. `[M]` If you do want any smart-home layer, start with a single well-supported protocol (Tuya/Govee local API, or Home Assistant as a client Hestia talks to rather than orchestrator) instead of building your own device layer.
243. `[Q]` Add a "focus mode" that's purely software-side to start (mutes notifications, silences Dionysus/entertainment suggestions) before wiring it to actual lights/plugs.
244. `[M]` Add battery/thermal-aware suggestions if you ever run parts of this on a phone (HEARTH.txt's "battery >38°C, suggest a break" idea) — cheap to check via Termux:API even without the rest of the phone-native build.

## 26. Documentation & Developer Experience

245. `[Q]` Add a `CONTRIBUTING.md` documenting the "add one line to intent_registry.py" pattern so future-you (or a collaborator) doesn't reintroduce the drift bug it just fixed.
246. `[M]` Add architecture decision records (ADRs) for the big calls already made (why SQLite+ChromaDB hybrid, why no Docker, why single-process) so the reasoning survives beyond memory.
247. `[Q]` Add module-level README files (`modules/pluto/README.md` etc.) summarizing what each entity actually does today, since `god_function.md`/`README.md` describe the system but not each module in isolation.
248. `[M]` Add auto-generated API docs from `api.py`'s FastAPI schema (free via `/docs`, just confirm it's exposed and documented for future-you).
249. `[Q]` Add a CHANGELOG.md and start using it — useful once you're iterating on 280 backlog items and need to remember what actually shipped.
250. `[M]` Add example `.env`/config files with inline comments for every optional integration (you already do this well for Telegram — extend the pattern to Google, market-data, and OMDB/Spotify keys).

## 27. Multi-Device & Sync

251. `[M]` Flesh out the existing `sync:` push/pull API (currently disabled by default) into a real two-way sync with conflict resolution, rather than the Dell-backup architecture HEARTH.txt describes.
252. `[L]` Add a lightweight mobile companion (even a simple PWA hitting the Flask API) instead of a native app, for the "access from your phone" need without a full HEARTH.txt-style Tailscale mesh.
253. `[Q]` Add offline queueing on the client side (mobile/web) so actions taken while the server's unreachable sync once it's back, mirroring HEARTH.txt's offline-mode idea.
254. `[M]` Add device-identity tagging on logged data (which device logged this workout/expense) once more than one entry point exists.

## 28. Gamification (scoped down from "Life RPG Mode")

255. `[M]` Add a single unified daily score combining habit consistency + health logging + productivity, shown once on the dashboard — the useful core of the XP idea without a full leveling system.
256. `[Q]` Add streak-milestone celebrations (7/30/100 days) as a Telegram/notification message, reusing Artemis's existing streak data.
257. `[Q]` Keep XP/leveling explicitly optional and toggleable — the audit found no gamification layer at all, so even a minimal, skippable version is a net addition without over-engineering it.

## 29. Testing the Assistant's Own Judgment

258. `[M]` Add a "confidence calibration" report — compare NLU's stated confidence to actual correctness over a labeled sample, and see if confidence scores mean what they claim to.
259. `[Q]` Add a feedback command ("that was wrong") that logs the query+wrong-response pair for later review, distinct from silent failure.
260. `[M]` Add A/B-able prompt variants for the trickiest intents (the ones the NLU prompt file already has extensive disambiguation notes for, like Ares's premortem-vs-risk split) and measure which phrasing performs better.

## 30. Ideas From Outside the Assistant Space

261. `[M]` Borrow from game-save design: add a single "checkpoint" command that snapshots all module state at once, for safe experimentation.
262. `[Q]` Borrow from git: add semantic commit-style tags to Mnemosyne's summaries (`#health`, `#finance`, `#research`) for fast filtering.
263. `[M]` Borrow from IDEs: add a command palette (fuzzy-searchable list of every intent) in the web UI for power-user access without needing exact phrasing.
264. `[M]` Borrow from spaced-repetition apps (Anki): apply the same due-card scheduling model to reminders that matter but aren't time-critical ("check on this application status") — not just quiz facts.
265. `[Q]` Borrow from postmortem culture (SRE): after any bad autonomous action (a wrong reminder, a bad recommendation spree), write a one-paragraph postmortem note into Mnemosyne so the pattern is remembered.
266. `[M]` Borrow from personal-finance apps: add "safe to spend today" style derived numbers in Pluto instead of raw totals only.
267. `[M]` Borrow from note-taking tools (Roam/Logseq): add backlinks between Mnemosyne facts and Athena documents that reference the same concept.
268. `[Q]` Borrow from CLI tool design: add `--help` text and examples to every module's exposed commands, not just the top-level CLI.
269. `[M]` Borrow from recommender systems: add explicit "more like this / less like this" feedback buttons to Dionysus recommendations instead of only binary dismiss.
270. `[L]` Borrow from multi-agent research (AutoGPT-style critique loops): let Metis automatically critique Orpheus's output once before showing it to you, as an optional "polish pass" toggle.
271. `[Q]` Borrow from habit-tracker apps (Streaks, Loop): add a "why did I break this streak" optional note field so the habit log captures context, not just pass/fail.
272. `[M]` Borrow from RSS readers: add a unified "inbox" view merging new papers (Athena), new reminders (Chronos), new recommendations (Dionysus) into one triage list instead of five separate places to check.
273. `[Q]` Borrow from password managers: add a "data you've given Hestia" audit screen broken down by module, for periodic review/cleanup.
274. `[M]` Borrow from journaling apps (Day One): add mood/weather/location auto-tagging on voice journal entries if you build that Mnemosyne feature, purely as context, never as a diagnosis.
275. `[Q]` Borrow from build tools: add a `--verbose`/`--quiet` global flag so CLI output can be dialed up for debugging or down for daily use.
276. `[M]` Borrow from feature-flag systems: wrap every experimental Tier-3-style feature in a flag so it can be disabled instantly without a code change, matching the "manual fallback" philosophy both your docs already emphasize.
277. `[Q]` Borrow from static site generators: add incremental rebuilds everywhere you currently do full rebuilds (Athena's BM25 index rebuild, Iris's stats) so large libraries don't force a full recompute on every small change.
278. `[M]` Borrow from observability tooling (Honeycomb/Datadog philosophy): treat every user correction as a labeled data point, not just a one-off fix — feed it back into the eval sets in #21.
279. `[Q]` Borrow from good onboarding flows: add a first-run wizard that walks through enabling each module and testing its connection (Google, Telegram, Ollama) instead of a wall of YAML.
280. `[M]` Borrow from resilience engineering: run periodic "chaos" tests locally — kill Ollama mid-query, corrupt a config value, unplug the network — and confirm every module fails the way its docstrings claim it does (`_err`/`_miss` responses, not stack traces).

---

## How to actually use this list

- **Don't build all 280.** Pick 5–10 per month, tied to what you actually feel the absence of day-to-day.
- **Cheapest, highest-leverage first pass** if you want a starting cluster: #5, #8, #14, #26 (observability/safety nets), #33–35 (spaced repetition, since GATE prep is a real near-term need), #180 (dashboard), #197 (wire up the secrets scanner you already wrote), #220–221 (doctor command + auto-restart).
- Every item here is additive to what already exists — none of them require undoing the architecture decisions (single process, SQLite/Chroma, no Docker) that are already working for you.
