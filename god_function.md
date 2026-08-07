Hestia (core) — the assistant's default persona: general chat, notes (take/get/delete), remembering your name, preferences, conversation history, system info.

Hecate — the router. Not user-facing; decides which god handles each query. No intents of its own.

Mnemosyne — long-term memory: remember/recall facts, get_facts, learn_fact, forget_fact, get_user_info. Also runs summarisation and semantic (vector) search over past interactions.

Hermes — Gmail & Google Calendar: read_email, send_email, list_events, create_event.

Hephaestus — browser automation: browser_action (open a URL), search_web, check_flight status. Runs a real headless browser, can't launch desktop apps.

Chronos — time-related: get_time, get_date, get_weather, set_reminder.

Athena — document/RAG search: athena_search, query_documents, search_documents — over your notes, PDFs, research files (ChromaDB-backed).

Iris — media search & ingestion: iris_search, iris_ingest, iris_analyse, iris_query, iris_status — photos, images, videos, galleries.

Artemis — habits & goals: add_habit, complete_habit, list_habits, add_goal, update_goal, productivity_summary.

Ares — strategic/analytical: analyse_risk, strategic_plan, swot_analysis, decision_support, premortem_analysis, competitive_analysis, contingency_plan, war_room_briefing.

Apollo — health tracking: log_workout, track_sleep, log_mood, log_health, get_health_summary.

Orpheus — creative writing: write_poem, brainstorm, creative_prompt, generate_lyrics, write_story, continue_writing (extends the user's own fiction/verse in-voice), critique_writing, rewrite_style (shifts creative tone/voice — for functional-text rewrites see Metis), generate_names, get_creations (recall past creative output).

Metis — writing assistance & editing: correct_text, improve_clarity, suggest_style, detect_tone, rewrite_text, draft_content, summarize_text, expand_text, shorten_text, generate_outline, check_plagiarism (best-effort only, no live web index), generate_citation, check_consistency, readability_report, writing_stats. Handles your own text and utilitarian drafting (emails, reports, cover letters); fiction/verse/lyrics stay with Orpheus.

Dionysus — entertainment/leisure: recommend_movie (OMDB-backed), find_restaurant, recommend_music, plan_outing.

Pluto — personal finance: log_expense, get_budget_summary, track_investment (live crypto/stock prices), spending_report (AI-generated).