# Hestia Heartbeat Checklist

Tasks Hestia checks on every heartbeat tick.

**Note on the `- [ ]` boxes:** these are NOT one-time to-dos. `core/heartbeat.py`
re-evaluates every `- [ ] ` line on every tick and never writes `- [x]` back to
this file — each line describes a *recurring* condition (e.g. "run once a day
in this hour window", "fire at most once per cooldown period") whose own
internal state (`_last_brief_date`, `_reminder_last_fired`, etc.) governs how
often it actually fires. Checking a box here would not do anything and is not
supported; leave every task as `- [ ]`.

- [ ] morning brief
- [ ] reminder: drink some water
- [ ] nightly summary