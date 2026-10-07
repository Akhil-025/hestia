"""
core/web_settings.py - edit config toggles from the web UI (backlog #188).

Two design constraints drive this file:

1. **The YAML keeps its comments.** laptop_config.yaml is mostly explanation.
   Loading and re-dumping it with PyYAML would erase every comment, so values
   are changed with a line-level edit that touches only the value text and
   leaves the trailing comment, indentation and line endings (the shipped file
   uses CRLF) alone. Before anything is written, the result is parsed again and
   compared with the original: exactly the requested keys may differ.

2. **Only a fixed list of keys can be changed.** ``SPEC`` below is the whole
   surface. It deliberately leaves out anything that sends data off the
   machine, widens what Hestia may do to the outside world (mailbox changes,
   travel providers, API tokens), carries paths or credentials, or could lock
   you out of this page (``webui.*``). Those stay edit-by-hand.

Hestia reads its config once at start-up, so a saved change applies after a
restart. The UI says so.
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Setting:
    key: str                      # dotted path in the YAML
    label: str
    group: str
    kind: str                     # "bool" | "int" | "float" | "enum"
    default: Any
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: tuple = ()


SPEC: tuple[Setting, ...] = (
    # --- Reminders & notifications
    Setting("chronos.scheduler_enabled", "Fire reminders automatically", "Reminders & notifications", "bool", True,
            "Off means reminders are stored but never announced."),
    Setting("chronos.default_snooze_minutes", "Default snooze (minutes)", "Reminders & notifications", "int", 10,
            "Used when you say \"snooze\" without a length.", 1, 240),
    Setting("chronos.skip_public_holidays", "Skip public holidays", "Reminders & notifications", "bool", False,
            "Recurring reminders skip public holidays of the configured country."),
    Setting("chronos.proactive_weather", "Mention weather in the daily agenda", "Reminders & notifications", "bool", False),
    Setting("telegram.push_notifications", "Push reminders to Telegram", "Reminders & notifications", "bool", False),
    # --- Health
    Setting("apollo.units.weight", "Weight unit", "Health", "enum", "kg", "", choices=("kg", "lb")),
    Setting("apollo.units.water", "Water unit", "Health", "enum", "ml", "", choices=("ml", "oz")),
    Setting("apollo.hydration.enabled", "Hydration nudges", "Health", "bool", True),
    Setting("apollo.weekly_summary.enabled", "Weekly health summary", "Health", "bool", True),
    Setting("apollo.goal_reminders.enabled", "Health goal reminders", "Health", "bool", False),
    Setting("apollo.burnout.enabled", "Burnout check-ins", "Health", "bool", True),
    # --- Habits
    Setting("artemis.nudges.enabled", "Habit nudges", "Habits", "bool", True),
    Setting("artemis.nudges.lateness_minutes", "Nudge after this many minutes late", "Habits", "int", 60,
            "", 0, 720),
    Setting("artemis.habit_grace_days", "Streak grace days", "Habits", "int", 0,
            "Missed days a streak survives.", 0, 7),
    # --- Voice
    Setting("tts.rate", "Speech rate (words per minute)", "Voice", "int", 175, "", 80, 300),
    Setting("tts.volume", "Speech volume", "Voice", "float", 1.0, "0 is silent, 1 is full.", 0.0, 1.0),
    Setting("stt.noise_filter", "Noise filter on the microphone", "Voice", "bool", True),
    Setting("barge_in.enabled", "Interrupt Hestia by speaking", "Voice", "bool", True),
    # --- Assistant behaviour
    Setting("dionysus.mood_aware", "Mood-aware suggestions", "Assistant", "bool", True),
    Setting("consensus.enabled", "Consensus answers", "Assistant", "bool", True),
    Setting("conference.enabled", "Conference mode", "Assistant", "bool", True),
    Setting("conference.llm_summary", "Summarise conferences with the language model", "Assistant", "bool", True),
    Setting("whatif.enabled", "What-if scenarios", "Assistant", "bool", True),
    Setting("writing.polish_pass", "Polish pass on drafts", "Assistant", "bool", False),
    Setting("writing.plagiarism_web_check", "Web check in the plagiarism scan", "Assistant", "bool", True,
            "Sends short excerpts to a search engine when on."),
    # --- Memory
    Setting("mnemosyne.papers.enabled", "Fetch new research papers", "Memory", "bool", False),
    Setting("mnemosyne.episodes.use_embeddings", "Embedding search for episodes", "Memory", "bool", False),
    # --- System
    Setting("heartbeat.enabled", "Background heartbeat", "System", "bool", True),
    Setting("maintenance.enabled", "Automatic maintenance", "System", "bool", True),
)

_BY_KEY = {s.key: s for s in SPEC}


class SettingsError(ValueError):
    """The request was refused; ``str(e)`` is safe to show the user."""


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _lookup(cfg: Any, dotted: str) -> tuple[bool, Any]:
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def read_settings(config_path: str | os.PathLike) -> list[dict]:
    """Every whitelisted setting with its current value, ready for the UI."""
    import yaml
    text = Path(config_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    out = []
    for s in SPEC:
        found, value = _lookup(cfg, s.key)
        out.append({
            "key": s.key, "label": s.label, "group": s.group, "kind": s.kind,
            "help": s.help, "default": s.default,
            "value": value if found and value is not None else s.default,
            "explicit": found and value is not None,
            "min": s.min, "max": s.max, "choices": list(s.choices),
        })
    return out


# --------------------------------------------------------------------------
# validating a requested value
# --------------------------------------------------------------------------

def coerce(setting: Setting, value: Any) -> Any:
    """Return *value* if it is acceptable for *setting*; else SettingsError."""
    if setting.kind == "bool":
        if isinstance(value, bool):
            return value
        raise SettingsError(f"{setting.label}: must be on or off.")
    if setting.kind in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{setting.label}: must be a number.")
        if setting.kind == "int":
            if int(value) != value:
                raise SettingsError(f"{setting.label}: must be a whole number.")
            value = int(value)
        else:
            value = float(value)
        if setting.min is not None and value < setting.min:
            raise SettingsError(f"{setting.label}: must be at least {setting.min:g}.")
        if setting.max is not None and value > setting.max:
            raise SettingsError(f"{setting.label}: must be at most {setting.max:g}.")
        return value
    if setting.kind == "enum":
        if value not in setting.choices:
            raise SettingsError(
                f"{setting.label}: choose one of {', '.join(setting.choices)}."
            )
        return value
    raise SettingsError(f"{setting.label}: unsupported setting type.")  # pragma: no cover


def _render(setting: Setting, value: Any) -> str:
    if setting.kind == "bool":
        return "true" if value else "false"
    if setting.kind == "int":
        return str(int(value))
    if setting.kind == "float":
        text = repr(float(value))
        return text if ("." in text or "e" in text) else text + ".0"
    return '"' + str(value) + '"'   # enums: choices contain no quotes


# --------------------------------------------------------------------------
# editing the text
# --------------------------------------------------------------------------

_KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z0-9_]+):(?P<gap> *)(?P<rest>[^\r\n]*)(?P<nl>\r?\n)?$")


def _scalar_span(rest: str) -> tuple[int, int]:
    """(start, end) of the value in *rest*, excluding any trailing comment."""
    if not rest or rest.startswith("#"):
        return 0, 0
    if rest[0] in "\"'":
        quote = rest[0]
        i = 1
        while i < len(rest):
            if rest[i] == quote:
                if quote == "'" and rest[i + 1:i + 2] == "'":
                    i += 2
                    continue
                return 0, i + 1
            if rest[i] == "\\" and quote == '"':
                i += 1
            i += 1
        return 0, len(rest)
    m = re.search(r"\s#", rest)
    end = m.start() if m else len(rest)
    return 0, len(rest[:end].rstrip())


def _index_lines(lines: list[str]) -> list[tuple[int, tuple[str, ...], re.Match, bool]]:
    """(line number, key path, match, is_section_header) for every key line."""
    stack: list[tuple[int, str]] = []
    out = []
    for n, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        m = _KEY_LINE.match(line)
        if not m:
            continue
        indent = len(m.group("indent"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple(k for _, k in stack) + (m.group("key"),)
        rest = m.group("rest")
        is_header = (not rest) or rest.startswith("#")
        out.append((n, path, m, is_header))
        if is_header:
            stack.append((indent, m.group("key")))
    return out


def set_value_text(text: str, dotted: str, rendered: str) -> str:
    """Return *text* with the value at *dotted* replaced by *rendered*.

    If the key is missing it is added under its parent section (the parent is
    created at the end of the file if need be).
    """
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += nl
    target = tuple(dotted.split("."))
    index = _index_lines(lines)

    for n, path, m, is_header in index:
        if path == target:
            if is_header and not m.group("rest").startswith("#") and not m.group("rest"):
                # Section header with children: not a scalar.
                if any(p[: len(target)] == target and len(p) > len(target) for _, p, _, _ in index):
                    raise SettingsError(f"{dotted} is a section, not a single value.")
            rest = m.group("rest")
            s, e = _scalar_span(rest)
            tail = rest[e:]
            if e == 0 and rest.startswith("#"):
                tail = " " + rest          # value was empty; keep the comment
            gap = m.group("gap") or " "
            lines[n] = f"{m.group('indent')}{m.group('key')}:{gap}{rendered}{tail}{m.group('nl') or nl}"
            return "".join(lines)

    # Missing: find the deepest existing ancestor section.
    depth = len(target) - 1
    while depth > 0:
        anc = target[:depth]
        hit = next((x for x in index if x[1] == anc and x[3]), None)
        if hit:
            n, _, m, _ = hit
            base = len(m.group("indent"))
            child_indent = None
            for n2, p2, m2, _ in index:
                if n2 > n and len(p2) == depth + 1 and p2[:depth] == anc:
                    child_indent = len(m2.group("indent"))
                    break
            pad = " " * (child_indent if child_indent is not None else base + 2)
            new_lines = []
            for i, part in enumerate(target[depth:]):
                ind = pad + "  " * i
                last = i == len(target[depth:]) - 1
                new_lines.append(f"{ind}{part}: {rendered}{nl}" if last else f"{ind}{part}:{nl}")
            lines[n + 1:n + 1] = new_lines
            return "".join(lines)
        depth -= 1

    # No ancestor at all: append the whole chain.
    chain = []
    for i, part in enumerate(target):
        last = i == len(target) - 1
        chain.append(f"{'  ' * i}{part}: {rendered}{nl}" if last else f"{'  ' * i}{part}:{nl}")
    if lines and lines[-1].strip():
        lines.append(nl)
    return "".join(lines + chain)


def _without(cfg: Any, dotted: str) -> Any:
    """*cfg* minus the key at *dotted*; sections left empty by that are dropped
    too, so "section did not exist" and "section exists but is now empty"
    compare equal (adding the first key of a new section must not look like a
    change to something else)."""
    out = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
    chain = [out]
    parts = dotted.split(".")
    for p in parts[:-1]:
        nxt = chain[-1].get(p) if isinstance(chain[-1], dict) else None
        if not isinstance(nxt, dict):
            return out
        chain.append(nxt)
    chain[-1].pop(parts[-1], None)
    for depth in range(len(parts) - 1, 0, -1):
        if not chain[depth]:
            chain[depth - 1].pop(parts[depth - 1], None)
    return out


def apply_changes(text: str, changes: dict[str, Any]) -> tuple[str, list[dict]]:
    """Apply validated *changes* to YAML *text*; returns (new_text, summary).

    Raises SettingsError if a key is not allowed, a value is invalid, or the
    edit would alter anything but the requested keys.
    """
    import yaml

    if not isinstance(changes, dict) or not changes:
        raise SettingsError("Nothing to change.")
    clean: dict[str, Any] = {}
    for key, value in changes.items():
        spec = _BY_KEY.get(key)
        if spec is None:
            raise SettingsError(f"{key} cannot be changed from the web UI.")
        clean[key] = coerce(spec, value)

    before = yaml.safe_load(text) or {}
    new_text = text
    for key, value in clean.items():
        new_text = set_value_text(new_text, key, _render(_BY_KEY[key], value))
    after = yaml.safe_load(new_text) or {}

    # Safety net: only the requested keys may differ.
    expect_before, expect_after = before, after
    for key in clean:
        expect_before = _without(expect_before, key)
        expect_after = _without(expect_after, key)
    if expect_before != expect_after:
        raise SettingsError("The edit would have changed other settings; nothing was saved.")
    summary = []
    for key, value in clean.items():
        found, got = _lookup(after, key)
        if not found or got != value:
            raise SettingsError(f"Could not safely write {key}; nothing was saved.")
        _, old = _lookup(before, key)
        summary.append({"key": key, "old": old, "new": value})

    # Refuse an edit that makes the config worse. Compared against the errors
    # it already had, so a file that was already missing something unrelated
    # can still have a toggle flipped.
    from core.config_validation import validate_config
    if set(validate_config(after).errors) - set(validate_config(before).errors):
        raise SettingsError("The resulting config would not validate; nothing was saved.")
    return new_text, summary


def save_settings(config_path: str | os.PathLike, changes: dict[str, Any]) -> list[dict]:
    """Validate, edit, back up and atomically replace the config file."""
    path = Path(config_path)
    with open(path, "r", encoding="utf-8", newline="") as fh:   # keep CRLF as-is
        text = fh.read()
    new_text, summary = apply_changes(text, changes)
    if new_text == text:
        return summary
    backup = path.with_name(path.name + ".bak")
    tmp = path.with_name(path.name + ".tmp")
    with open(backup, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_text)
    os.replace(tmp, path)
    return summary
