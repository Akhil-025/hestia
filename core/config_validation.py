"""
core/config_validation.py

Fail-fast validation for ``config/laptop_config.yaml`` (backlog #14).

Why this exists
---------------
Before this, a malformed or partially-filled config didn't fail at
startup — it failed *deep inside a module*, minutes later, in a form that
didn't name the config key responsible. Concretely:

  - ``ollama.port: "11434"`` (a string, because YAML quoting is easy to get
    wrong) surfaced as a connection error from ``core/ollama_client.py``,
    not as "port must be an integer".
  - a missing ``database.path`` surfaced as an sqlite3 error about an empty
    filename from whichever module happened to touch the DB first.
  - ``webui.enabled: "false"`` (the *string* "false", which is truthy in
    Python) silently started the web UI anyway.

The validator is intentionally conservative: it checks the keys Hestia
genuinely cannot run without, plus the *types* of widely-used optional
keys, and otherwise stays out of the way. Unknown keys are allowed (they
may belong to a module added later) but reported as warnings, which is how
typos like ``databse:`` get caught.

Design notes
------------
- Returns a report instead of raising, so callers choose the policy:
  ``main.py`` raises on errors at startup, while ``--dry-run`` and the
  tests just print them.
- Every message names the offending key path (``ollama.port``), the
  problem, and what was expected — that's the entire point of failing here
  rather than later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# (dotted key path, expected python type(s), required?)
#
# "Required" means Hestia cannot start coherently without it. Everything
# else is type-checked only when present, so an existing working config
# never becomes invalid by virtue of this file being added.
_SCHEMA: tuple[tuple[str, tuple[type, ...], bool], ...] = (
    # Inference stack — no Ollama block means no NLU and no chat.
    ("ollama", (dict,), True),
    ("ollama.model", (str,), True),
    ("ollama.host", (str,), False),
    ("ollama.port", (int,), False),

    # Persistence — every module's writes land here.
    ("database", (dict,), True),
    ("database.path", (str,), True),

    # Optional NLU override block (see HestiaBuilder.build_nlu).
    ("nlu", (dict,), False),
    ("nlu.model", (str,), False),
    ("nlu.host", (str,), False),
    ("nlu.port", (int,), False),
    ("nlu.prompt_path", (str,), False),

    # Timezone drives Chronos, Hermes and CoreModule alike; a wrong type
    # here offsets every reminder and calendar event.
    ("chronos", (dict,), False),
    ("chronos.timezone", (str,), False),

    # Feature flags whose truthiness decides whether a subsystem boots.
    ("webui", (dict,), False),
    ("webui.enabled", (bool,), False),
    ("webui.host", (str,), False),
    ("webui.port", (int,), False),
    ("browser", (dict,), False),
    ("browser.enabled", (bool,), False),
    ("browser.headless", (bool,), False),
    ("google", (dict,), False),
    ("google.enabled", (bool,), False),
    ("telegram", (dict,), False),
    ("telegram.enabled", (bool,), False),
    ("sync", (dict,), False),
    ("sync.enabled", (bool,), False),
    ("athena", (dict,), False),
    ("iris", (dict,), False),
    ("mnemosyne", (dict,), False),
)

# Top-level keys the app knows about. Anything else is reported as a
# warning (likely a typo), never an error.
_KNOWN_TOP_LEVEL: frozenset[str] = frozenset({
    "ollama", "nlu", "llm", "database", "chronos", "webui", "browser",
    "google", "telegram", "sync", "athena", "iris", "mnemosyne", "stt",
    "tts", "wake_word", "barge_in", "scheduler", "heartbeat", "skills",
    "hephaestus", "pluto", "dionysus", "apollo", "artemis", "logging",
    "observability",
})


@dataclass
class ValidationReport:
    """Outcome of validating a config mapping."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        """Multi-line, human-readable summary suitable for a log or stderr."""
        lines: list[str] = []
        if self.errors:
            lines.append(f"{len(self.errors)} configuration error(s):")
            lines.extend(f"  ERROR  {m}" for m in self.errors)
        if self.warnings:
            lines.append(f"{len(self.warnings)} configuration warning(s):")
            lines.extend(f"  WARN   {m}" for m in self.warnings)
        if not lines:
            lines.append("Configuration OK.")
        return "\n".join(lines)


class ConfigError(ValueError):
    """Raised by :func:`validate_or_raise` when the config is unusable."""


def _lookup(cfg: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    """
    Resolve a dotted key path.

    Returns ``(found, value)``. ``found`` is False if any segment is
    missing, which is how "required key absent" is distinguished from
    "key present but set to null".
    """
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _type_name(types: Iterable[type]) -> str:
    return " or ".join(t.__name__ for t in types)


def validate_config(cfg: Any) -> ValidationReport:
    """
    Validate a loaded config mapping and return a :class:`ValidationReport`.

    Never raises — callers decide whether a given error is fatal.
    """
    report = ValidationReport()

    if not isinstance(cfg, dict):
        report.errors.append(
            f"Top-level config must be a YAML mapping, got {type(cfg).__name__}."
        )
        return report

    for dotted, types, required in _SCHEMA:
        found, value = _lookup(cfg, dotted)

        if not found:
            if required:
                report.errors.append(
                    f"{dotted}: required key is missing. Add it to your config "
                    f"(see config/laptop_config.example.yaml) — expected "
                    f"{_type_name(types)}."
                )
            continue

        if value is None:
            # Explicit null. Fatal for required keys, harmless otherwise
            # (the builders all pass a default to .get()).
            if required:
                report.errors.append(
                    f"{dotted}: is set to null but is required — expected "
                    f"{_type_name(types)}."
                )
            continue

        # bool is a subclass of int in Python, so an `enabled: 1` would
        # sail through an (int,) check and a `port: true` through a
        # (bool,) one. Check bool explicitly in both directions.
        if bool in types and not isinstance(value, bool):
            report.errors.append(
                f"{dotted}: expected a YAML boolean (true/false), got "
                f"{type(value).__name__} {value!r}. Note that the quoted "
                f'string "false" is truthy in Python and would silently '
                f"enable this."
            )
            continue
        if bool not in types and isinstance(value, bool):
            report.errors.append(
                f"{dotted}: expected {_type_name(types)}, got boolean {value!r}."
            )
            continue

        if not isinstance(value, types):
            hint = ""
            if int in types and isinstance(value, str) and value.strip().isdigit():
                hint = (
                    f" It looks like a quoted number — write it as "
                    f"{value.strip()} without quotes."
                )
            report.errors.append(
                f"{dotted}: expected {_type_name(types)}, got "
                f"{type(value).__name__} {value!r}.{hint}"
            )
            continue

        if str in types and isinstance(value, str) and not value.strip():
            report.errors.append(f"{dotted}: must not be an empty string.")

    # Port sanity — a valid int in an impossible range still fails later.
    for dotted in ("ollama.port", "nlu.port", "webui.port"):
        found, value = _lookup(cfg, dotted)
        if found and isinstance(value, int) and not isinstance(value, bool):
            if not (1 <= value <= 65535):
                report.errors.append(
                    f"{dotted}: {value} is outside the valid TCP port range "
                    f"(1-65535)."
                )

    # Typo detection on top-level keys.
    for key in cfg:
        if key not in _KNOWN_TOP_LEVEL:
            report.warnings.append(
                f"{key}: unrecognised top-level config section — ignored by "
                f"Hestia. If this is a typo, nothing under it takes effect."
            )

    return report


def validate_or_raise(cfg: Any, source: str = "config") -> ValidationReport:
    """
    Validate *cfg*, raising :class:`ConfigError` on any error.

    This is what ``main._load_config`` calls, so a bad config fails at
    startup with every problem listed at once, rather than one-at-a-time
    over successive restarts.
    """
    report = validate_config(cfg)
    if not report.ok:
        raise ConfigError(
            f"Invalid configuration in {source}:\n{report.format()}"
        )
    return report
