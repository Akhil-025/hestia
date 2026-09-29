"""
core/module_loader.py

Plugin/skill auto-registration (backlog #9).

Why
---
Adding a new full module (Pluto, Athena, ...) genuinely needs main.py's
attention — it has its own config section, its own constructor
dependencies (memory, browser_agent, ollama_cfg), sometimes its own DB.
That's a deliberate decision, made explicitly in
`HestiaBuilder.build_orchestrator`, and this loader does NOT change it —
none of the 14 existing modules move here.

What it does add is a lighter-weight tier for a *skill*: a single-file
`BaseModule` with a small, uniform dependency list (at most `ollama_cfg`
and `memory`), that you want to be able to drop into `skills/` and have
picked up on the next restart without editing `main.py` at all.

Convention
----------
Every `*.py` file directly under the configured skills directory
(`skills.path` in `laptop_config.yaml`, default `"skills/"`) that defines
a top-level class named exactly `Skill` is a candidate:

    # skills/word_of_the_day.py
    from modules.base import BaseModule

    class Skill(BaseModule):
        name = "word_of_the_day"
        _INTENTS = {"get_word_of_the_day"}

        def __init__(self, ollama_cfg=None, memory=None):
            ...

        def can_handle(self, intent):
            return intent in self._INTENTS

        def handle(self, intent, entities, context):
            ...

`discover_skills()` only *finds and instantiates* candidates — it never
touches `sys.path` beyond what's needed to import a file by path, never
executes anything outside the configured directory, and constructs each
class with only the keyword arguments its own `__init__` declares (so a
skill that wants neither `ollama_cfg` nor `memory` can have a bare
`__init__(self)`).

A skill that fails to import, isn't a `BaseModule` subclass, has a name
collision with an already-registered module, or raises during
construction is skipped with a logged warning — one broken skill file
must never stop the other nine, or Hestia herself, from starting.
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from modules.base import BaseModule

logger = logging.getLogger(__name__)

_SKILL_CLASS_NAME = "Skill"
_DEFAULT_SKILLS_DIR = Path("skills")


def discover_skills(
    directory: str | Path | None = _DEFAULT_SKILLS_DIR,
    *,
    ollama_cfg: Optional[dict] = None,
    memory: Any = None,
    skip_names: Iterable[str] = (),
) -> list[BaseModule]:
    """
    Import every `*.py` file directly under *directory*, instantiate its
    `Skill` class (if any), and return the successfully constructed
    instances.

    Not recursive — a skill is a single file, deliberately, to keep the
    convention obvious and to avoid accidentally importing a skill's own
    private helper modules as if they were skills.

    `skip_names` is the set of module names already registered (the 14
    built-in modules plus anything registered earlier); a skill claiming
    one of those names is skipped rather than silently replacing it —
    unlike `HestiaOrchestrator.register()`, which allows replacement for
    modules registered directly by code, a *dropped-in file* silently
    shadowing a built-in module is a much easier mistake to make by
    accident (a copy-pasted filename) and a much worse one to debug.
    """
    skip = set(skip_names)
    path = Path(directory) if directory is not None else None
    if path is None or not path.is_dir():
        logger.debug("No skills directory at %s; skipping skill discovery.", path)
        return []

    skills: list[BaseModule] = []
    for file_path in sorted(path.glob("*.py")):
        if file_path.name.startswith("_"):
            continue  # leading underscore = private helper, not a skill

        skill_cls = _load_skill_class(file_path)
        if skill_cls is None:
            continue

        instance = _instantiate(skill_cls, file_path, ollama_cfg, memory)
        if instance is None:
            continue

        if instance.name in skip:
            logger.warning(
                "Skill %s (from %s) claims name %r, which is already "
                "registered; skipping it to avoid silently shadowing an "
                "existing module. Rename the skill's `name` attribute.",
                skill_cls.__name__, file_path.name, instance.name,
            )
            continue

        skip.add(instance.name)  # a second file can't claim the same name either
        skills.append(instance)
        logger.info(
            "Loaded skill %r from %s.", instance.name, file_path.name
        )

    return skills


def _load_skill_class(file_path: Path) -> Optional[type[BaseModule]]:
    """Import *file_path* as an isolated module and return its Skill class."""
    module_name = f"hestia_skill_{file_path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            logger.warning("Could not build an import spec for %s.", file_path)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        logger.exception("Skill file %s raised while importing; skipping.", file_path)
        return None

    skill_cls = getattr(module, _SKILL_CLASS_NAME, None)
    if skill_cls is None:
        logger.debug(
            "%s defines no top-level `Skill` class; not a skill file.", file_path.name
        )
        return None
    if not (inspect.isclass(skill_cls) and issubclass(skill_cls, BaseModule)):
        logger.warning(
            "%s's `Skill` is not a modules.base.BaseModule subclass; skipping.",
            file_path.name,
        )
        return None
    return skill_cls


def _instantiate(
    skill_cls: type[BaseModule],
    file_path: Path,
    ollama_cfg: Optional[dict],
    memory: Any,
) -> Optional[BaseModule]:
    """
    Construct *skill_cls*, passing only the kwargs its own `__init__`
    declares.

    Inspecting the signature rather than always passing both keyword
    arguments lets a skill with `def __init__(self):` work without
    raising a TypeError for an unexpected keyword — the whole point of
    keeping the skill contract smaller than a full module's.
    """
    try:
        params = inspect.signature(skill_cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}

    kwargs: dict[str, Any] = {}
    if "ollama_cfg" in params:
        kwargs["ollama_cfg"] = ollama_cfg or {}
    if "memory" in params:
        kwargs["memory"] = memory

    try:
        instance = skill_cls(**kwargs)
    except Exception:
        logger.exception(
            "Skill class %s (from %s) raised during construction; skipping.",
            skill_cls.__name__, file_path.name,
        )
        return None

    if not isinstance(instance, BaseModule):  # pragma: no cover - defensive
        logger.warning(
            "Skill %s (from %s) did not produce a BaseModule instance; skipping.",
            skill_cls.__name__, file_path.name,
        )
        return None
    return instance
