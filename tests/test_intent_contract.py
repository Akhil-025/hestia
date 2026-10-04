# tests/test_intent_contract.py
"""
Registry <-> module contract (backlog #210).

Invariants, each of which has silently broken routing before:

1. DECLARED => REGISTERED (static). Every intent a module lists in an
   ``*_INTENTS`` class attribute must exist in modules/hecate/intent_registry.py
   for that same module (as written, or with the module prefix, because the
   orchestrator strips ``hermes_`` etc. before dispatch). An intent a module
   handles but the registry doesn't know can never be produced by the NLU, so
   the code behind it is unreachable by voice or chat.

2. REGISTERED => ACCEPTED (dynamic). Every intent the registry sends to a
   module must be accepted by that module's ``can_handle()`` once the prefix is
   stripped. If it isn't, the orchestrator quietly reroutes the query to
   another module or to chat and the user just gets a wrong answer.

3. Every module name the registry points at is a real module.

(1) reads the source with ``ast``, so it needs no third-party packages and
runs everywhere. (2) imports the module classes; one whose dependencies aren't
installed in this environment is skipped by name rather than silently passed.

If you add an intent and (1) or (2) fails: add it to the registry and to
config/nlu_prompt.txt (tests/test_registry_contract.py checks the second).
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from modules.hecate.intent_registry import INTENT_MODULE_MAP, strip_module_prefix

_ROOT = Path(__file__).resolve().parent.parent
_MODULES_DIR = _ROOT / "modules"

# Intents a module accepts that the NLU is deliberately never asked to emit.
# The list is closed: a new entry needs a reason, and an entry that stops being
# true fails test_alias_allowlist_has_no_stale_entries so it can't rot.
DOCUMENTED_ALIASES: dict[str, dict[str, str]] = {
    "artemis": {"list_goals": "legacy spelling of the registered get_goals"},
    "athena": {
        "query_documents": "legacy spelling of athena_search",
        "search_documents": "legacy spelling of athena_search",
        "ingest_documents": "legacy spelling of athena_ingest",
    },
    "mnemosyne": {
        # Internal names: Hecate's text-trigger tier dispatches "recall" directly
        # (see the comment above "learn_fact" in intent_registry.py), so the NLU
        # is deliberately never asked to emit them.
        "recall": "dispatched by Hecate's mnemosyne text trigger, not by the NLU",
        "remember": "internal; reached through learn_fact and the recall path",
        "get_facts": "internal; reached through the recall path",
        # Registered to core on purpose: core is a superset and wins (test_hecate.py).
        "get_user_info": "registered to core by design",
        "get_history": "registered to core by design",
    },
}


class _Unresolved(Exception):
    pass


def _module_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            out[node.target.id] = node.value
    return out


def _resolve_import(path: Path, node: ast.ImportFrom, name: str) -> Optional[tuple[Path, str]]:
    """Source file and original name for ``from .x import name`` (relative only)."""
    if node.level == 0:
        return None
    base = path.parent
    for _ in range(node.level - 1):
        base = base.parent
    parts = (node.module or "").split(".") if node.module else []
    target = base.joinpath(*parts)
    for cand in (target.with_suffix(".py"), target / "__init__.py"):
        if cand.exists():
            return cand, name
    return None


def _eval_set(node: ast.expr, path: Path, tree: ast.Module, depth: int = 0) -> frozenset[str]:
    """Evaluate a set expression made of literals, same-file names, relative
    imports and ``|`` / ``+`` / ``-``. Anything else is unresolved."""
    if depth > 6:
        raise _Unresolved("too deep")
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        vals = []
        for elt in node.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                raise _Unresolved("non-literal element")
            vals.append(elt.value)
        return frozenset(vals)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("frozenset", "set"):
        return _eval_set(node.args[0], path, tree, depth + 1) if node.args else frozenset()
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.Add, ast.Sub)):
        left = _eval_set(node.left, path, tree, depth + 1)
        right = _eval_set(node.right, path, tree, depth + 1)
        return left - right if isinstance(node.op, ast.Sub) else left | right
    if isinstance(node, ast.Name):
        local = _module_assignments(tree)
        if node.id in local:
            return _eval_set(local[node.id], path, tree, depth + 1)
        for imp in tree.body:
            if isinstance(imp, ast.ImportFrom):
                for alias in imp.names:
                    if (alias.asname or alias.name) == node.id:
                        found = _resolve_import(path, imp, alias.name)
                        if found:
                            other, orig = found
                            other_tree = ast.parse(other.read_text(encoding="utf-8"))
                            other_assign = _module_assignments(other_tree)
                            if orig in other_assign:
                                return _eval_set(other_assign[orig], other, other_tree, depth + 1)
    raise _Unresolved(ast.dump(node)[:60])


def _class_attr(cls: ast.ClassDef, attr: str) -> Optional[ast.expr]:
    for st in cls.body:
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if isinstance(t, ast.Name) and t.id == attr:
                    return st.value
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == attr:
            return st.value
    return None


def discover_declarations() -> tuple[dict[str, set[str]], dict[str, str], list[str]]:
    """``(declared, class_files, unresolved)``.

    ``declared[module_name]`` is every intent in any ``*_INTENTS`` set on a
    class whose ``name = "..."`` is a module name. ``unresolved`` lists
    attributes that couldn't be evaluated statically (reported, never ignored).
    ``*_INTENT_ALIASES`` dicts are not scanned: they map a spelling onto an
    intent the module already declares.
    """
    declared: dict[str, set[str]] = {}
    files: dict[str, str] = {}
    unresolved: list[str] = []
    for path in sorted(_MODULES_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            unresolved.append(f"{path}: unparsable")
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            name_node = _class_attr(cls, "name")
            if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
                continue
            mod_name = name_node.value
            for st in cls.body:
                targets = []
                if isinstance(st, ast.Assign):
                    targets = [t for t in st.targets if isinstance(t, ast.Name)]
                elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    targets = [st.target]
                for t in targets:
                    if not t.id.endswith("_INTENTS") or st.value is None:
                        continue
                    files.setdefault(mod_name, str(path.relative_to(_ROOT)))
                    try:
                        declared.setdefault(mod_name, set()).update(_eval_set(st.value, path, tree))
                    except _Unresolved as exc:
                        unresolved.append(f"{path.relative_to(_ROOT)}::{cls.name}.{t.id} ({exc})")
    return declared, files, unresolved


DECLARED, CLASS_FILES, UNRESOLVED = discover_declarations()


def _registered_for(module: str, intent: str) -> bool:
    """Is *intent* (as a module declares it) registered to *module*?"""
    return any(INTENT_MODULE_MAP.get(c) == module for c in (intent, f"{module}_{intent}"))


# --------------------------------------------------------- 1. declared => registered

def test_the_scan_found_the_modules_it_should():
    # A scanner that finds nothing would make every check below vacuously pass.
    assert len(DECLARED) >= 8, sorted(DECLARED)
    assert {"apollo", "hermes", "hephaestus", "mnemosyne"} <= set(DECLARED)
    assert sum(len(v) for v in DECLARED.values()) > 100


def test_every_declared_set_could_be_read():
    assert UNRESOLVED == [], (
        "These _INTENTS attributes are built in a way the checker can't read; make them a "
        f"literal set or a union of named sets so the contract is checkable: {UNRESOLVED}"
    )


def test_mnemosyne_extension_intents_are_resolved_through_the_import():
    # mnemosyne declares frozenset({...}) | EXTENSION_INTENTS from .extensions
    assert "start_quiz" in DECLARED["mnemosyne"] and "recall" in DECLARED["mnemosyne"]


@pytest.mark.parametrize("module", sorted(DECLARED))
def test_declared_intents_are_in_the_registry(module):
    allowed = DOCUMENTED_ALIASES.get(module, {})
    missing = sorted(i for i in DECLARED[module]
                     if not _registered_for(module, i) and i not in allowed)
    assert not missing, (
        f"{CLASS_FILES.get(module)} declares {missing} but modules/hecate/intent_registry.py "
        f"has no entry for {module!r} with that name (or '{module}_<name>'). The NLU can never "
        "emit them, so nothing can reach that code. Register them, drop them, or, if they are "
        "deliberate legacy aliases, add them to DOCUMENTED_ALIASES with a reason."
    )


def test_alias_allowlist_has_no_stale_entries():
    for module, aliases in DOCUMENTED_ALIASES.items():
        assert module in DECLARED, f"{module} no longer declares any intents"
        for intent, reason in aliases.items():
            assert reason.strip(), f"{module}.{intent} needs a reason"
            assert intent in DECLARED[module], f"{module} no longer declares {intent}; remove it"
            assert not _registered_for(module, intent), (
                f"{module}.{intent} is registered now, so it isn't an alias; remove it from the list"
            )


# --------------------------------------------------------- 3. registry module names

def test_every_registry_target_is_a_real_module():
    real = set(DECLARED) | {"core", "iris"}      # iris declares its set inline in can_handle()
    unknown = sorted({m for m in INTENT_MODULE_MAP.values() if m not in real})
    assert not unknown, f"registry routes to modules with no class of that name: {unknown}"


# --------------------------------------------------------- 2. registered => accepted

_CLASSES = {
    "apollo": ("modules.apollo.engine", "ApolloEngine"),
    "ares": ("modules.ares.engine", "AresEngine"),
    "artemis": ("modules.artemis.engine", "ArtemisEngine"),
    "athena": ("modules.athena.engine", "AthenaEngine"),
    "chronos": ("modules.chronos.engine", "ChronosEngine"),
    "core": ("modules.hestia.core_module", "CoreModule"),
    "dionysus": ("modules.dionysus.engine", "DionysusEngine"),
    "hephaestus": ("modules.hephaestus.engine", "HephaestusEngine"),
    "hermes": ("modules.hermes.engine", "HermesEngine"),
    "iris": ("modules.iris.iris_engine", "IrisEngine"),
    "metis": ("modules.metis.engine", "MetisEngine"),
    "mnemosyne": ("modules.mnemosyne.engine", "MnemosyneEngine"),
    "orpheus": ("modules.orpheus", "OrpheusEngine"),
    "pluto": ("modules.pluto.engine", "PlutoEngine"),
}


def test_every_registry_module_has_a_class_to_check():
    assert set(INTENT_MODULE_MAP.values()) <= set(_CLASSES), (
        "add the new module to _CLASSES: "
        f"{sorted(set(INTENT_MODULE_MAP.values()) - set(_CLASSES))}"
    )


@pytest.mark.parametrize("module", sorted(_CLASSES))
def test_module_accepts_every_intent_registered_to_it(module):
    mod_name, cls_name = _CLASSES[module]
    try:
        cls = getattr(importlib.import_module(mod_name), cls_name)
    except ImportError as exc:                     # dependency missing in THIS environment
        pytest.skip(f"{mod_name} can't be imported here: {exc}")
    # can_handle() only reads class-level sets in every module, so an
    # uninitialised instance avoids opening databases and models.
    probe = object.__new__(cls)
    rejected = []
    for intent, owner in INTENT_MODULE_MAP.items():
        if owner != module:
            continue
        stripped = strip_module_prefix(intent)      # what the orchestrator passes in
        try:
            ok = probe.can_handle(stripped)
        except AttributeError as exc:
            pytest.skip(f"{cls_name}.can_handle needs instance state ({exc}); can't probe without a full build")
        if not ok:
            rejected.append(f"{intent} (as {stripped!r})")
    assert not rejected, (
        f"{cls_name}.can_handle() rejects intents the registry routes to it: {rejected}. "
        "The orchestrator would silently reroute those queries."
    )


def test_a_made_up_intent_is_rejected_by_the_modules_that_can_be_probed():
    # Guards the guard: a can_handle that returns True for everything would
    # pass the check above without proving anything.
    probed = 0
    for module, (mod_name, cls_name) in _CLASSES.items():
        if module == "core":                         # CoreModule is the catch-all by design
            continue
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
            result = object.__new__(cls).can_handle("zz_definitely_not_an_intent_zz")
        except (ImportError, AttributeError):
            continue
        probed += 1
        assert result is False, f"{cls_name}.can_handle accepts anything"
    assert probed >= 6
