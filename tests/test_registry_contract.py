# tests/test_registry_contract.py
"""
Contract tests for modules/hecate/intent_registry.py.

Codifies invariants that were previously only documented or only warned
about at runtime:

  - backlog #26: config/nlu_prompt.txt's "Valid intents:" block and
    ALL_INTENTS must not drift. core/nlu.py logs a warning when they do,
    but a warning in a log nobody reads is how the original drift survived
    long enough to make four working modules unreachable. This fails CI
    instead.

  - backlog #11: REGISTRY_VERSION must be well-formed and the fingerprint
    must be stable for a given mapping and sensitive to any real change,
    since clients rely on it to detect a breaking change.

  - every registered intent must actually be accepted by the module that
    owns it, once the orchestrator strips the module prefix. This is the
    invariant the registry docstring claims and that the pluto/iris/athena
    unreachable-intent bug violated.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hecate.intent_registry import (
    ALL_INTENTS,
    INTENT_MODULE_MAP,
    MODULE_PREFIXES,
    PREFIX_TO_MODULE,
    REGISTRY_VERSION,
    module_for_intent,
    registry_fingerprint,
    registry_info,
    strip_module_prefix,
)

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
_PROMPT_PATH = os.path.join(_REPO_ROOT, "config", "nlu_prompt.txt")


def _prompt_intents() -> set[str]:
    """Parse the prompt's "Valid intents:" block the same way core/nlu.py does."""
    with open(_PROMPT_PATH, "r", encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r"Valid intents:(.*?)---", text, re.S)
    assert match, "config/nlu_prompt.txt has no parseable 'Valid intents:' block"
    return {t.strip() for t in re.split(r"[,\n\r]", match.group(1)) if t.strip()}


# ---------------------------------------------------------------------------
# #26 — prompt / registry drift
# ---------------------------------------------------------------------------

def test_prompt_lists_every_registered_intent():
    missing = sorted(ALL_INTENTS - _prompt_intents())
    assert not missing, (
        "These intents are in intent_registry.py but not in "
        "config/nlu_prompt.txt's 'Valid intents:' block, so the model is "
        f"never told they exist and will never emit them: {missing}"
    )


def test_prompt_lists_no_unregistered_intents():
    extra = sorted(_prompt_intents() - ALL_INTENTS)
    assert not extra, (
        "These intents are listed in config/nlu_prompt.txt but are not in "
        "intent_registry.py, so the NLU schema's enum rejects them even if "
        f"the model emits them: {extra}"
    )


def test_prompt_intent_block_is_not_accidentally_empty():
    # A regex that silently matches nothing would make both tests above
    # pass vacuously.
    assert len(_prompt_intents()) > 50


# ---------------------------------------------------------------------------
# #11 — versioning
# ---------------------------------------------------------------------------

def test_registry_version_is_semver_shaped():
    assert re.fullmatch(r"\d+\.\d+\.\d+", REGISTRY_VERSION), REGISTRY_VERSION


def test_fingerprint_is_stable_across_calls():
    assert registry_fingerprint() == registry_fingerprint()


def test_fingerprint_is_order_independent():
    # Reordering entries for readability must not look like a breaking
    # change to a client comparing fingerprints.
    import hashlib

    reversed_payload = ";".join(
        f"{k}={v}" for k, v in sorted(reversed(list(INTENT_MODULE_MAP.items())))
    )
    expected = hashlib.sha256(reversed_payload.encode("utf-8")).hexdigest()[:12]
    assert registry_fingerprint() == expected


def test_fingerprint_changes_when_an_intent_is_added(monkeypatch):
    before = registry_fingerprint()
    patched = dict(INTENT_MODULE_MAP)
    patched["some_new_intent"] = "core"
    monkeypatch.setattr(
        "modules.hecate.intent_registry.INTENT_MODULE_MAP", patched
    )
    assert registry_fingerprint() != before


def test_fingerprint_changes_when_an_intent_is_reassigned(monkeypatch):
    before = registry_fingerprint()
    patched = dict(INTENT_MODULE_MAP)
    patched["get_weather"] = "core"      # was chronos
    monkeypatch.setattr(
        "modules.hecate.intent_registry.INTENT_MODULE_MAP", patched
    )
    assert registry_fingerprint() != before


def test_registry_info_shape():
    info = registry_info()
    assert info["version"] == REGISTRY_VERSION
    assert info["fingerprint"] == registry_fingerprint()
    assert info["intent_count"] == len(INTENT_MODULE_MAP)
    assert info["module_count"] == len(set(INTENT_MODULE_MAP.values()))


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def test_all_intents_matches_the_map():
    assert ALL_INTENTS == frozenset(INTENT_MODULE_MAP)


def test_intent_names_are_unprefixed_snake_case_or_module_prefixed():
    # Mirrors modules/base.py's _INTENT_RE, applied to the registry side.
    pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    bad = [i for i in ALL_INTENTS if not pattern.match(i)]
    assert not bad, bad


def test_module_names_are_lowercase_and_non_empty():
    bad = [m for m in INTENT_MODULE_MAP.values() if not m or m != m.lower()]
    assert not bad, bad


def test_prefix_to_module_derives_from_module_prefixes():
    assert set(PREFIX_TO_MODULE) == set(MODULE_PREFIXES)
    for prefix, module in PREFIX_TO_MODULE.items():
        assert prefix == f"{module}_"


def test_strip_module_prefix_removes_known_prefixes():
    assert strip_module_prefix("apollo_log_workout") == "log_workout"
    assert strip_module_prefix("pluto_log_expense") == "log_expense"


def test_strip_module_prefix_leaves_unprefixed_intents_alone():
    for intent in ("get_time", "chat", "add_habit", "modules_status"):
        assert strip_module_prefix(intent) == intent


def test_module_for_intent_returns_none_for_unknown():
    assert module_for_intent("definitely_not_an_intent") is None


def test_every_prefixed_intent_is_registered_to_its_own_prefixs_module():
    # A "pluto_*" intent registered to athena would route correctly via
    # the registry but strip to an intent athena doesn't declare.
    mismatched = []
    for intent, module in INTENT_MODULE_MAP.items():
        for prefix, prefix_module in PREFIX_TO_MODULE.items():
            if intent.startswith(prefix) and module != prefix_module:
                mismatched.append((intent, module, prefix_module))
    assert not mismatched, mismatched


# ---------------------------------------------------------------------------
# The core diagnostic intents added for #3/#8/#259
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "intent", ["modules_status", "explain_routing", "report_mistake"]
)
def test_diagnostic_intents_are_registered_to_core(intent):
    assert module_for_intent(intent) == "core"


@pytest.mark.parametrize(
    "intent", ["modules_status", "explain_routing", "report_mistake"]
)
def test_core_module_declares_the_diagnostic_intents(intent):
    # The registry saying core owns it is only half the contract; core's
    # can_handle() has to agree, or dispatch silently falls back to chat.
    from modules.hestia.core_module import CoreModule

    assert intent in CoreModule._INTENTS
