# tests/test_module_loader.py
"""
Tests for core/module_loader.py (backlog #9).

Each test writes real .py files into a tmp_path directory and points
discover_skills() at it, rather than mocking importlib — the whole point
of this module is "does a file on disk actually get imported and
instantiated correctly", which a mock can't verify.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.module_loader import discover_skills
from modules.base import BaseModule


def write_skill(tmp_path, filename, source):
    (tmp_path / filename).write_text(source, encoding="utf-8")


_GOOD_SKILL = '''
from modules.base import BaseModule

class Skill(BaseModule):
    name = "word_of_the_day"
    _INTENTS = {"get_word_of_the_day"}

    def can_handle(self, intent):
        return intent in self._INTENTS

    def handle(self, intent, entities, context):
        return {"response": "serendipity", "data": {}, "confidence": 0.9}
'''

_SKILL_WITH_DEPS = '''
from modules.base import BaseModule

class Skill(BaseModule):
    name = "needs_deps"
    _INTENTS = {"x"}

    def __init__(self, ollama_cfg=None, memory=None):
        self.ollama_cfg = ollama_cfg
        self.memory = memory

    def can_handle(self, intent):
        return True

    def handle(self, intent, entities, context):
        return {"response": "ok", "data": {}, "confidence": 0.9}
'''


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_discovers_and_instantiates_a_valid_skill(tmp_path):
    write_skill(tmp_path, "wotd.py", _GOOD_SKILL)
    skills = discover_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "word_of_the_day"
    assert isinstance(skills[0], BaseModule)


def test_discovered_skill_works(tmp_path):
    write_skill(tmp_path, "wotd.py", _GOOD_SKILL)
    skill = discover_skills(tmp_path)[0]
    assert skill.can_handle("get_word_of_the_day") is True
    result = skill.handle("get_word_of_the_day", {}, {})
    assert result["response"] == "serendipity"


def test_multiple_skill_files_are_all_loaded(tmp_path):
    write_skill(tmp_path, "a.py", _GOOD_SKILL.replace("word_of_the_day", "skill_a"))
    write_skill(tmp_path, "b.py", _GOOD_SKILL.replace("word_of_the_day", "skill_b"))
    names = {s.name for s in discover_skills(tmp_path)}
    assert names == {"skill_a", "skill_b"}


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

def test_only_declared_kwargs_are_passed(tmp_path):
    # _GOOD_SKILL's __init__ (inherited, takes no args) must not receive
    # ollama_cfg/memory, or construction would raise a TypeError.
    write_skill(tmp_path, "wotd.py", _GOOD_SKILL)
    skills = discover_skills(tmp_path, ollama_cfg={"model": "x"}, memory=object())
    assert len(skills) == 1


def test_skill_declaring_deps_receives_them(tmp_path):
    write_skill(tmp_path, "deps.py", _SKILL_WITH_DEPS)
    sentinel_memory = object()
    skills = discover_skills(
        tmp_path, ollama_cfg={"model": "mistral"}, memory=sentinel_memory
    )
    assert skills[0].ollama_cfg == {"model": "mistral"}
    assert skills[0].memory is sentinel_memory


def test_no_ollama_cfg_or_memory_supplied_defaults_sensibly(tmp_path):
    write_skill(tmp_path, "deps.py", _SKILL_WITH_DEPS)
    skills = discover_skills(tmp_path)
    assert skills[0].ollama_cfg == {}
    assert skills[0].memory is None


# ---------------------------------------------------------------------------
# Robustness — one bad file must not break the others
# ---------------------------------------------------------------------------

def test_file_with_no_skill_class_is_ignored(tmp_path):
    write_skill(tmp_path, "not_a_skill.py", "x = 1\n")
    assert discover_skills(tmp_path) == []


def test_syntax_error_in_one_file_does_not_block_others(tmp_path):
    write_skill(tmp_path, "broken.py", "def (((\n")
    write_skill(tmp_path, "good.py", _GOOD_SKILL)
    names = {s.name for s in discover_skills(tmp_path)}
    assert names == {"word_of_the_day"}


def test_skill_that_raises_on_construction_is_skipped(tmp_path):
    source = '''
from modules.base import BaseModule

class Skill(BaseModule):
    name = "explodes"
    def __init__(self):
        raise RuntimeError("nope")
    def can_handle(self, intent):
        return True
    def handle(self, intent, entities, context):
        return {"response": "", "data": {}, "confidence": 0.0}
'''
    write_skill(tmp_path, "explodes.py", source)
    write_skill(tmp_path, "good.py", _GOOD_SKILL)
    names = {s.name for s in discover_skills(tmp_path)}
    assert names == {"word_of_the_day"}


def test_non_basemodule_skill_class_is_skipped(tmp_path):
    write_skill(tmp_path, "wrong_type.py", "class Skill:\n    pass\n")
    assert discover_skills(tmp_path) == []


def test_skill_missing_can_handle_is_skipped_not_raised(tmp_path):
    # BaseModule is an ABC — a subclass missing an abstract method can't
    # even be instantiated. Must be caught, not propagated.
    source = '''
from modules.base import BaseModule

class Skill(BaseModule):
    name = "incomplete"
    def handle(self, intent, entities, context):
        return {"response": "", "data": {}, "confidence": 0.0}
'''
    write_skill(tmp_path, "incomplete.py", source)
    assert discover_skills(tmp_path) == []


# ---------------------------------------------------------------------------
# File-selection convention
# ---------------------------------------------------------------------------

def test_files_starting_with_underscore_are_not_treated_as_skills(tmp_path):
    write_skill(tmp_path, "_helpers.py", _GOOD_SKILL)
    assert discover_skills(tmp_path) == []


def test_subdirectories_are_not_recursed_into(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    write_skill(nested, "wotd.py", _GOOD_SKILL)
    assert discover_skills(tmp_path) == []


def test_non_python_files_are_ignored(tmp_path):
    (tmp_path / "README.md").write_text("not a skill", encoding="utf-8")
    assert discover_skills(tmp_path) == []


# ---------------------------------------------------------------------------
# Name collisions with existing modules
# ---------------------------------------------------------------------------

def test_skill_colliding_with_a_registered_module_name_is_skipped(tmp_path):
    write_skill(tmp_path, "wotd.py", _GOOD_SKILL)
    skills = discover_skills(tmp_path, skip_names={"word_of_the_day"})
    assert skills == []


def test_two_skills_claiming_the_same_name_only_the_first_wins(tmp_path):
    write_skill(tmp_path, "a_first.py", _GOOD_SKILL)
    write_skill(tmp_path, "b_second.py", _GOOD_SKILL)
    # sorted() glob order makes "a_first.py" load before "b_second.py".
    skills = discover_skills(tmp_path)
    assert len(skills) == 1


def test_skip_names_defaults_to_empty(tmp_path):
    write_skill(tmp_path, "wotd.py", _GOOD_SKILL)
    assert len(discover_skills(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Missing / disabled directory
# ---------------------------------------------------------------------------

def test_missing_directory_returns_empty_list_without_raising(tmp_path):
    assert discover_skills(tmp_path / "does_not_exist") == []


def test_none_directory_returns_empty_list():
    assert discover_skills(None) == []


def test_empty_directory_returns_empty_list(tmp_path):
    assert discover_skills(tmp_path) == []


def test_a_file_where_a_directory_is_expected_does_not_raise(tmp_path):
    not_a_dir = tmp_path / "skills_file"
    not_a_dir.write_text("oops", encoding="utf-8")
    assert discover_skills(not_a_dir) == []
