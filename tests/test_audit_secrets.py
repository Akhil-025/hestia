"""Backlog #197: the secret scanner as a gate (exit code, baseline, path mode)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import audit_secrets as A  # noqa: E402

# Built from pieces so this file itself never looks like a committed key.
FAKE_AWS = "AK" + "IA" + "ABCDEFGHIJKLMNOP"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "core").mkdir()
    (tmp_path / "tests").mkdir()
    return tmp_path


def run(args, capsys):
    code = A.main(list(args) + ["--no-report"])
    return code, capsys.readouterr()


def test_default_mode_reports_but_exits_zero(repo, capsys):
    (repo / "core" / "a.py").write_text(f'KEY = "{FAKE_AWS}"\n')
    code, out = run([], capsys)
    assert code == 0 and "AWS Access Key" in out.out


def test_fail_on_high_exits_one_and_lists_only_blocking(repo, capsys):
    (repo / "core" / "a.py").write_text(f'KEY = "{FAKE_AWS}"\nuser_token = "abcdefgh"\n')
    code, out = run(["--fail-on", "high"], capsys)
    assert code == 1
    assert "AWS Access Key" in out.out and "Hardcoded Secret" not in out.out
    assert "FAILED" in out.err


def test_clean_repo_passes_gate(repo, capsys):
    (repo / "core" / "a.py").write_text("x = 1\n")
    code, out = run(["--fail-on", "low"], capsys)
    assert code == 0 and "No secrets found" in out.out


def test_medium_is_gated_only_when_asked(repo, capsys):
    (repo / "core" / "a.py").write_text('db_password="hunter2hunter2"\n')
    assert run(["--fail-on", "high"], capsys)[0] == 0
    assert run(["--fail-on", "medium"], capsys)[0] == 1


def test_inline_ignore_marker_skips_the_line(repo, capsys):
    (repo / "core" / "a.py").write_text(f'KEY = "{FAKE_AWS}"  # audit: ignore\n')
    assert run(["--fail-on", "high"], capsys)[0] == 0


def test_baseline_accepts_reviewed_findings_but_not_new_ones(repo, capsys):
    f = repo / "core" / "a.py"
    f.write_text(f'KEY = "{FAKE_AWS}"\n')
    assert run(["--write-baseline"], capsys)[0] == 0
    data = json.loads((repo / A.DEFAULT_BASELINE).read_text())
    assert len(data["accepted"]) == 1 and "preview" not in json.dumps(data)
    assert run(["--fail-on", "high"], capsys)[0] == 0
    # moving the line does not un-accept it (fingerprint ignores line numbers)
    f.write_text(f'# comment\n\nKEY = "{FAKE_AWS}"\n')
    assert run(["--fail-on", "high"], capsys)[0] == 0
    # a different secret is new
    f.write_text(f'KEY = "{FAKE_AWS}"\nOTHER = "AK' + 'IA' + 'ZZZZZZZZZZZZZZZZ"\n')
    assert run(["--fail-on", "high"], capsys)[0] == 1


def test_paths_mode_scans_only_given_files_in_scope(repo, capsys):
    (repo / "core" / "bad.py").write_text(f'KEY = "{FAKE_AWS}"\n')
    (repo / "core" / "ok.py").write_text("x = 1\n")
    (repo / "tests" / "fake.py").write_text(f'KEY = "{FAKE_AWS}"\n')
    assert run(["--fail-on", "high", "core/ok.py"], capsys)[0] == 0
    assert run(["--fail-on", "high", "core/bad.py"], capsys)[0] == 1
    # tests/ is outside the scan scope in both modes, so pre-commit and CI agree
    assert run(["--fail-on", "high", "tests/fake.py"], capsys)[0] == 0
    assert run(["--fail-on", "high"], capsys)[0] == 1  # core/bad.py found repo-wide


def test_corrupt_baseline_is_ignored_not_fatal(repo, capsys):
    (repo / A.DEFAULT_BASELINE).write_text("{not json")
    (repo / "core" / "a.py").write_text(f'KEY = "{FAKE_AWS}"\n')
    assert run(["--fail-on", "high"], capsys)[0] == 1


def test_the_shipped_repo_passes_its_own_gate(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    assert A.main(["--fail-on", "high", "--no-report"]) == 0


def test_precommit_and_ci_run_the_gate():
    pc = (ROOT / ".pre-commit-config.yaml").read_text()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "audit_secrets.py --fail-on high" in pc
    assert "audit_secrets.py --fail-on high" in ci
