"""Backlog #198 (encryption at rest) and #199 (rotation checklist)."""
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core import crypto_store as cs  # noqa: E402
import scripts.rotate_secrets as rs  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    cs.reset()
    yield
    cs.reset()


def key():
    return Fernet.generate_key().decode()


def raw(path, sql):
    return sqlite3.connect(path).execute(sql).fetchall()


# ── crypto_store ─────────────────────────────────────────────────────────────

def test_round_trip_and_plain_passthrough():
    cs.configure(enabled=True, key_material=key())
    token = cs.encrypt_text("felt dizzy")
    assert token.startswith(cs.PREFIX) and "dizzy" not in token
    assert cs.decrypt_text(token) == "felt dizzy"
    assert cs.decrypt_text("old plain row") == "old plain row"
    assert cs.encrypt_text(token) == token            # idempotent
    assert cs.encrypt_text(None) is None and cs.encrypt_text("") == ""
    assert cs.encrypt_text(42) == 42


def test_disabled_writes_plain_but_enabled_key_still_reads():
    k = key()
    cs.configure(enabled=True, key_material=k)
    token = cs.encrypt_text("secret")
    cs.configure(enabled=False, key_material=k)
    assert cs.encrypt_text("x") == "x"
    assert cs.decrypt_text(token) == "secret"


def test_missing_or_wrong_key_reads_as_notice_never_raises_or_leaks():
    cs.configure(enabled=True, key_material=key())
    token = cs.encrypt_text("secret")
    cs.configure(enabled=True, key_material=key())    # a different key
    assert cs.decrypt_text(token) == cs.UNREADABLE
    cs.reset()
    assert cs.decrypt_text(token) == cs.UNREADABLE
    assert cs.decrypt_text(cs.PREFIX + "garbage") == cs.UNREADABLE


def test_enabling_without_a_key_fails_fast():
    with pytest.raises(cs.CryptoConfigError):
        cs.configure(enabled=True, key_material="")
    assert not cs.is_enabled()
    with pytest.raises(cs.CryptoConfigError):
        cs.configure(enabled=True, key_material="not-a-fernet-key")


def test_key_rotation_new_first_old_still_reads():
    old, new = key(), key()
    cs.configure(enabled=True, key_material=old)
    token = cs.encrypt_text("note")
    cs.configure(enabled=True, key_material=f"{new},{old}")
    assert cs.decrypt_text(token) == "note"
    assert cs.needs_rewrite(token)                     # still under the old key
    assert cs.decrypt_text(cs.encrypt_text("fresh")) == "fresh"
    assert not cs.needs_rewrite(cs.encrypt_text("fresh"))


def test_configure_from_config_and_key_sources(tmp_path):
    k = key()
    assert cs.configure_from_config({}, env={}) is False
    assert cs.configure_from_config({"security": {"encrypt_at_rest": True}},
                                    env={"HESTIA_DATA_KEY": k}) is True
    f = tmp_path / "k"
    f.write_text(k + "\n")
    assert cs.configure_from_config({"security": {"encrypt_at_rest": True, "key_file": str(f)}}, env={})
    with pytest.raises(cs.CryptoConfigError):
        cs.configure_from_config({"security": {"encrypt_at_rest": True, "key_file": str(tmp_path / "nope")}}, env={})
    with pytest.raises(cs.CryptoConfigError):
        cs.configure_from_config({"security": {"encrypt_at_rest": True}}, env={})


def test_decrypting_row_covers_name_index_dict_and_iteration():
    cs.configure(enabled=True, key_material=key())
    conn = sqlite3.connect(":memory:")
    conn.row_factory = cs.DecryptingRow
    conn.execute("CREATE TABLE t (a TEXT, n INTEGER, b TEXT)")
    conn.execute("INSERT INTO t VALUES (?, ?, ?)", (cs.encrypt_text("one"), 7, "plain"))
    r = conn.execute("SELECT * FROM t").fetchone()
    assert r["a"] == "one" and r[0] == "one" and r["n"] == 7 and r[2] == "plain"
    assert dict(r) == {"a": "one", "n": 7, "b": "plain"}
    assert list(r) == ["one", 7, "plain"]
    assert r[0:2] == ("one", 7)


def test_encrypt_existing_converts_plain_rows_once(tmp_path):
    cs.configure(enabled=True, key_material=key())
    conn = sqlite3.connect(tmp_path / "x.db")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, notes TEXT)")
    conn.executemany("INSERT INTO t (notes) VALUES (?)", [("a",), (None,), ("",), ("b",)])
    assert cs.encrypt_existing(conn, "t", ["notes"]) == 2
    assert cs.encrypt_existing(conn, "t", ["notes"]) == 0
    stored = [r[0] for r in conn.execute("SELECT notes FROM t ORDER BY id")]
    assert stored[1] is None and stored[2] == "" and all(cs.is_encrypted(x) for x in (stored[0], stored[3]))
    cs.reset()
    with pytest.raises(cs.CryptoConfigError):
        cs.encrypt_existing(conn, "t", ["notes"])


# ── wired into Apollo and Pluto ──────────────────────────────────────────────

def test_apollo_notes_encrypted_on_disk_plain_through_the_api(tmp_path):
    from modules.apollo.db import ApolloDB
    cs.configure(enabled=True, key_material=key())
    path = tmp_path / "apollo.db"
    db = ApolloDB(str(path))
    db.log_workout("run", 30, "knee hurt again")
    db.log_sleep(7.0, "ok", "woke twice")
    db.log_mood("low", "argument at work")
    db.log_weight(70.0, "after dinner")
    db.log_meal("dal", 300, notes="skipped lunch")
    db.log_pain("knee", 4, "stairs")
    for table in ("workouts", "sleep_logs", "mood_logs", "weight_logs", "meal_logs", "pain_logs"):
        (stored,) = raw(path, f"SELECT notes FROM {table}")[0]
        assert cs.is_encrypted(stored), table
    assert db.get_workouts()[0]["notes"] == "knee hurt again"
    assert db.get_mood()[0]["notes"] == "argument at work"
    assert db.get_meals()[0]["notes"] == "skipped lunch"
    assert db.get_pain()[0]["notes"] == "stairs"
    assert db.get_workouts()[0]["duration"] == 30      # numbers stay plain and usable
    assert raw(path, "SELECT mood FROM mood_logs")[0][0] == "low"


def test_apollo_works_unchanged_when_encryption_is_off(tmp_path):
    from modules.apollo.db import ApolloDB
    path = tmp_path / "apollo.db"
    db = ApolloDB(str(path))
    db.log_workout("run", 30, "plain note")
    assert raw(path, "SELECT notes FROM workouts")[0][0] == "plain note"
    assert db.get_workouts()[0]["notes"] == "plain note"


def test_pluto_expense_description_encrypted_and_totals_unaffected(tmp_path):
    from modules.pluto.db import PlutoDB
    cs.configure(enabled=True, key_material=key())
    path = tmp_path / "pluto.db"
    db = PlutoDB(str(path))
    db.log_expense(250.0, "therapy session", "health")
    db.log_expense(100.0, "coffee", "food", logged_at="2026-01-05 10:00:00")
    assert all(cs.is_encrypted(r[0]) for r in raw(path, "SELECT description FROM expenses"))
    descs = sorted(r["description"] for r in db.get_expenses())
    assert descs == ["coffee", "therapy session"]
    assert db.get_grand_total() == 350.0
    assert {r["category"]: r["total"] for r in db.get_totals_by_category()} == {"health": 250.0, "food": 100.0}
    assert db.get_expenses_between("2026-01-01", "2026-02-01")[0]["description"] == "coffee"


def test_config_accepts_security_section():
    from core.config_validation import validate_config
    cfg = {"security": {"encrypt_at_rest": False, "key_file": "k"}}
    rep = validate_config(cfg)
    assert not [w for w in rep.warnings if "security" in w] and not [e for e in rep.errors if "security" in e]
    bad = validate_config({"security": {"encrypt_at_rest": "yes"}})
    assert any("security.encrypt_at_rest" in e for e in bad.errors)


def test_reencrypt_command_converts_and_backs_up(tmp_path, capsys, monkeypatch):
    from modules.apollo.db import ApolloDB
    k = key()
    path = tmp_path / "apollo.db"
    db = ApolloDB(str(path))
    db.log_workout("run", 30, "plain note")             # written while encryption was off
    monkeypatch.setenv("HESTIA_DATA_KEY", k)
    code = rs.main(["--env-file", str(tmp_path / "none.env"), "reencrypt",
                    "--apollo-db", str(path), "--pluto-db", str(tmp_path / "missing.db")])
    assert code == 0
    assert cs.is_encrypted(raw(path, "SELECT notes FROM workouts")[0][0])
    assert (tmp_path / "apollo.db.pre-encrypt.bak").is_file()
    assert "pluto" in capsys.readouterr().out
    cs.configure(enabled=False, key_material=k)
    assert ApolloDB(str(path)).get_workouts()[0]["notes"] == "plain note"


def test_reencrypt_without_key_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("HESTIA_DATA_KEY", raising=False)
    monkeypatch.delenv("HESTIA_DATA_KEY_FILE", raising=False)
    assert rs.main(["--env-file", str(tmp_path / "none.env"), "reencrypt"]) == 2


# ── rotate_secrets ───────────────────────────────────────────────────────────

TODAY = date(2026, 10, 8)


def rows(env, state):
    return {r.name: r for r in rs.evaluate(env, state, TODAY)}


def test_status_not_set_unknown_ok_soon_overdue():
    env = {"TELEGRAM_BOT_TOKEN": "x", "HESTIA_SYNC_API_KEY": "y", "TODOIST_API_TOKEN": "z",
           "ANTHROPIC_API_KEY": "w"}
    state = {"telegram": "2026-09-01",                                   # 37 days of 180: ok
             "sync_api": (TODAY - timedelta(days=170)).isoformat(),      # 10 left: due soon
             "anthropic": (TODAY - timedelta(days=200)).isoformat()}     # overdue
    r = rows(env, state)
    assert r["omdb"].status == "not set" and not r["omdb"].needs_attention
    assert r["telegram"].status == "ok" and not r["telegram"].needs_attention
    assert r["sync_api"].status == "due soon" and r["sync_api"].needs_attention
    assert r["anthropic"].status == "overdue" and "20 days past" in r["anthropic"].detail
    assert r["todoist"].status == "unknown age" and r["todoist"].needs_attention


def test_data_key_has_no_age_limit_and_kite_is_daily():
    env = {"HESTIA_DATA_KEY": "k", "KITE_API_KEY": "a", "KITE_ACCESS_TOKEN": "t"}
    r = rows(env, {"data_key": "2020-01-01", "kite": "2026-10-07"})
    assert r["data_key"].status == "ok" and not r["data_key"].needs_attention
    assert r["kite"].status == "ok"
    r = rows(env, {"data_key": "2020-01-01", "kite": "2026-10-05"})
    assert r["kite"].status == "overdue"


def test_values_are_never_printed(capsys, tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text("TELEGRAM_BOT_TOKEN=super-secret-value-123\n")
    rs.main(["--env-file", str(envfile), "--state", str(tmp_path / "s.json"),
             "--google-token", str(tmp_path / "none.json")], today=TODAY)
    out = capsys.readouterr().out
    assert "telegram" in out and "super-secret-value-123" not in out


def test_record_then_status_and_strict_exit(tmp_path, capsys):
    envfile = tmp_path / ".env"
    envfile.write_text("TELEGRAM_BOT_TOKEN=abc\n")
    common = ["--env-file", str(envfile), "--state", str(tmp_path / "s.json"),
              "--google-token", str(tmp_path / "none.json")]
    assert rs.main(common + ["status", "--strict"], today=TODAY) == 1       # unknown age
    assert rs.main(common + ["record", "telegram"], today=TODAY) == 0
    assert json.loads((tmp_path / "s.json").read_text())["rotated"]["telegram"] == "2026-10-08"
    assert rs.main(common + ["status", "--strict"], today=TODAY) == 0
    assert rs.main(common + ["record", "telegram", "--date", "2027-01-01"], today=TODAY) == 2
    assert rs.main(common + ["record", "nonsense"], today=TODAY) == 2
    assert rs.main(common + ["checklist", "nonsense"], today=TODAY) == 2


def test_checklist_prints_steps(capsys, tmp_path):
    assert rs.main(["--state", str(tmp_path / "s.json"), "checklist", "telegram"]) == 0
    out = capsys.readouterr().out
    assert "BotFather" in out and "record telegram" in out


def test_google_token_row(tmp_path):
    now = datetime(2026, 10, 8, tzinfo=timezone.utc)
    assert rs.google_token_row(tmp_path / "none.json", now) is None
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"refresh_token": "r", "expiry": "2026-10-07T10:00:00Z"}))
    assert rs.google_token_row(p, now).status == "ok"
    p.write_text(json.dumps({"refresh_token": "r", "expiry": "2026-09-01T10:00:00Z"}))
    assert rs.google_token_row(p, now).needs_attention
    p.write_text(json.dumps({"access_token": "a"}))
    assert "no refresh token" in rs.google_token_row(p, now).detail
    p.write_text("{broken")
    assert rs.google_token_row(p, now).needs_attention


def test_env_example_covers_every_secret_the_rotation_script_tracks():
    text = (ROOT / ".env.example").read_text()
    for spec in rs.SECRETS:
        for var in spec.env:
            assert var in text, var
    assert "HESTIA_DATA_KEY" in text
