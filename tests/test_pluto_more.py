# tests/test_pluto_more.py
"""
Tests for Pluto backlog items #131 (broker sync), #132 (price/news alerts) and
#136 (multi-currency net worth). No network: Kite, Yahoo and FX are fakes.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.pluto import alerts as al
from modules.pluto import brokers as bk
from modules.pluto import networth as nw
from modules.pluto.db import PlutoDB
from modules.pluto.market_data import MarketDataError, MarketDataThrottled, PriceSeries
from modules.pluto.throttle import MONITOR


@pytest.fixture
def db(tmp_path):
    return PlutoDB(str(tmp_path / "pluto.db"))


@pytest.fixture(autouse=True)
def _reset_monitor():
    MONITOR.reset()
    yield
    MONITOR.reset()


# ---------------------------------------------------------------- #136 net worth

class TestCurrencyParsing:
    @pytest.mark.parametrize("raw,code", [("$", "USD"), ("dollars", "USD"), ("eur", "EUR"), ("in pounds", "GBP"),
                                          ("INR", "INR"), ("\u20b9", "INR"), ("AED", "AED")])
    def test_known(self, raw, code):
        assert nw.parse_currency(raw) == code

    @pytest.mark.parametrize("raw", ["", None, 5, "banana", "XYZ"])
    def test_unknown(self, raw):
        assert nw.parse_currency(raw) is None


class TestNetWorth:
    def _nw(self, db, prices=None, rates=None):
        prices = prices or {}
        rates = {"USD": 80.0} if rates is None else rates

        def fx(src, dst):
            if src not in rates:
                raise RuntimeError("no rate")
            return rates[src]
        return nw.NetWorth(db, "INR", "\u20b9", price_fn=lambda n, t: prices.get(n), fx_fn=fx)

    def test_empty(self, db):
        assert "haven't tracked" in self._nw(db).net_worth()["response"]

    def test_mixed_currencies_are_converted(self, db):
        db.log_investment("RELIANCE.NS", "stock", 10, 100)      # 1000 INR at cost
        db.log_investment("AAPL", "stock", 2, 150)
        n = self._nw(db, prices={"AAPL": 200.0})
        assert n.set_currency({"name": "aapl", "currency": "USD"})["data"]["currency"] == "USD"
        r = n.net_worth()
        assert r["data"]["total"] == pytest.approx(1000 + 2 * 200 * 80)
        assert r["data"]["by_currency"]["USD"] == pytest.approx(400)
        assert r["data"]["at_cost"] == 1 and "at cost" in r["response"]

    def test_lots_of_one_holding_are_summed(self, db):
        db.log_investment("TCS.NS", "stock", 1, 10)
        db.log_investment("TCS.NS", "stock", 2, 20)
        assert self._nw(db).net_worth()["data"]["total"] == pytest.approx(50)

    def test_missing_rate_is_named_and_total_is_a_floor(self, db):
        db.log_investment("RELIANCE.NS", "stock", 1, 100)
        db.log_investment("SAP", "stock", 1, 100)
        n = self._nw(db, rates={})
        n.set_currency({"name": "SAP", "currency": "EUR"})
        r = n.net_worth()
        assert r["data"]["total"] == 100 and r["data"]["skipped"] == ["SAP (EUR)"]
        assert "floor" in r["response"] and "at least" in r["response"]

    def test_unset_currency_is_reported_as_assumed(self, db):
        db.log_investment("INFY", "stock", 1, 100)
        r = self._nw(db).net_worth()
        assert r["data"]["assumed_currency"] == 1 and "no currency set" in r["response"]

    def test_crypto_live_price_is_inr_regardless_of_held_currency(self, db):
        db.log_investment("bitcoin", "crypto", 1, 10)
        n = self._nw(db, prices={"bitcoin": 5_000_000.0})
        n.set_currency({"name": "bitcoin", "currency": "USD"})
        assert n.net_worth()["data"]["total"] == pytest.approx(5_000_000)

    def test_set_currency_errors(self, db):
        db.log_investment("INFY", "stock", 1, 100)
        n = self._nw(db)
        assert "Which holding" in n.set_currency({})["response"]
        assert "don't have" in n.set_currency({"name": "nope", "currency": "USD"})["response"]
        assert "didn't recognise" in n.set_currency({"name": "INFY", "currency": "zzz"})["response"]
        n.set_currency({"name": "INFY", "currency": "USD"})
        assert "back to the default" in n.set_currency({"name": "INFY", "currency": "clear"})["response"]
        assert db.get_setting(nw.SETTING_KEY) == "{}"

    def test_corrupt_setting_does_not_crash(self, db):
        db.log_investment("INFY", "stock", 1, 100)
        db.set_setting(nw.SETTING_KEY, "not json")
        assert self._nw(db).net_worth()["data"]["total"] == 100

    def test_zero_or_negative_rate_is_treated_as_missing(self, db):
        db.log_investment("X", "stock", 1, 100)
        n = self._nw(db, rates={"USD": 0})
        n.set_currency({"name": "X", "currency": "USD"})
        assert n.net_worth()["data"]["skipped"] == ["X (USD)"]


# ---------------------------------------------------------------- #131 broker sync

KITE = {"status": "success", "data": [
    {"tradingsymbol": "RELIANCE", "exchange": "NSE", "quantity": 10, "t1_quantity": 0, "average_price": 2500.0},
    {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 4, "t1_quantity": 1, "average_price": 3500.0},
    {"tradingsymbol": "BADROW", "exchange": "NSE", "quantity": 0, "average_price": 10},
    {"tradingsymbol": "NOPRICE", "exchange": "NSE", "quantity": 3, "average_price": "abc"},
    "junk",
]}


class TestKiteParsing:
    def test_parses_and_skips_bad_rows(self):
        h = bk.parse_kite_holdings(KITE)
        assert [(x.symbol, x.quantity, x.exchange) for x in h] == [("RELIANCE", 10, "NS"), ("TCS", 5, "NS")]

    def test_bse_exchange(self):
        h = bk.parse_kite_holdings({"status": "success", "data": [
            {"tradingsymbol": "X", "exchange": "BSE", "quantity": 1, "average_price": 1}]})
        assert h[0].exchange == "BO"

    @pytest.mark.parametrize("payload", [None, [], {"status": "error"}])
    def test_error_payloads(self, payload):
        with pytest.raises(bk.BrokerError):
            bk.parse_kite_holdings(payload)

    def test_nan_and_negative_rejected(self):
        h = bk.parse_kite_holdings({"data": [
            {"tradingsymbol": "A", "quantity": float("nan"), "average_price": 1},
            {"tradingsymbol": "B", "quantity": 1, "average_price": -5}]})
        assert h == []


class _Resp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._p, self.headers = status, payload, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http")

    def json(self):
        return self._p


class TestKiteFetch:
    def test_sends_documented_headers(self):
        seen = {}

        def get(url, headers, timeout):
            seen.update(url=url, headers=headers, timeout=timeout)
            return _Resp(200, KITE)
        assert len(bk.fetch_kite_holdings("k", "t", get=get)) == 2
        assert seen["url"] == bk.KITE_URL and seen["headers"]["Authorization"] == "token k:t"
        assert seen["headers"]["X-Kite-Version"] == "3"

    def test_expired_token_message(self):
        with pytest.raises(bk.BrokerError, match="expires every day"):
            bk.fetch_kite_holdings("k", "t", get=lambda *a, **kw: _Resp(403))

    def test_429_starts_cooldown_and_next_call_fails_fast(self):
        calls = []

        def get(*a, **kw):
            calls.append(1)
            return _Resp(429, headers={"Retry-After": "30"})
        with pytest.raises(bk.BrokerError, match="rate-limiting"):
            bk.fetch_kite_holdings("k", "t", get=get)
        with pytest.raises(bk.BrokerError, match="slow down"):
            bk.fetch_kite_holdings("k", "t", get=get)
        assert len(calls) == 1

    def test_network_error_does_not_leak_details(self):
        def get(*a, **kw):
            raise ConnectionError("secret-token-in-url")
        with pytest.raises(bk.BrokerError) as e:
            bk.fetch_kite_holdings("k", "t", get=get)
        assert "secret" not in str(e.value)


class TestCsv:
    def _write(self, tmp_path, text, name="h.csv"):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_groww_style_with_title_lines(self, tmp_path):
        p = self._write(tmp_path, "Holdings Statement\nfor 2026\n\nStock Name,ISIN,Quantity,Average buy price\n"
                                  "Reliance Industries,INE002A01018,10,\"2,500.50\"\nTCS,INE467B01029,3,3400\n")
        h = bk.parse_holdings_csv(p)
        assert [(x.symbol, x.quantity, x.avg_price) for x in h] == [("RELIANCEINDUSTRIES", 10, 2500.5), ("TCS", 3, 3400)]

    def test_zerodha_console_headers(self, tmp_path):
        p = self._write(tmp_path, "Instrument,Qty.,Avg. cost,LTP\nINFY,5,1500,1600\n")
        assert bk.parse_holdings_csv(p)[0].symbol == "INFY"

    @pytest.mark.parametrize("text", ["a,b\n1,2\n", "Symbol,Quantity,Average price\n,,\n"])
    def test_unusable(self, tmp_path, text):
        with pytest.raises(bk.BrokerError):
            bk.parse_holdings_csv(self._write(tmp_path, text))

    def test_rejects_non_csv_and_missing(self, tmp_path):
        with pytest.raises(bk.BrokerError, match="csv"):
            bk.parse_holdings_csv(str(tmp_path / "x.xlsx"))
        with pytest.raises(bk.BrokerError, match="can't find"):
            bk.parse_holdings_csv(str(tmp_path / "missing.csv"))


class TestSync:
    def _sync(self, db, env=None, holdings=None):
        h = holdings if holdings is not None else bk.parse_kite_holdings(KITE)
        return bk.BrokerSync(db, "\u20b9", env={"KITE_API_KEY": "k", "KITE_ACCESS_TOKEN": "t"} if env is None else env,
                             kite_fn=lambda k, t: h)

    def test_needs_credentials_and_never_takes_them_from_chat(self, db):
        r = self._sync(db, env={}).sync({"broker": "zerodha", "access_token": "abc"})
        assert "KITE_API_KEY" in r["response"] and db.get_investments() == []

    def test_preview_writes_nothing(self, db):
        r = self._sync(db).sync({})
        assert "Would add 2" in r["response"] and db.get_investments() == []
        assert r["data"]["applied"] is False

    def test_confirm_saves_with_ns_suffix(self, db):
        r = self._sync(db).sync({"confirm": "yes"})
        assert r["data"]["applied"] is True
        names = {i["name"]: (i["quantity"], i["buy_price"]) for i in db.get_investments()}
        assert names == {"RELIANCE.NS": (10, 2500.0), "TCS.NS": (5, 3500.0)}

    def test_second_sync_is_a_noop(self, db):
        s = self._sync(db)
        s.sync({"confirm": True})
        r = s.sync({"confirm": True})
        assert "2 already match" in r["response"] and "Nothing to change" in r["response"]
        assert len(db.get_investments()) == 2

    def test_only_the_shortfall_is_added_matching_by_loose_name(self, db):
        db.log_investment("Reliance", "stock", 4, 2000.0)          # tracked by hand, 4 of 10
        r = self._sync(db).sync({"confirm": "true"})
        lots = [i for i in db.get_investments() if i["name"] == "Reliance"]
        assert sorted(i["quantity"] for i in lots) == [4, 6]       # delta lot joins the existing name
        # broker cost 10*2500 = 25000; tracked 8000; delta 6 -> price (25000-8000)/6
        assert [i["buy_price"] for i in lots if i["quantity"] == 6][0] == pytest.approx(17000 / 6)
        assert r["data"]["added"][0]["quantity"] == 6

    def test_broker_showing_fewer_is_reported_not_removed(self, db):
        db.log_investment("TCS.NS", "stock", 50, 100)
        r = self._sync(db).sync({"confirm": "yes"})
        assert "nothing removed" in r["response"]
        assert sum(i["quantity"] for i in db.get_investments() if i["name"] == "TCS.NS") == 50

    def test_failures_are_messages(self, db):
        def boom(k, t):
            raise bk.BrokerError("Kite rejected the login.")
        s = bk.BrokerSync(db, env={"KITE_API_KEY": "k", "KITE_ACCESS_TOKEN": "t"}, kite_fn=boom)
        r = s.sync({})
        assert "rejected" in r["response"] and r["confidence"] < 0.5

    def test_unknown_broker_and_csv_path(self, db, tmp_path):
        assert "Zerodha" in self._sync(db).sync({"broker": "upstox"})["response"]
        p = tmp_path / "h.csv"
        p.write_text("Symbol,Quantity,Average price\nINFY,2,1500\n")
        r = self._sync(db, env={}).sync({"csv_path": str(p), "confirm": "yes"})
        assert db.get_investments()[0]["name"] == "INFY.NS" and "your CSV" in r["response"]

    def test_write_failure_changes_nothing(self, db):
        class Boom:
            def get_investments(self):
                return []

            def transaction(self):
                raise RuntimeError("disk")
        s = bk.BrokerSync(Boom(), env={"KITE_API_KEY": "k", "KITE_ACCESS_TOKEN": "t"},
                          kite_fn=lambda k, t: bk.parse_kite_holdings(KITE))
        assert "nothing was changed" in s.sync({"confirm": "yes"})["response"]


# ---------------------------------------------------------------- #132 alerts

def _series(last, prev, sym="X.NS"):
    return PriceSeries(sym, ["2026-10-01", "2026-10-02"], [prev, last])


class _Alerts:
    """Builds PriceAlerts with a controllable clock, time of day and data."""

    def __init__(self, db, hour=11, moves=None, news=None):
        self.db, self.hour, self.t = db, hour, 0.0
        self.moves = moves or {}
        self.calls = []

        def hist(sym, rng):
            self.calls.append(sym)
            v = self.moves.get(sym)
            if isinstance(v, Exception):
                raise v
            return _series(*v, sym=sym) if v else PriceSeries(sym, [], [])
        self.a = al.PriceAlerts(db, history_fn=hist, news_fn=news,
                                now_fn=lambda: datetime(2026, 10, 6, self.hour, 0),
                                clock=lambda: self.t)


class TestAlertConfig:
    def test_off_until_configured(self, db):
        assert "off" in al.PriceAlerts(db).configure({})["response"]
        assert al.PriceAlerts(db).check() is None

    def test_set_threshold_and_validation(self, db):
        a = al.PriceAlerts(db)
        assert a.configure({"threshold": "4 percent"})["data"]["threshold"] == 4
        assert a.configure({"threshold": "4%"})["data"]["threshold"] == 4
        for bad in ("0", "-3", "99", "abc"):
            assert "between" in a.configure({"threshold": bad})["response"]
        assert al.PriceAlerts(db)._threshold() == 4
        assert "off" in a.configure({"threshold": "off"})["response"]
        assert a._threshold() is None

    def test_watchlist_add_remove_and_default_threshold(self, db):
        a = al.PriceAlerts(db)
        r = a.configure({"watch": "Tata Motors"})
        assert r["data"]["watch"] == ["Tata Motors"] and r["data"]["threshold"] == al.DEFAULT_THRESHOLD
        a.configure({"watch": "tata motors"})
        assert a._watch() == ["Tata Motors"]
        a.configure({"unwatch": "TATA MOTORS"})
        assert a._watch() == []

    def test_watchlist_cap_and_junk_names(self, db):
        a = al.PriceAlerts(db)
        for i in range(al.MAX_WATCH):
            a.configure({"watch": f"S{i}"})
        assert "at most" in a.configure({"watch": "ONE-TOO-MANY"})["response"]
        db.set_setting(al.WATCH_KEY, "garbage")
        assert a._watch() == []
        assert "<script>" not in str(a.configure({"watch": "<script>x"}))


class TestAlertChecks:
    def _on(self, db, thr="5"):
        db.set_setting(al.THRESHOLD_KEY, thr)

    def test_big_move_alerts_once_per_day_and_direction(self, db):
        self._on(db)
        db.log_investment("X", "stock", 1, 1)
        h = _Alerts(db, moves={"X.NS": (106.0, 100.0)})
        msg = h.a.check()
        assert "X.NS is up 6.0%" in msg and msg.startswith("Heads up")
        h.t += al.CHECK_INTERVAL_S + 1
        assert h.a.check() is None                          # same day, same direction: quiet

    def test_small_move_is_quiet_and_down_move_reads_down(self, db):
        self._on(db)
        db.log_investment("X", "stock", 1, 1)
        h = _Alerts(db, moves={"X.NS": (101.0, 100.0)})
        assert h.a.check() is None
        h.moves["X.NS"] = (90.0, 100.0)
        h.t += al.CHECK_INTERVAL_S + 1
        assert "down 10.0%" in h.a.check()

    def test_quiet_hours_do_not_consume_the_alert(self, db):
        self._on(db)
        db.log_investment("X", "stock", 1, 1)
        h = _Alerts(db, hour=23, moves={"X.NS": (110.0, 100.0)})
        assert h.a.check() is None and h.calls == []
        h.hour = 9
        assert "up 10.0%" in h.a.check()

    def test_interval_limits_fetching(self, db):
        self._on(db)
        db.log_investment("X", "stock", 1, 1)
        h = _Alerts(db, moves={"X.NS": (101.0, 100.0)})
        h.a.check()
        h.t += 60
        h.a.check()
        assert h.calls == ["X.NS"]

    def test_throttle_stops_the_round(self, db):
        self._on(db)
        for n in "ABC":
            db.log_investment(n, "stock", 1, 1)
        h = _Alerts(db, moves={"A.NS": MarketDataThrottled("slow down"), "B.NS": (150.0, 100.0)})
        assert h.a.check() is None and h.calls == ["A.NS"]

    def test_failures_and_empty_series_are_skipped(self, db):
        self._on(db)
        db.log_investment("A", "stock", 1, 1)
        db.log_investment("B", "stock", 1, 1)
        h = _Alerts(db, moves={"A.NS": MarketDataError("x"), "B.NS": (0.0, 0.0)})
        assert h.a.check() is None

    def test_only_stocks_are_checked_and_watchlist_included(self, db):
        self._on(db)
        db.log_investment("bitcoin", "crypto", 1, 1)
        db.log_investment("X", "stock", 1, 1)
        db.set_setting(al.WATCH_KEY, '["Y"]')
        h = _Alerts(db, moves={})
        h.a.check()
        assert sorted(h.calls) == ["X.NS", "Y.NS"]

    def test_at_most_three_items_per_alert_rest_next_round(self, db):
        self._on(db)
        moves = {}
        for n in "ABCDE":
            db.log_investment(n, "stock", 1, 1)
            moves[f"{n}.NS"] = (120.0, 100.0)
        h = _Alerts(db, moves=moves)
        assert h.a.check().count("up 20.0%") == 3
        h.t += al.CHECK_INTERVAL_S + 1
        assert h.a.check().count("up 20.0%") == 2

    def test_news_keyword_alert_is_recent_deduped_and_labelled(self, db):
        self._on(db)
        db.log_investment("X", "stock", 1, 1)
        items = [{"title": "SEBI probe into X Ltd widens", "published": "2026-10-05"},
                 {"title": "Old probe story", "published": "2026-09-01"},
                 {"title": "X launches new product", "published": "2026-10-06"}]
        h = _Alerts(db, moves={"X.NS": (100.0, 100.0)}, news=lambda s: items)
        msg = h.a.check()
        assert "SEBI probe into X Ltd widens" in msg and "keyword match only" in msg
        assert "Old probe" not in msg and "launches" not in msg
        h.t += al.CHECK_INTERVAL_S + 1
        assert h.a.check() is None

    def test_news_can_be_switched_off_and_failures_are_ignored(self, db):
        self._on(db)
        db.log_investment("X", "stock", 1, 1)
        items = [{"title": "Fraud probe", "published": "2026-10-06"}]
        h = _Alerts(db, moves={"X.NS": (100.0, 100.0)}, news=lambda s: items)
        db.set_setting(al.NEWS_KEY, "off")
        assert h.a.check() is None

        def boom(s):
            raise RuntimeError("x")
        h2 = _Alerts(db, moves={"X.NS": (100.0, 100.0)}, news=boom)
        db.set_setting(al.NEWS_KEY, "on")
        assert h2.a.check() is None

    def test_headline_without_a_date_is_not_alerted(self, db):
        self._on(db)
        db.log_investment("X", "stock", 1, 1)
        h = _Alerts(db, moves={"X.NS": (100.0, 100.0)}, news=lambda s: [{"title": "Fraud probe", "published": None}])
        assert h.a.check() is None


# ---------------------------------------------------------------- engine wiring

class TestEngineWiring:
    def _engine(self, db):
        from modules.pluto.engine import PlutoEngine
        stub = SimpleNamespace(db=db, get_context=lambda: {})
        eng = PlutoEngine.__new__(PlutoEngine)
        eng.pf_manager = stub
        eng.networth = nw.NetWorth(db, price_fn=lambda n, t: None, fx_fn=lambda a, b: 80.0)
        eng.broker_sync = bk.BrokerSync(db, env={})
        eng.price_alerts = al.PriceAlerts(db)
        return eng

    def test_can_handle_and_aliases(self, db):
        eng = self._engine(db)
        for i in ("sync_broker", "set_price_alert", "set_holding_currency", "net_worth",
                  "sync_zerodha", "price_alerts", "networth", "set_currency"):
            assert eng.can_handle(i), i

    def test_routes(self, db):
        eng = self._engine(db)
        db.log_investment("INFY", "stock", 1, 100)
        assert eng.handle("net_worth", {}, {})["data"]["total"] == 100
        assert eng.handle("networth", {}, {})["data"]["total"] == 100
        assert "KITE_API_KEY" in eng.handle("sync_broker", {}, {})["response"]
        assert eng.handle("set_price_alert", {"threshold": "3"}, {})["data"]["threshold"] == 3
        assert eng.handle("set_holding_currency", {"name": "INFY", "currency": "USD"}, {})["data"]["currency"] == "USD"

    def test_check_price_alerts_never_raises(self, db):
        eng = self._engine(db)
        eng.price_alerts = SimpleNamespace(check=lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert eng.check_price_alerts() is None


# ---------------------------------------------------------------- registration

class TestRegistration:
    NEW_PLUTO = ("sync_broker", "set_price_alert", "set_holding_currency", "net_worth")

    def test_registry(self):
        from modules.hecate.intent_registry import INTENT_MODULE_MAP
        for i in self.NEW_PLUTO:
            assert INTENT_MODULE_MAP[f"pluto_{i}"] == "pluto"

    def test_nlu_prompt(self):
        text = open(os.path.join(os.path.dirname(__file__), "..", "config", "nlu_prompt.txt"), encoding="utf-8").read()
        for i in self.NEW_PLUTO:
            assert f"pluto_{i}" in text
        for i in ("set_outing_preferences", "plan_group_outing", "clear_group_outing"):
            assert f"dionysus_{i}" in text

    def test_heartbeat_hook_speaks_and_survives_errors(self):
        from unittest.mock import MagicMock
        from core.heartbeat import HestiaHeartbeat as Heartbeat
        hb = Heartbeat.__new__(Heartbeat)
        hb.pluto = SimpleNamespace(check_price_alerts=lambda: "Heads up: X is up 6%.")
        import core.heartbeat as hbm
        spoken = []
        orig = hbm.bus.emit
        hbm.bus.emit = lambda ev, data=None: spoken.append((ev, data))
        try:
            hb._maybe_run_pluto_price_alerts()
            hb.pluto = SimpleNamespace(check_price_alerts=lambda: (_ for _ in ()).throw(RuntimeError("x")))
            hb._maybe_run_pluto_price_alerts()
            hb.pluto = None
            hb._maybe_run_pluto_price_alerts()
        finally:
            hbm.bus.emit = orig
        assert spoken == [("speak", {"text": "Heads up: X is up 6%."})]


# ---------------------------------------------------------------- web routes

class TestWebRoutes:
    def _client(self, pluto):
        from unittest.mock import MagicMock
        from web_ui import HestiaWebUI
        return HestiaWebUI(memory=MagicMock(spec=["get_stats"]), pluto=pluto).app.test_client()

    def test_networth_route(self):
        from unittest.mock import MagicMock
        assert self._client(None).get("/api/pluto/networth").status_code == 503
        p = MagicMock()
        p.networth.net_worth.return_value = {"response": "r", "data": {}, "confidence": 1}
        assert self._client(p).get("/api/pluto/networth").get_json()["response"] == "r"

    def test_receipt_upload_validation_and_temp_cleanup(self, tmp_path):
        import io
        from unittest.mock import MagicMock
        seen = {}

        def ingest(entities):
            seen.update(entities)
            seen["existed"] = os.path.isfile(entities["image_path"])
            return {"response": "ok", "data": {"saved": True}, "confidence": 0.9}
        p = MagicMock()
        p.receipts.ingest.side_effect = ingest
        c = self._client(p)
        assert self._client(None).post("/api/pluto/receipt").status_code == 503
        assert c.post("/api/pluto/receipt").status_code == 400
        bad = c.post("/api/pluto/receipt", data={"image": (io.BytesIO(b"x"), "evil.exe")},
                     content_type="multipart/form-data")
        assert bad.status_code == 400
        ok = c.post("/api/pluto/receipt", data={"image": (io.BytesIO(b"\x89PNGdata"), "../../etc/passwd.png"),
                                                "amount": "450"}, content_type="multipart/form-data")
        assert ok.status_code == 200 and ok.get_json()["data"]["saved"] is True
        assert seen["existed"] is True and seen["amount"] == "450"
        assert "passwd" not in seen["image_path"] and seen["image_path"].endswith(".png")
        assert not os.path.exists(seen["image_path"])            # temp file removed afterwards

    def test_receipt_failure_is_generic_and_still_cleans_up(self):
        import io
        from unittest.mock import MagicMock
        paths = []

        def boom(entities):
            paths.append(entities["image_path"])
            raise RuntimeError("secret detail")
        p = MagicMock()
        p.receipts.ingest.side_effect = boom
        r = self._client(p).post("/api/pluto/receipt", data={"image": (io.BytesIO(b"x"), "a.jpg")},
                                 content_type="multipart/form-data")
        assert r.status_code == 500 and "secret" not in r.get_data(as_text=True)
        assert not os.path.exists(paths[0])

    def test_portfolio_page_has_new_panels(self):
        html = open(os.path.join(os.path.dirname(__file__), "..", "templates", "index.html"), encoding="utf-8").read()
        for needle in ("id=\"qs-run\"", "id=\"rc-run\"", "id=\"pluto-networth\"", "/api/pluto/quant-score", "/api/pluto/receipt"):
            assert needle in html


# ---------------------------------------------------------------- #137 tax hints

class TestExpenseHints:
    from modules.pluto import planning as _pl

    @pytest.mark.parametrize("cat,desc,needle", [
        ("health", "doctor", "80D"),                       # category wins
        ("Other", "donation to relief fund", "80G"),
        ("Bills", "house rent october", "HRA"),
        ("Other", "health insurance premium", "80D"),
        ("Other", "home loan interest", "24(b)"),
        ("Other", "education loan interest", "80E"),
        ("Other", "PPF deposit", "80C"),
    ])
    def test_hints(self, cat, desc, needle):
        assert needle in self._pl.expense_hint(cat, desc)

    @pytest.mark.parametrize("cat,desc", [("food", "dinner"), ("Other", "parental guidance movie"),
                                          ("Other", "current account"), ("Other", ""), ("", None)])
    def test_no_false_hint(self, cat, desc):
        assert self._pl.expense_hint(cat, desc if desc is not None else "") == ""

    def test_csv_uses_description_hints(self, tmp_path):
        e, _ = self._pl.write_tax_csvs(tmp_path, 2026, [
            {"logged_at": "2026-05-01", "description": "donation to ngo", "category": "Other", "amount": 500}], [])
        assert "80G" in open(e, encoding="utf-8").read()
