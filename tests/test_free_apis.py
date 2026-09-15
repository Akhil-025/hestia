# tests/test_free_apis.py
"""
Regression tests for core/free_apis.py.

Run with:  pytest tests/test_free_apis.py -v

None of these tests hit the real network — every `requests.get` call is
mocked at the module boundary (`core.free_apis.requests.get`). The goal
is to lock down each function's own logic (parameter building, response
parsing, graceful-degradation behaviour) rather than the third-party
APIs themselves, which are already documented as "no test coverage" in
the sense that nothing here previously verified this file's parsing at
all.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.free_apis import (
    FreeAPIError,
    convert_currency,
    fx_rate,
    sec_company_facts,
    fred_series_latest,
    food_lookup,
    public_holidays,
    is_public_holiday,
    suggest_activity,
    _ACTIVITY_FALLBACKS,
    country_info,
)


def _ok_response(json_body):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_body
    return resp


# ---------------------------------------------------------------------------
# convert_currency / fx_rate
# ---------------------------------------------------------------------------

def test_convert_currency_same_currency_short_circuits_without_a_request():
    with patch("core.free_apis.requests.get") as mock_get:
        result = convert_currency(100, "usd", "USD")

    assert result == 100
    mock_get.assert_not_called()


def test_convert_currency_returns_converted_amount():
    with patch(
        "core.free_apis.requests.get",
        return_value=_ok_response({"rates": {"EUR": 91.5}}),
    ):
        result = convert_currency(100, "usd", "eur")

    assert result == 91.5


def test_convert_currency_uppercases_and_strips_codes():
    with patch(
        "core.free_apis.requests.get",
        return_value=_ok_response({"rates": {"EUR": 91.5}}),
    ) as mock_get:
        convert_currency(100, " usd ", " eur ")

    params = mock_get.call_args.kwargs["params"]
    assert params["from"] == "USD"
    assert params["to"] == "EUR"


def test_convert_currency_raises_free_api_error_when_rate_missing():
    with patch(
        "core.free_apis.requests.get",
        return_value=_ok_response({"rates": {}}),
    ):
        with pytest.raises(FreeAPIError):
            convert_currency(100, "usd", "eur")


def test_fx_rate_delegates_to_convert_currency_with_amount_one():
    with patch(
        "core.free_apis.requests.get",
        return_value=_ok_response({"rates": {"EUR": 0.915}}),
    ) as mock_get:
        rate = fx_rate("usd", "eur")

    assert rate == 0.915
    assert mock_get.call_args.kwargs["params"]["amount"] == 1.0


# ---------------------------------------------------------------------------
# _get() error translation (exercised via convert_currency)
# ---------------------------------------------------------------------------

def test_network_error_is_translated_to_free_api_error():
    with patch(
        "core.free_apis.requests.get",
        side_effect=requests.ConnectionError("refused"),
    ):
        with pytest.raises(FreeAPIError):
            convert_currency(100, "usd", "eur")


def test_non_json_response_is_translated_to_free_api_error():
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.side_effect = ValueError("not json")
    with patch("core.free_apis.requests.get", return_value=resp):
        with pytest.raises(FreeAPIError):
            convert_currency(100, "usd", "eur")


def test_http_error_status_is_translated_to_free_api_error():
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError("404")
    with patch("core.free_apis.requests.get", return_value=resp):
        with pytest.raises(FreeAPIError):
            convert_currency(100, "usd", "eur")


# ---------------------------------------------------------------------------
# sec_company_facts — CIK zero-padding
# ---------------------------------------------------------------------------

def test_sec_company_facts_pads_cik_to_ten_digits():
    with patch(
        "core.free_apis.requests.get", return_value=_ok_response({})
    ) as mock_get:
        sec_company_facts("320193")

    url = mock_get.call_args.args[0]
    assert url.endswith("CIK0000320193.json")


def test_sec_company_facts_strips_leading_zeros_before_repadding():
    with patch(
        "core.free_apis.requests.get", return_value=_ok_response({})
    ) as mock_get:
        sec_company_facts("0000320193")

    url = mock_get.call_args.args[0]
    assert url.endswith("CIK0000320193.json")


def test_sec_company_facts_sends_required_user_agent_header():
    with patch(
        "core.free_apis.requests.get", return_value=_ok_response({})
    ) as mock_get:
        sec_company_facts("320193")

    headers = mock_get.call_args.kwargs["headers"]
    assert "User-Agent" in headers


# ---------------------------------------------------------------------------
# fred_series_latest — graceful degradation without an API key
# ---------------------------------------------------------------------------

def test_fred_series_latest_returns_none_without_api_key():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FRED_API_KEY", None)
        with patch("core.free_apis.requests.get") as mock_get:
            result = fred_series_latest("CPIAUCSL")

    assert result is None
    mock_get.assert_not_called()


def test_fred_series_latest_returns_latest_value_with_api_key():
    with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
        with patch(
            "core.free_apis.requests.get",
            return_value=_ok_response(
                {"observations": [{"value": "310.3"}]}
            ),
        ):
            result = fred_series_latest("CPIAUCSL")

    assert result == 310.3


def test_fred_series_latest_returns_none_when_no_observations():
    with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
        with patch(
            "core.free_apis.requests.get",
            return_value=_ok_response({"observations": []}),
        ):
            result = fred_series_latest("CPIAUCSL")

    assert result is None


# ---------------------------------------------------------------------------
# food_lookup — response simplification
# ---------------------------------------------------------------------------

def test_food_lookup_simplifies_product_records():
    raw = {
        "products": [
            {
                "product_name": "Peanut Butter",
                "brands": "Acme",
                "nutriments": {
                    "energy-kcal_100g": 588,
                    "proteins_100g": 25,
                    "sugars_100g": 9,
                },
                "nutriscore_grade": "d",
            }
        ]
    }
    with patch("core.free_apis.requests.get", return_value=_ok_response(raw)):
        results = food_lookup("peanut butter")

    assert results == [{
        "name": "Peanut Butter",
        "brand": "Acme",
        "calories_kcal_100g": 588,
        "protein_g_100g": 25,
        "sugar_g_100g": 9,
        "nutriscore": "d",
    }]


def test_food_lookup_falls_back_to_generic_name_and_handles_missing_nutriments():
    raw = {"products": [{"generic_name": "Mystery Snack"}]}
    with patch("core.free_apis.requests.get", return_value=_ok_response(raw)):
        results = food_lookup("snack")

    assert results[0]["name"] == "Mystery Snack"
    assert results[0]["calories_kcal_100g"] is None


# ---------------------------------------------------------------------------
# public_holidays / is_public_holiday
# ---------------------------------------------------------------------------

def test_is_public_holiday_returns_local_name_on_match():
    holidays = [{"date": "2026-01-26", "localName": "Republic Day", "name": "Republic Day"}]
    with patch("core.free_apis.requests.get", return_value=_ok_response(holidays)):
        result = is_public_holiday("2026-01-26", "IN")

    assert result == "Republic Day"


def test_is_public_holiday_returns_none_when_no_match():
    holidays = [{"date": "2026-01-01", "localName": "New Year's Day"}]
    with patch("core.free_apis.requests.get", return_value=_ok_response(holidays)):
        result = is_public_holiday("2026-03-15", "IN")

    assert result is None


def test_is_public_holiday_returns_none_on_api_failure_instead_of_raising():
    with patch(
        "core.free_apis.requests.get",
        side_effect=requests.ConnectionError("down"),
    ):
        result = is_public_holiday("2026-03-15", "IN")

    assert result is None


# ---------------------------------------------------------------------------
# suggest_activity — offline fallback
# ---------------------------------------------------------------------------

def test_suggest_activity_uses_remote_result_when_available():
    with patch(
        "core.free_apis.requests.get",
        return_value=_ok_response({"activity": "Go for a walk", "type": "recreational"}),
    ):
        result = suggest_activity()

    assert result == {"activity": "Go for a walk", "type": "recreational"}


def test_suggest_activity_falls_back_to_static_list_on_failure():
    with patch(
        "core.free_apis.requests.get",
        side_effect=requests.ConnectionError("down"),
    ):
        result = suggest_activity()

    assert result in _ACTIVITY_FALLBACKS


# ---------------------------------------------------------------------------
# country_info
# ---------------------------------------------------------------------------

def test_country_info_extracts_expected_fields():
    raw = [{
        "name": {"common": "Japan"},
        "capital": ["Tokyo"],
        "region": "Asia",
        "currencies": {"JPY": {}},
        "languages": {"jpn": "Japanese"},
        "population": 125000000,
    }]
    with patch("core.free_apis.requests.get", return_value=_ok_response(raw)):
        result = country_info("Japan")

    assert result == {
        "name": "Japan",
        "capital": "Tokyo",
        "region": "Asia",
        "currencies": ["JPY"],
        "languages": ["Japanese"],
        "population": 125000000,
    }


def test_country_info_returns_none_for_empty_result():
    with patch("core.free_apis.requests.get", return_value=_ok_response([])):
        result = country_info("Nowhereland")

    assert result is None
