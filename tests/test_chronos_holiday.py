# tests/test_chronos_holiday.py
"""
Regression tests for ChronosEngine's `get_holiday` intent, which wires
Nager.Date (via core/free_apis.py, part of the public-apis integration)
into Chronos.

Run with:  pytest tests/test_chronos_holiday.py -v
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chronos.engine import ChronosEngine
from core.free_apis import FreeAPIError


class TestChronosHoliday(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ChronosEngine(local_tz="Asia/Kolkata")

    def test_can_handle_get_holiday(self) -> None:
        self.assertTrue(self.engine.can_handle("get_holiday"))

    @patch("modules.chronos.engine.is_public_holiday")
    def test_specific_date_is_holiday(self, mock_is_holiday) -> None:
        mock_is_holiday.return_value = "Republic Day"
        result = self.engine.handle(
            "get_holiday", {"date": "2026-01-26", "country": "IN"}, {}
        )
        self.assertTrue(result["data"]["is_holiday"])
        self.assertEqual(result["data"]["name"], "Republic Day")
        self.assertIn("Republic Day", result["response"])
        mock_is_holiday.assert_called_once_with("2026-01-26", "IN")

    @patch("modules.chronos.engine.is_public_holiday")
    def test_specific_date_not_holiday(self, mock_is_holiday) -> None:
        mock_is_holiday.return_value = None
        result = self.engine.handle(
            "get_holiday", {"date": "2026-03-02", "country": "IN"}, {}
        )
        self.assertFalse(result["data"]["is_holiday"])
        self.assertIn("not a public holiday", result["response"])

    @patch("modules.chronos.engine.is_public_holiday")
    def test_remote_failure_is_graceful(self, mock_is_holiday) -> None:
        mock_is_holiday.side_effect = FreeAPIError("boom")
        result = self.engine.handle(
            "get_holiday", {"date": "2026-01-26", "country": "IN"}, {}
        )
        # Never raises; degrades to a normal error-shaped response.
        self.assertIn("response", result)
        self.assertNotIn("is_holiday", result.get("data", {}))

    @patch("modules.chronos.engine.public_holidays")
    def test_no_date_lists_year(self, mock_holidays) -> None:
        mock_holidays.return_value = [
            {"date": "2026-01-01", "localName": "New Year's Day"},
            {"date": "2026-01-26", "localName": "Republic Day"},
        ]
        result = self.engine.handle("get_holiday", {"country": "IN"}, {})
        self.assertEqual(result["data"]["country"], "IN")
        self.assertEqual(len(result["data"]["holidays"]), 2)
        self.assertIn("2", result["response"])  # "2 public holidays"


if __name__ == "__main__":
    unittest.main()