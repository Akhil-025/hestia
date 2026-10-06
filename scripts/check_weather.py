#!/usr/bin/env python
"""
scripts/check_weather.py - one-time live check of the Open-Meteo calls behind
weather-triggered suggestions (#88).

The build sandbox could not reach Open-Meteo, so the rain assessment has only
ever run against a hand-made forecast. This makes the two real requests Hestia
makes (current weather and the hourly forecast) and tells you:

  * whether each call worked,
  * which hourly fields came back (and whether the weather-code field is named
    ``weathercode`` or ``weather_code`` - Hestia reads either),
  * the peak rain chance it would report for today and tomorrow.

``--save-fixture PATH`` writes the raw hourly forecast so it can be dropped into
the tests exactly as Open-Meteo sent it.

    python scripts/check_weather.py
    python scripts/check_weather.py --lat 19.07 --lon 72.88 --tz Asia/Kolkata --save-fixture meteo.json

Exit code 0 only if both calls worked and the forecast covered today.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lat", type=float, default=19.0760)
    ap.add_argument("--lon", type=float, default=72.8777)
    ap.add_argument("--tz", default="Asia/Kolkata")
    ap.add_argument("--save-fixture", metavar="PATH")
    args = ap.parse_args(argv)

    from modules.chronos import agenda
    from modules.chronos.engine import WeatherFetchError, _fetch_forecast, _fetch_weather

    ok = True
    try:
        cur = _fetch_weather(args.lat, args.lon)
        print(f"current_weather: OK  keys={sorted(cur)}")
    except WeatherFetchError as exc:
        print(f"current_weather: FAILED  {exc}")
        ok = False

    try:
        hourly = _fetch_forecast(args.lat, args.lon, args.tz)
    except WeatherFetchError as exc:
        print(f"hourly forecast: FAILED  {exc}")
        return 1
    print(f"hourly forecast: OK  fields={sorted(hourly)}  hours={len(hourly.get('time', []))}")
    code_field = next((k for k in ("weathercode", "weather_code") if k in hourly), None)
    print(f"weather-code field: {code_field or 'MISSING (only the rain probability will be used)'}")
    if "precipitation_probability" not in hourly:
        print("precipitation_probability: MISSING - rain warnings would never fire")
        ok = False

    tz = ZoneInfo(args.tz)
    today = datetime.now(tz).date()
    covered = False
    for label, day in (("today", today), ("tomorrow", today + timedelta(days=1))):
        out = agenda.rain_outlook(hourly, day, tz)
        if out is None:
            print(f"{label}: forecast does not cover 08:00-20:00")
        else:
            covered = covered or day == today
            print(f"{label}: peak rain chance {out[0]}% around {out[1]}")
    ok = ok and covered

    if args.save_fixture:
        Path(args.save_fixture).write_text(json.dumps(hourly, indent=2), encoding="utf-8")
        print(f"saved raw hourly forecast to {args.save_fixture}")
    print("RESULT:", "OK" if ok else "PROBLEMS FOUND (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
