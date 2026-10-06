#!/usr/bin/env python
"""
scripts/check_ieee.py - one-time live check of the IEEE Xplore paper source (#47).

Makes ONE Metadata Search call (it costs one of your 200 calls/day), then tells
you:
  * whether the key works (and the HTTP status if not),
  * which of the response fields Hestia's parser relies on are really present,
  * how many papers the parser extracted from it.

``--save-fixture PATH`` writes the raw JSON response (the key is not part of
the response body) so it can be added to the tests exactly as IEEE sent it.

The key is read from IEEE_API_KEY (environment or .env) and is never printed.

    python scripts/check_ieee.py
    python scripts/check_ieee.py --query "graph neural networks" --save-fixture ieee_sample.json

Exit code 0 only if the call succeeded AND at least one paper was parsed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fields the parser reads, with what it does when one is missing.
EXPECTED_FIELDS = {
    "article_number": "REQUIRED - the article is skipped without it",
    "title": "REQUIRED - the article is skipped without it",
    "abstract": "summary is empty",
    "authors": "author list is empty",
    "publication_date": "date falls back to publication_year",
    "publication_year": "no date: the paper is treated as recent",
    "html_url": "link is built from the article number",
    "pdf_url": "no PDF link",
    "doi": "no DOI (cross-source de-duplication uses the title instead)",
    "publication_title": "no venue shown",
}


def _default_fetch(url: str) -> tuple[int, str]:
    import requests
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Hestia-paper-monitor/1.0 (personal assistant)"})
    return resp.status_code, resp.text


def main(argv: Optional[list[str]] = None,
         fetch: Optional[Callable[[str], tuple[int, str]]] = None,
         out=print) -> int:
    ap = argparse.ArgumentParser(description="One-time live check of the IEEE Xplore source.")
    ap.add_argument("--query", default="machine learning")
    ap.add_argument("--save-fixture", metavar="PATH", help="write the raw response body here")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("IEEE_API_KEY", "").strip()
    if not key:
        out("IEEE_API_KEY is not set. Put it in your environment or .env (never in the YAML config).")
        return 2

    from modules.mnemosyne.ieee_source import IeeeSource, parse_ieee_response
    src = IeeeSource(key)
    url = src.build_url(args.query, 3)
    out(f"Calling IEEE Xplore once (query: {args.query!r}) ...")
    try:
        status, body = (fetch or _default_fetch)(url)
    except Exception as exc:
        out(f"Request failed: {src.redact(f'{type(exc).__name__}: {exc}')}")
        return 1

    out(f"HTTP status: {status}")
    if status == 401:
        out("The key was rejected (401). Check it on https://developer.ieee.org/apps/myapps - it may still be pending approval.")
        return 1
    if status in (403, 429):
        out(f"Refused ({status}): the key may be pending, restricted, or over its daily limit.")
        return 1
    if status != 200:
        out(f"Unexpected status. First 200 characters of the reply: {src.redact(body[:200])!r}")
        return 1

    if args.save_fixture:
        Path(args.save_fixture).write_text(body, encoding="utf-8")
        out(f"Saved the raw response to {args.save_fixture}")

    try:
        data = json.loads(body)
    except ValueError:
        out("The reply was not JSON. The parser cannot read it.")
        return 1
    articles = data.get("articles") if isinstance(data, dict) else None
    if not isinstance(articles, list):
        out(f"No 'articles' list in the reply. Top-level keys: {sorted(data) if isinstance(data, dict) else type(data).__name__}")
        return 1
    out(f"Articles returned: {len(articles)} (total_records: {data.get('total_records', '?')})")

    problems = 0
    for field, consequence in EXPECTED_FIELDS.items():
        present = sum(1 for a in articles if isinstance(a, dict) and a.get(field) not in (None, "", [], {}))
        flag = "ok     " if present == len(articles) and articles else "MISSING" if present == 0 else "partial"
        if flag != "ok     ":
            problems += 1
        out(f"  {flag} {field:18} in {present}/{len(articles)}" + ("" if flag == "ok     " else f"  -> {consequence}"))

    papers = parse_ieee_response(body)
    out(f"Parsed {len(papers)} paper(s).")
    for p in papers[:3]:
        out(f"  - {p.title[:70]}  [{p.published or 'no date'} ({p.date_precision})]  id={p.external_id}")
    if not papers:
        return 1
    out("All fields present - the parser matches the live format." if not problems else
        "The parser works, but see the fields flagged above; send me the saved fixture if anything is MISSING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
