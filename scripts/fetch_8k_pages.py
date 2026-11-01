#!/usr/bin/env python3
"""Download a ticker's EDGAR submissions pages to the JSON that run_real.py --pages-json expects.

    python scripts/fetch_8k_pages.py --ticker NVDA --user-agent "Your Name you@example.com" --out data/nvda_8k_pages.json

Raw responses are cached under edgar_cache/<ticker>/ so reruns are offline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l1 import edgar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--user-agent", required=True, help='"Name email", required by the SEC')
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()
    cik = edgar.cik_for_ticker(a.ticker, a.user_agent)
    pages = edgar.fetch_submissions(cik, a.user_agent, cache_dir=a.cache_dir or f"edgar_cache/{a.ticker.upper()}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(pages))
    n = sum(len(p.get("form", [])) for p in pages)
    print(f"{a.ticker} CIK {cik}: {len(pages)} pages, {n} filings -> {a.out}")


if __name__ == "__main__":
    main()
