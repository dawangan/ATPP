"""
Pull XNAS.ITCH trades from Databento for the days the 8-K census says we need.

Default is a DRY RUN: it builds the day list, asks Databento what it would cost, prints the
total and exits. Nothing is downloaded until you add --yes.

The API key is read from the environment (never put it in code, a file in the repo, or chat):
    Windows PowerShell:   $env:DATABENTO_API_KEY = "db-..."
    macOS/Linux:          export DATABENTO_API_KEY=db-...

Usage:
    # 1) census first
    python scripts/edgar_census.py --ticker-file tickers.txt --user-agent "You you@x.com" --out census
    # 2) dry run: cost estimate for event days + 25 sessions of lead-in + 100 random control days per ticker
    python scripts/pull_databento.py --plan census/pull_plan.csv --out dbn
    # 3) real download, refusing to spend more than $40
    python scripts/pull_databento.py --plan census/pull_plan.csv --out dbn --yes --max-cost 40

    # or: whole window for a few tickers (no census needed)
    python scripts/pull_databento.py --tickers AAPL NVDA --full --start 2023-09-19 --end 2025-09-18 --out dbn

Files land at  <out>/<TICKER>/xnas-itch-YYYYMMDD.trades.dbn  (same naming as your MSFT folder).
Each file covers one America/New_York calendar day, 00:00-24:00 ET, so pre-market (04:00) and
after-hours (to 20:00) are included. Existing files are skipped, so the script is resumable.
Weekdays with no trading (exchange holidays) return nothing and are skipped.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "XNAS.ITCH"
SCHEMA = "trades"


def weekdays(a: pd.Timestamp, b: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(a, b)


def days_for_ticker(event_days: list[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp,
                    lead: int, controls: int, seed: int, full: bool) -> list[pd.Timestamp]:
    """Event days + `lead` weekdays before each (the trailing 20-session level needs them)
    + `controls` random other weekdays in the window (for the diurnal curve and placebos)."""
    allw = weekdays(start, end)
    if full:
        return list(allw)
    keep = set()
    for d in event_days:
        d = pd.Timestamp(d).normalize()
        if not (start <= d <= end):
            continue
        keep.add(d)
        keep.add(d + pd.offsets.BDay(1))     # an after-hours filing also moves the next morning's open
        keep.update(pd.bdate_range(end=d - pd.Timedelta(days=1), periods=lead))
    keep = {d for d in keep if start <= d <= end}
    rest = [d for d in allw if d not in keep]
    rng = np.random.default_rng(seed)
    if controls > 0 and rest:
        pick = rng.choice(len(rest), size=min(controls, len(rest)), replace=False)
        keep.update(rest[i] for i in pick)
    return sorted(keep)


def to_ranges(days: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge into contiguous runs of weekdays (Fri->Mon counts as contiguous) for cost queries."""
    out, days = [], sorted(days)
    if not days:
        return out
    s = p = days[0]
    for d in days[1:]:
        if len(pd.bdate_range(p, d)) <= 2:          # p and d adjacent business days
            p = d
        else:
            out.append((s, p))
            s = p = d
    out.append((s, p))
    return out


def et_day_bounds(d: pd.Timestamp):
    s = pd.Timestamp(d).normalize().tz_localize("America/New_York")
    e = (pd.Timestamp(d).normalize() + pd.Timedelta(days=1)).tz_localize("America/New_York")
    return s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", help="census/pull_plan.csv (columns ticker,event_day)")
    ap.add_argument("--tickers", nargs="*", default=[], help="with --full, or to restrict --plan")
    ap.add_argument("--full", action="store_true", help="every weekday in the window for each ticker")
    ap.add_argument("--start", default="2023-09-19")
    ap.add_argument("--end", default="2025-09-18")
    ap.add_argument("--lead", type=int, default=25, help="weekdays before each event day")
    ap.add_argument("--controls", type=int, default=100, help="random extra control weekdays per ticker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dbn")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--yes", action="store_true", help="actually download (default is a dry run)")
    ap.add_argument("--max-cost", type=float, default=None, help="abort if estimated total USD exceeds this")
    a = ap.parse_args()

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("Set DATABENTO_API_KEY in your environment first (see the docstring).")
    import databento as db
    client = db.Historical(key)

    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    if a.plan:
        plan = pd.read_csv(a.plan, parse_dates=["event_day"])
        if a.tickers:
            plan = plan[plan["ticker"].isin([t.upper() for t in a.tickers])]
        by_tk = {tk: list(g["event_day"]) for tk, g in plan.groupby("ticker")}
    elif a.tickers and a.full:
        by_tk = {t.upper(): [] for t in a.tickers}
    else:
        raise SystemExit("give --plan, or --tickers with --full")

    # ---- build the day list per ticker and price it ---------------------------------
    todo, total_cost, total_gb = {}, 0.0, 0.0
    for tk, evd in sorted(by_tk.items()):
        days = days_for_ticker(evd, start, end, a.lead, a.controls, a.seed, a.full)
        todo[tk] = days
        cost = size = 0.0
        for s, e in to_ranges(days):
            kw = dict(dataset=a.dataset, symbols=[tk], schema=SCHEMA, stype_in="raw_symbol",
                      start=et_day_bounds(s)[0], end=et_day_bounds(e)[1])
            cost += client.metadata.get_cost(**kw)
            size += client.metadata.get_billable_size(**kw)
        total_cost += cost
        total_gb += size / 1e9
        print(f"{tk:6s} days={len(days):4d} est_cost=${cost:8.2f} est_size={size/1e9:6.2f} GB")
    print(f"\nTOTAL est_cost=${total_cost:.2f}  est_size={total_gb:.1f} GB  tickers={len(todo)}")

    if a.max_cost is not None and total_cost > a.max_cost:
        raise SystemExit(f"estimate ${total_cost:.2f} exceeds --max-cost {a.max_cost}; nothing downloaded")
    if not a.yes:
        print("Dry run only. Re-run with --yes to download.")
        return

    # ---- download, one file per ET day, resumable ----------------------------------------
    out = Path(a.out)
    for tk, days in todo.items():
        (out / tk).mkdir(parents=True, exist_ok=True)
        got = skipped = empty = failed = 0
        for d in days:
            path = out / tk / f"xnas-itch-{d:%Y%m%d}.trades.dbn"
            if path.exists() and path.stat().st_size > 1000:
                skipped += 1
                continue
            s, e = et_day_bounds(d)
            try:
                client.timeseries.get_range(dataset=a.dataset, symbols=[tk], schema=SCHEMA,
                                            stype_in="raw_symbol", start=s, end=e, path=str(path))
            except Exception as ex:                  # holidays / outside licensed range / transient
                failed += 1
                print(f"  [fail] {tk} {d:%Y-%m-%d}: {str(ex)[:120]}")
                if path.exists():
                    path.unlink()
                continue
            if not path.exists() or path.stat().st_size <= 1000:   # holiday: header only
                empty += 1
                if path.exists():
                    path.unlink()
            else:
                got += 1
        print(f"[{tk}] downloaded={got} skipped_existing={skipped} empty={empty} failed={failed}")


if __name__ == "__main__":
    main()
