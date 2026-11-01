#!/usr/bin/env python3
"""A3: stack every ticker's per-event results with census metadata -> panel_events.parquet.
SKELETON: argparse and main() wiring are done; the function bodies are yours (docs/PART_A_BUILD.md, A3-1..A3-6).

    python scripts/pool_events.py --out panel_events.parquet

Inputs (after the A2r re-runs, which add ticker/accession/is_earnings to every row):
  results_<tk>/g1/per_event_events.csv        LONG: one row per event x OLD window
  results_<tk>/g1/per_event_extra_events.csv  LONG: one row per event x NEW outcome (anch_15m, day24),
                                              plus dens_30m_prints, anchor_lag_s, truncated
Exclude results_msft/ (old G1 event file); MSFT comes from results_msft_census/.
n_event is in the --measure unit (default 'events' = prints collapsed at 1 ms); dens_30m_prints is raw prints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l1.windows import source_class  # noqa: E402

KEY = ["ticker", "accession"]
STATS = ["n_event", "control_median", "ratio", "rank_p", "n_controls"]
EXPLORE_END = pd.Timestamp("2025-09-18")     # PREREGISTRATION.md §4: spec frozen on 2023-09-19 .. 2025-09-18


# ---------------------------------------------------------------- A3-1
def read_per_event(path: str | Path) -> pd.DataFrame:
    """Read one long per-event CSV; guarantee a 'ticker' column (upper case) and an 'accession' column.
    Raise ValueError naming the file if accession is missing (= it predates A2-8: re-run A2r)."""
    raise NotImplementedError("A3-1")


# ---------------------------------------------------------------- A3-2
def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, accession). Columns: KEY + cls + is_earnings + items + accept, then
    '<stat>_<window>' for stat in STATS, e.g. ratio_post_15m, n_event_anch_15m."""
    raise NotImplementedError("A3-2")


# ---------------------------------------------------------------- A3-3
def attach_census(wide: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """Left-merge census metadata on KEY, one_to_one; print the _merge counts; raise if any
    event is 'left_only' (an event the census doesn't know = a timing or ticker bug)."""
    raise NotImplementedError("A3-3")


# ---------------------------------------------------------------- A3-4
def add_density_gate(df: pd.DataFrame, col: str, min_density: float) -> pd.DataFrame:
    """Adds dens_ok (row: df[col] >= min_density), ticker_median_density, g2_pass (ticker median >= min).
    FLAGS only: nothing is dropped here (the G2 decision is still open)."""
    raise NotImplementedError("A3-4")


# ---------------------------------------------------------------- A3-5
def add_period(df: pd.DataFrame, date_col: str = "event_day") -> pd.DataFrame:
    """period = 'explore' if date <= EXPLORE_END else 'holdout'; assert no holdout rows (tapes end
    2025-09-18, so a holdout row means a date bug)."""
    raise NotImplementedError("A3-5")


# ---------------------------------------------------------------- A3-6
def ticker_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: events by cls, by source, median density, g2_pass. Counts only, NO outcome columns:
    this table is printed before the B5 pre-registration is frozen."""
    raise NotImplementedError("A3-6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-event", nargs="+", default=None,
                    help="default: results_*/g1/per_event_events.csv + per_event_extra_events.csv, minus results_msft/")
    ap.add_argument("--census", default="census/events.csv")
    ap.add_argument("--out", default="panel_events.parquet")
    ap.add_argument("--density-col", default="dens_30m_prints")
    ap.add_argument("--min-density", type=float, default=200)
    a = ap.parse_args()

    if a.per_event:
        paths = a.per_event
    else:
        paths = sorted(str(p) for p in Path(".").glob("results_*/g1/per_event*_events.csv")
                       if p.parts[0] != "results_msft")
    print(f"{len(paths)} per-event files")
    long = pd.concat([read_per_event(p) for p in paths], ignore_index=True)
    wide = to_wide(long)
    census = pd.read_csv(a.census, parse_dates=["event_day"])
    census["ticker"] = census["ticker"].str.upper()
    df = attach_census(wide, census)
    df["source"] = df["items"].map(source_class)
    df = add_density_gate(df, a.density_col, a.min_density)
    df = add_period(df)
    pd.set_option("display.width", 200)
    print(ticker_report(df).to_string())
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} events -> {a.out}")


if __name__ == "__main__":
    main()
