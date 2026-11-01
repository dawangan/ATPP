#!/usr/bin/env python3
"""A5 (optional, plan v4 [M]2 / D2 precursor): does a SOURCE firm's 8-K move the PEER firm's tape?
MARA <-> RIOT first: both tapes are on disk; 45 of MARA's 49 timed event days are on RIOT's tape,
37 of RIOT's 37 are on MARA's. STARTER: argparse done, you write the TODOs (part-a guide, A5).

    python scripts/peer_pilot.py --source MARA --peer RIOT --counts ext_RIOT.npz \
        --events census/events.csv --out results_peer/MARA_to_RIOT

!! Pre-register the peer outcome and the confound rule (TODO 3) BEFORE the first run. !!
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from after_hours_pilot import analyse, load_events  # noqa: E402


# ---------------------------------------------------------------- A5-1
def confound_filter(src_dates, peer_dates, calendar: pd.DatetimeIndex, k: int = 1) -> np.ndarray:
    """Boolean keep-mask over src_dates: False if any peer 8-K date is within +-k CALENDAR SESSIONS
    (not calendar days: a Friday source and a Monday peer filing are 1 session apart)."""
    raise NotImplementedError("A5-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--counts", required=True, help="the PEER's ext_*.npz")
    ap.add_argument("--events", default="census/events.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--measure", default="events", choices=["events", "prints", "shares"])
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--calendar", default="ext_MSFT.npz")
    a = ap.parse_args()

    # TODO 1  src = load_events(a.events, a.source); own = load_events(a.events, a.peer)
    # TODO 2  load the peer's npz: counts = z[a.measure], dates, close_s (same three lines as the pilot's main)
    # TODO 3  (A5-1) keep = confound_filter(src['date'], own['date'].unique(), calendar); src = src[keep]
    #         confound rule: drop source events whose date is within +-1 SESSION of any peer 8-K date
    #         (MARA and RIOT both file monthly production updates in the first days of the month;
    #         without this the "peer" burst is the peer's own news). Print how many you drop.
    # TODO 4  summ, per = analyse(counts, dates, close_s, src, B=a.perms, exclude_dates=own["date"].unique())
    #         Why exclude_dates: controls must be quiet for BOTH firms, not just the source.
    # TODO 5  add columns source, peer to both frames; write summary_/per_event_ CSVs to a.out; print summ
    raise NotImplementedError("A5: write the TODOs above")


if __name__ == "__main__":
    main()
