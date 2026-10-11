"""
Stage 6: falsification. A significant gamma means little until these are silent.

placebo_events : same number, kind and time-of-day as the real events, moved to
                 random NON-event days. Keeps the open-heavy timing of real
                 events, so a gamma that only reflects "the open is fast" shows
                 up here too and gets exposed.
shift_events   : move every event by delta seconds. A real effect should peak
                 at delta ~ 0 and vanish for delta < 0 (before the filing). A
                 response already present at delta = -10 min means the
                 information reached the market before the 8-K acceptance
                 (press release on the wire first), i.e. the 8-K timestamp lags
                 the information, which matters for any lead-time claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .trades import TZ


def placebo_events(events: pd.DataFrame, trading_days, rng=None, avoid_days=None) -> pd.DataFrame:
    """Fake event table with the same rows (kind, items, ...) as `events`, but each
    event moved to a random trading day that has NO real event, keeping its
    local time of day (so 09:30-mapped events stay at 09:30).

    Return a copy of events with eff_ns (int64 ns, UTC) and eff_date replaced,
    sorted by eff_ns. Draw days with replacement from the pool of non-event days
    (also excluding avoid_days if given). Raise ValueError if the pool is empty.

    Why match time of day: most scheduled events sit at the open, where
    durations are shortest. A placebo at random times would not reproduce that,
    so a gamma that merely reflects "the open is fast" would look significant
    against it. Matched placebos expose that.

    HINTS
    -----
    - local = pd.to_datetime(eff_ns, utc=True).tz_convert(TZ); tod = local - local.normalize().
    - new = (DatetimeIndex(picked_days) + tod).tz_localize(TZ) -> tz_convert("UTC")
      -> .as_unit("ns").asi8.
    """
    rng = np.random.default_rng(rng)
    out = events.copy()
    eff = pd.to_datetime(out["eff_ns"].to_numpy(np.int64), utc=True).tz_convert(TZ)
    tod = eff - eff.normalize()

    days = pd.DatetimeIndex(pd.to_datetime(list(trading_days))).normalize()
    bad = pd.DatetimeIndex(eff.tz_localize(None).normalize()).unique()
    if avoid_days is not None:
        bad = bad.union(pd.DatetimeIndex(pd.to_datetime(list(avoid_days))).normalize())
    pool = days[~days.isin(bad)]
    if len(pool) == 0:
        raise ValueError("no non-event trading days available")

    picked = pool[rng.integers(0, len(pool), size=len(out))]
    new = (picked + tod).tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    out["eff_ns"] = new.as_unit("ns").asi8
    out["eff_date"] = picked
    out = out[new.notna()]
    return out.sort_values("eff_ns").reset_index(drop=True)


def shift_events(events: pd.DataFrame, delta_s: float) -> pd.DataFrame:
    out = events.copy()
    out["eff_ns"] = out["eff_ns"].to_numpy(np.int64) + int(delta_s * 1e9)
    return out
