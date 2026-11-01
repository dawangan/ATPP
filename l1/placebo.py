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
    """Move each event to a random non-event trading day at the same LOCAL clock time.
    Days are drawn without replacement when the pool allows (real events sit on
    distinct days, so placebos should too)."""
    rng = np.random.default_rng(rng)
    days = pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_days))).normalize().unique().sort_values()
    ev_days = pd.DatetimeIndex(pd.to_datetime(events["eff_date"])).normalize()
    pool = days.difference(ev_days)
    # PATCH: `if avoid_days:` raises on a DatetimeIndex (ambiguous truth value)
    if avoid_days is not None and len(avoid_days):
        pool = pool.difference(pd.DatetimeIndex(pd.to_datetime(pd.Index(avoid_days))).normalize())
    if len(pool) == 0:
        raise ValueError("placebo day pool is empty")
    pos = rng.choice(len(pool), size=len(events), replace=len(pool) < len(events))
    picked = pool[pos]

    # PATCH: was picked.to_datetime(...) (no such method -> AttributeError)
    local = pd.DatetimeIndex(pd.to_datetime(events["eff_ns"].to_numpy(), utc=True)).tz_convert(TZ)
    tod = local - local.normalize()                    # wall-clock offset from local midnight
    new_local = (picked + tod).tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    new_ns = new_local.tz_convert("UTC").as_unit("ns").asi8

    out = events.copy()
    out["eff_ns"] = new_ns
    out["eff_date"] = picked
    if "eff_utc" in out.columns:
        out["eff_utc"] = pd.to_datetime(new_ns, utc=True)
    # PATCH: the sorted frame was computed and discarded; build_design needs sorted times
    return out.sort_values("eff_ns", kind="stable").reset_index(drop=True)
  
def shift_events(events: pd.DataFrame, delta_s: float) -> pd.DataFrame:
    out = events.copy()
    out["eff_ns"] = out["eff_ns"].to_numpy(np.int64) + int(delta_s * 1e9)
    return out
