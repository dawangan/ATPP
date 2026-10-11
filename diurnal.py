"""
Stage 2: remove the deterministic intraday pattern.

Decomposition (Engle & Russell 1998):   x_i = L_d * phi(tod_{i-1}) * x~_i
  L_d     day-level scale, estimated from PAST days only (trailing mean)
  phi(.)  time-of-day factor, cubic regression spline, knots every 30 min
  x~_i    adjusted duration, what the ACD model sees (mean ~ 1)

Two look-ahead traps this module is built to avoid:
  1. Normalizing each day by its OWN mean duration (a common recipe in the
     literature). If an 8-K makes the whole day faster, same-day normalization
     divides the effect away. Use a trailing level instead.
  2. Fitting phi on event days. After-hours filings map to 09:30, so events
     cluster at the open; fitting phi on all days absorbs part of the event
     effect into "the open is fast". Fit phi on non-event days.

Tests: pytest tests/test_2_diurnal.py -q
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_lsq_spline

from .trades import RTH_CLOSE_S, RTH_OPEN_S


def daily_level(dur: pd.DataFrame, window: int = 20, min_periods: int = 5) -> pd.Series:

  daily = dur.groupby("date")["duration"].mean()
  past = daily.shift(1)
  level = past.rolling(window, min_periods=min_periods).mean()
  level = level.rename("level")
  return level
@dataclass
class DiurnalModel:
    spline: BSpline
    level: pd.Series
    lo: float
    hi: float

    def phi(self, tod) -> np.ndarray:
        t = np.clip(np.asarray(tod, float), self.lo, self.hi)
        return np.maximum(self.spline(t), 1e-6)


def _knots(lo: float, hi: float, every_s: float) -> np.ndarray:
    """Full knot vector for a cubic spline on [lo, hi]: boundary knots repeated
    k+1 = 4 times, interior knots every `every_s` seconds."""
    interior = np.arange(lo + every_s, hi - 1e-9, every_s)
    return np.r_[[lo] * 4, interior, [hi] * 4]


def fit_diurnal(dur, knot_minutes=30, exclude_dates=None, window=20,
                lo=RTH_OPEN_S, hi=RTH_CLOSE_S):
    level = daily_level(dur, window=window)          # level from ALL days
    df = dur.assign(level=dur["date"].map(level)).dropna(subset=["level"])

    if exclude_dates is not None:                    # drop event days from the FIT only
        ex = pd.to_datetime(list(exclude_dates)).normalize()
        df = df[~df["date"].dt.normalize().isin(ex)]

    y = (df["duration"] / df["level"]).to_numpy()
    x = df["tod_prev"].to_numpy(float)

    xu, inv = np.unique(x, return_inverse=True)      # sorted unique x, row -> unique index
    cnt = np.bincount(inv)                           # rows per unique x
    ymean = np.bincount(inv, weights=y) / cnt        # mean y per unique x

    spline = make_lsq_spline(xu, ymean, t=_knots(lo, hi, knot_minutes * 60),
                             k=3, w=np.sqrt(cnt))
    return DiurnalModel(spline, level, lo, hi)

  

def adjust(dur, model):
    lv = dur["date"].map(model.level)
    ph = model.phi(dur["tod_prev"])
    out = dur.assign(level=lv, phi=ph)
    out["x"] = out["duration"] / (out["level"] * out["phi"])
    return out.dropna(subset=["level"])


def diurnal_profile(dur, bucket_minutes=5, col="duration"):
    width = bucket_minutes * 60
    bucket = (dur["tod_prev"] // width * width).rename("bucket")

    prof = dur.groupby(bucket)[col].agg(["mean", "median", "count"]).reset_index()

    h = (prof["bucket"] // 3600).astype(int)
    m = ((prof["bucket"] % 3600) // 60).astype(int)
    prof["time"] = h.map("{:02d}".format) + ":" + m.map("{:02d}".format)

    return prof[["bucket", "mean", "median", "count", "time"]]