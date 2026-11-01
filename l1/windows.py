"""Part A2 / A2e / A3e helpers for scripts/after_hours_pilot.py and scripts/pool_events.py.
SKELETON: every function body is yours. Build steps: docs/PART_A_BUILD.md (same IDs as below).

Conventions (match the existing pilot, not the old tutorial):
  t      seconds since ET midnight (the pilot's e["t_a"]); 04:00 = 14400, 09:30 = 34200, 16:00 = 57600
  k      column index into a counts row: k = ceil(t - T0), T0 = 14400; a row has N = 57600 columns
  cum    cumulative counts with a leading zero column, shape [n_days, N + 1];
         total over columns [k_start, k_end) = cum[:, k_end] - cum[:, k_start]
  close  z["close_s"][row]: 57600 normally, 46800 on half-days
  windows are half-open [lo, hi) seconds relative to acceptance, like the pilot's WINDOWS
"""
from __future__ import annotations

import numpy as np
import pandas as pd

T0 = 4 * 3600                 # 04:00, column 0
T_OPEN = 9 * 3600 + 30 * 60   # 09:30
T_END = 20 * 3600             # 20:00, one past the last column
N = T_END - T0                # 57600 columns
CROSS_GUARD_S = 1             # the opening/closing cross prints in the single second at 09:30:00 / 16:00:00
                              # (MSFT median at 16:00:00: 3.1M shares vs 0 at 16:00:01). Segments that
                              # START at a cross skip that second.

# name: (lo, hi, anchored). Run next to the pilot's WINDOWS so the old tables still reproduce.
NEW_WINDOWS = {
    "anch_15m": (-900, 900, True),   # start = max(segment start, t - 15 min): catches release-timed bursts
}
DENSITY_S = 1800                     # G2 density: raw prints in [t, t + 30 min) on the event day
EARN = "2.02"


# ---------------------------------------------------------------- A2-1
def segment_bounds(close_s: float) -> list[tuple[str, float, float]]:
    """-> [('pre_open', 14400, 34200), ('intraday', 34201, close_s), ('after_hours', close_s + 1, 72000)]
    (with CROSS_GUARD_S = 1). Seconds since midnight, half-open [start, end)."""
    #temporary; will need to modify 
    return [('pre_open', T0, T_OPEN), ('intraday', T_OPEN+CROSS_GUARD_S, close_s), ('after_hours', close_s + 1, T_END)]



# ---------------------------------------------------------------- A2-2
def segment_of(t: float, close_s: float) -> tuple[str, float, float] | None:
    """The (name, start, end) from segment_bounds whose UNGUARDED span holds t, else None.
    An 8-K at exactly 16:00:00 is after_hours, even though its segment's counting starts at 16:00:01."""
    for name, start, end in segment_bounds(close_s): 
        ung_s = start
        if name == "pre_open": 
            ung_s = start
        else: 
            ung_s = start-CROSS_GUARD_S
        
        if ung_s <= t < end: 
            return (name, start, end)


# ---------------------------------------------------------------- A2-3
def item_codes(items) -> list[str]:
    """'2.02, 9.01' -> ['2.02', '9.01']; NaN/None/'' -> []."""
    if items is None or pd.isna(items):
        return []
    return [c.strip() for c in str(items).split(",") if c.strip()]


# ---------------------------------------------------------------- A2-4
def classify_event(t: float, items, close_s: float) -> tuple[str | None, bool]:
    is_earnings = EARN in item_codes(items)
    event_time = segment_of(t, close_s)
    if not event_time:
        return (None, False)
    if event_time[0] == "after_hours":
        if is_earnings:
            label = "AH_earnings_2.02"
        else:
            label = "AH_other"
    else:
        label = event_time[0]
    return (label, is_earnings)

# ---------------------------------------------------------------- A2-5
def window_bounds(t: float, lo: int, hi: int, anchored: bool, seg_start: float, seg_end: float
                  ) -> tuple[int, int, bool]:
    start = t+lo
    if anchored: 
        start = max(start, seg_start) 
        
    end =t+hi 
    truncated = end> seg_end
    end = min(end, seg_end)
    k_start = int(np.ceil(start - T0))
    k_end = int(np.ceil(end - T0))
    return k_start, k_end, bool(truncated)


# ---------------------------------------------------------------- A2-6
def event_outcomes(cum: np.ndarray, row: int, ctrl_rows: np.ndarray, k_start: int, k_end: int) -> dict:
    n = float(cum[row, k_end] - cum[row, k_start])
    c = cum[ctrl_rows, k_end] - cum[ctrl_rows, k_start]
    med = float(np.median(c))
    lr_ctrl = np.log((c + 1) / (med + 1))
    logr = float(np.log((n + 1) / (med + 1)))
    ratio = (n + 1) / (med + 1)
    rank_p = (1 + (c >= n).sum()) / (1 + len(c))
    return {"n": n, "med": med, "logr": logr, "ratio": ratio, "rank_p": rank_p,
            "n_ctrl": len(c), "lr_ctrl": lr_ctrl}


# ---------------------------------------------------------------- A2e-1
def next_session_row(dates: pd.DatetimeIndex, row: int, calendar: pd.DatetimeIndex) -> int | None:
    """Row in `dates` (this ticker's tape) of the calendar session after dates[row], or None if that
    session is not on the tape or dates[row] is the last calendar session."""
    row_acc = dates[row]
    try:
        i = calendar.get_loc(row_acc)
    except KeyError:
        return None
    if i == len(calendar) - 1:
        return None
    next_trading_day = calendar[i + 1]
    try:
        return int(dates.get_loc(next_trading_day))
    except KeyError:
        return None

# ---------------------------------------------------------------- A2e-2
def day24_total(cum, row, nrow, k):
    if nrow is None:
        return np.nan
    nc = cum.shape[1] - 1                       # last column of cum = the full-day total
    return (cum[row, nc] - cum[row, k]) + cum[nrow, k]


# ---------------------------------------------------------------- A3e-1
SOURCE_ORDER = (("earnings_release", {"2.02"}), ("reg_fd", {"7.01"}),
                ("agreement", {"1.01", "1.02", "2.01"}), ("personnel", {"5.02"}),
                ("annual_meeting_vote", {"5.07"}))     # new: negative control

def source_class(items):
    codes = set(item_codes(items))
    for name, required in SOURCE_ORDER:
        if codes & required:                    # sets overlap
            return name
    return "other_filing"


