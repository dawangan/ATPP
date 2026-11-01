"""
Stage 1: Databento trades -> clean trade events -> intraday durations.

Input: the DataFrame you already have (Databento `trades` schema, columns
ts_event, rtype, publisher_id, instrument_id, action, side, depth, price, size,
flags, ts_in_delta, sequence, symbol; ts_recv usually the index).

Output of the stage (compute_durations): one row per duration x_i = t_i - t_{i-1},
both trades on the same session date, with the clock of the START of the
interval (t_{i-1}) attached, because everything we condition on must be known
at t_{i-1}.

Tests: pytest tests/test_1_trades.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/New_York"
RTH_OPEN_S = 9 * 3600 + 30 * 60      # 09:30:00 local, seconds after midnight
RTH_CLOSE_S = 16 * 3600              # 16:00:00 local
PRICE_UNDEF = np.iinfo(np.int64).max  # Databento UNDEF_PRICE sentinel (fixed-point)

# Trade-reporting-facility (off-exchange) publishers, from Databento's
# metadata.list_publishers(). TRF prints are *reported* late (FINRA allows up to
# 10 s), so their ts_event is report time, not execution time. For a timing
# study they are noise; drop them. Check your own frame with
# df.publisher_id.value_counts() first.
TRF_PUBLISHER_IDS = frozenset({17, 18, 19, 54, 55, 56, 68, 69, 70, 82, 83, 130, 131, 132})


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Any Databento trades frame -> int64 ns `ts`, float dollar `price`, sorted.

    Accepts to_df() output (datetime ts_event, float price) and raw output
    (int ns ts_event, fixed-point int price in units of 1e-9 dollars).
    Never modifies the caller's frame."""
    # PATCH: reset_index on a copy, not in place (the old version mutated the caller's df)
    out = df.reset_index() if "ts_event" not in df.columns else df.copy()

    ts = out["ts_event"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        ts_ns = pd.to_datetime(ts, utc=True).dt.as_unit("ns").astype("int64")   # naive -> treated as UTC
    elif pd.api.types.is_integer_dtype(ts):
        ts_ns = ts.astype("int64")
    else:
        raise TypeError("ts_event must be integer ns or datetime")
    out["ts"] = ts_ns.to_numpy()

    # PATCH: fixed-point prices were left as 1e-9-dollar integers, and the UNDEF
    # sentinel (int64 max) survived as a real price. Convert and drop it.
    p = out["price"]
    if pd.api.types.is_integer_dtype(p):
        p = p.astype("int64")
        out["price"] = (p.where(p != PRICE_UNDEF) / 1e9).astype(float)
    out = out.dropna(subset=["price"])
    out = out[out["size"] > 0]
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def drop_publishers(tr: pd.DataFrame, ids=TRF_PUBLISHER_IDS) -> pd.DataFrame:
    return tr[~tr["publisher_id"].isin(list(ids))].reset_index(drop=True)


def add_session_clock(tr: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    local = pd.to_datetime(tr["ts"], utc=True).dt.tz_convert(tz)   # aware, local wall clock
    midnight = local.dt.normalize()                                 # aware, local midnight

    out = tr.copy()
    out["tod"] = (local - midnight).dt.total_seconds()
    out["date"] = midnight.dt.tz_localize(None)                     # naive local date
    return out 


def filter_rth(tr: pd.DataFrame, open_buffer_s: float = 0.0, close_buffer_s: float = 0.0) -> pd.DataFrame:
    """Keep trades with RTH_OPEN_S + open_buffer_s <= tod < RTH_CLOSE_S - close_buffer_s.
    Needs add_session_clock first. A small open buffer (5-30 s) removes the
    opening-cross print burst; decide on your data whether you need it."""
    
    keep = (tr["tod"] >= RTH_OPEN_S + open_buffer_s) & (tr["tod"] < RTH_CLOSE_S - close_buffer_s)
    return tr.loc[keep].reset_index(drop=True)      # PATCH: no in-place op on a .loc slice


EARLY_CLOSE_S = 13 * 3600


def detect_closes(tr: pd.DataFrame) -> pd.Series:
    """ADDED. Session close (seconds after local midnight) per date.

    Half-days (day after Thanksgiving, Christmas Eve, July 3) close at 13:00, but
    extended-hours trading continues afterwards, so 'where trading stops' cannot
    detect them. Two signatures together identify a half-day:
      1. a closing-cross print in [12:59:30, 13:05:00) at least 10x the day's
         99th-percentile print size (the opening cross can be bigger, so the
         day's single largest print is NOT used), and
      2. activity collapses after it: prints in 13:05-16:00 are < 10% of prints
         in 09:30-13:00 (on a normal day the afternoon is ~half the volume)."""
    lo, hi = EARLY_CLOSE_S - 30, EARLY_CLOSE_S + 300

    def one(g):
        t, s = g["tod"].to_numpy(), g["size"].to_numpy()
        win = (t >= lo) & (t < hi)
        if not win.any():
            return RTH_CLOSE_S
        cross = s[win].max() >= 10 * np.quantile(s, 0.99)
        after = ((t >= hi) & (t < RTH_CLOSE_S)).sum()
        before = ((t >= RTH_OPEN_S) & (t < EARLY_CLOSE_S)).sum()
        return EARLY_CLOSE_S if (cross and after < 0.1 * max(before, 1)) else RTH_CLOSE_S
    return tr.groupby("date").apply(one, include_groups=False).rename("close_s").astype(float)


def filter_session(tr: pd.DataFrame, closes: pd.Series | None = None, open_buffer_s: float = 0.0) -> pd.DataFrame:
    """ADDED. Like filter_rth but cuts each day at ITS close (closes: date -> seconds)."""
    if closes is None:
        return filter_rth(tr, open_buffer_s=open_buffer_s)
    close_row = tr["date"].map(closes).fillna(RTH_CLOSE_S).to_numpy()
    keep = (tr["tod"].to_numpy() >= RTH_OPEN_S + open_buffer_s) & (tr["tod"].to_numpy() < close_row)
    return tr.loc[keep].reset_index(drop=True)


def collapse_simultaneous(tr: pd.DataFrame, window_ns: int = 0) -> pd.DataFrame:
    """Merge prints whose gap to the PREVIOUS print is <= window_ns (same session
    date) into one event: first ts, total size, VWAP price, side of the largest fill.
    Note: gaps chain, so with window_ns > 0 a run of prints each 0.4 ms apart
    merges even if first-to-last spans more than the window."""
    tr = tr.reset_index(drop=True)                   # PATCH: idxmax labels must index tr_copy
    ts = tr["ts"].to_numpy()
    gap = np.diff(ts, prepend=ts[0] - window_ns - 1)
    new_cluster = (gap > window_ns) | tr["date"].ne(tr["date"].shift()).to_numpy()
    cluster_id = np.cumsum(new_cluster) - 1

    # PATCH: ITCH reports hidden-order fills with side 'N' even inside an aggressive
    # sweep whose displayed fills say 'A'/'B'. Take the side of the largest fill that
    # HAS a side; fall back to 'N' only if no fill in the cluster has one.
    sided = tr["side"].isin(["A", "B"]).to_numpy()
    tr_copy = tr.assign(pxsz=tr["price"] * tr["size"], cid=cluster_id,
                        size_sided=np.where(sided, tr["size"].to_numpy().astype(np.int64), -1))
    g = tr_copy.groupby("cid", sort=True)
    big = g["size_sided"].idxmax()                   # largest sided fill (or any fill if none sided)

    out = pd.DataFrame({
        "ts": g["ts"].first(),
        "size": g["size"].sum(),
        "price": g["pxsz"].sum() / g["size"].sum(),  # VWAP
        # PATCH: was g.loc[...] (GroupBy has no .loc -> AttributeError); look up in the frame
        "side": np.where(g["size_sided"].max().to_numpy() > 0,
                         tr_copy.loc[big.to_numpy(), "side"].to_numpy(), "N"),
        "n_fills": g["ts"].count(),
    })
    for col in ("date", "tod", "publisher_id"):
        if col in tr_copy.columns:                   # PATCH: was g.columns
            out[col] = g[col].first()
    return out.reset_index(drop=True)
  

def compute_durations(tr: pd.DataFrame) -> pd.DataFrame:
    ts = tr["ts"].to_numpy()
    date = tr["date"].to_numpy()
    tod = tr["tod"].to_numpy()

    same_day = date[1:] == date[:-1]
    dur = (ts[1:] - ts[:-1])[same_day] / 1e9      # ns -> seconds

    if (dur <= 0).any():
        raise ValueError("non-positive duration: collapse_simultaneous skipped or data unsorted")

    return pd.DataFrame({
        "date": date[1:][same_day],
        "ts_prev": ts[:-1][same_day],
        "ts": ts[1:][same_day],
        "tod_prev": tod[:-1][same_day],
        "duration": dur,
    })

def prepare(df: pd.DataFrame, drop_trf: bool = True, open_buffer_s: float = 0.0,
            window_ns: int = 0, detect_early_close: bool = True) -> pd.DataFrame:
    """Convenience: normalize -> drop TRF -> clock -> session filter -> collapse -> durations.
    PATCH: with detect_early_close, half-days are cut at 13:00 (closing cross) instead
    of keeping three hours of extended-hours prints as if they were regular trading."""
    tr = normalize_trades(df)
    if drop_trf:
        tr = drop_publishers(tr)
    tr = add_session_clock(tr)
    closes = detect_closes(tr) if detect_early_close else None
    tr = filter_session(tr, closes, open_buffer_s=open_buffer_s)
    tr = collapse_simultaneous(tr, window_ns=window_ns)
    return compute_durations(tr)
