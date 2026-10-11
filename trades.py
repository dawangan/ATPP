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
    """Canonical form: ts = int64 ns UTC, price = float dollars, size = int64, side = str.
    Accepts both to_df()-style (datetime ts, float price) and raw (int ns, 1e-9 fixed-point)."""
    if "ts_event" not in df.columns:
        df = df.reset_index()                      # non-inplace: do not mutate the caller's frame
    ts = df["ts_event"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        ts_ns = pd.to_datetime(ts, utc=True).dt.as_unit("ns").astype("int64")
    elif pd.api.types.is_integer_dtype(ts):
        ts_ns = ts.astype("int64")
    else:
        raise TypeError("ts_event must be integer ns or datetime")

    price = df["price"]
    if pd.api.types.is_integer_dtype(price):       # raw fixed-point: 1e-9 dollars, with an UNDEF sentinel
        price = price.where(price != PRICE_UNDEF).astype("float64") / 1e9
    else:
        price = price.astype("float64")

    out = df.assign(ts=ts_ns, price=price, size=df["size"].astype("int64"), side=df["side"].astype(str))
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
    
    tr = tr.loc[(tr['tod'] >= RTH_OPEN_S+open_buffer_s) & (tr['tod'] < RTH_CLOSE_S - close_buffer_s)]
    return tr.reset_index(drop=True)
    
    
    
   

def collapse_simultaneous(tr: pd.DataFrame, window_ns: int = 0) -> pd.DataFrame:
    
  ts = tr["ts"].to_numpy()
  gap = np.diff(ts, prepend=ts[0] - window_ns - 1)
  gap_flag = gap > window_ns
  date_flag = tr["date"].ne(tr["date"].shift()).to_numpy()
  new_cluster = gap_flag | date_flag
  cluster_id = np.cumsum(new_cluster) - 1
  
  tr_copy= tr.assign(pxsz = tr["price"]*tr['size'])
  tr_copy['cid'] = cluster_id
  
  g = tr_copy.groupby('cid')
  big = g["size"].idxmax()                      # row label of the largest fill per cluster

  out = pd.DataFrame({
        "ts": g["ts"].first(),
        "size": g["size"].sum(),
        "price": g["pxsz"].sum() / g["size"].sum(),   # VWAP
        "side": tr_copy.loc[big.to_numpy(), "side"].to_numpy(),
        "n_fills": g["ts"].count(),
    })
  for col in ("date", "tod", "publisher_id"):
        if col in tr_copy.columns:
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
            window_ns: int = 0) -> pd.DataFrame:
    """Convenience: normalize -> drop TRF -> clock -> RTH -> collapse -> durations."""
    tr = normalize_trades(df)
    if drop_trf:
        tr = drop_publishers(tr)
    tr = add_session_clock(tr)
    tr = filter_rth(tr, open_buffer_s=open_buffer_s)
    tr = collapse_simultaneous(tr, window_ns=window_ns)
    return compute_durations(tr)
