#!/usr/bin/env python3
"""Daily Databento .dbn(.zst) trade files -> one parquet of RTH durations + a per-day diagnostic table.

    python scripts/load_dbn.py --dir path/to/MSFT --out msft_durations.parquet

Each file is read with DBNStore.to_df() (datetime ts_event, float dollar price) and
passed through trades.prepare() one day at a time, so memory stays at one day.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l1 import trades  # noqa: E402


def one_day(path: Path, open_buffer_s: float, window_ns: int):
    import databento as db
    df = db.DBNStore.from_file(path).to_df()
    tr = trades.add_session_clock(trades.drop_publishers(trades.normalize_trades(df)))
    closes = trades.detect_closes(tr)
    rth = trades.filter_session(tr, closes, open_buffer_s=open_buffer_s)
    col = trades.collapse_simultaneous(rth, window_ns=window_ns)
    dur = trades.compute_durations(col)
    raw_gap = np.diff(rth["ts"].to_numpy())
    first = rth.iloc[0] if len(rth) else None
    diag = {
        "file": path.name,
        "date": col["date"].iloc[0] if len(col) else pd.NaT,
        "close_s": float(closes.iloc[0]) if len(closes) else np.nan,
        "n_raw": len(df), "n_rth_prints": len(rth), "n_events": len(col), "n_durations": len(dur),
        "share_zero_gap_prints": float(np.mean(raw_gap == 0)) if len(raw_gap) else np.nan,
        "share_dur_lt_1ms": float(np.mean(dur["duration"] < 1e-3)) if len(dur) else np.nan,
        "mean_dur_s": float(dur["duration"].mean()) if len(dur) else np.nan,
        "median_dur_s": float(dur["duration"].median()) if len(dur) else np.nan,
        "first_tod": float(rth["tod"].min()) if len(rth) else np.nan,
        "last_tod": float(rth["tod"].max()) if len(rth) else np.nan,
        "first_print_size": int(first["size"]) if first is not None else 0,
        "max_print_size": int(rth["size"].max()) if len(rth) else 0,
        "share_side_N": float((rth["side"] == "N").mean()) if len(rth) else np.nan,
    }
    keep = ["date", "ts_prev", "ts", "tod_prev", "duration"]
    return dur[keep], diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag", default=None)
    ap.add_argument("--open-buffer", type=float, default=0.0)
    ap.add_argument("--collapse-window-ns", type=int, default=0)
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob("*.trades.dbn.zst"))
    durs, diags = [], []
    for i, f in enumerate(files):
        d, g = one_day(f, a.open_buffer, a.collapse_window_ns)
        durs.append(d); diags.append(g)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(files)} {f.name}", flush=True)
    dur = pd.concat(durs, ignore_index=True)
    dur["date"] = pd.to_datetime(dur["date"])
    dur.to_parquet(a.out, index=False)
    pd.DataFrame(diags).to_csv(a.diag or str(Path(a.out).with_suffix(".diag.csv")), index=False)
    print(f"{len(dur):,} durations over {dur['date'].nunique()} sessions -> {a.out}")


if __name__ == "__main__":
    main()
