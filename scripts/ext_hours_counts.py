#!/usr/bin/env python3
"""Per-second extended-hours activity counts from daily Databento trade files.

For every session file: keep 04:00:00-20:00:00 ET, collapse simultaneous prints (default 1 ms),
and count, per second of the day, (a) collapsed executions, (b) raw prints, (c) shares.

    python scripts/ext_hours_counts.py --dir path/to/MSFT --out msft_ext_counts.npz [--procs 2]

Output .npz: dates (YYYY-MM-DD), close_s (session close, 57600 or 46800 on half-days),
events/prints (uint16, shape [n_days, 57600]), shares (uint32), tod0=14400.
Index k of a row is second 04:00:00 + k of that ET day.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l1 import trades  # noqa: E402

T0, T1 = 4 * 3600, 20 * 3600
N = T1 - T0


NAME_RE = re.compile(r"xnas-itch-(\d{8})\.trades\.dbn(\.zst)?")


def list_session_files(folder, pattern):
    """One file per session date. Skips (and reports) directories, unparseable names, empty files,
    and duplicate dates (keeps the .zst original over an uncompressed copy)."""
    log = []
    by_date = defaultdict(list)
    for p in sorted(Path(folder).glob(pattern)):
        if not p.is_file():
            log.append(f"skip (not a file): {p.name}")
            continue
        m = NAME_RE.fullmatch(p.name)
        if m is None:
            log.append(f"skip (name): {p.name}")
            continue
        if p.stat().st_size == 0:
            log.append(f"skip (empty): {p.name}")
            continue
        by_date[m.group(1)].append(p)
    files = []
    for d in sorted(by_date):
        cands = sorted(by_date[d], key=lambda p: p.suffix != ".zst")   # .zst first
        files.append(cands[0])
        for extra in cands[1:]:
            log.append(f"skip (duplicate date {d}, kept {cands[0].name}): {extra.name}")
    return files, log


def one(args):
    """Never raises: returns ('ok', row) or ('err', filename, message), so one bad file
    cannot kill the pool."""
    path, window_ns = args
    try:
        r = _one(path, window_ns)
    except Exception as exc:                                   # noqa: BLE001
        return ("err", Path(path).name, f"{type(exc).__name__}: {exc}")
    return ("ok", r)


def _one(path, window_ns):
    import databento as db
    df = db.DBNStore.from_file(path).to_df()
    tr = trades.add_session_clock(trades.drop_publishers(trades.normalize_trades(df)))
    if tr.empty:
        return None
    closes = trades.detect_closes(tr)
    # a session file can touch two ET dates only if it spans midnight; keep the modal date
    date = tr["date"].mode().iloc[0]
    tr = tr[tr["date"] == date]
    ext = tr[(tr["tod"] >= T0) & (tr["tod"] < T1)].reset_index(drop=True)
    col = trades.collapse_simultaneous(ext, window_ns=window_ns)
    sec_raw = np.floor(ext["tod"].to_numpy() - T0).astype(int)
    sec_col = np.floor(col["tod"].to_numpy() - T0).astype(int)
    prints = np.bincount(sec_raw, minlength=N)[:N].astype(np.uint16)
    events = np.bincount(sec_col, minlength=N)[:N].astype(np.uint16)
    shares = np.bincount(sec_raw, weights=ext["size"].to_numpy(), minlength=N)[:N].astype(np.uint32)
    close_s = float(closes.get(date, trades.RTH_CLOSE_S))
    return str(pd.Timestamp(date).date()), close_s, events, prints, shares, float(ext["tod"].min()), float(ext["tod"].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-ms", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--glob", default="xnas-itch-*.trades.dbn*")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N files only")
    a = ap.parse_args()

    files, log = list_session_files(a.dir, a.glob)
    if a.limit:
        files = files[: a.limit]
    w = int(a.window_ms * 1e6)
    rows = []
    with Pool(a.procs) as pool:
        for i, res in enumerate(pool.imap(one, [(f, w) for f in files], chunksize=1), 1):
            if res[0] == "err":
                log.append(f"FAILED {res[1]}: {res[2]}")
            elif res[1] is not None:
                rows.append(res[1])
            if i % 25 == 0 or i == len(files):
                print(f"{i}/{len(files)} {files[i-1].name}", flush=True)
    rows.sort(key=lambda r: r[0])
    dates = [r[0] for r in rows]
    assert len(dates) == len(set(dates)), "duplicate session dates in output"
    if dates:   # weekdays with no session: holidays are expected here, anything else is a missing download
        have = set(dates)
        gaps = [d for d in pd.bdate_range(dates[0], dates[-1]).strftime("%Y-%m-%d") if d not in have]
        log.append(f"weekdays without a session ({len(gaps)}; holidays expected): {' '.join(gaps)}")
    log_path = Path(str(a.out) + ".log.txt")
    log_path.write_text("\n".join(log) + "\n")
    n_fail = sum(s.startswith("FAILED") for s in log)
    n_skip = sum(s.startswith("skip") for s in log)
    print(f"{n_skip} skipped, {n_fail} failed -> details in {log_path}")
    np.savez_compressed(
        a.out,
        dates=np.array([r[0] for r in rows]),
        close_s=np.array([r[1] for r in rows]),
        events=np.stack([r[2] for r in rows]),
        prints=np.stack([r[3] for r in rows]),
        shares=np.stack([r[4] for r in rows]),
        first_tod=np.array([r[5] for r in rows]),
        last_tod=np.array([r[6] for r in rows]),
        tod0=T0,
    )
    print(f"{len(rows)} sessions -> {a.out}")


if __name__ == "__main__":
    main()
