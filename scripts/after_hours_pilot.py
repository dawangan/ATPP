#!/usr/bin/env python3
"""Test G1 (PREREGISTRATION.md section 3): is there a timeable extended-hours reaction to an 8-K?

    python scripts/after_hours_pilot.py --counts msft_ext_counts.npz --events results_msft/res_msft/events.csv \
        --out results_msft/g1_after_hours [--measure events]
    # other ticker, census events file:
    python scripts/after_hours_pilot.py --counts ext_NVDA.npz --events census/events.csv --ticker NVDA --out results_nvda

Per event e (acceptance time t_a, ET) and window (a, b) seconds relative to t_a:
    n_e   = count in [t_a + a, t_a + b) on the event day
    c_ej  = count in the SAME clock window on control day j
Controls: weekdays within +-90 calendar days with no 8-K accepted that day, the day before or after;
not half-days, not third Fridays of Mar/Jun/Sep/Dec, not the last Friday of June.
Effect = mean_e log[(n_e+1) / (median_j c_ej + 1)]; one-sided permutation p (draw a random control
day for every event, 20,000 times). Per-event rank p = (1 + #{c_ej >= n_e}) / (1 + J).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l1 import windows as W  # noqa: E402  (A2: segment_of, classify_event, window_bounds, event_outcomes, ...)

T0 = 4 * 3600
N = 16 * 3600
WINDOWS = {"post_1m": (0, 60), "post_5m": (0, 300), "post_15m": (0, 900), "post_30m": (0, 1800),
           "post_60m": (0, 3600), "lead_5m": (-300, 0), "lead_1m": (-60, 0)}
PRIMARY = "post_15m"


def last_friday_june(d: pd.Timestamp) -> bool:
    return d.month == 6 and d.dayofweek == 4 and (d + pd.Timedelta(days=7)).month == 7


def third_friday_quarter(d: pd.Timestamp) -> bool:
    return d.month in (3, 6, 9, 12) and d.dayofweek == 4 and 15 <= d.day <= 21


def control_days(ev_date, dates, close_s, event_dates):
    """Indices of valid control sessions for an event on ev_date."""
    pos = {d: i for i, d in enumerate(dates)}
    bad = set()
    for d in event_dates:                       # 8-K day, previous and next session
        i = pos.get(d)
        if i is None:
            continue
        bad.update({i, max(i - 1, 0), min(i + 1, len(dates) - 1)})
    out = []
    for i, d in enumerate(dates):
        if i in bad or abs((d - ev_date).days) > 90 or d == ev_date:
            continue
        if close_s[i] < 16 * 3600 or third_friday_quarter(d) or last_friday_june(d):
            continue
        out.append(i)
    return np.array(out, dtype=int)


def window_sum(cum, row, k0, a, b):
    lo, hi = k0 + a, k0 + b
    if lo < 0 or hi > N:
        return np.nan
    return cum[row, hi] - cum[row, lo]


# ---------------------------------------------------------------- A2-7
def perm_pvalue(lrs, B, rng):
    """lrs = [(obs_log_ratio_e, lr_ctrl_e array), ...] -> (obs_mean, one-sided p).
    Must consume rng EXACTLY as the inline code in analyse() does (one rng.integers call per event,
    in event order), or tests/test_11_pilot_regression.py fails."""
    raise NotImplementedError("A2-7")


def analyse(counts, dates, close_s, ev, B=20000, seed=0, windows=WINDOWS, exclude_dates=()):
    """counts: [n_days, N] array; ev: DataFrame with columns date (Timestamp), t_a (seconds ET), cls."""
    dates = pd.DatetimeIndex(dates)
    cum = np.zeros((counts.shape[0], N + 1), dtype=np.float64)
    cum[:, 1:] = np.cumsum(counts, axis=1)
    pos = {d: i for i, d in enumerate(dates)}
    event_dates = list(ev["date"].unique()) + list(exclude_dates)   # exclude_dates: other 8-K days (null calibration)
    rng = np.random.default_rng(seed)

    per_event, summary = [], []
    for cls, g in ev.groupby("cls"):
        for wname, (a, b) in windows.items():
            lrs, rows = [], []
            for _, e in g.iterrows():
                if e["date"] not in pos:
                    continue
                row = pos[e["date"]]
                k0 = int(np.ceil(e["t_a"] - T0))
                ctrl = control_days(e["date"], dates, close_s, event_dates)
                n_e = window_sum(cum, row, k0, a, b)
                c = np.array([window_sum(cum, j, k0, a, b) for j in ctrl])
                ok = ~np.isnan(c)
                c = c[ok]
                if np.isnan(n_e) or len(c) < 20:
                    continue
                med = np.median(c)
                lr_ctrl = np.log((c + 1) / (med + 1))
                # TODO A2-8: add ticker=e.get("ticker"), accession=e.get("accession"),
                #            is_earnings=e.get("is_earnings") to this dict (pool_events joins on them)
                rows.append(dict(cls=cls, window=wname, accept=e["accept_et"], items=e["items"], n_event=n_e,
                                 control_median=med, ratio=(n_e + 1) / (med + 1),
                                 rank_p=(1 + (c >= n_e).sum()) / (1 + len(c)), n_controls=len(c)))
                lrs.append((np.log((n_e + 1) / (med + 1)), lr_ctrl))
            if not lrs:
                continue
            # TODO A2-7: replace these six lines with  obs, p = perm_pvalue(lrs, B, rng)
            obs = np.mean([x[0] for x in lrs])
            null = np.zeros(B)
            for obs_i, lr_ctrl in lrs:
                null += lr_ctrl[rng.integers(0, len(lr_ctrl), size=B)]
            null /= len(lrs)
            p = (1 + (null >= obs).sum()) / (1 + B)
            summary.append(dict(cls=cls, window=wname, n_events=len(lrs), ratio=float(np.exp(obs)),
                                perm_p_one_sided=float(p),
                                median_event_ratio=float(np.median([r["ratio"] for r in rows])),
                                frac_events_ratio_gt1=float(np.mean([r["ratio"] > 1 for r in rows]))))
            per_event += rows
    return pd.DataFrame(summary), pd.DataFrame(per_event)


# ---------------------------------------------------------------- A2-10
def analyse_extra(counts, dates, close_s, ev, calendar, B=20000, seed=1, windows=W.NEW_WINDOWS,
                  exclude_dates=(), density_counts=None, day24=True):
    """NEW outcomes, kept out of analyse() so the old tables reproduce bit for bit:
      * windows in W.NEW_WINDOWS (anch_15m): segment-aware, anchored, truncated at the segment end
      * day24 (A2e): from acceptance to the same clock time on the next calendar session
      * dens_30m_prints: raw prints in [t, t+30 min) on the event day (G2 density; no controls)
    -> (summary DataFrame, per_event DataFrame) with the same columns as analyse() plus
       anchor_lag_s, truncated, k_start, k_end, dens_30m_prints.
    calendar: pd.DatetimeIndex of ALL sessions (MSFT's dates), not this ticker's tape."""
    raise NotImplementedError("A2-10")


def profile(counts, dates, close_s, ev, lo=-600, hi=1800, bin_s=30):
    """Pooled ratio of event count to control-median count in bin_s-second bins relative to t_a, plus
    the onset estimate (first 30-s bin starting at or after t_a - 10 min whose pooled ratio to the control mean is >= 2 for 3 consecutive bins)."""
    dates = pd.DatetimeIndex(dates)
    pos = {d: i for i, d in enumerate(dates)}
    event_dates = list(ev["date"].unique())
    secs = np.arange(lo, hi)
    ne = np.zeros(len(secs))
    nc = np.zeros(len(secs))
    for _, e in ev.iterrows():
        if e["date"] not in pos:
            continue
        k0 = int(np.ceil(e["t_a"] - T0))
        idx = k0 + secs
        if idx.min() < 0 or idx.max() >= N:
            continue
        ctrl = control_days(e["date"], dates, close_s, event_dates)
        ne += counts[pos[e["date"]], idx]
        nc += counts[ctrl][:, idx].mean(axis=0)     # control MEAN: a per-second median of sparse counts is 0
    onset_cum = None                                 # dropped: see PREREGISTRATION.md section 3 (amendment)
    nb = len(secs) // bin_s
    eb = ne[: nb * bin_s].reshape(nb, bin_s).sum(1)
    cb = nc[: nb * bin_s].reshape(nb, bin_s).sum(1)
    ratio = (eb + 1) / (cb + 1)
    tb = secs[: nb * bin_s : bin_s]
    onset_bins = next((int(tb[i]) for i in range(nb - 2) if (ratio[i:i + 3] >= 2).all()), None)
    return pd.DataFrame(dict(t_rel_s=tb, event=eb, control_median=cb, ratio=ratio)), onset_cum, onset_bins


def load_events(path, ticker=None):
    """Reads either the Layer 1 events.csv (accept_utc, items) or the census events.csv
    (accepted_utc, items, ticker). With ticker=..., keeps that ticker only."""
    ev = pd.read_csv(path)
    if "accepted_utc" in ev.columns and "accept_utc" not in ev.columns:
        ev = ev.rename(columns={"accepted_utc": "accept_utc"})
    if ticker is not None:
        ev = ev[ev["ticker"].str.upper() == ticker.upper()].reset_index(drop=True)
    # TODO A2-9: replace the bucket/is_earn/cls block below with W.classify_event per row, so `intraday`
    #            becomes a class; set ev["is_earnings"]; keep 'ticker' and 'accession' if present.
    #            Weekend acceptances still map to None (dropped).
    acc = pd.to_datetime(ev["accept_utc"], utc=True).dt.tz_convert("America/New_York")
    ev["accept_et"] = acc
    ev["date"] = acc.dt.tz_localize(None).dt.normalize()
    ev["t_a"] = acc.dt.hour * 3600 + acc.dt.minute * 60 + acc.dt.second + acc.dt.microsecond / 1e6
    wd = acc.dt.dayofweek < 5
    bucket = np.where(wd & (ev["t_a"] >= 16 * 3600) & (ev["t_a"] < 20 * 3600), "after_hours",
              np.where(wd & (ev["t_a"] >= 4 * 3600) & (ev["t_a"] < 9.5 * 3600), "pre_open", "other"))
    is_earn = ev["items"].astype(str).str.contains(r"(?:^|,)2\.02(?:,|$)")
    ev["cls"] = np.where(bucket == "after_hours", np.where(is_earn, "AH_earnings_2.02", "AH_other"), bucket)
    # TODO A4-4: if "form" in ev.columns: keep ev["form"].isin(forms) (new `forms` arg, default ("8-K",))
    return ev[ev["cls"] != "other"].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticker", default=None, help="with the census events.csv: keep one ticker")
    ap.add_argument("--measure", default="events", choices=["events", "prints", "shares"])
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--extra", action="store_true", help="A2-11: also run analyse_extra (anch_15m, day24, density)")
    ap.add_argument("--calendar", default="ext_MSFT.npz", help="npz whose dates are the full session calendar")
    a = ap.parse_args()

    z = np.load(a.counts)
    counts = z[a.measure].astype(np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    ev = load_events(a.events, a.ticker)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ, per = analyse(counts, dates, z["close_s"], ev, B=a.perms)
    summ.to_csv(out / f"summary_{a.measure}.csv", index=False)
    per.to_csv(out / f"per_event_{a.measure}.csv", index=False)
    meta = {}
    for cls in ev["cls"].unique():
        prof, onset_cum, onset_bins = profile(counts, dates, z["close_s"], ev[ev["cls"] == cls])
        prof.to_csv(out / f"profile_{cls}_{a.measure}.csv", index=False)
        meta[cls] = dict(onset_cumulative_excess_s=onset_cum, onset_ratio_ge2_3bins_s=onset_bins)
    (out / f"onset_{a.measure}.json").write_text(json.dumps(meta, indent=2))
    if a.extra:
        # TODO A2-11: calendar = pd.DatetimeIndex(pd.to_datetime(np.load(a.calendar)["dates"]))
        #   summ_x, per_x = analyse_extra(counts, dates, z["close_s"], ev, calendar, B=a.perms,
        #                                 density_counts=z["prints"].astype(np.float64))
        #   write summary_extra_{measure}.csv and per_event_extra_{measure}.csv; print summ_x
        raise NotImplementedError("A2-11")
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
