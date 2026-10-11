#!/usr/bin/env python3
"""
End-to-end Layer 1 run: ticks + 8-Ks -> diurnally adjusted durations -> Log-ACD
with event covariates -> tests -> placebo distribution -> offset scan.

Real data:
    df.to_parquet("ticks.parquet")        # in your notebook, the frame you already have
    python scripts/run_layer1.py --ticks ticks.parquet --ticker XYZ \
        --user-agent "Your Name your@email.com" --out results/

Offline dry run on synthetic data (use this first; it must finish cleanly):
    python scripts/run_layer1.py --simulate --out results_sim/

Pre-registered choices are the defaults: half-life 1800 s, exponential QMLE,
robust Wald test on gamma, placebo randomization p-value. Anything else the
script prints (half-life profile, Weibull fit, offset scan) is diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l1 import acd, diurnal, edgar, features, placebo, simulate, trades  # noqa: E402

KINDS = ("scheduled", "surprise")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_ticks(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".pkl", ".pickle"):
        return pd.read_pickle(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"unsupported file type {p.suffix}")


def session_closes(dur: pd.DataFrame) -> pd.Series:
    """Detect early-close sessions (13:00 ET) from where trading stops."""
    last = dur.groupby("date")["tod_prev"].max()
    return pd.Series(np.where(last < 13.5 * 3600, 13 * 3600, trades.RTH_CLOSE_S), index=last.index)


def fit_with(adj: pd.DataFrame, ev: pd.DataFrame, half_life_s: float, dist="exp", robust=True):
    Z = features.build_design(adj, ev, kinds=KINDS, half_life_s=half_life_s)
    a = pd.concat([adj, Z], axis=1)
    return acd.fit(acd.ACDData.from_frame(a, zcols=Z.columns), dist=dist, zcols=list(Z.columns),
                   robust_se=robust)


def gamma_of(res, k):
    return float(res.theta[res.names.index(f"gamma_z_{k}")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks")
    ap.add_argument("--ticker")
    ap.add_argument("--user-agent", help='"Your Name your@email.com" (SEC requires it)')
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="edgar_cache")
    ap.add_argument("--assume-tz", default="auto", choices=["auto", "utc", "eastern"])
    ap.add_argument("--half-life", type=float, default=1800.0, help="pre-registered decay half-life, seconds")
    ap.add_argument("--profile-half-lives", default="300,900,1800,3600,7200")
    ap.add_argument("--open-buffer", type=float, default=0.0, help="drop the first N seconds after 09:30")
    ap.add_argument("--collapse-window-ns", type=int, default=0)
    ap.add_argument("--control-days", type=int, default=0,
                    help="0 = estimate on all days; N = event days + N random non-event days per event day")
    ap.add_argument("--placebos", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    R: dict = {"args": vars(args)}

    # ---------------------------------------------------------------- 1. ticks
    if args.simulate:
        df, pages, truth = simulate.simulate_market(n_days=60, n_scheduled=10, n_surprise=10, seed=args.seed)
        R["truth"] = {"gamma": truth["gamma"], "theta": truth["theta"].tolist()}
        log("simulated", len(df), "rows")
    else:
        df = load_ticks(args.ticks)
        log("loaded", len(df), "rows")
        if "publisher_id" in df.columns:
            log("rows by publisher_id:", df["publisher_id"].value_counts().to_dict())
    dur = trades.prepare(df, open_buffer_s=args.open_buffer, window_ns=args.collapse_window_ns)
    days = pd.DatetimeIndex(sorted(pd.to_datetime(dur["date"].unique())))
    closes = session_closes(dur)
    log(f"{len(dur):,} durations over {len(days)} sessions {days.min().date()} .. {days.max().date()}")
    R["n_durations"], R["n_sessions"] = len(dur), len(days)
    R["first_day"], R["last_day"] = str(days.min().date()), str(days.max().date())

    # ---------------------------------------------------------------- 2. filings
    if not args.simulate:
        cik = edgar.cik_for_ticker(args.ticker, args.user_agent)
        pages = edgar.fetch_submissions(cik, args.user_agent, cache_dir=Path(args.cache) / cik)
        R["cik"] = cik
    fil = edgar.filings_frame(pages)
    diag = edgar.diagnose_acceptance_tz(fil["acceptanceDateTime"])
    log("acceptance-time diagnosis:\n" + diag.to_string(index=False))
    tz = args.assume_tz
    if tz == "auto":
        d = diag.set_index("assume")["in_edgar_window"]
        tz = d.idxmax()
        if abs(d["utc"] - d["eastern"]) < 0.01:
            log("WARNING: diagnosis cannot tell the conventions apart; defaulting to", tz,
                "- check a few filings by hand on the EDGAR index page (it shows ET)")
    R["acceptance_tz"], R["tz_diagnosis"] = tz, diag.to_dict("records")
    ev = edgar.build_event_table(fil, days, tz, closes=closes)
    ev.to_csv(out / "events.csv", index=False)
    counts = ev.groupby(["kind", "intraday"]).size().unstack(fill_value=0)
    log("in-sample 8-Ks by kind x intraday:\n" + counts.to_string())
    R["event_counts"] = {f"{k}|intraday={i}": int(v) for (k, i), v in counts.stack().items()}
    if ev["kind"].isin(KINDS).sum() == 0:
        log("no scheduled/surprise events in sample; stopping"); return

    # ---------------------------------------------------------------- 3. diurnal
    event_days = pd.DatetimeIndex(pd.to_datetime(ev["eff_date"].unique()))
    model = diurnal.fit_diurnal(dur, exclude_dates=event_days)
    adj = diurnal.adjust(dur, model)
    diurnal.diurnal_profile(dur).to_csv(out / "profile_raw.csv", index=False)
    diurnal.diurnal_profile(adj, col="x").to_csv(out / "profile_adjusted.csv", index=False)
    pr = diurnal.diurnal_profile(dur, bucket_minutes=30)
    R["raw_open_midday_ratio"] = float(pr["mean"].max() / pr["mean"].min())
    log(f"raw diurnal max/min (30-min buckets): {R['raw_open_midday_ratio']:.2f}")

    if args.control_days > 0:
        pool = days[~days.isin(event_days) & days.isin(pd.to_datetime(adj["date"].unique()))]
        k = min(len(pool), args.control_days * len(event_days))
        keep = event_days.union(pd.DatetimeIndex(rng.choice(pool, size=k, replace=False)))
        adj = adj[pd.to_datetime(adj["date"]).isin(keep)].reset_index(drop=True)
        log(f"estimation sample: {len(keep)} days, {len(adj):,} durations")

    # ---------------------------------------------------------------- 4. fits
    base = acd.fit(acd.ACDData.from_frame(adj), robust_se=False)
    full = fit_with(adj, ev, args.half_life)
    log("exponential QMLE, pre-registered half-life:\n" + full.table().to_string())
    wald = acd.wald_test(full, [f"gamma_z_{k}" for k in KINDS])
    lr = acd.lr_test(full, base, len(KINDS))
    lre = {k: acd.long_run_effect(full, f"gamma_z_{k}") for k in KINDS}
    lb = acd.ljung_box(acd.residuals(full, acd.ACDData.from_frame(
        pd.concat([adj, features.build_design(adj, ev, KINDS, args.half_life)], axis=1),
        zcols=[f"z_{k}" for k in KINDS])))
    R["exp_fit"] = full.table().reset_index().to_dict("records")
    R["wald_robust"], R["lr_naive"], R["long_run"] = wald, lr, lre
    R["ljung_box_resid"] = {"Q": lb["Q"], "p": lb["p"]}
    log(f"robust Wald p={wald['p']:.3g}  naive LR p={lr['p']:.3g}")
    for k, v in lre.items():
        log(f"long-run duration multiplier [{k}]: {v['multiplier']:.3f} (log-effect {v['effect']:.3f} +/- {v['se']:.3f})")

    wb = fit_with(adj, ev, args.half_life, dist="weibull")
    R["weibull_fit"] = wb.table().reset_index().to_dict("records")
    R["weibull_k"] = float(np.exp(wb.theta[-1]))
    log(f"Weibull shape k = {R['weibull_k']:.3f} (k<1: decreasing hazard / overdispersion)")

    prof = []
    for h in [float(s) for s in args.profile_half_lives.split(",")]:
        r = fit_with(adj, ev, h, robust=False)
        prof.append({"half_life_s": h, "loglik": r.loglik, **{f"gamma_{k}": gamma_of(r, k) for k in KINDS}})
    R["half_life_profile"] = prof
    log("half-life profile (descriptive, not a test):\n" + pd.DataFrame(prof).to_string(index=False))

    # ---------------------------------------------------------------- 5. placebo distribution
    obs = {k: gamma_of(full, k) for k in KINDS}
    draws = {k: [] for k in KINDS}
    for i in range(args.placebos):
        pe = placebo.placebo_events(ev[ev["kind"].isin(KINDS)], pd.to_datetime(adj["date"].unique()),
                                    rng=rng.integers(1 << 31))
        r = fit_with(adj, pe, args.half_life, robust=False)
        for k in KINDS:
            draws[k].append(gamma_of(r, k))
        if (i + 1) % 10 == 0:
            log(f"placebo {i + 1}/{args.placebos}")
    R["placebo"] = {}
    for k in KINDS:
        g = np.array(draws[k])
        if len(g):
            p = (1 + np.sum(np.abs(g) >= abs(obs[k]))) / (len(g) + 1)
            R["placebo"][k] = {"observed": obs[k], "placebo_mean": float(g.mean()), "placebo_sd": float(g.std()),
                               "randomization_p": float(p)}
            log(f"[{k}] gamma={obs[k]:.4f}  placebo mean={g.mean():.4f} sd={g.std():.4f}  randomization p={p:.3f}")
    pd.DataFrame(draws).to_csv(out / "placebo_draws.csv", index=False)

    # ---------------------------------------------------------------- 6. offset scan (intraday events only)
    intr = ev[ev["intraday"] & ev["kind"].isin(KINDS)]
    if len(intr):
        scan = []
        for delta in (-3600, -1800, -600, -120, 0, 120, 600, 1800):
            r = fit_with(adj, placebo.shift_events(intr, delta), args.half_life, robust=False)
            scan.append({"delta_s": delta, "loglik": r.loglik,
                         **{f"gamma_{k}": gamma_of(r, k) for k in KINDS if (intr["kind"] == k).any()}})
        R["offset_scan"] = scan
        log("offset scan, intraday events (effect should switch on at delta ~ 0, not before):\n"
            + pd.DataFrame(scan).to_string(index=False))

    (out / "results.json").write_text(json.dumps(R, indent=2, default=str))
    log("wrote", out / "results.json")


if __name__ == "__main__":
    main()
