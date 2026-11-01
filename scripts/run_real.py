#!/usr/bin/env python3
"""Layer 1 on a real multi-day durations panel (output of scripts/load_dbn.py), memory-lean.

Pre-registered core (unchanged): 1800 s half-life, exponential QMLE, scheduled/surprise
classes from item codes, placebo randomization p-value. Additions, all labelled:
  * day-clustered SEs next to trade-level robust SEs
  * estimation on event days + N random non-event control days (run_layer1 --control-days)
  * placebos warm-started from the real fit, run in parallel, kept off ALL 8-K days
  * a model-free check: event-day vs placebo-day mean adjusted duration in the first 30 min

    python scripts/run_real.py --durations msft_dur.parquet --closes-csv msft_dur.diag.csv \
        --pages-json msft_8k_pages.json --out results_msft
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l1 import acd, diurnal, edgar, features, placebo  # noqa: E402

KINDS = ("scheduled", "surprise")
ET = "America/New_York"
_G = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def data_for(adj, ev, h):
    Z = features.build_design(adj, ev, kinds=KINDS, half_life_s=h)
    return acd.ACDData(x=adj["x"].to_numpy(), Z=Z.to_numpy(), day_start=_G["start"],
                       l0=_G["l0"]), list(Z.columns)


def fit_ev(adj, ev, h, theta0=None, robust=True, dist="exp"):
    d, cols = data_for(adj, ev, h)
    return acd.fit(d, dist=dist, zcols=cols, theta0=theta0, robust_se=robust)


def _placebo_job(args):
    seed, h, theta0, mode = args
    adj, ev_k, days, avoid = _G["adj"], _G["ev_k"], _G["days"], _G["avoid"]
    pe = placebo.placebo_events(ev_k, days, rng=seed, avoid_days=avoid)
    if mode == "controls_only":  # refit on control days only: no real-event days anywhere in the sample
        A = _G["adj_ctrl"]
        Zp = features.build_design(A, pe, kinds=KINDS, half_life_s=h).to_numpy()
        d = acd.ACDData(x=A["x"].to_numpy(), Z=Zp, day_start=_G["start_ctrl"], l0=_G["l0_ctrl"])
        r = acd.fit(d, theta0=theta0, robust_se=False)
        return [float(r.theta[3]), float(r.theta[4]), r.loglik]
    if mode == "plain":          # original design: real events dropped from the refit
        r = fit_ev(adj, pe, h, theta0=theta0, robust=False)
        return [float(r.theta[r.names.index(f"gamma_z_{k}")]) for k in KINDS] + [r.loglik]
    # "nuisance": keep the REAL event covariates in the model and add the placebo ones.
    # Otherwise the real event days' fast opens sit in the residual of the placebo fit,
    # pull the baseline down and bias every placebo gamma upward (seen: mean +0.08).
    Zr = features.build_design(adj, _G["ev_real"], kinds=KINDS, half_life_s=h).to_numpy()
    Zp = features.build_design(adj, pe, kinds=KINDS, half_life_s=h).to_numpy()
    d = acd.ACDData(x=adj["x"].to_numpy(), Z=np.c_[Zr, Zp], day_start=_G["start"], l0=_G["l0"])
    r = acd.fit(d, theta0=theta0, robust_se=False)
    return [float(r.theta[5]), float(r.theta[6]), r.loglik]


def open_window_stat(adj, ev, minutes=30):
    """Mean adjusted duration in the first `minutes` after each event (model-free)."""
    out = []
    t = adj["ts_prev"].to_numpy()
    x = adj["x"].to_numpy()
    for tau in ev["eff_ns"].to_numpy():
        lo, hi = np.searchsorted(t, tau), np.searchsorted(t, tau + minutes * 60 * 10**9)
        out.append(np.mean(np.log(x[lo:hi])) if hi > lo else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", required=True)
    ap.add_argument("--closes-csv", required=True)
    ap.add_argument("--pages-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--half-life", type=float, default=1800.0)
    ap.add_argument("--control-days", type=int, default=5)
    ap.add_argument("--placebos", type=int, default=200)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--placebo-mode", choices=["plain", "nuisance", "controls_only"], default="nuisance")
    ap.add_argument("--skip-extras", action="store_true", help="skip Weibull, half-life profile")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    R = {"args": vars(a)}

    dur = pd.read_parquet(a.durations, columns=["date", "ts_prev", "tod_prev", "duration"])
    dur["date"] = pd.to_datetime(dur["date"])
    days = pd.DatetimeIndex(sorted(dur["date"].unique()))
    closes = pd.read_csv(a.closes_csv, parse_dates=["date"]).set_index("date")["close_s"]
    fil = edgar.filings_frame(json.loads(Path(a.pages_json).read_text()))
    diag = edgar.diagnose_acceptance_tz(fil["acceptanceDateTime"]).set_index("assume")
    tz = diag["in_edgar_window"].idxmax()
    # TODO A1-5: the diagnose step picks 'utc' vs 'eastern' but cannot see the +4/5 h shift. Before
    # build_event_table: acc = edgar.parse_acceptance(fil["acceptanceDateTime"], tz);
    # acc, rep = edgar.fix_acceptance_tz(acc, fil["filingDate"]); log(rep.to_string());
    # fil["acceptanceDateTime"] = acc.dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"); tz = "utc"
    ev = edgar.build_event_table(fil, days, tz, closes=closes)
    ev.to_csv(out / "events.csv", index=False)
    R["acceptance_tz"] = tz
    R["event_counts"] = ev.groupby(["kind", "intraday"]).size().rename("n").reset_index().to_dict("records")
    log(f"{len(dur):,} durations, {len(days)} sessions; acceptance tz={tz}")
    log("events by kind x intraday:\n" + ev.groupby(["kind", "intraday"]).size().to_string())

    ev_days = pd.DatetimeIndex(pd.to_datetime(ev["eff_date"].unique()))
    m = diurnal.fit_diurnal(dur, exclude_dates=ev_days)
    adj = diurnal.adjust(dur, m)[["date", "ts_prev", "tod_prev", "x"]]
    del dur
    prof = diurnal.diurnal_profile(adj, bucket_minutes=30, col="x")
    R["adjusted_profile_max_over_min"] = float(prof["mean"].max() / prof["mean"].min())
    log(f"adjusted 30-min profile max/min = {R['adjusted_profile_max_over_min']:.3f}")

    # ---- estimation sample: event days + control days (pre-registered option)
    have = pd.DatetimeIndex(adj["date"].unique())
    ev_in = ev_days[ev_days.isin(have)]
    pool = have[~have.isin(ev_days)]
    ctrl = pd.DatetimeIndex(rng.choice(pool, size=min(len(pool), a.control_days * len(ev_in)), replace=False))
    keep = ev_in.union(ctrl)
    adj = adj[adj["date"].isin(keep)].reset_index(drop=True)
    R["n_event_days_in_sample"], R["n_control_days"], R["n_durations"] = len(ev_in), len(ctrl), len(adj)
    log(f"estimation sample: {len(ev_in)} event days + {len(ctrl)} control days = {len(adj):,} durations")
    day = adj["date"].to_numpy()
    _G["start"] = np.r_[True, day[1:] != day[:-1]]
    _G["l0"] = float(np.log(adj["x"].mean()))

    # ---- fits
    d0 = acd.ACDData(x=adj["x"].to_numpy(), Z=np.zeros((len(adj), 0)), day_start=_G["start"], l0=_G["l0"])
    base = acd.fit(d0, robust_se=False)
    full = fit_ev(adj, ev, a.half_life, theta0=np.r_[base.theta, 0, 0])
    tab = full.table()
    tab["se_cluster_day"] = full.se_cluster
    tab["p_cluster_day"] = 2 * stats.norm.sf(np.abs(full.theta / full.se_cluster))
    log("exponential QMLE (pre-registered h):\n" + tab.to_string())
    R["exp_fit"] = tab.reset_index().to_dict("records")
    R["converged"] = full.converged
    names = [f"gamma_z_{k}" for k in KINDS]
    R["wald_robust"] = acd.wald_test(full, names)
    R["wald_cluster_day"] = acd.wald_test(full, names, which="cluster")
    R["lr_naive"] = acd.lr_test(full, base, 2)
    R["long_run"] = {k: {w: acd.long_run_effect(full, f"gamma_z_{k}", which=w) for w in ("robust", "cluster")}
                     for k in KINDS}
    for k in KINDS:
        lr = R["long_run"][k]["cluster"]
        log(f"[{k}] long-run log effect {lr['effect']:+.3f} (cluster se {lr['se']:.3f}) -> duration x{lr['multiplier']:.3f}")
    d_full, _ = data_for(adj, ev, a.half_life)
    eps = acd.residuals(full, d_full)
    lb = acd.ljung_box(eps)
    R["resid"] = {"mean": float(eps.mean()), "var": float(eps.var()), "ljung_box_Q": lb["Q"], "ljung_box_p": lb["p"],
                  "acf1_5": lb["acf"][:5].tolist()}
    log(f"residual eps: mean {eps.mean():.3f} var {eps.var():.2f}; Ljung-Box(20) Q={lb['Q']:.0f}")

    if not a.skip_extras:
        wb = fit_ev(adj, ev, a.half_life, theta0=np.r_[full.theta, 0.0], dist="weibull")
        R["weibull_k"] = float(np.exp(wb.theta[-1]))
        R["weibull_gamma"] = {k: float(wb.theta[wb.names.index(f"gamma_z_{k}")]) for k in KINDS}
        R["weibull_minus_exp_loglik_per_obs"] = (wb.loglik - full.loglik) / full.n
        log(f"Weibull k={R['weibull_k']:.3f}; gain over exp {R['weibull_minus_exp_loglik_per_obs']:.3f} nats/obs")

        prof_h = []
        for h in (300, 900, 1800, 3600, 7200):
            r = fit_ev(adj, ev, h, theta0=full.theta, robust=False)
            prof_h.append({"half_life_s": h, "loglik_minus_base": r.loglik - base.loglik,
                           **{f"gamma_{k}": float(r.theta[r.names.index(f'gamma_z_{k}')]) for k in KINDS}})
        R["half_life_profile"] = prof_h
        log("half-life profile:\n" + pd.DataFrame(prof_h).to_string(index=False))

    # ---- placebos
    ev_k = ev[ev["kind"].isin(KINDS)].reset_index(drop=True)
    _G.update(adj=adj, ev_k=ev_k, days=keep, avoid=ev_days, ev_real=ev)
    if a.placebo_mode == "controls_only":
        A = adj[~adj["date"].isin(ev_days)].reset_index(drop=True)
        dd = A["date"].to_numpy()
        _G.update(adj_ctrl=A, start_ctrl=np.r_[True, dd[1:] != dd[:-1]], l0_ctrl=float(np.log(A["x"].mean())))
    theta0 = np.r_[full.theta, 0, 0] if a.placebo_mode == "nuisance" else np.r_[base.theta, 0, 0]
    jobs = [(int(s), a.half_life, theta0, a.placebo_mode) for s in rng.integers(1 << 31, size=a.placebos)]
    R["placebo_mode"] = a.placebo_mode
    if a.placebos:
        t0 = time.time()
        with Pool(a.procs) as p:
            draws = np.array(p.map(_placebo_job, jobs, chunksize=4))
        log(f"{a.placebos} placebos ({a.placebo_mode}) in {time.time() - t0:.0f}s")
        pd.DataFrame(draws, columns=[f"gamma_{k}" for k in KINDS] + ["loglik"]).to_csv(out / "placebo_draws.csv", index=False)
        R["placebo"] = {}
        for j, k in enumerate(KINDS):
            g_obs = float(full.theta[full.names.index(f"gamma_z_{k}")])
            g = draws[:, j]
            R["placebo"][k] = {"observed": g_obs, "placebo_mean": float(g.mean()), "placebo_sd": float(g.std()),
                               "p_two_sided": float((1 + np.sum(np.abs(g) >= abs(g_obs))) / (len(g) + 1)),
                               "p_one_sided_negative": float((1 + np.sum(g <= g_obs)) / (len(g) + 1)),
                               "z_vs_placebo": float((g_obs - g.mean()) / g.std()),
                               "mean_within_1sd_of_zero": bool(abs(g.mean()) <= g.std())}
            log(f"[{k}] gamma={g_obs:+.4f}  placebo mean={g.mean():+.4f} sd={g.std():.4f}  "
                f"p(two-sided)={R['placebo'][k]['p_two_sided']:.3f}  p(gamma<=obs)={R['placebo'][k]['p_one_sided_negative']:.3f}")

    # ---- model-free: first 30 min after the open, event days vs placebo days
    real = open_window_stat(adj, ev_k)
    R["open30"] = {}
    for k in KINDS:
        msk = ev_k["kind"].to_numpy() == k
        r_mean = float(np.nanmean(real[msk]))
        # placebo_events sorts by time, so re-derive kind per placebo draw from its own frame
        draws_k = []
        for s in rng.integers(1 << 31, size=200):
            pe = placebo.placebo_events(ev_k, keep, rng=int(s), avoid_days=ev_days)
            v = open_window_stat(adj, pe)
            draws_k.append(np.nanmean(v[pe["kind"].to_numpy() == k]))
        draws_k = np.array(draws_k)
        R["open30"][k] = {"event_mean_log_x": r_mean, "placebo_mean": float(draws_k.mean()),
                          "placebo_sd": float(draws_k.std()),
                          "p_one_sided_faster": float((1 + np.sum(draws_k <= r_mean)) / (len(draws_k) + 1)),
                          "per_event": [None if np.isnan(v) else float(v) for v in real[msk]]}
        log(f"[{k}] first-30-min mean log(x): events {r_mean:+.3f} vs placebo {draws_k.mean():+.3f} "
            f"(sd {draws_k.std():.3f}), p(faster)={R['open30'][k]['p_one_sided_faster']:.3f}")

    (out / "results.json").write_text(json.dumps(R, indent=2, default=str))
    log("wrote", out / "results.json")


if __name__ == "__main__":
    main()
