"""
Synthetic data shaped exactly like your inputs, with KNOWN truth.

simulate_market() returns
  * a DataFrame in Databento `trades` schema (as DBNStore.to_df() gives it, or
    the raw int form), including the junk real data has: pre-market prints,
    off-exchange TRF prints, multi-fill sweeps sharing one ts_event;
  * EDGAR-style submissions pages (column-oriented dicts, like data.sec.gov);
  * the truth: diurnal shape, daily levels, ACD parameters, event effects.

Every stage of the pipeline is tested by running it on this and checking it
recovers the truth. That is the same logic as your checkpoints A/B/C, extended
to the preprocessing, where most real-data bugs actually live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .trades import RTH_CLOSE_S, RTH_OPEN_S, TZ


def u_shape(tod_s: np.ndarray, depth: float = 0.6) -> np.ndarray:
    """Mean-one diurnal factor for expected duration: SHORT at open/close, LONG
    midday (an inverted U in durations = a U in trade intensity).
    phi(u) = c * (1 - depth + 4*depth*u*(1-u)), u = session fraction, c makes the mean 1."""
    u = (np.asarray(tod_s, float) - RTH_OPEN_S) / (RTH_CLOSE_S - RTH_OPEN_S)
    u = np.clip(u, 0, 1)
    raw = (1 - depth) + 4 * depth * u * (1 - u)
    mean = (1 - depth) + 4 * depth / 6
    return raw / mean


def _fmt(ts_utc: pd.Timestamp, convention: str) -> str:
    t = ts_utc if convention == "utc" else ts_utc.tz_convert(TZ)
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def simulate_market(n_days: int = 40, start: str = "2025-03-03", seed: int = 0,
                    mean_dur_s: float = 6.0, theta=(0.0, 0.06, 0.92),
                    gamma: dict | None = None, half_life_s: float = 1800.0,
                    dist: str = "exp", weibull_k: float = 0.8,
                    n_scheduled: int = 6, n_surprise: int = 6, n_routine: int = 4,
                    dup_prob: float = 0.05, price_as_int: bool = False,
                    ts_as_datetime: bool = True, acceptance_convention: str = "utc",
                    holiday_index: int = 7, diurnal_depth: float = 0.6, level_sd: float = 0.08):
    rng = np.random.default_rng(seed)
    gamma = {"scheduled": -0.04, "surprise": -0.03} if gamma is None else gamma

    days = pd.bdate_range(start, periods=n_days + 1)
    holiday = days[holiday_index]
    days = days.delete(holiday_index)                 # a market holiday inside the sample

    # daily level: persistent log-AR(1) around mean_dur_s
    lv = np.empty(len(days)); a = 0.0
    for i in range(len(days)):
        a = 0.8 * a + rng.normal(0, level_sd) if level_sd > 0 else 0.0
        lv[i] = mean_dur_s * np.exp(a)
    level = pd.Series(lv, index=days)

    # ---- filings (acceptance times) and their true market-effective times
    filings, eff = [], []   # eff: (ns, kind)
    later_days = days[25:]  # leave the first 25 days event-free (trailing level warm-up + clean diurnal fit)

    def local(d, sec):
        # whole seconds: EDGAR acceptance times have 1 s resolution
        return (pd.Timestamp(d) + pd.Timedelta(seconds=int(sec))).tz_localize(TZ).tz_convert("UTC")

    sched_days = rng.choice(len(later_days), size=min(n_scheduled, len(later_days)), replace=False)
    for j, k in enumerate(sched_days):
        d = later_days[k]
        if j % 2 == 0:   # after the close the day before (16:05-16:15 ET)
            prev = days[days.get_loc(d) - 1]
            acc = local(prev, 16 * 3600 + 5 * 60 + rng.integers(0, 600))
        else:            # before the open the same day (07:00-08:00 ET)
            acc = local(d, 7 * 3600 + rng.integers(0, 3600))
        filings.append((acc, "2.02,9.01"))
        eff.append((local(d, RTH_OPEN_S).value, "scheduled"))
    surp_days = rng.choice(len(later_days), size=min(n_surprise, len(later_days)), replace=False)
    for k in surp_days:
        d = later_days[k]
        sec = rng.uniform(10.5 * 3600, 15.0 * 3600)
        acc = local(d, sec)
        filings.append((acc, "5.02"))
        eff.append((acc.value, "surprise"))
    for k in rng.choice(len(later_days), size=n_routine, replace=False):
        acc = local(later_days[k], rng.uniform(11 * 3600, 14 * 3600))
        filings.append((acc, "7.01,9.01"))
    # one filing accepted on the holiday evening -> must map to the next session's open
    filings.append((local(holiday, 18 * 3600), "8.01"))
    eff.append((local(days[holiday_index], RTH_OPEN_S).value, "surprise"))
    # one 8-K/A that must be ignored
    filings.append((local(later_days[0], 12 * 3600), "AMEND"))

    ev_ns = {k: np.array(sorted(v for v, kk in eff if kk == k), dtype=np.int64) for k in gamma}

    omega, alpha, beta = theta
    lam = 1.0
    if dist == "weibull":
        from scipy.special import gamma as G
        lam = 1.0 / G(1 + 1 / weibull_k)

    rows_t, rows_px, rows_sz, rows_side, rows_pub = [], [], [], [], []
    inv_len = 1.0 / (RTH_CLOSE_S - RTH_OPEN_S)
    inv_mean = 1.0 / ((1 - diurnal_depth) + 4 * diurnal_depth / 6)
    px = 100.0
    l0 = 0.0
    for di, d in enumerate(days):
        # pre-market junk
        for s in np.sort(rng.uniform(8 * 3600, RTH_OPEN_S - 1, 20)):
            rows_t.append(local(d, s).value); rows_px.append(px); rows_sz.append(100); rows_side.append("N"); rows_pub.append(2)
        open_ns = local(d, RTH_OPEN_S).value
        t = open_ns + int(rng.integers(1, 1000)) * 1000
        close_ns = local(d, RTH_CLOSE_S).value
        # events that can touch this day
        day_ev = {k: v[(v > open_ns - 40 * half_life_s * 1e9) & (v < close_ns)] for k, v in ev_ns.items()}
        has_ev = any(len(v) for v in day_ev.values())
        B = 200_000
        eps_blk = rng.exponential(1.0, B) if dist == "exp" else lam * rng.weibull(weibull_k, B)
        u_blk = rng.random(B); nf_blk = rng.integers(1, 4, B)
        lvl = level.iloc[di]
        l = l0; logx_prev = None; j = 0
        while True:
            tod = RTH_OPEN_S + (t - open_ns) / 1e9
            if logx_prev is not None:
                zz = 0.0
                if has_ev:
                    for k, e in day_ev.items():
                        e = e[e <= t]
                        if len(e):
                            zz += gamma[k] * np.exp2(-(t - e) / (half_life_s * 1e9)).sum()
                l = omega + alpha * logx_prev + (beta - alpha) * l + zz
            xt = np.exp(l) * eps_blk[j]
            logx_prev = np.log(xt)
            uu = min(max((tod - RTH_OPEN_S) * inv_len, 0.0), 1.0)
            dur = lvl * ((1 - diurnal_depth) + 4 * diurnal_depth * uu * (1 - uu)) * inv_mean * xt
            nfill = 1 + (nf_blk[j] if u_blk[j] < dup_prob else 0)
            j += 1
            for f in range(nfill):
                rows_t.append(t); rows_px.append(px); rows_sz.append(100 * (1 + f))
                rows_side.append("A" if (j + f) % 2 else "B"); rows_pub.append(2)
            px = max(1.0, px + 0.01 * (1 if eps_blk[j] > 0.69 else -1))
            t = t + max(int(dur * 1e9), 1)
            if t >= close_ns:
                break
        # off-exchange (TRF) prints scattered through the day
        for s in rng.uniform(RTH_OPEN_S, RTH_CLOSE_S, 200):
            rows_t.append(local(d, s).value); rows_px.append(px); rows_sz.append(100); rows_side.append("N"); rows_pub.append(83)

    ts = np.array(rows_t, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    price = np.array(rows_px)[order]
    df = pd.DataFrame({
        "ts_event": pd.to_datetime(ts, utc=True) if ts_as_datetime else ts,
        "rtype": 0, "publisher_id": np.array(rows_pub)[order], "instrument_id": 1, "action": "T",
        "side": np.array(rows_side)[order], "depth": 0,
        "price": np.round(price * 1e9).astype(np.int64) if price_as_int else price,
        "size": np.array(rows_sz)[order], "flags": 0, "ts_in_delta": 15000,
        "sequence": np.arange(len(ts)), "symbol": "SYNTH",
    })
    recv = ts + 20_000
    df.index = pd.Index(pd.to_datetime(recv, utc=True) if ts_as_datetime else recv, name="ts_recv")

    # ---- EDGAR-style pages
    fl = sorted(filings, key=lambda r: r[0], reverse=True)
    page = {"accessionNumber": [], "form": [], "filingDate": [], "reportDate": [],
            "acceptanceDateTime": [], "items": [], "primaryDocument": []}
    for i, (acc, items) in enumerate(fl):
        page["accessionNumber"].append(f"0000000000-25-{i:06d}")
        page["form"].append("8-K/A" if items == "AMEND" else "8-K")
        page["filingDate"].append(acc.tz_convert(TZ).strftime("%Y-%m-%d"))
        page["reportDate"].append(acc.tz_convert(TZ).strftime("%Y-%m-%d"))
        page["acceptanceDateTime"].append(_fmt(acc, acceptance_convention))
        page["items"].append("5.02" if items == "AMEND" else items)
        page["primaryDocument"].append("doc.htm")
    # split into a "recent" page and one older page, like the real API
    half = len(fl) // 2
    pages = [{k: v[:half] for k, v in page.items()}, {k: v[half:] for k, v in page.items()}]

    truth = {"theta": np.array(theta), "gamma": gamma, "half_life_s": half_life_s, "level": level,
             "days": days, "holiday": holiday, "event_ns": ev_ns, "dist": dist, "weibull_k": weibull_k,
             "diurnal_depth": diurnal_depth}
    return df, pages, truth
