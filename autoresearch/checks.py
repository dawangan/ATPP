"""Deterministic adversarial checks. These need no LLM and cannot be talked out of a verdict.

Each returns {id, status, detail} with status in pass | concern | fail | na.
`fail` blocks a *positive* claim (claims_effect=True); for null / method / defect findings a failed
check is attached as a caveat instead, because weak evidence does not invalidate "we found nothing".
"""
from __future__ import annotations

import math

from .core import code_hash
from .experiments import KINDS


def _r(cid, status, detail, **ev):
    return {"id": cid, "status": status, "detail": detail, **({"evidence": ev} if ev else {})}


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def spec_drift(res, ctx):
    d = res.get("drift") or []
    if d:
        return _r("spec_drift", "concern", "run overrides the frozen spec -> exploratory, ineligible for the "
                  "pre-registered verdict: " + "; ".join(d))
    return _r("spec_drift", "pass", "no frozen-spec field overridden")


def holdout(res, ctx):
    if res["data_kind"] == "synthetic":
        return _r("holdout", "na", "synthetic data")
    last = res["meta"].get("last_day")
    if last is None:
        return _r("holdout", "concern", "results carry no last_day; cannot confirm the holdout is untouched")
    return _r("holdout", "pass", f"last_day={last} <= explore_end={ctx['pr']['explore_end']} (enforced at ingest)")


def convergence(res, ctx):
    if res["meta"].get("converged") is False:
        return _r("convergence", "fail", "optimizer reported non-convergence for the main fit")
    return _r("convergence", "pass" if res["meta"].get("converged") else "na", "")


def se_validity(res, ctx):
    """The pre-registered test uses trade-level robust SEs, which the repo's own Monte Carlo says are
    ~3x too small when a day-level shock hits every trade of a day."""
    h = ctx["pr"]["harness"]
    wr, wc, alpha = res.get("wald_robust"), res.get("wald_cluster"), ctx["pr"]["criteria"]["alpha"]
    if not wr or not wc:
        return _r("se_validity", "na", "need both robust and day-clustered Wald results")
    ratios = {}
    for k in KINDS:
        lr = res["estimates"][k]["long_run"]
        if lr.get("se_robust") and lr.get("se_cluster"):
            ratios[k] = lr["se_cluster"] / lr["se_robust"]
    worst = max(ratios.values()) if ratios else None
    if wr["p"] < alpha <= wc["p"]:
        return _r("se_validity", "concern", f"joint Wald passes under robust SEs (p={wr['p']:.2g}) but NOT under "
                  f"day-clustered SEs (p={wc['p']:.2g}); cluster/robust SE ratio up to {worst:.1f}" if worst else "",
                  p_robust=wr["p"], p_cluster=wc["p"], se_ratio=ratios)
    if worst and worst > h["se_inflation_concern"]:
        return _r("se_validity", "concern", f"day-clustered SEs are {worst:.1f}x the trade-level robust SEs; the "
                  "pre-registered test is anti-conservative", se_ratio=ratios)
    return _r("se_validity", "pass", "robust and day-clustered inference agree", se_ratio=ratios)


def size_calibration(res, ctx):
    """Does the PRE-REGISTERED test hold its nominal size on data with no effect? Evidence comes from the
    latest mc_calibration (gamma=0) in the ledger for the CURRENT code."""
    h = ctx["pr"]["harness"]
    want = code_hash()
    best = None
    for e in ctx["ledger"].entries("experiment"):
        mc = e.get("mc")
        if (e.get("experiment") == "mc_calibration" and mc and mc.get("is_null") and e.get("code_hash") == want
                and mc.get("level_sd") == h["calibration_level_sd"]):
            best = e
    if best is None:
        return _r("size_calibration", "concern", "no null Monte Carlo (gamma=0, day-level shocks level_sd="
                  f"{h['calibration_level_sd']}) on file for the current code: run task 'q-method-null' before trusting any p-value from this pipeline")
    mc = best["mc"]
    if mc["binom_p_robust_above_nominal"] < h["size_fail_binom_p"]:
        return _r("size_calibration", "fail",
                  f"under a TRUE NULL the robust joint Wald rejected {mc['reject_robust']}/{mc['n_sims']} times at "
                  f"nominal {mc['nominal']:.0%} (binomial p={mc['binom_p_robust_above_nominal']:.2g}); day-clustered: "
                  f"{mc['reject_cluster']}/{mc['n_sims']}. Under THIS simulated DGP (day-level shocks, level_sd={mc['level_sd']}) the "
                  "pre-registered Wald criterion is not a valid test; whether real data has comparable day-level dependence is empirical.",
                  ledger_seq=best["seq"], **{k: mc[k] for k in ("reject_robust", "reject_cluster", "n_sims", "nominal")})
    return _r("size_calibration", "pass", f"robust Wald size {mc['reject_robust']}/{mc['n_sims']} at nominal "
              f"{mc['nominal']:.0%}", ledger_seq=best["seq"])


def power_events(res, ctx):
    if res["data_kind"] == "synthetic" or not res.get("event_counts"):
        return _r("power_events", "na", "")
    mn = ctx["pr"]["harness"]["min_events_per_class"]
    low = {k: res["event_counts"].get(k, 0) for k in KINDS if res["event_counts"].get(k, 0) < mn}
    if low:
        return _r("power_events", "concern", f"effect is identified by the number of event windows, not trades: "
                  f"{low} < {mn}. A null here is not evidence of absence; a hit is fragile.", counts=res["event_counts"])
    return _r("power_events", "pass", f"event windows per class: {res['event_counts']}")


def truth_recovery(res, ctx):
    t = res["meta"].get("truth")
    if not t or res["data_kind"] != "synthetic" or "gamma" not in t:
        return _r("truth_recovery", "na", "")
    beta = t["theta"][2]
    bad, rows = [], {}
    for k in KINDS:
        truth = t["gamma"][k] / (1 - beta)
        lr = res["estimates"][k]["long_run"]
        se = lr.get("se_cluster") or lr.get("se_robust")
        if lr.get("effect") is None or not se:
            continue
        z = (lr["effect"] - truth) / se
        rows[k] = {"truth": truth, "est": lr["effect"], "z_vs_truth": z}
        if abs(z) > 3:
            bad.append(f"{k}: est {lr['effect']:+.3f} vs truth {truth:+.3f} (z={z:+.1f})")
    if bad:
        return _r("truth_recovery", "concern", "long-run effect misses the simulated truth by >3 SE: " + "; ".join(bad), **rows)
    return _r("truth_recovery", "pass", "long-run effects within 3 SE of simulated truth", **rows)


def replication(res, ctx):
    rep = ctx.get("replicate")
    if rep is None:
        return _r("replication", "na", "no independent replicate available for this data")
    a, b = res["estimates"], rep["estimates"]
    flips = [k for k in KINDS if a[k]["gamma"] is not None and b[k]["gamma"] is not None and (a[k]["gamma"] < 0) != (b[k]["gamma"] < 0)]
    if flips:
        return _r("replication", "concern", f"sign of gamma flips on re-run with a new seed for: {flips}")
    return _r("replication", "pass", "signs agree on an independent replicate (new seed)")


def multiplicity(res, ctx):
    if res["data_kind"] == "synthetic":
        return _r("multiplicity", "na", "synthetic data")
    n = len(ctx["ledger"].spec_trials("H1_stage1")) or 1
    ps = [t["p"] for t in res.get("tests", []) if t.get("p") is not None]
    if not ps:
        return _r("multiplicity", "na", "no p-values")
    binding, alpha = max(ps), ctx["pr"]["criteria"]["alpha"]
    if n > 1 and binding < alpha <= binding * n:
        return _r("multiplicity", "concern", f"{n} distinct specs have been looked at for H1_stage1; the binding p="
                  f"{binding:.3g} clears alpha={alpha} only without a Bonferroni correction (alpha/{n}={alpha/n:.4f})", n_specs=n)
    return _r("multiplicity", "pass", f"{n} spec(s) tried for H1_stage1; binding p={binding:.3g}", n_specs=n)


def mc_sanity(res, ctx):
    mc = res.get("mc")
    if not mc:
        return _r("mc_sanity", "na", "")
    if mc["n_sims"] < 12:
        return _r("mc_sanity", "concern", f"only {mc['n_sims']} simulations: size estimates are very coarse")
    if mc["n_nonconverged"]:
        return _r("mc_sanity", "concern", f"{mc['n_nonconverged']} of {mc['n_sims']} fits did not converge")
    return _r("mc_sanity", "pass", f"{mc['n_sims']} converged simulations")


CHECKS = [spec_drift, holdout, convergence, se_validity, size_calibration, power_events,
          truth_recovery, replication, multiplicity, mc_sanity]


def run_checks(res: dict, ctx: dict) -> list[dict]:
    out = []
    for fn in CHECKS:
        try:
            out.append(fn(res, ctx))
        except Exception as e:                      # a broken check must be visible, never silent
            out.append(_r(fn.__name__, "concern", f"check crashed: {type(e).__name__}: {e}"))
    return out
