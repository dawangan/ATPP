"""Registered experiments: the ONLY way a researcher gets numbers out of the data.

Each experiment is a named, deterministic function of (params, outdir). Researchers cannot run
arbitrary analyses; they can only request registered ones, and every request is ledgered. New
analyses must be added here by a human (this is the forking-paths control).

Experiments run in a fresh subprocess (`python -m autoresearch.worker`) for isolation, timeouts
and fork-safety under the orchestrator's thread pool.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import prereg as P
from .core import ROOT, code_hash, dump_json, file_sha256, git_state, load_json, sha256_json
from .ledger import Ledger

REGISTRY: dict = {}
KINDS = ("scheduled", "surprise")


def experiment(name: str, family: str, data_kind: str):
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "family": family, "data_kind": data_kind}
        return fn
    return deco


def resolve_path(p: str) -> Path | None:
    """Resolve against the repo root, then ATPP_DATA_ROOTS (os.pathsep-separated): this is how the
    harness finds results/parquet files that live on the user's own machine."""
    cand = Path(p).expanduser()
    if cand.is_absolute():
        return cand if cand.exists() else None
    roots = [ROOT] + [Path(r).expanduser() for r in os.environ.get("ATPP_DATA_ROOTS", "").split(os.pathsep) if r]
    for r in roots:
        if (r / cand).exists():
            return r / cand
    return None


# ------------------------------------------------------------------ normalization
def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_layer1(R: dict, source: str, data_kind: str) -> dict:
    """Map the results.json of scripts/run_layer1.py OR scripts/run_real.py onto one schema."""
    args = R.get("args", {})
    fit = {r.get("index"): r for r in R.get("exp_fit", [])}
    est = {}
    for k in KINDS:
        row = fit.get(f"gamma_z_{k}", {})
        lr = R.get("long_run", {}).get(k, {})
        lr_rob = lr["robust"] if "robust" in lr else lr
        lr_cl = lr["cluster"] if "cluster" in lr else R.get("long_run_cluster_day", {}).get(k, {})
        est[k] = {"gamma": row.get("estimate"), "se_robust": row.get("se_robust"),
                  "se_cluster": _first(row, "se_cluster_day"),
                  "long_run": {"effect": lr_rob.get("effect"), "se_robust": lr_rob.get("se"),
                               "se_cluster": lr_cl.get("se"), "effect_cluster": lr_cl.get("effect")}}
    n_pl = int(args.get("placebos") or 0)
    pl = {}
    for k, v in R.get("placebo", {}).items():
        pl[k] = {"observed": v.get("observed"), "mean": v.get("placebo_mean"), "sd": v.get("placebo_sd"),
                 "p": _first(v, "p_two_sided", "randomization_p"), "p_one_sided_negative": v.get("p_one_sided_negative")}
    ec = R.get("event_counts", {})
    counts: dict = {}
    if isinstance(ec, dict):
        for key, n in ec.items():
            counts[key.split("|")[0]] = counts.get(key.split("|")[0], 0) + int(n)
    else:
        for rec in ec:
            counts[rec["kind"]] = counts.get(rec["kind"], 0) + int(rec["n"])
    spec = {k: args[k] for k in ("half_life", "control_days", "open_buffer", "collapse_window_ns", "placebo_mode") if k in args}
    res = {
        "source": source, "data_kind": data_kind, "spec": spec,
        "estimates": est, "wald_robust": R.get("wald_robust"), "wald_cluster": R.get("wald_cluster_day"),
        "placebo": pl, "n_placebos": n_pl, "offset_scan": R.get("offset_scan"), "event_counts": counts,
        "meta": {"n_sessions": R.get("n_sessions"), "n_durations": R.get("n_durations"),
                 "first_day": R.get("first_day"), "last_day": R.get("last_day"),
                 "acceptance_tz": R.get("acceptance_tz"), "converged": R.get("converged"),
                 "truth": R.get("truth"), "seed": args.get("seed")},
    }
    res["tests"] = (
        ([{"name": "wald_joint_robust", "p": res["wald_robust"]["p"]}] if res["wald_robust"] else []) +
        [{"name": f"placebo_p_{k}", "p": v["p"]} for k, v in pl.items() if v["p"] is not None])
    return res


# ------------------------------------------------------------------ experiments
@experiment("stage1_synthetic", family="H1_stage1", data_kind="synthetic")
def stage1_synthetic(p: dict, out: Path) -> dict:
    """scripts/run_layer1.py --simulate, end to end. Validates the machinery against known truth;
    says nothing about H1 on real data."""
    cmd = [sys.executable, str(ROOT / "scripts" / "run_layer1.py"), "--simulate", "--out", str(out / "layer1"),
           "--placebos", str(int(p.get("placebos", 200))), "--seed", str(int(p.get("seed", 0))),
           "--half-life", str(float(p.get("half_life", 1800.0)))]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (out / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode:
        raise RuntimeError(f"run_layer1.py failed ({r.returncode}); see {out/'stdout.txt'}")
    return normalize_layer1(load_json(out / "layer1" / "results.json"), "scripts/run_layer1.py --simulate", "synthetic")


@experiment("ingest_results", family="H1_stage1", data_kind="real")
def ingest_results(p: dict, out: Path) -> dict:
    """Vet a results.json that run_real.py / run_layer1.py already produced (e.g. on the local
    machine, where the Databento parquet lives). The harness never re-fits real data itself."""
    path = resolve_path(p["path"])
    if path is None:
        raise FileNotFoundError(f"{p['path']} not found under repo root or ATPP_DATA_ROOTS")
    res = normalize_layer1(load_json(path), f"ingested:{p['path']}", p.get("data_kind", "real"))
    res["meta"]["input_file"] = str(path)
    res["meta"]["input_sha256"] = file_sha256(path)
    return res


def _one_sim(a):
    seed, gamma, n_days, half_life, level_sd = a
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from l1 import acd, diurnal, edgar, features, simulate, trades
    df, pages, truth = simulate.simulate_market(n_days=n_days, n_scheduled=10, n_surprise=10, seed=seed, gamma=gamma,
                                             level_sd=level_sd)
    dur = trades.prepare(df)
    days = pd.DatetimeIndex(sorted(dur["date"].unique()))
    ev = edgar.build_event_table(edgar.filings_frame(pages), days, "utc")
    adj = diurnal.adjust(dur, diurnal.fit_diurnal(dur, exclude_dates=ev["eff_date"].unique()))
    Z = features.build_design(adj, ev, half_life_s=half_life)
    r = acd.fit(acd.ACDData.from_frame(pd.concat([adj, Z], axis=1), zcols=Z.columns), zcols=list(Z.columns))
    names = [f"gamma_z_{k}" for k in KINDS]
    row = {"seed": seed, "converged": bool(r.converged),
           "p_robust": acd.wald_test(r, names)["p"], "p_cluster": acd.wald_test(r, names, which="cluster")["p"]}
    beta_true = truth["theta"][2]
    for k in KINDS:
        rb, cl = acd.long_run_effect(r, f"gamma_z_{k}", "robust"), acd.long_run_effect(r, f"gamma_z_{k}", "cluster")
        row.update({f"lr_{k}": rb["effect"], f"se_rob_{k}": rb["se"], f"se_cl_{k}": cl["se"],
                    f"truth_{k}": gamma[k] / (1 - beta_true)})
    return row


@experiment("mc_calibration", family="method", data_kind="synthetic")
def mc_calibration(p: dict, out: Path) -> dict:
    """Monte Carlo of the full pipeline on simulated worlds with KNOWN effects.
    gamma = 0  -> empirical size of the joint Wald tests (should equal the nominal level)
    gamma != 0 -> bias and CI coverage of the long-run effect, plus power."""
    import numpy as np
    from scipy import stats
    gamma = p.get("gamma", {"scheduled": 0.0, "surprise": 0.0})
    M, nom = int(p.get("n_sims", 16)), float(p.get("nominal", 0.05))
    procs = int(p.get("procs", min(4, os.cpu_count() or 1)))
    jobs = [(int(p.get("seed", 1000)) + i, gamma, int(p.get("n_days", 60)), float(p.get("half_life", 1800.0)),
             float(p.get("level_sd", 0.08))) for i in range(M)]
    with ProcessPoolExecutor(procs) as ex:
        rows = list(ex.map(_one_sim, jobs))
    dump_json(out / "mc_rows.json", rows)
    rej_r = int(sum(r["p_robust"] < nom for r in rows))
    rej_c = int(sum(r["p_cluster"] < nom for r in rows))
    null = all(v == 0 for v in gamma.values())
    mc = {"n_sims": M, "gamma": gamma, "nominal": nom, "is_null": null, "level_sd": float(p.get("level_sd", 0.08)),
          "reject_robust": rej_r, "reject_cluster": rej_c,
          "rate_robust": rej_r / M, "rate_cluster": rej_c / M,
          "binom_p_robust_above_nominal": float(stats.binomtest(rej_r, M, nom, alternative="greater").pvalue),
          "binom_p_cluster_above_nominal": float(stats.binomtest(rej_c, M, nom, alternative="greater").pvalue),
          "n_nonconverged": int(sum(not r["converged"] for r in rows)), "per_kind": {}}
    for k in KINDS:
        lr, tr = np.array([r[f"lr_{k}"] for r in rows]), rows[0][f"truth_{k}"]
        cov = lambda se: float(np.mean([abs(r[f"lr_{k}"] - tr) <= 1.96 * r[se] for r in rows]))
        mc["per_kind"][k] = {"truth_long_run": tr, "mean_est": float(lr.mean()), "bias": float(lr.mean() - tr),
                             "sd_across_sims": float(lr.std(ddof=1)), "rmse": float(np.sqrt(np.mean((lr - tr) ** 2))),
                             "mean_se_robust": float(np.mean([r[f"se_rob_{k}"] for r in rows])),
                             "mean_se_cluster": float(np.mean([r[f"se_cl_{k}"] for r in rows])),
                             "coverage95_robust": cov(f"se_rob_{k}"), "coverage95_cluster": cov(f"se_cl_{k}")}
    return {"source": "autoresearch.mc_calibration", "data_kind": "synthetic", "spec": {}, "tests": [],
            "estimates": {}, "placebo": {}, "n_placebos": 0, "mc": mc,
            "meta": {"seed": p.get("seed", 1000), "last_day": None}}


# ------------------------------------------------------------------ runner
class ExperimentRefused(RuntimeError):
    pass


def run_experiment(name: str, params: dict, outdir: Path, ledger: Ledger, task_id: str,
                   pr: dict, timeout: int = 3600) -> dict:
    """Run one registered experiment in a subprocess; ledger it; return its normalized result.
    Raises ExperimentRefused for unregistered experiments, missing data or holdout violations."""
    if name not in REGISTRY:
        raise ExperimentRefused(f"'{name}' is not a registered experiment (registered: {sorted(REGISTRY)})")
    reg = REGISTRY[name]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dump_json(outdir / "params.json", params)
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run([sys.executable, "-m", "autoresearch.worker", name, str(outdir / "params.json"), str(outdir)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        msg = (outdir / "error.txt").read_text() if (outdir / "error.txt").exists() else proc.stderr[-2000:]
        ledger.append("failure", task_id=task_id, experiment=name, params=params, error=msg[-1500:])
        if "FileNotFoundError" in msg:
            raise ExperimentRefused(msg.strip().splitlines()[-1])
        raise RuntimeError(f"experiment {name} failed:\n{msg[-1500:]}")
    res = load_json(outdir / "result.json")
    if res["data_kind"] != "synthetic":
        try:
            P.assert_explore(res["meta"].get("last_day"), pr)
        except P.HoldoutViolation as e:
            ledger.append("refusal", task_id=task_id, experiment=name, reason=str(e))
            raise ExperimentRefused(str(e))
    spec = {**{k: v for k, v in params.items() if k in pr["spec_keys"]}, **res.get("spec", {})}
    res["spec"], res["drift"] = spec, P.drift(spec, pr)
    res["spec_hash"] = sha256_json({"family": reg["family"], "experiment": name, "spec": spec,
                                    "input": res["meta"].get("input_sha256")})
    run = ledger.append("experiment", task_id=task_id, experiment=name, family=reg["family"],
                        data_kind=res["data_kind"], params=params, spec=spec, spec_hash=res["spec_hash"],
                        seed=params.get("seed"), code_hash=code_hash(), git=git_state(),
                        input_sha256=res["meta"].get("input_sha256"), tests=res.get("tests", []),
                        drift=res["drift"], mc=res.get("mc"), outdir=str(outdir))
    res["run"] = {"ledger_seq": run["seq"], "outdir": str(outdir), "code_hash": run["code_hash"], "git": run["git"]}
    dump_json(outdir / "result.json", res)
    return res
