"""Pre-registered Stage 1 verdict, computed by code from a normalized result.

Nothing in here is LLM-judged. The five criteria are the README's:
  1. scheduled and surprise effects both negative
  2. joint robust Wald p < alpha
  3. placebo randomization p < alpha for each class, with >= min_placebos placebos
  4. placebo mean within one placebo SD of zero
  5. (intraday events) log-likelihood over event-time offsets peaks within +-2 minutes
"""
from __future__ import annotations


def _c(cid, text, status, value=None, detail=""):
    return {"id": cid, "text": text, "status": status, "value": value, "detail": detail}


def evaluate_stage1(res: dict, pr: dict) -> dict:
    cr = pr["criteria"]
    kinds, alpha = cr["kinds"], cr["alpha"]
    est, pl = res.get("estimates", {}), res.get("placebo", {})
    out = []

    # 1 ----------------------------------------------------------------- signs
    g = {k: est.get(k, {}).get("gamma") for k in kinds}
    if any(v is None for v in g.values()):
        out.append(_c(1, "both effects negative", "unknown", g, "gamma missing"))
    else:
        out.append(_c(1, "both effects negative", "pass" if all(v < 0 for v in g.values()) else "fail", g))

    # 2 ------------------------------------------------------- joint robust Wald
    w = res.get("wald_robust")
    if not w:
        out.append(_c(2, f"joint robust Wald p < {alpha}", "unknown", None, "no wald_robust in results"))
    else:
        out.append(_c(2, f"joint robust Wald p < {alpha}", "pass" if w["p"] < alpha else "fail", w["p"]))

    # 3 ----------------------------------------------------- placebo randomization
    n_pl = res.get("n_placebos") or 0
    if not pl:
        out.append(_c(3, f"placebo p < {alpha} per class, >= {cr['min_placebos']} placebos", "unknown", None, "no placebo block"))
    else:
        ps = {k: pl[k]["p"] for k in kinds if k in pl}
        floor = 1.0 / (n_pl + 1) if n_pl else 1.0
        if n_pl < cr["min_placebos"]:
            out.append(_c(3, f"placebo p < {alpha} per class, >= {cr['min_placebos']} placebos", "fail", ps,
                          f"only {n_pl} placebos (pre-registered minimum {cr['min_placebos']}); smallest attainable p = {floor:.4f}"))
        else:
            ok = len(ps) == len(kinds) and all(p < alpha for p in ps.values())
            out.append(_c(3, f"placebo p < {alpha} per class, >= {cr['min_placebos']} placebos",
                          "pass" if ok else "fail", ps, f"n_placebos={n_pl}"))

    # 4 ------------------------------------------------------- placebo centred at 0
    if not pl:
        out.append(_c(4, "placebo mean within 1 placebo SD of 0", "unknown"))
    else:
        d = {k: {"mean": pl[k]["mean"], "sd": pl[k]["sd"]} for k in kinds if k in pl}
        ok = len(d) == len(kinds) and all(abs(v["mean"]) <= cr["placebo_mean_within_sd"] * v["sd"] for v in d.values())
        out.append(_c(4, "placebo mean within 1 placebo SD of 0", "pass" if ok else "fail", d))

    # 5 ----------------------------------------------------------- offset-scan peak
    scan = res.get("offset_scan")
    if not scan:
        out.append(_c(5, "offset-scan peak within +-2 min", "na", None,
                      "no offset scan in these results (scripts/run_real.py does not run one; scripts/run_layer1.py does)"))
    else:
        best = max(scan, key=lambda r: r["loglik"])
        tol = cr["offset_peak_tolerance_s"]
        out.append(_c(5, "offset-scan peak within +-2 min", "pass" if abs(best["delta_s"]) <= tol else "fail",
                      {"peak_delta_s": best["delta_s"], "grid": [r["delta_s"] for r in scan]},
                      "grid resolution is coarse: only -120, 0, +120 s fall inside the tolerance"))

    statuses = [c["status"] for c in out]
    overall = "fail" if "fail" in statuses else ("incomplete" if any(s in ("unknown", "na") for s in statuses) else "pass")
    return {"criteria": out, "overall": overall}
