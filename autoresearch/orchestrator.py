"""Orchestrator: plan -> research (parallel) -> review (parallel, adversarial) -> vet -> finalize.

Separation of powers:
  researchers  interpret registered experiments, audit code, search literature. They cannot run
               unregistered analyses and cannot touch the holdout.
  reviewers    independent sessions that see artifacts but never the researcher's reasoning, each
               assigned a different attack lens; plus deterministic checks that need no LLM.
  vet()        pure code. Status is a function of check results and reviewer verdicts, not of anyone's
               opinion about the finding, and VERIFIED is impossible without a real LLM review.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import prereg as P
from . import schemas as S
from .checks import run_checks
from .core import PROMPTS, ROOT, STATE, dump_json, git_state, load_json
from .experiments import REGISTRY, ExperimentRefused, resolve_path, run_experiment
from .ledger import Ledger
from .llm import BudgetExceeded
from .verdict import evaluate_stage1

FIX_HINTS = {
    "size_calibration": "Run q-method-null with n_sims >= 40 and decide, BEFORE any real-data gate is declared, whether the "
                        "pre-registered robust Wald must be replaced by the day-clustered Wald (dated, ledgered amendment).",
    "se_validity": "Report the day-clustered Wald next to the robust Wald in every result; treat disagreement as a failed gate.",
    "power_events": "Extend the panel (more tickers / event windows) before interpreting this class of result.",
    "spec_drift": "Re-run at the frozen spec, or keep the run explicitly exploratory and exclude it from the verdict.",
    "multiplicity": "Apply the Bonferroni/Holm correction for the number of specs in the ledger, or stop adding specs.",
    "replication": "Investigate why an independent replicate disagrees before relying on this estimate.",
    "convergence": "Fix the optimizer failure (start values, bounds) and re-fit.",
}


def read(name: str) -> str:
    return (PROMPTS / name).read_text()


def render(template: str, **kw) -> str:
    out = template.replace("<<CONTEXT>>", read("_context.md"))
    for k, v in kw.items():
        out = out.replace(f"<<{k.upper()}>>", v if isinstance(v, str) else json.dumps(v, indent=1, default=str))
    return out


def _clip(obj, n=14000) -> str:
    s = json.dumps(obj, indent=1, default=str)
    return s if len(s) <= n else s[:n] + "\n... [truncated; read the artifact on disk]"


def load_tasks(path=None, status="active") -> list[dict]:
    tasks = load_json(path or STATE / "questions.json")["tasks"]
    return [t for t in tasks if t.get("status", "active") == status]


# ---------------------------------------------------------------- deterministic finding template
def template_finding(task, res, verdict, rep) -> dict:
    caveats, ev = [], []
    if res.get("mc"):
        mc = res["mc"]
        if mc["is_null"]:
            claim = (f"Under a TRUE NULL ({mc['n_sims']} simulated worlds) the joint Wald test rejected "
                     f"{mc['reject_robust']}/{mc['n_sims']} times with trade-level robust SEs and "
                     f"{mc['reject_cluster']}/{mc['n_sims']} with day-clustered SEs at nominal {mc['nominal']:.0%} "
                     f"(binomial p vs nominal: robust {mc['binom_p_robust_above_nominal']:.2g}, "
                     f"cluster {mc['binom_p_cluster_above_nominal']:.2g}).")
        else:
            pk = mc["per_kind"]
            claim = ("With a known effect, long-run bias is " + ", ".join(f"{k} {v['bias']:+.3f}" for k, v in pk.items())
                     + "; 95% CI coverage (robust / cluster SE): "
                     + ", ".join(f"{k} {v['coverage95_robust']:.0%} / {v['coverage95_cluster']:.0%}" for k, v in pk.items())
                     + f"; joint Wald power at {mc['nominal']:.0%}: robust {mc['rate_robust']:.0%}, cluster {mc['rate_cluster']:.0%}.")
        ev.append({"ref": "mc", "note": "see result.mc"})
        return {"claim": claim, "kind": "method", "claims_effect": False, "confidence": "medium",
                "evidence": ev, "caveats": ["synthetic worlds from l1/simulate.py: validates the machinery, not H1",
                                            f"{mc['n_sims']} simulations: coarse"], "followups": []}
    c = {x["id"]: x["status"] for x in verdict["criteria"]} if verdict else {}
    claim = (f"Pre-registered Stage 1 verdict: {verdict['overall'].upper()} "
             f"(criteria: " + ", ".join(f"{i}={s}" for i, s in c.items()) + ").") if verdict else "No verdict available."
    synthetic = res["data_kind"] == "synthetic"
    claims_effect = bool(verdict and verdict["overall"] == "pass" and not res.get("drift") and not synthetic)
    if synthetic:
        caveats.append("synthetic data: validates the pipeline against known truth; says nothing about H1")
    return {"claim": claim, "kind": "method" if synthetic else ("effect" if claims_effect else
            ("null" if verdict and verdict["overall"] == "fail" else "inconclusive")),
            "claims_effect": claims_effect, "confidence": "low", "evidence": [{"ref": "verdict", "note": "computed by code"}],
            "caveats": caveats, "followups": []}


# ---------------------------------------------------------------- repro runner (code-audit lane)
def run_repro(code: str, workdir: Path, idx: int, timeout: int = 120) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    f = workdir / f"repro_{idx}.py"
    f.write_text(code)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    try:
        p = subprocess.run([sys.executable, str(f)], cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"verified": False, "why": f"repro timed out after {timeout}s"}
    if p.returncode == 0:
        return {"verified": False, "why": "repro exited 0: the bug is not demonstrated"}
    if "AssertionError" in p.stderr:
        return {"verified": True, "why": "AssertionError raised: bug demonstrated", "stderr": p.stderr[-600:]}
    return {"verified": False, "why": "repro crashed for a reason other than AssertionError", "stderr": p.stderr[-600:]}


# ---------------------------------------------------------------- vetting (pure code)
def vet(finding, checks, reviews, llm_ran: bool, final: bool) -> dict:
    claims = bool(finding.get("claims_effect"))
    fails = [c for c in checks if c["status"] == "fail"]
    cons = [c for c in checks if c["status"] == "concern"]
    blocks = [r for r in reviews if r["verdict"] == "block"]
    rconcerns = [r for r in reviews if r["verdict"] == "concern"]
    reasons = []
    if claims and fails:
        return {"status": "REJECTED", "reasons": [f"check {c['id']} failed: {c['detail']}" for c in fails]}
    if blocks:
        why = [f"{r['lens']} blocked: " + "; ".join(i["claim"] for i in r["issues"] if i["severity"] == "blocker") for r in blocks]
        if not final:
            return {"status": "NEEDS_REVISION", "reasons": why}
        unanimous = len(blocks) == len(reviews)
        return {"status": "REJECTED" if unanimous else "NEEDS_HUMAN",
                "reasons": why + ([] if unanimous else ["reviewers disagree after revision; a human must adjudicate"])}
    if not llm_ran:
        return {"status": "DETERMINISTIC_ONLY", "reasons": ["no LLM review ran; deterministic checks only"]}
    reasons += [f"check {c['id']}: {c['detail']}" for c in fails + cons]
    reasons += [f"{r['lens']} concern: " + "; ".join(i["claim"] for i in r["issues"]) for r in rconcerns]
    return {"status": "VERIFIED_WITH_CAVEATS" if reasons else "VERIFIED", "reasons": reasons}


# ---------------------------------------------------------------- orchestrator
class Orchestrator:
    def __init__(self, executor, ledger: Ledger, pr: dict, out_root: Path, max_parallel=3, max_rounds=1):
        self.ex, self.ledger, self.pr = executor, ledger, pr
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(out_root) / self.run_id
        self.max_parallel, self.max_rounds = max_parallel, max_rounds
        self.mc_procs = max(1, (os.cpu_count() or 1))

    # -- plan ------------------------------------------------------------------
    def plan(self, tasks, only=None):
        run, blocked = [], []
        for t in sorted(tasks, key=lambda t: t.get("priority", 5)):
            if only and t["id"] not in only:
                continue
            lane = t["lane"]
            if lane == "experiment":
                if t.get("experiment") not in REGISTRY:
                    blocked.append((t, f"unregistered experiment {t.get('experiment')!r}"))
                    continue
                path = t.get("params", {}).get("path")
                if path and resolve_path(path) is None:
                    blocked.append((t, f"missing data: {path} (not under the repo or ATPP_DATA_ROOTS)"))
                    continue
            elif not self.ex.runs_llm:
                blocked.append((t, f"{lane} lane needs an LLM executor (re-run with --executor claude)"))
                continue
            run.append(t)
        return run, blocked

    # -- LLM role wrappers -----------------------------------------------------
    def _llm(self, role, prompt, schema, tools="readonly"):
        try:
            return self.ex.call(role, prompt, schema, tools=tools)
        except BudgetExceeded as e:
            return {"ok": False, "data": None, "cost": 0.0, "error": str(e), "budget": True}

    def _write_finding(self, task, res, verdict, rep, tdir):
        lane = task["lane"]
        if lane == "experiment":
            base = template_finding(task, res, verdict, rep)
            if not self.ex.runs_llm:
                return base, None
            r = self._llm("researcher", render(read("researcher_experiment.md"), task=task, result=_clip(res),
                                               verdict=verdict or "n/a (method experiment)", replicate=_clip(rep) if rep else "none"),
                          S.FINDING)
            if r["ok"]:
                f = r["data"]
                if verdict and verdict["overall"] != "pass":
                    f["claims_effect"] = False            # code outranks the researcher: a failed gate cannot be claimed as an effect
                if res["data_kind"] == "synthetic":
                    f["claims_effect"] = False
                return f, r
            base["caveats"].append(f"LLM researcher failed ({r['error']}); deterministic template used")
            return base, r
        prompt = render(read("researcher_literature.md" if lane == "literature" else "researcher_code_audit.md"), task=task)
        r = self._llm("researcher", prompt, S.FINDING, tools="web" if lane == "literature" else "readonly")
        return (r["data"] if r["ok"] else None), r

    def _verify_defects(self, finding, tdir):
        """Harness-side verification: a defect only counts if its repro demonstrably fails."""
        keep, dropped = [], []
        for i, d in enumerate(finding.get("defects", [])):
            v = run_repro(d["repro_code"], tdir / "repros", i)
            d["repro_result"] = v
            (keep if v["verified"] else dropped).append(d)
        finding["defects"] = keep
        # Code decides the claim: the LLM's prose may mention defects no repro substantiated.
        finding["researcher_claim"] = finding.get("claim")
        listed = "; ".join(f"{d['file']}:{d['line']} [{d['severity']}, {d.get('repro_kind')}]" for d in keep) or "none"
        finding["claim"] = (f"{len(keep)} defect(s) demonstrated by a harness-run repro: {listed}. {len(dropped)} further "
                            "claimed defect(s) were NOT substantiated by a repro (that is not the same as false; see caveats).")
        if dropped:
            finding.setdefault("caveats", []).append(
                f"{len(dropped)} claimed defect(s) discarded: repro did not demonstrate the bug "
                f"({'; '.join(d['file'] + ':' + str(d['line']) + ' - ' + d['repro_result']['why'] for d in dropped)})")
        beh = sum(d.get("repro_kind") == "behavioral" for d in keep)
        return [{"id": "repro_verified", "status": "pass" if beh else "concern",
                 "detail": f"{beh} behavioral + {len(keep) - beh} structural repro(s) hold, {len(dropped)} discarded. "
                           "A structural repro shows a code fact (e.g. a call is missing), not that any result changes."}]

    def _citation_form(self, finding):
        cites = finding.get("citations", [])
        if not cites:
            return [{"id": "citation_form", "status": "concern",
                     "detail": "no citation with a fetched URL and verbatim quote: the claim rests on search snippets alone"}]
        bad = [c for c in cites if not c["url"].startswith("http") or len(c["quote"]) < 20]
        if bad:
            return [{"id": "citation_form", "status": "concern", "detail": f"{len(bad)} citation(s) lack a URL or a substantive verbatim quote"}]
        return [{"id": "citation_form", "status": "pass" if finding.get("citations") else "na", "detail": ""}]

    def _review(self, task, finding, res, checks, tdir):
        lenses = S.LENSES[task["lane"]] if self.ex.runs_llm else []
        paths = {"task_dir": str(tdir), "result_json": str(tdir / "exp" / "result.json") if res else None,
                 "ledger": str(self.ledger.path), "repo_code": ["l1/", "scripts/"]}

        def one(lens):
            prompt = render(read(f"reviewer_{lens}.md"), task=task, finding=finding, result=_clip(res) if res else "n/a",
                            checks=checks, paths=paths)
            r = self._llm(f"reviewer_{lens}", prompt, S.REVIEW, tools="web" if lens == "citation" else "readonly")
            if r["ok"]:
                r["data"]["lens"] = lens
            return lens, r
        with ThreadPoolExecutor(max(1, len(lenses))) as pool:
            out = list(pool.map(one, lenses))
        reviews, cost, failed = [], 0.0, []
        for lens, r in out:
            cost += r["cost"]
            (reviews if r["ok"] else failed).append(r["data"] if r["ok"] else {"lens": lens, "error": r["error"]})
        return reviews, failed, cost

    # -- one task, end to end -------------------------------------------------
    def process(self, task) -> dict:
        tid, lane = task["id"], task["lane"]
        tdir = self.dir / "tasks" / tid
        tdir.mkdir(parents=True, exist_ok=True)
        b = {"task": task, "status": None, "reasons": [], "finding": None, "checks": [], "reviews": [],
             "reviewer_failures": [], "verdict": None, "result": None, "replicate": None, "cost": 0.0, "rounds": 0}
        res = rep = verdict = None
        try:
            if lane == "experiment":
                params = dict(task.get("params", {}))
                if task["experiment"] == "mc_calibration":
                    params.setdefault("procs", getattr(self, "mc_procs", 1))
                res = run_experiment(task["experiment"], params, tdir / "exp", self.ledger, tid, self.pr)
                if REGISTRY[task["experiment"]]["family"] == "H1_stage1":
                    verdict = evaluate_stage1(res, self.pr)
                if task.get("replicate") and res["data_kind"] == "synthetic":
                    rep = run_experiment(task["experiment"], {**params, "seed": int(params.get("seed", 0)) + 1000},
                                         tdir / "exp_rep", self.ledger, tid + "#rep", self.pr)
            finding, r = self._write_finding(task, res, verdict, rep, tdir)
            b["cost"] += (r or {}).get("cost", 0.0)
            if finding is None:
                b["status"], b["reasons"] = "RESEARCHER_FAILED", [(r or {}).get("error", "no finding produced")]
                return b
            ctx = {"pr": self.pr, "ledger": self.ledger, "replicate": rep}
            checks = run_checks(res, ctx) if res else []
            if lane == "code_audit":
                checks += self._verify_defects(finding, tdir)
            if lane == "literature":
                checks += self._citation_form(finding)

            final = False
            while True:
                reviews, failed, cost = self._review(task, finding, res, checks, tdir)
                b["cost"] += cost
                llm_ran = self.ex.runs_llm and len(reviews) > 0
                final = b["rounds"] >= self.max_rounds
                v = vet(finding, checks, reviews, llm_ran, final)
                if v["status"] != "NEEDS_REVISION":
                    break
                b["rounds"] += 1
                issues = [i for r_ in reviews if r_["verdict"] == "block" for i in r_["issues"]]
                rv = self._llm("researcher", render(read("researcher_revise.md"), task=task, finding=finding, issues=issues),
                               S.FINDING, tools="web" if lane == "literature" else "readonly")
                b["cost"] += rv["cost"]
                if not rv["ok"]:
                    v = vet(finding, checks, reviews, llm_ran, True)
                    break
                if res and res["data_kind"] == "synthetic" or (verdict and verdict["overall"] != "pass"):
                    rv["data"]["claims_effect"] = False
                finding = rv["data"]
                if lane == "code_audit":
                    checks = [c for c in checks if c["id"] != "repro_verified"] + self._verify_defects(finding, tdir)
            if failed:
                v["reasons"].append(f"{len(failed)} reviewer(s) failed to run: " + ", ".join(f['lens'] for f in failed))
            b.update(status=v["status"], reasons=v["reasons"], finding=finding, checks=checks, reviews=reviews,
                     reviewer_failures=failed, verdict=verdict, result=res, replicate=rep)
        except ExperimentRefused as e:
            b["status"], b["reasons"] = "BLOCKED", [str(e)]
        except Exception as e:                                           # noqa: BLE001
            b["status"], b["reasons"] = "ERROR", [f"{type(e).__name__}: {e}"]
        dump_json(tdir / "bundle.json", b)
        self.ledger.append("vet", task_id=tid, status=b["status"], reasons=b["reasons"], rounds=b["rounds"],
                           reviewers=[r["lens"] for r in b["reviews"]], cost_usd=round(b["cost"], 4),
                           checks={c["id"]: c["status"] for c in b["checks"]})
        return b

    # -- next steps ---------------------------------------------------------
    def next_steps(self, bundles, blocked):
        steps = []
        for b in bundles:
            tid = b["task"]["id"]
            for r in b["reviews"]:
                for i in r["issues"]:
                    if i["severity"] in ("major", "blocker"):
                        steps.append({"title": i["suggested_fix"], "why": f"{r['lens']} reviewer on {tid}: {i['claim']}", "source": "reviewer", "priority": 1})
            for c in b["checks"]:
                if c["status"] in ("fail", "concern") and c["id"] in FIX_HINTS:
                    steps.append({"title": FIX_HINTS[c["id"]], "why": f"{tid}: {c['detail']}", "source": "check", "priority": 1 if c["status"] == "fail" else 2})
            if b["verdict"]:
                for c in b["verdict"]["criteria"]:
                    if c["status"] in ("unknown", "na", "fail"):
                        steps.append({"title": f"Pre-registered criterion {c['id']} ({c['text']}) is {c['status']}", "why": f"{tid}: {c['detail'] or c['value']}", "source": "prereg", "priority": 2})
            for d in (b["finding"] or {}).get("defects", []):
                steps.append({"title": f"Fix {d['file']}:{d['line']} ({d['severity']})", "why": d["why"], "source": "audit", "priority": 1})
            for f in (b["finding"] or {}).get("followups", []):
                steps.append({"title": f["question"], "why": f["why"], "source": "researcher", "priority": 3, "lane": f["lane"], "experiment": f.get("experiment")})
        for t, why in blocked:
            steps.append({"title": f"Unblock {t['id']}: {t.get('unblock') or why}", "why": why, "source": "plan", "priority": 2})
        seen, out = set(), []
        for s in sorted(steps, key=lambda s: s["priority"]):
            k = " ".join(s["title"].lower().split())[:120]
            if k not in seen:
                seen.add(k)
                out.append(s)
        return out

    def _propose(self, steps):
        path = STATE / "proposed_tasks.json"
        cur = load_json(path) if path.exists() else {"note": "Researcher-proposed follow-ups. A human promotes one to questions.json to make it runnable.", "tasks": []}
        have = {t["question"] for t in cur["tasks"]}
        for s in steps:
            if s["source"] == "researcher" and s["title"] not in have:
                cur["tasks"].append({"id": f"p-{self.run_id}-{len(cur['tasks'])}", "status": "proposed", "lane": s.get("lane", "experiment"),
                                     "experiment": s.get("experiment"), "question": s["title"], "why": s["why"], "proposed_in": self.run_id})
        dump_json(path, cur)

    # -- full cycle ----------------------------------------------------------
    def run(self, tasks, only=None) -> dict:
        from .report import render_markdown
        self.dir.mkdir(parents=True, exist_ok=True)
        runnable, blocked = self.plan(tasks, only)
        self.ledger.append("run_start", run_id=self.run_id, executor=self.ex.name, tasks=[t["id"] for t in runnable],
                           blocked=[t["id"] for t, _ in blocked], git=git_state(), prereg_hash=P.prereg_hash(self.pr))
        # Wave 1: Monte Carlo calibration, so wave-2 checks always see current size/coverage evidence (no race).
        wave1 = [t for t in runnable if t.get("experiment") == "mc_calibration"]
        wave2 = [t for t in runnable if t not in wave1]
        self.mc_procs = max(1, (os.cpu_count() or 1) // max(1, min(self.max_parallel, len(wave1))))
        bundles = []
        for wave in (wave1, wave2):
            with ThreadPoolExecutor(self.max_parallel) as pool:
                bundles += list(pool.map(self.process, wave))
        steps = self.next_steps(bundles, blocked)
        summary, ranked = None, None
        if self.ex.runs_llm and bundles:
            cyc = [{"task": b["task"]["id"], "question": b["task"]["question"], "status": b["status"], "reasons": b["reasons"],
                    "claim": (b["finding"] or {}).get("claim"), "verdict": (b["verdict"] or {}).get("overall")} for b in bundles]
            r = self._llm("finalizer", render(read("finalizer.md"), cycle={"findings": cyc, "mechanical_next_steps": steps[:20]}), S.FINAL)
            if r["ok"]:
                summary, ranked = r["data"]["summary"], r["data"]["ranked_next_steps"]
        ok, bad = self.ledger.verify()
        report = {"run_id": self.run_id, "created": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()), "executor": self.ex.name,
                  "llm_review_ran": bool(self.ex.runs_llm), "git": git_state(), "prereg_hash": P.prereg_hash(self.pr),
                  "ledger": {"entries": len(self.ledger.entries()), "chain_ok": ok, "first_bad_seq": bad,
                             "h1_specs_tried": len(self.ledger.spec_trials("H1_stage1"))},
                  "cost_usd": round(sum(b["cost"] for b in bundles), 4), "summary": summary, "llm_next_steps": ranked,
                  "findings": [{k: b[k] for k in ("task", "status", "reasons", "finding", "checks", "reviews", "reviewer_failures", "verdict", "rounds")}
                               | {"result": b["result"]} for b in bundles],
                  "blocked": [{"task": t, "why": why} for t, why in blocked], "next_steps": steps}
        dump_json(self.dir / "report.json", report)
        (self.dir / "REPORT.md").write_text(render_markdown(report))
        self._propose(steps)
        self.ledger.append("run_end", run_id=self.run_id, statuses={b["task"]["id"]: b["status"] for b in bundles},
                           cost_usd=report["cost_usd"])
        return report
