"""CLI:  python -m autoresearch {plan,run,ledger-verify,ledger-summary} ..."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from . import prereg as P
from .core import PKG
from .ledger import Ledger
from .llm import ClaudeCLI, NullExecutor
from .orchestrator import Orchestrator, load_tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autoresearch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        s = sub.add_parser(name)
        s.add_argument("--tasks", help="comma-separated task ids (default: all active)")
        s.add_argument("--tasks-file", default=None)
        s.add_argument("--executor", choices=["none", "claude"], default="none",
                       help="none = deterministic checks only (free); claude = headless `claude -p` agents")
        if name == "run":
            s.add_argument("--parallel", type=int, default=3, help="tasks processed concurrently")
            s.add_argument("--rounds", type=int, default=1, help="max revision rounds after reviewer blocks")
            s.add_argument("--max-usd", type=float, default=10.0, help="total LLM budget for the run")
            s.add_argument("--model", default=None, help="researcher/finalizer model")
            s.add_argument("--reviewer-model", default=None, help="reviewer model (use a different one for decorrelated errors)")
            s.add_argument("--effort", default=None)
            s.add_argument("--out", default=str(PKG / "runs"))
    sub.add_parser("ledger-verify")
    sub.add_parser("ledger-summary")
    a = ap.parse_args(argv)

    led = Ledger()
    if a.cmd == "ledger-verify":
        ok, bad = led.verify()
        print("ledger chain OK" if ok else f"LEDGER TAMPERED at seq {bad}")
        return 0 if ok else 2
    if a.cmd == "ledger-summary":
        es = led.entries()
        print(f"{len(es)} entries; kinds: {dict(Counter(e['kind'] for e in es))}")
        print(f"distinct real-data H1 specs tried: {len(led.spec_trials('H1_stage1'))}")
        return 0

    pr = P.load()
    ex = ClaudeCLI(model=getattr(a, "model", None), reviewer_model=getattr(a, "reviewer_model", None),
                   effort=getattr(a, "effort", None), max_total_usd=getattr(a, "max_usd", 10.0)) \
        if a.executor == "claude" else NullExecutor()
    orch = Orchestrator(ex, led, pr, Path(getattr(a, "out", PKG / "runs")),
                        max_parallel=getattr(a, "parallel", 3), max_rounds=getattr(a, "rounds", 1))
    tasks = load_tasks(a.tasks_file)
    only = a.tasks.split(",") if a.tasks else None
    if a.cmd == "plan":
        run, blocked = orch.plan(tasks, only)
        print("RUNNABLE:", *[f"  {t['id']:<22} [{t['lane']}] {t['question'][:90]}" for t in run], sep="\n")
        print("BLOCKED:", *[f"  {t['id']:<22} {why}" for t, why in blocked], sep="\n")
        return 0
    rep = orch.run(tasks, only)
    print(f"run {rep['run_id']}: " + ", ".join(f"{f['task']['id']}={f['status']}" for f in rep["findings"]))
    print(f"blocked: {[b['task']['id'] for b in rep['blocked']]}  | LLM cost ${rep['cost_usd']:.2f}")
    print(f"report: {orch.dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
