"""LLM executors. The harness talks to models only through `Executor.call`.

ClaudeCLI  : headless `claude -p` (uses the caller's existing Claude Code auth; no API-key plumbing).
             Every call is a fresh, non-persisted session, so reviewers share no context with
             researchers. Tool access is allow-listed and enforced with --permission-mode dontAsk.
NullExecutor: no LLM. The orchestrator then runs deterministic checks only and caps every
             finding's status at DETERMINISTIC_ONLY: it never reports VERIFIED without LLM review.
"""
from __future__ import annotations

import json
import subprocess
import threading

from .core import ROOT

TOOLSETS = {
    "readonly": ("Read,Grep,Glob", "Read Grep Glob"),
    "web": ("Read,Grep,Glob,WebSearch,WebFetch", "Read Grep Glob WebSearch WebFetch"),
}


class BudgetExceeded(RuntimeError):
    pass


class NullExecutor:
    name = "none"
    runs_llm = False
    spent = 0.0

    def call(self, role, prompt, schema, tools="readonly", model=None):
        return {"ok": False, "data": None, "cost": 0.0, "error": "no LLM executor (deterministic-only run)"}


class ClaudeCLI:
    name = "claude-cli"
    runs_llm = True

    def __init__(self, model=None, reviewer_model=None, effort=None, max_total_usd=10.0,
                 per_call_usd=1.5, max_turns=15, timeout=900):
        self.model, self.reviewer_model, self.effort = model, reviewer_model, effort
        self.max_total_usd, self.per_call_usd = max_total_usd, per_call_usd
        self.max_turns, self.timeout = max_turns, timeout
        self.spent, self._lock = 0.0, threading.Lock()
        self.calls: list[dict] = []

    def call(self, role, prompt, schema, tools="readonly", model=None):
        with self._lock:
            if self.spent >= self.max_total_usd:
                raise BudgetExceeded(f"total LLM budget ${self.max_total_usd:.2f} reached")
        tool_list, allow = TOOLSETS[tools]
        m = model or (self.reviewer_model if role.startswith("reviewer") else self.model)
        cmd = ["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
               "--permission-mode", "dontAsk", "--tools", tool_list, "--allowedTools", allow,
               "--max-turns", str(self.max_turns), "--no-session-persistence", "--disable-slash-commands",
               "--max-budget-usd", str(self.per_call_usd)]
        if m:
            cmd += ["--model", m]
        if self.effort:
            cmd += ["--effort", self.effort]
        try:
            p = subprocess.run(cmd, input=prompt, cwd=ROOT, capture_output=True, text=True, timeout=self.timeout)
            out = json.loads(p.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            return {"ok": False, "data": None, "cost": 0.0, "error": f"{type(e).__name__}: {e}"}
        cost = float(out.get("total_cost_usd") or 0.0)
        with self._lock:
            self.spent += cost
            self.calls.append({"role": role, "cost": cost, "turns": out.get("num_turns")})
        data = out.get("structured_output")
        if data is None and isinstance(out.get("result"), str):
            try:
                data = json.loads(out["result"])
            except json.JSONDecodeError:
                data = None
        ok = bool(data) and not out.get("is_error")
        return {"ok": ok, "data": data, "cost": cost,
                "error": None if ok else (out.get("result") or out.get("subtype") or "no structured output")[:500]}
