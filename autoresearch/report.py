"""Markdown rendering. Every number here comes from code-computed artifacts, never from LLM prose."""
from __future__ import annotations

ICON = {"pass": "PASS", "fail": "FAIL", "concern": "CONCERN", "na": "n/a", "unknown": "UNKNOWN",
        "block": "BLOCK"}


def _cell(x, n=140):
    s = str(x).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _val(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_val(x)}" for k, x in v.items() if not isinstance(x, (list, dict)))
    return str(v)


def render_markdown(rep: dict) -> str:
    L = []
    a = L.append
    a(f"# ATPP auto-research report: {rep['run_id']}\n")
    if not rep["llm_review_ran"]:
        a("> **NO LLM REVIEW RAN.** Deterministic checks only. Nothing in this report can be `VERIFIED`; "
          "the best status is `DETERMINISTIC_ONLY`. Re-run with `--executor claude` for adversarial review.\n")
    g = rep["git"]
    a(f"- executor: `{rep['executor']}` | git `{g['sha'][:10]}`{' (dirty)' if g['dirty'] else ''} | "
      f"prereg hash `{rep['prereg_hash'][:12]}` | LLM cost ${rep['cost_usd']:.2f}")
    lg = rep["ledger"]
    a(f"- ledger: {lg['entries']} entries, chain {'OK' if lg['chain_ok'] else 'BROKEN at seq ' + str(lg['first_bad_seq'])}; "
      f"**{lg['h1_specs_tried']} distinct real-data spec(s) tried for H1** (multiplicity budget)\n")

    a("## Summary\n")
    if rep.get("summary"):
        a(rep["summary"] + "\n")
    else:
        from collections import Counter
        c = Counter(f["status"] for f in rep["findings"])
        a("Statuses: " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) + f"; blocked/not run: {len(rep['blocked'])}.\n")
    a("| task | lane | status | headline |\n|---|---|---|---|")
    for f in rep["findings"]:
        a(f"| {f['task']['id']} | {f['task']['lane']} | **{f['status']}** | {_cell((f['finding'] or {}).get('claim', '-'), 160)} |")
    for b in rep["blocked"]:
        a(f"| {b['task']['id']} | {b['task']['lane']} | BLOCKED | {_cell(b['why'], 160)} |")
    a("")

    a("## Findings\n")
    for f in rep["findings"]:
        t, fd = f["task"], f["finding"] or {}
        a(f"### {t['id']}: {t['question']}\n")
        a(f"**Status: {f['status']}**" + (f" (after {f['rounds']} revision round(s))" if f["rounds"] else ""))
        for r in f["reasons"]:
            a(f"- {r}")
        if fd:
            a(f"\n**Claim** ({fd['kind']}, confidence {fd['confidence']}, claims_effect={fd['claims_effect']}): {fd['claim']}\n")
        if f["verdict"]:
            a(f"**Pre-registered verdict (code): {f['verdict']['overall'].upper()}**\n")
            a("| # | criterion | status | value | note |\n|---|---|---|---|---|")
            for c in f["verdict"]["criteria"]:
                a(f"| {c['id']} | {c['text']} | {ICON[c['status']]} | {_cell(_val(c['value']), 90)} | {_cell(c['detail'], 110)} |")
            a("")
        if f["checks"]:
            a("**Deterministic adversarial checks**\n\n| check | status | detail |\n|---|---|---|")
            for c in f["checks"]:
                a(f"| {c['id']} | {ICON[c['status']]} | {_cell(c['detail'], 200)} |")
            a("")
        if f["reviews"]:
            a("**Independent reviewers**\n")
            for r in f["reviews"]:
                a(f"- **{r['lens']}: {r['verdict'].upper()}**; would change my mind: {_cell(r['what_would_change_my_mind'], 200)}")
                for i in r["issues"]:
                    a(f"  - [{i['severity']}] {i['claim']} (evidence: {_cell(i['evidence'], 160)})")
        for x in f["reviewer_failures"]:
            a(f"- reviewer `{x['lens']}` FAILED to run: {_cell(x['error'])}")
        if fd.get("caveats"):
            a("\n**Caveats**\n" + "\n".join(f"- {c}" for c in fd["caveats"]))
        if fd.get("defects"):
            a("\n**Defects (repro run by the harness)**\n")
            for d in fd["defects"]:
                a(f"- `{d['file']}:{d['line']}` [{d['severity']}, {d.get('repro_kind', '?')} repro] {d['why']}")
        if fd.get("citations"):
            a("\n**Citations**\n" + "\n".join(f"- {c['claim']} ({c['url']})" for c in fd["citations"]))
        res = f.get("result")
        if res:
            a(f"\n_Reproduce:_ ledger seq {res['run']['ledger_seq']}, code `{res['run']['code_hash'][:10]}`, artifacts `{res['run']['outdir']}`\n")

    if rep["blocked"]:
        a("\n## Blocked / not run\n")
        for b in rep["blocked"]:
            a(f"- **{b['task']['id']}**: {b['why']}" + (f". Unblock: {b['task']['unblock']}" if b["task"].get("unblock") else ""))
        a("")

    a("\n## Next steps (ranked)\n")
    if rep.get("llm_next_steps"):
        a("_Finalizer ranking:_\n")
        for s in sorted(rep["llm_next_steps"], key=lambda s: s["priority"]):
            a(f"{s['priority']}. **{s['title']}**: {s['why']}")
        a("\n_Mechanical list (from reviewers, checks, unmet criteria, blocked tasks):_\n")
    for i, s in enumerate(rep["next_steps"][:25], 1):
        a(f"{i}. [{s['source']}] {s['title']} (why: {_cell(s['why'], 200)})")
    a("\nResearcher-proposed follow-ups were written to `autoresearch/state/proposed_tasks.json`. "
      "They are NOT runnable until a human promotes them into `questions.json` (forking-paths control).\n")

    a("## How to read the statuses\n")
    a("- `VERIFIED`: survived every deterministic check and every adversarial reviewer. Not 'proven'; it means we could not break it under the registered spec.\n"
      "- `VERIFIED_WITH_CAVEATS`: survived, but with listed concerns that must travel with the claim.\n"
      "- `REJECTED`: a failed check on a positive claim, or unanimous reviewer blocks after revision. Rejected findings stay in the report and the ledger.\n"
      "- `NEEDS_HUMAN`: reviewers disagree after revision.  `DETERMINISTIC_ONLY`: no LLM review ran.  `BLOCKED`: data or executor missing.\n")
    return "\n".join(L)
