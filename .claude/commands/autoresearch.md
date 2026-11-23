---
description: Run the ATPP auto-research cycle (parallel researchers, adversarial reviewers, vetted report)
argument-hint: [comma-separated task ids | blank for all active tasks]
allowed-tools: Bash(python -m autoresearch:*), Bash(python -m pytest autoresearch:*), Read
---
Run the ATPP auto-research harness and report back. Do NOT edit research code, `prereg.json`, or `questions.json`.

1. `python -m autoresearch plan` and tell me what is runnable vs blocked (and how to unblock).
2. `python -m autoresearch run --executor claude --max-usd 15 $ARGUMENTS` (omit `--tasks` if no arguments were given; otherwise pass `--tasks $ARGUMENTS`).
3. Read the newest `autoresearch/runs/*/REPORT.md` and summarise: statuses, anything REJECTED or NEEDS_HUMAN and why,
   what the pre-registered verdict says, and the top 5 next steps. Quote numbers from the report; do not restate them from memory.
4. `python -m autoresearch ledger-verify`.
