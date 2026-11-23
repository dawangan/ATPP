# autoresearch: orchestrated, adversarial research harness for ATPP

**Principle: LLM agents propose and critique; code decides.**
ATPP is a pre-registered study with a holdout. The main risk of automating research on it is not
bugs, it is *forking paths*: an agent that tries many specifications and reports the best one.
So the harness is built around constraints, not around cleverness.

```
                         questions.json  (human-curated backlog; agents may only PROPOSE)
                                 |
                          ORCHESTRATOR.plan   -- blocks tasks whose data / executor is missing, says how to unblock
                                 |
        wave 1: Monte-Carlo calibration (null size, bias, coverage)   <- evidence every later check consumes
        wave 2: tasks in parallel (--parallel N)
        +--------------+--------------------+---------------------+
        | experiment   | literature         | code_audit          |   RESEARCHERS (parallel lanes)
        | registered   | WebSearch/Fetch,   | reads l1/, scripts/ |
        | experiments  | verbatim quotes,   | returns repro code; |
        | only         | searches logged    | HARNESS runs it     |
        +------+-------+---------+----------+----------+----------+
               |                 |                     |
               v                 v                     v
        deterministic checks (no LLM) + independent LLM reviewers, each a fresh non-persisted session
        statistician | timing/leakage auditor | skeptic/red-team | citation verifier   (they never see the
                                                                                      researcher's reasoning)
                                 |
                       vet()  -- pure code -> VERIFIED | VERIFIED_WITH_CAVEATS | REJECTED | NEEDS_HUMAN | ...
                                 |                      one bounded revision round if a reviewer blocks
                       FINALIZER + report.py -> REPORT.md, report.json, ranked next steps, proposed_tasks.json
                                 |
                       LEDGER (hash-chained, append-only): every run, failure, refusal, verdict
```

## What stops p-hacking

| control | where |
|---|---|
| Researchers can only run **registered** experiments; new analyses are added by a human | `experiments.py` |
| Every run is appended to a **hash-chained ledger** (also failures, refusals); tamper -> `ledger-verify` fails | `ledger.py` |
| Distinct real-data specs tried for H1 are counted; a pass that only clears alpha without the correction is flagged | `checks.multiplicity` |
| **Holdout is locked**: results reaching past `explore_end` are refused, and there is no unlock in the harness | `prereg.py` |
| Frozen-spec drift (e.g. half-life != 1800 s) marks a run *exploratory*: ineligible for the pre-registered verdict | `prereg.drift` |
| The pre-registered pass/fail is **computed by code** from your 5 criteria; the LLM cannot override it | `verdict.py` |
| Agent-proposed follow-ups land in `state/proposed_tasks.json`, **not runnable** until a human promotes them | `orchestrator._propose` |
| `VERIFIED` is impossible without a real LLM review (`--executor none` caps at `DETERMINISTIC_ONLY`) | `orchestrator.vet` |
| Code-audit defects count only if the harness runs the repro and it raises `AssertionError` | `orchestrator.run_repro` |

## Adversarial checks that need no LLM (`checks.py`)

`size_calibration` runs the whole pipeline on simulated worlds with **zero** true effect and tests whether your
pre-registered joint robust Wald holds its nominal size (binomial test). `se_validity` compares trade-level robust vs
day-clustered inference. Also: `truth_recovery`, `replication` (independent seed), `multiplicity`, `power_events`
(effect is identified by the number of event windows, not trades), `spec_drift`, `holdout`, `convergence`.

## Run it

```bash
pip install numpy scipy pandas pyarrow pytest            # the repo has no requirements file
python -m pytest autoresearch/tests -q                   # fast, offline
python -m autoresearch plan                              # what is runnable / blocked, and why
python -m autoresearch run                               # deterministic only, free: caps at DETERMINISTIC_ONLY
python -m autoresearch run --executor claude --max-usd 15 --parallel 3 \
       --reviewer-model <a different model than --model>  # full adversarial cycle
python -m autoresearch ledger-verify | ledger-summary
```

Outputs: `autoresearch/runs/<run_id>/REPORT.md` (+ `report.json`, per-task `bundle.json`, raw experiment artifacts).
Needs `claude` (Claude Code) on PATH and logged in; each agent call is a headless `claude -p` with an allow-listed
toolset under `--permission-mode dontAsk`: reviewers get Read/Grep/Glob only (citation reviewer adds web), researchers
never get Bash or Write. Prompts are in `prompts/`.

## Connecting your local machine

The real data (Databento parquet, `results_*/`, `census/`, `ext_*.npz`, `PREREGISTRATION.md`) is not in git. The
harness never re-fits real data itself; it **vets what your scripts already produced**:

```bash
python scripts/run_real.py --durations msft_dur.parquet --closes-csv msft_dur.diag.csv \
       --pages-json msft_8k_pages.json --out results_msft          # as you do today (>=200 placebos)
export ATPP_DATA_ROOTS=/path/containing/results_msft              # os.pathsep-separated list
python -m autoresearch run --executor claude --tasks q-real-msft,q-method-null
```
Add a task per ticker/panel to `state/questions.json` using `"experiment": "ingest_results"`. `prereg.json` is a
machine-readable **mirror** of the README's Stage-1 criteria: diff it against `PREREGISTRATION.md` before trusting
any verdict, and change it only with a dated amendment. Commit `state/ledger.jsonl` if you want one audit trail
shared across machines; the multiplicity count is only as good as the ledger it can see.

## Extending

* New analysis: add `@experiment("name", family=..., data_kind=...)` in `experiments.py` and a task in `questions.json`.
* New adversarial angle: add `prompts/reviewer_<lens>.md`, list it in `schemas.LENSES`, and/or add a function to `checks.CHECKS`.
* Nightly: schedule `python -m autoresearch run --executor claude` (cron, or a Claude Code scheduled task).

## Known limits (v1)

* Reviewers and researchers share a model family unless you set `--reviewer-model`; errors correlate. Use different models.
* A `VERIFIED` finding means "survived adversarial review under the registered spec", not "true".
* `prereg.json` encodes the README's criteria, not the (unseen) PREREGISTRATION.md.
* Offset scan (criterion 5) is only produced by `run_layer1.py`; `run_real.py` results are `incomplete` on it.
* Synthetic-data findings validate machinery only. The simulator embeds the model's own functional form.
* A code-audit defect that is *discarded* (its repro exited 0 or crashed) is "not substantiated", not "false": a true
  structural fact can be discarded because the repro was badly written. Read the caveats list.
* In a restricted cloud container `WebFetch` may be blocked by the outbound proxy; the literature lane then degrades to
  search-only and the harness flags it (`citation_form` concern, no quotes). It fetches normally on an unrestricted machine.
* Observed cost with the default model: about $0.35-0.40 per task (2 tasks $0.73, 4 tasks incl. literature $1.58).
  `--max-usd` is a hard stop checked before every call; budget for reviewers, which dominate.
* The Monte Carlo defaults (16 sims) are coarse; raise `n_sims` in `questions.json` (>= 40) before acting on size numbers.
