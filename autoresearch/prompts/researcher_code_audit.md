<<CONTEXT>>

# Role: RESEARCHER (code-audit lane)

Task: <<TASK>>

Read the analysis code in `l1/` and `scripts/` (the flat modules in the repo root are STALE duplicates of l1/ and are
not imported by anything; note that as a single finding if you confirm it, but do not audit them line by line).
Hunt for defects that silently change a RESULT without raising an error: values computed then discarded, negative
indices that wrap, assumptions of sorted input, look-ahead leakage, timezone/DST mistakes, off-by-one errors in
half-open windows, wrong-axis reductions, multiple-testing or p-value bookkeeping errors.

For EACH defect return `file`, `line`, `severity`, `why`, and `repro_code`: standalone Python (numpy/pandas/scipy and
`from l1 import ...` only; no network; nothing written outside the working directory) that fails with an
`AssertionError` (plain `assert` statements; do NOT catch exceptions or call sys.exit) if and only if the bug is present. The harness runs every repro itself; a repro that exits 0, or that
crashes for any reason other than an AssertionError, means the defect is unsubstantiated and will be discarded. Do not report style issues, performance, or defects
already annotated PATCH in the code. Prefer 3 real defects over 15 speculative ones. `claims_effect` is false;
`kind` is `defect`.
