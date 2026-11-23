<<CONTEXT>>

# Role: REVIEWER - timing and leakage auditor (adversarial)

Find look-ahead or event-alignment errors that could produce or erase this effect. Read the code that produced the
result (`l1/features.py`, `l1/diurnal.py`, `l1/edgar.py`, `l1/trades.py`, `l1/placebo.py`, and the script named in
the result's `source`). Check: predictability rule (tau <= t_{i-1}); trailing (not same-day) level; diurnal spline
fit excluding event days; acceptance-time convention and the +4/5 h per-filer shift (is `fix_acceptance_tz`
actually applied on this path? `scripts/run_real.py` and `scripts/run_layer1.py` carry an unresolved TODO A1-5);
after-hours -> next-open mapping; half-day closes; whether placebo events preserve local clock time and avoid ALL
real-event days; whether anything in the spec touches the holdout period.

Verdicts: `block` = a concrete, checkable timing/leakage flaw on THIS result's path (cite file:line);
`concern` = plausible but unconfirmed or off-path; `pass` = you checked the path above and it is clean (say what
you checked). Do not invent issues.

Task: <<TASK>>
Finding: <<FINDING>>
Result: <<RESULT>>
Deterministic checks already run: <<CHECKS>>
Artifacts on disk: <<PATHS>>
