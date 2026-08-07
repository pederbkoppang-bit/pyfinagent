# live_check evidence — step 61.2 (decision-input integrity)

Captured 2026-08-07 (cycle 173). The immutable live_check names three post-fix
signals; the criterion-3 signal is capturable NOW (its fix deployed 2026-07-09),
the other two require the operator's flag promotion — Sections C/D below record
exactly what remains and why.

## Section A — criterion 3 (company_name): FIXED AND LIVE-PROVEN (retires prior-Q/A blocker #1)

Query (read-only; `analysis_date`/`company_name` on `financial_reports.analysis_results`):

```sql
SELECT DATE(analysis_date) d, COUNTIF(company_name IS NULL) null_names, COUNT(*) n
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE analysis_date >= '2026-07-01' AND analysis_date < '2026-08-07'
GROUP BY d ORDER BY d
```

Verbatim result summary:

```
days 07-01..07-08 with NULL company_name (d, nulls, rows): [('2026-07-01', 6, 6), ('2026-07-02', 6, 6), ('2026-07-03', 5, 5), ('2026-07-06', 2, 5), ('2026-07-07', 5, 5), ('2026-07-08', 4, 5)]
days 07-09..08-06 with NULL company_name: NONE -- exactly 0 on every day
days with rows in range: 27
```

The ungated fix at `autonomous_loop.py:2926-2934` deployed at the 2026-07-08/09
restart; every row since carries a company_name. This is the evidence the prior
Q/A said could not exist yet.

## Section B — the fabrication baseline (criterion 1's "before" half; the defect is FIRING)

Query:

```sql
SELECT COUNT(*) total_n,
  COUNTIF(final_score = 0.0 AND UPPER(recommendation) = 'HOLD'
          AND JSON_EXTRACT_SCALAR(full_report_json, '$.final_synthesis.error') IS NOT NULL) fabricated_n,
  STRING_AGG(DISTINCT JSON_EXTRACT_SCALAR(full_report_json, '$.final_synthesis.error') LIMIT 3) error_strings,
  MAX(...) last_fabricated
FROM `financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 40 DAY)
```

Verbatim result:

```
{'total_n': 170, 'lite_n': 0, 'fabricated_n': 142, 'error_strings': 'Failed to parse final report.', 'last_fabricated': datetime.datetime(2026, 8, 6, 19, 20, 56, 402576, tzinfo=datetime.timezone.utc)}
```

**142 of 170 rows in 40 days (83.5%) are fabricated 0.00/HOLD** with the single
error string `Failed to parse final report.`, last one 2026-08-06 19:20:56Z.
(Derivation-rule CORRECTION, cycle-2 Q/A: my "could not reproduce the lite
split" disclosure was WRONG — the predicate is `$._path = 'lite'`, which
reproduces the researcher's split exactly: **14 lite / 156 full over the same
40d window, lite share 8.2%** — and that IS the AWS REL05-BP01 fallback-share
baseline the promotion ask turns on: post-promotion the lite path would carry
up to 91.8%. The fabrication count 142 and the last-fired timestamp agree
across all derivations. The original wrong note is superseded by this text.)

The laundering mechanism: `orchestrator.py:2280` stamps
`compute_weighted_score({}) = 0.0` onto the synthesis error dict; the fix is
BUILT (commit 6186784c) and DARK behind `paper_synthesis_integrity_enabled`
(settings.py:198, default False).

## Section C — post-promotion signals (BLOCKED on the operator's flag decision)

Requires `paper_synthesis_integrity_enabled` ON + backend restart + ≥1 scheduled
full-path cycle: (a) zero new 0.0+error rows; (b) the lite-fallback share of
traffic (measured baseline 8.2% via $._path; expect a jump toward up to 91.8% — the AWS REL05-BP01
"untested fallback becomes the primary path" risk, stated not buried); (c)
non-constant conviction in `paper_trades.signals`; (d) the 2-consecutive-all-
fallback WARN evidence; (e) the RiskJudge sector-context log line (criterion 6,
flag-gated the same way). Ask item filed with this evidence (operator_ask
#10).

## Section D — UI capture (prior-Q/A blocker #3)

The degraded-row badge (`reports-columns.tsx`) can only exist AFTER promotion
produces a degraded row; Playwright capture deferred with Section C.

## Test-state repair evidence (this cycle's executor work)

The immutable command was red for reasons OUTSIDE the step: two 61.2 tests and
two phase-50.2 tests called `execute_buy` without kill-switch injection, so
their outcomes depended on the operator's LIVE pause state (paused=True,
reason='manual' today — the phase-36.13 gate consults the real on-disk audit
when nothing is injected). Repaired via the documented `kill_switch_state`
seam; the classified sweep [CORRECTED per the cycle-2 Q/A's forced-uninjected
re-derivation] found **9 kill-switch-coupled failures across 5 files** (my
earlier 15 counted unclassified reds); 4 same-class failures remain in 2 files
and 13 FILES total are queued as step 36.28, incl. one MIS-classified member
(test_adjust_cash_and_mtm — a stale stub-signature TypeError, now explicitly
in 36.28's scope). Also repaired: a test-isolation defect in MY OWN
83.0 test (global `sys.modules` scan made it order-dependent under this `-k`
selection). After the repairs, the immutable command verbatim:

```
$ python -m pytest backend/tests -k 'synthesis or persist or downgrade or meta_scorer or 61_2' -q
71 passed, 2829 deselected, 1 warning in 18.99s
```

(and `test -f handoff/current/live_check_61.2.md` — this file.)
