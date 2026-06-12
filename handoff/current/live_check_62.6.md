# live_check -- phase-62.6: ops hygiene batch

Date: 2026-06-12. Status: criteria 1+2 COMPLETE (Q/A-verified incl. independent
reproductions); criterion 3 PENDING the 39.1 strict path (3 post-fix SCHEDULED nights,
06-13/14/15 -- collected automatically by the away PM sessions; fresh Q/A then closes
both). Q/A spawn-1 verdict: CONDITIONAL on exactly that coupling + this file's absence.

## Criterion 1 -- rotation (verbatim, Q/A-reproduced)

    $ test $(stat -f%z backend.log) -lt 52428800 && python -c "import langchain_huggingface; print('lh OK')"
    lh OK

385MB -> handoff/logs/backend.log.20260612T104931Z.gz (19.18MB, gunzip-intact,
check-ignored AND untracked -- the pre-redaction FRED-key archive cannot reach git).
Live-log growth probe (Q/A): 51,579 -> 51,999 bytes with NO restart -- O_APPEND claim
proven. Ongoing: healthcheck.sh rotation block on the 30-min watchdog cadence,
log_rotated field in the health JSON line.

## Criterion 2 -- autoresearch at $0 + ablation (Q/A-reproduced)

Q/A ran the REAL nightly entrypoint itself: exit 0, preflight-only line, memo count
33 -> 33 (zero spend), langchain-core HELD ==1.2.30; run_memo.py:208-211 returns before
the only LLM path (:213); the flag is hardwired in run_nightly.sh:37 -- a full run is
IMPOSSIBLE without editing the script (structural no-spend guarantee). .env-sourcing
regression: raw-vs-sanitized repro reproduced the unexpected-EOF death and the fix.
Ablation: disposition accepted-as-intent (last exit 0, all-tested branch
run_ablation.py:329-331; the criterion's failing-job premise dissolved; original
traceback unrecoverable -- documented, job stays loaded).

## Criterion 3 -- 39.1 closure: HELD for strict evidence

Q/A ruling: the literal 39.1 command is structurally dead (0 PASS files ever; pipeline
exit vacuous), so evidence-by-output is the only path -- but criterion-a demands
post-fix SCHEDULED nights and there are 0 of 3 (today's runs were manual; the 11-night
streak ran the superseded skip-path script; the step's own 2026-05-25 "source fixed"
precedent was followed by 6 scheduled ERROR nights). Criteria b+c met today
(live_check_39.1.md). The PM sessions record 06-13/14/15 scheduled outcomes; a fresh
Q/A closes 39.1 and thereby 62.6 ~06-15/16. Tonight's 02:00 run exercises COMMITTED
code (interim commit per Q/A recommendation).
