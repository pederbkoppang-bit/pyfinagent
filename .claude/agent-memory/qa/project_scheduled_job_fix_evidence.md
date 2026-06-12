---
name: scheduled-job-fix-evidence
description: Cron/launchd fix steps need POST-FIX SCHEDULED-run evidence; manual dry runs insufficient (39.1 burned twice); grep|head verification pipelines are vacuous exit gates
metadata:
  type: project
---

Closure criteria like "exit 0 for N consecutive nights" on launchd/cron fix
steps must be evidenced by SCHEDULED runs of the CURRENT (post-fix)
configuration. Manual dry runs and pre-fix workaround streaks do not count.

**Why:** phase-39.1 (autoresearch nightly) burned on this twice: (1)
2026-05-25 root_cause.md declared "SOURCE FIXED; calendar-bound" and the job
errored 6 MORE scheduled nights (05-26..05-31, second failure layer
ModuleNotFoundError visible only in the launchd context); (2) the 2026-06-12
lenient closure attempt offered an 11-night exit-0 streak that ran the
SUPERSEDED script on the deps-missing 51.4 skip path -- workaround evidence,
not fix evidence. Q/A held for 3 post-fix scheduled nights (06-13/14/15,
collected free by PM sessions).

**How to apply:** when evaluating any step whose criterion is a scheduled-job
exit/streak: (a) check whether the streak nights ran the code under
evaluation or a predecessor; (b) absence of ERROR artifacts is NOT exit-0
proof (pre-log deaths leave no artifact; logs get truncated by housekeeping
-- check the log's earliest surviving line); (c) launchd-env bugs (PATH, env,
sourcing) only surface in scheduled context. Related: verification commands
of the form `... | grep ... | head -1` are VACUOUS as exit gates (pipeline
exit is head's = 0 even on no match) -- judge by output presence, and rule
evidence-by-output against success_criteria when the literal command is
structurally dead (criteria immutability cuts both ways). See
[[committed-criterion-gitignore-check]] for the same evidence-vs-letter
discipline on file-tracking criteria.
