---
name: criterion-wording-existence-vs-completion
description: Rule wall-clock criteria on their literal verbs -- "writes rows" = existence at eval time; do not import completion requirements (66.1 closing)
metadata:
  type: project
---

Immutable criteria with live-evidence verbs are ruled on their LITERAL wording.
Phase-66.1 criterion 3 ("a SCHEDULED cycle after deploy WRITES ok=true rows to
llm_call_log") was ruled SATISFIED while the cycle was still status=started:
the writes existed and were independently reproduced (32 ok rows, nonzero
tokens, sole cycle_id today, cron-shaped started_at). Cycle completion was not
in the text, so requiring it would be an unwritten extra criterion; conversely,
counts GROWING between live_check snapshot and Q/A reproduction is expected,
not drift, when the criterion is existential.

**Why:** two symmetric evaluator errors exist -- (a) FAILing an existence
criterion because a process has not finished (importing requirements), and
(b) PASSing a completion/durability claim on mere existence evidence. The
criterion's verb decides which evidence shape is owed.

**How to apply:** quote the criterion verbatim from masterplan.json, identify
its verb class (exists/writes vs completes/sustains/N-day streak), and match
the evidence to that class. Streak-class criteria (e.g. 66.2's ">=5 healthy-
rail trading days") still need the full window per
[[scheduled-job-fix-evidence]].
