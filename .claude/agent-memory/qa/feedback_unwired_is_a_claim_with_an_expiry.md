---
name: unwired-is-a-claim-with-an-expiry
description: "X is inert/unwired/hypothetical" in a limitations section is a claim about the world that a SIBLING step can falsify mid-session -- re-derive it against HEAD and the live hook registry, and check the module NAME is not doing the work
metadata:
  type: feedback
---

A limitations section ("what I could NOT verify") is where scope honesty is
graded, so treat every sentence in it as a claim to REPRODUCE, not as a
disclaimer to accept. In particular, re-derive any claim of the form
"*X is still inert and unwired, so that consumer remains hypothetical*"
against **HEAD** and against `.claude/settings.json`'s hook block -- not
against the module's own file.

**Why:** on 86.85 cycle 10, `experiment_results` §4 item 4 said the ledger's
"second intended consumer remains hypothetical" because `attempt_budget.py`
was unwired. True about that FILENAME. But commit `192ef652` -- an ancestor of
HEAD, made by the same session **34 minutes earlier** -- had registered a
DIFFERENT file, `scripts/harness/attempt_gate.py`, as a live PreToolUse hook,
and that file does `from verdict_ledger_write import emit_sequence` against the
REAL `handoff/verdict_ledger.jsonl`. The graded module was on the live
tool-call path while its own artifact called that consumer hypothetical. The
module NAME was carrying the sentence's truth; the SUBSTANCE was false.

**How to apply:** for each "unwired / no caller / out of scope / hypothetical"
claim, run a positive-controlled search for callers of the *symbol*, not the
*filename* (`grep -rn "<symbol>" --include='*.py' --include='*.js'`), and grep
the hook registry (`grep -n "<script>" .claude/settings.json`). Then date the
wiring commit and compare it to the artifact's mtime -- if the commit predates
the write, the author had the information. Then **measure the direction of
harm before choosing severity**: here the new consumer wrapped the call in
`except Exception: return []`, swallowing every loud refusal the step had
built, but that path feeds only a PASS exception and `disposition()` checks
PASS first, so it is fail-CLOSED -- WARN, not BLOCK. A stale scope claim with
a safe direction is still a finding; a stale scope claim assumed unsafe is a
plausible-but-wrong finding. Related:
[[recheck-head-before-returning-a-scoped-grade]],
[[queued-is-a-claim-that-must-reproduce]],
[[structural-fix-needs-a-mechanism]].
