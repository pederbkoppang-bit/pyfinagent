---
name: a-provenance-fix-that-only-logs
description: A provenance fix whose whole observable effect is a logger.warning -- then, one cycle later, an additive key that reaches no persisted artifact; capture the PERSIST call's kwargs
metadata:
  type: feedback
---

When a step claims it corrected **provenance** at a seam, do not grade the seam.
Load the value the seam produces into the function that CONSUMES it and print the
consumer's answer for the fixed input and for the input it is supposed to be
distinguishable from. If the two answers are identical, the fix is a log line.

**Why:** 86.88 added `if risk_dict == _LITE_RISK_DEFAULT: log("resolving as
ABSENT"); return 3.0` and three artifacts said "now recorded as ABSENT rather
than as an explicit SIZE". MEASURED: the persisted `risk_assessment` still
carries `recommended_position_pct: 3.0`, `_resolve_position_pct` on it returns
`PositionVerdict(SIZE, 3.0)` -- byte-identical to a judge that really said 3% --
and `decide_trades` emits the same BUY under both flag states. The code never
constructs an ABSENT verdict; it early-returns the default float BEFORE the
resolver runs. Nothing durable changed. The tests could not catch it because the
only assertion available was a `caplog` substring on the log message.

**How to apply:** three probes, in this order. (1) build the persisted artifact
via the real producer; (2) feed it to the real downstream resolver AND to the
order/decision function; (3) do the same for the input the fix claims to
distinguish it from, and diff. Ask whether the criterion's verb is *log*,
*record*, or *persist* -- "persisting as X rather than Y" is a claim about the
stored record, and a log is not a record. Then check the tension is real: here
an ADDITIVE provenance key would have satisfied both the provenance criterion
and the no-order-movement criterion, so the gap was a choice, not a constraint.
**THE SEQUEL, and it is the more useful half (86.88 cycle 2).** Told the fix was
"only a log line", the author added the additive key I had named -- and the SAME
claim moved one level out. `judge_verdict_absent` is set correctly in the
in-memory dict and is guarded by three tests that kill four mutants, yet it
reaches NO persisted artifact: the lite path's `full_report` is
`{source, analysis, market_data}` and does not contain `risk_assessment` at all,
so the flag is absent from `full_report_json`, and `save_report`'s named columns
(`risk_judge_decision`/`risk_level`/`recommended_position_pct`) are identical for
judge-failed and judge-said-3%. Measured: persisted blob sha **identical** for both
states. The docstring meanwhile said "no auditor reading the persisted row can see
it" about the OLD fix -- implying the new one is row-visible. A repo census settled
it: the key appeared in 1 production line and 3 test assertions, **zero consumers**.

**The probe that decides it in one shot:** stub the persistence collaborator to
CAPTURE its kwargs (`class FakeBQ: def save_report(self, **kw): captured.append(kw)`),
drive the REAL producer twice -- once for the state the fix claims to distinguish,
once for its twin -- run the REAL persist function on both, and diff the captured
payloads. If the only differing field predates the change, the fix is invisible
downstream. Do this BEFORE grading the guard: a key with excellent mutation
coverage and no consumer is a closed loop between a producer and its own tests.

Corollary worth keeping: a surviving weakening of such a key (here a subset match
ignoring one field) is inert *only because* nothing reads it -- it becomes a real
defect the moment the key is threaded to persistence. Say so rather than dropping it.

Related: [[feedback_assert_the_property_not_a_proxy]],
[[feedback_guards_stop_one_seam_short]],
[[feedback_drive_the_real_thing_for_behavioural_claims]],
[[feedback_slice_and_exec_with_the_collaborator_stubbed]].
