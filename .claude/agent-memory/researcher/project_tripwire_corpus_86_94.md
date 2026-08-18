---
name: tripwire-corpus-86-94
description: 86.94's mentions_reviewed tripwire is RED (42/3) because two session-ops notes named the files; its corpus is 89.5% gitignored so it is unreproducible in the same class the step exists to close; the fail-closed branch has ZERO mutation cells
metadata:
  type: project
---

Measured 2026-08-17 against `scripts/qa/verify_no_sliding_windows_86_94.py`
(the masterplan PARK note says "ALL GREEN 45/0"; that is stale).

**The guard is RED: 42 passed, 3 failed.** All three are the `mentions_reviewed`
equality at `:544-551` disagreeing with the pins at `:227-231` -- 282->283,
6->9, 49->50.

**It fired on incidental change.** Since the pinning commit `964b0255`
(2026-08-17T00:51:13+02:00) exactly 3 handoff files were added; two mention the
guarded names -- `handoff/current/day_halt.md` and
`handoff/current/day_report_2026-08-17.md`. Both are session narration. Neither
quotes a figure from any window. That is the brittle/change-detector shape:
a failure on an unrelated change that introduces no bug.

**The tripwire's corpus is not a defined set -- the same defect class the step
closes.** `MENTIONS` at `:511` walks `(REPO/'handoff').rglob('*.md')`, i.e. the
WORKING TREE: 49,094 `.md` under `handoff/`, only 5,167 tracked, **43,927
(89.5%) gitignored** via `.gitignore:80` (`handoff/archive/_quarantine_*/`).
For `frontend_route_inventory.py` only **5 of 50** mention-sites are tracked.
The allowlist's own smoking-gun citation at `:203-205`
(`handoff/archive/_quarantine_2026-04-21/phase-3.7.5-v22/experiment_results.md`)
**is itself gitignored** -- on a fresh clone the count is 5 and the evidence for
`quoted_as_evidence: True` is absent. A count over "whatever .md is on this
disk" is a number about a machine exactly as `--since=<bare date>` is a number
about a clock. Fix direction: `git ls-files handoff`, and bind the claim to the
adjudicated (file, figure) pairs rather than to a filename.

**Kill attribution -- my hypothesis was WRONG and the measurement was better.**
I predicted some `[4]` cells were killed by the fail-closed `<unparsed>` branch
rather than by `classify()`. Measured across all 11 injections: **all 11 are
killed by `classify()` on a parsed VALUE; none reaches `<unparsed>`.** The real
gap is the inverse -- **the fail-closed branch at `:374-379` has ZERO cells**,
and its own comment is stale (`--since 2026-08-11` now parses, because
`PLAUSIBLE_VALUE` matches). Four shapes measured to reach it:
`subprocess.run(["git","log","--since", win])`, the f-string-element form,
`--since=` empty, `--after` + variable. Every `[4]` assertion is
`bool(hits)` filtered only on `h[3]=="SLIDING"` (`:608-610`) and never asserts
`h[2]`, so no cell can tell the two mechanisms apart.

**Why:** step 86.94 PARKED at the cap (FAIL, FAIL, CONDITIONAL) on
evidence-integrity, not on the product; cycle 4 needs the tripwire's predicate
and corpus fixed, not its numbers bumped.

**How to apply:** never bump `mentions_reviewed` to 283/9/50 -- the entry's own
text forbids it and it leaves the corpus defect untouched. External backing:
in-toto binds subjects **by digest** ("matched purely by digest"), SLSA requires
the verifier to check the subject digest and to reject unrecognized fields, and
mutation-kill reasons are separable (assertion vs exception/timeout).

Related: [[research-gate-discipline]], [[reproducible-windows-86-94]],
[[uncalled-function-86-97]].
