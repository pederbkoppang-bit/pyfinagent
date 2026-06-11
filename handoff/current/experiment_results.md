# Experiment Results — Step 59.3 (GENERATE)

**Step:** 59.3 — Stress-test doctrine run (bare Fable 5 vs the 55.2 harness chain). **Date:** 2026-06-11.

## What was built

| Artifact | Content |
|---|---|
| `handoff/current/59.3-harness-free-output.md` | The bare run's verbatim output (282 lines; its single permitted write) — 10 findings AW-1..AW-10 |
| `handoff/current/59.3-stress-test-comparison.md` | Selection justification, leakage disclosure, all 6 dimension scores with verified examples from both artifacts, per-component verdicts via the pre-registered rules, the model-vs-harness confound, the operator-gated follow-up proposal |
| `handoff/current/live_check_59.3.md` | Telemetry, the 11/11 fabrication spot-test, the dimension table, verdicts, the 6 new bug candidates |

Mechanics: pinned worktree `70a8242b` (blinding verified), single run (310K tokens / 126 tool uses / 35.4 min), worktree torn down.

## Verification command output (verbatim)

```
$ test -f handoff/current/59.3-stress-test-comparison.md && test -f handoff/current/live_check_59.3.md && echo PASS
PASS
```

## Headline result

The bare Fable 5 pass scored **10/10 on the QA-verified anchors (zero confident-wrong), 3/3 on premise probes, 11/11 on fabrication spot-tests, 8.5/9 coverage** — and went beyond the harness chain in five places, including two verified REVISIONS of the harness's own findings (the Approve error's true source is the OpenClaw gateway; 55.2's "$0.40 metered" summed a cumulative column) and a re-diagnosis of F-H (silent full→lite fallback driven by a retired Gemini model 404 + KR CIK gaps, not a checkbox desync). It also surfaced the probable primary money-bleed mechanism of the away week (the sentinel-conviction churn engine) which no harness step had found.

Per the PRE-REGISTERED rules: researcher-gate and contract thresholds for PRUNE-candidate were MET; the recommendations are deliberately the prudent MODIFY readings (adaptive tiering / scope-by-step-class), the Q/A verdict is MODIFY-at-most by the rule's own "verification ≠ generation" clause, handoff files KEEP, turn caps MODIFY (already begun in 59.1). **Nothing was removed from the harness; every verdict is operator-gated**, with a cheap component-at-a-time confirmation run proposed before acting (this run measured the joint effect — attributed, not isolated; n=1, analysis-class only; the Fable-5-vs-Opus model upgrade is confounded with the harness removal by design of the doctrine).

## Honest limitations

- Attributed-not-isolated; n=1 task; analysis class only — code/money steps untested and Q/A's value concentrates there.
- Leakage residuals disclosed (backend.log reachable; low-impact — no fix-era echoes; the run's biggest findings exist in no answer key).
- The bare run did zero external research — the researcher gate's literature half was never exercised and remains mandatory where criteria demand it.
- The 6 new bug candidates (incl. two P0s) enter the NORMAL masterplan flow as findings — not fixed here (review-only step).
