---
name: stress-test-blinded-rerun
description: How to run the recurring harness stress-test (doctrine fires on every model release) — blinded re-run design, the worktree partial-brief gotcha, and the per-component attribution caveat
metadata:
  type: project
---

Phase-59.3 (2026-06-11, Fable 5 release) designed the first doctrine-mandated
harness-free re-run. The design generalizes to every future model-release
stress-test.

**Candidate class:** only ANALYSIS steps are safely re-runnable (a bare CODE
step writes conflicting code). Best candidate = one whose archived Q/A critique
independently re-derived claims against live BQ/code (hostile-tested ground
truth). 55.2 was chosen: 7-file chain, 10 QA-verified anchor facts, 4 premise
corrections to probe.

**The worktree gotcha (L13):** pinning a `git worktree` at the commit BEFORE
the step's PASS commit is NOT clean — write-first incremental briefs get
captured by intermediate chore auto-commits (70a8242b at 18:35 already held a
partial `research_brief_55.2.md`). Setup must `rm -rf <wt>/handoff` +
`<wt>/.claude/masterplan.json`, then restore ONLY the inputs the original chain
had (input parity). Teardown: `git worktree remove --force`.

**Why the pin is mandatory at all:** later fix-phases INVERT code ground truth
(56.2 removed the dead button; 57.1 made REJECT binding) — a bare agent reading
live code would be CORRECT to contradict the archived findings, making accuracy
scoring incoherent. The pin = temporal-cutoff contamination handling
(arXiv:2502.17521 taxonomy). BQ side: date-filter <= step date; beware
post-hoc restatements (56.1 backfill restated 9 KR paper_trades rows) and
post-fix telemetry (llm_call_log now logs the rail).

**Scoring design:** pre-register rubrics BEFORE the run (the research brief IS
the pre-registration); weight PROCESS dims (premise-catching, evidence rigor,
coverage, calibration) over answer-matching (leakage inflates the latter);
fact-anchored checklists, never holistic pairwise preference (position bias is
MAX when both candidates are good, arXiv:2406.07791; same-family
perplexity-familiarity bias, arXiv:2410.21819). Main authors the spawn prompt
— paste verbatim from the brief or Main's answer-laden context leaks hints.

**Attribution caveat (must appear in every comparison doc):** Anthropic's
documented method is component-AT-A-TIME removal; an all-at-once bare run
measures the JOINT effect only — per-component keep/prune verdicts are
attributed-not-isolated ("attribution is well-posed iff each component is
lifted out", arXiv:2604.07236, which also found stronger models SUBSTITUTE for
structure rather than erase its marginal value). Any prune recommendation
should propose a component-at-a-time confirmation run as the operator-gated
follow-up.

**The QA-component tension:** Fable 5's announcement claims "at the highest
effort, [it] reflects on and validates its own work" — this is the vendor's
hypothesis about the model under test, NOT evidence against the separate-QA
component; harness-design's "agents confidently praise their own work" is the
empirical counter. Adjudicate, don't assume.

See [[fable5-adoption]] for model economics (free window ended 2026-06-22).
