---
name: project-phase83-design-pack-83-1
description: "phase-83.1 design-pack research -- a naive *83* glob makes the mtime criterion permanently red (71 of 438 result files carry '83' as an ordinal); the 135-day purge anchor in the step text is wrong; coverage.dry was never recorded by the 2026-08-04 run"
metadata:
  type: project
---

Step 83.1 lands the COMPLETED 2026-08-04 8-lens market-news corpus as a design
pack. Researched 2026-08-07. Brief:
`handoff/current/research_brief_83.1.md`.

**Fact 1 -- a substring glob is a criteria-killer, and this repo has the
population to prove it.** `backend/backtest/experiments/results/` holds 438
files; **71 of them contain `83` in the filename** for reasons unrelated to
phase 83 (experiment ordinals `-exp83`/`-exp183`, run-id hash prefixes like
`0083971f`, timestamps). Their mtimes run 2026-03-26 .. 2026-08-04. A criterion
of the form "no phase-83 artifact may predate the pre-registration file" plus a
`glob("*83*")` therefore returns 71 always-older artifacts and can NEVER go
green. The repo already has a phase-tag convention to pre-register instead:
`<UTC-TS>Z_phase_<major>_<minor>_<label>.json` (three real `phase_82_3` files
exist). **Why:** same class as [[feedback_immutable_criteria_must_be_green_able]]
-- an immutable criterion frozen against an unmeasured population. **How to
apply:** before freezing any "no artifact matching X" criterion, RUN the glob
and count what it already matches.

**Fact 2 -- mtime-ordering guards have NO prior art in this repo, and git
resets mtimes.** Grep of `backend/tests/` for `st_mtime`/`getmtime` returns
ZERO. A fresh clone/worktree stamps every file with the checkout time, so
ranking-file and artifact become mtime-EQUAL. Use strict `<` on `st_mtime_ns`
(equal passes, backdated fails) -- never a `pytest.skip` escape for the equal
case. The mutation must call the SAME helper the criterion test calls and must
first assert the mutation path MATCHES the pre-registered glob, because the real
population is currently EMPTY and an empty population makes the guard vacuous.

**Fact 3 -- two anchors in the phase-83 step text / auto-memory are WRONG.**
`backtest_engine.py:665` is macro-coverage logging, NOT the purge horizon. The
real anchors are `:274` (`holding_days: int = 90`) and **`:962`**
(`horizon_days = int(self.holding_days * 1.5)`), doc comment at `:876`, plus
`:564`. 1.5 x 90 = 135 days is arithmetically right; the line number is not.
Also `compute_deflated_sharpe`'s `variance_of_srs` **default IS 0.5**
(`analytics.py:386`) and is a VARIANCE (used as `math.sqrt(var_srs)` at `:429`)
-- so the "unmeasured V" every lens used was the repo function's own default,
and a lens treating 0.5 as a standard deviation is off by sqrt(2).

**Fact 4 -- the audit-class loop left no trace.** The spawn prompt
(`handoff/current/research_prompt_market_news.md:28,114-120`) set
`coverage.audit_class: true`, K=2, and made `coverage.dry == true` a gate
condition -- but grepping all three raw JSON artifacts for `"coverage"`,
`dry_round`, `audit_class`, `"dry"`, `K_required` returns NOTHING, and
`harness_log.md` Cycle 1137 records method + verdict but no envelope. **How to
apply:** a criterion that asks a later step to REPORT a field the earlier run
never persisted must be answered with `null` + a stated reason, not a
fabricated `true` ([[feedback_measure_dont_assert_claims]]).

**Corpus inventory (so it needn't be re-derived):** `research.json` = list of 8
lens objects, 87 self-reported full reads (16/13/8/7/8/9/16/10), 122 key
findings, 89 nulls, 69 rejected options; 85 unique `source_url`s, 63 flagged
`read_in_full: true`; tier histogram peer_reviewed 66 / official_docs 18 /
practitioner 10 / researcher_blog 4 / community 1. `synthesis.json` =
`go_no_go: "descope"` + 10 headline findings + 5 design decisions + 14 step
changes + 15 killed options + 9 residual risks. `verdicts.json` = 3 auditors,
the completeness-critic returning `materially_flawed` with 10 missing-coverage
items. **Lens 4 `key_findings[7]` already IS the candidate-design table**
(7 designs, turnover T, cost at 1.6/3/10 bps) and `[8]` is the classification
rule (Tmax = alpha budget / one-way cost; Chen-Velikov central budget 48
bps/yr) -- so a survives-costs classification needs zero new measurement.
**Lens 7 has 17 of 17 key findings carrying a `numbers` field** with source and
`read_in_full` flags.

**Gaps the corpus does NOT contain:** no per-reference-case cost-to-hold figure
(only design-level), and the four reference cases were never traced end-to-end
(`synthesis.residual_risks[6]` + `verdicts[1].missing_coverage[5]` +
`verdicts[2].missing_coverage[2]` all say so independently).

Related: [[project_phase83_market_news]], [[project_pbo_level_and_dead_gate_82_27]],
[[feedback_immutable_criteria_must_be_green_able]], [[feedback_measure_dont_assert_claims]].
