---
name: empty-hold-86-69
description: phase-86.69 root cause -- a RESTORATION commit caused the regression; a new provenance field is a free deploy marker; the dark 61.2 guard separates the live populations at 100/100
metadata:
  type: project
---

Root-cause of the 81%-empty-`HOLD` regression, measured 2026-08-17 for step
86.69. Brief: `handoff/current/research_brief_86.69.md`.

**A commit that RESTORES a path can be the cause of a regression.** phase-60.1
(`fa62b5fe`, 2026-06-11) repinned the server-side-discontinued
`gemini-2.0-flash` -> `gemini-2.5-flash`. Until then every full-pipeline call
404'd and the book ran **entirely on the healthy lite path**. Restoring the
full path restored a defect that had been waiting in it. Searching `git log`
for a *breaking* change in the window finds nothing -- the commit that did it
reads as a fix.

**A newly-added provenance field is a free, exact deploy marker.** 60.1 added
`full_report["_path"]`. No row before the deploy can carry it, so
`MIN(analysis_date) WHERE JSON_VALUE(...,'$._path') IS NOT NULL` dates the
deploy to the minute (last unstamped 2026-06-10 18:38Z, first stamped
2026-06-11 10:17Z) **without any log**. Use this whenever a step needs a deploy
time: look for a field the suspect commit introduced. It also moved the break
date one to four days earlier than the prior diagnosis had it.

**The zero-score series is a sharper break detector than the BUY rate** --
`0,0,0,0` then `5,3,7,4` where the BUY rate still showed 1 and 2 during the
mixed changeover day.

**The CC rail does not honour a Gemini structured-output config.** The full
pipeline runs `rail=claude_code` (257/259 rows). `_SYNTHESIS_STRUCTURED_CONFIG`
sets `response_schema`, `response_mime_type`, `max_output_tokens`,
`temperature` -- and per phase-78.1 the rail's `--json-schema` is **post-hoc
validated with re-prompting, not constrained decoding**, with no temperature or
output-token flag at all. So the synthesis has no schema guarantee and
`_parse_json_with_fallback` is the only thing between prose and the book. I
first hypothesised Gemini-2.5-pro thinking-budget truncation; that was REFUTED
by reading the live `deep_think_model` from the running process.

**A dark flag's own predicate can be validated against the live population.**
The phase-61.2 guard tests `synthesis.get("error") or "scoring_matrix" not in
synthesis`. Against production that is 211/211 sensitivity and 38/38
specificity. Before designing a new fix, test whether an existing dark one
already discriminates -- it is far stronger evidence than a unit test.

**One row signature, two different root causes, five weeks apart.** The same
`0.0`/`HOLD`/`''` shape ran at ~78% in mid-May from the
`final_score`-vs-`final_weighted_score` key bug (documented at
`autonomous_loop.py:2180-2189`), was fixed, then resumed 06-11 from an unrelated
cause. **A criterion asserting "the signature is gone" does not prove the
mechanism was fixed.** The clean baseline is only 14 days, and the full-path
zero-rate drifts unexplained (2026-06 97.0% -> 2026-07 96.0% -> 2026-08 51.5%),
so a single pre-window number cannot size an improvement.

**Sequencing hazard:** `_DOWNGRADE_RECS = {"HOLD","SELL","STRONG_SELL"}`
(`portfolio_manager.py:62,264`), so a fabricated `HOLD` on a HELD position is a
SELL trigger, not merely a non-buy. Arming
`paper_position_recommendation_fix_enabled` or
`paper_recommendation_vocab_fix_enabled` while
`paper_synthesis_integrity_enabled` is OFF converts 84% of analyses into sell
pressure on healthy holdings.

**The honest-absence shape is already plumbed** -- BQ nullable ->
`api/models.py:100` `Optional[float]` -> `types.ts:125` `number | null`, both
with comments saying NULL means degraded. Only the writer fabricates
(`autonomous_loop.py:2179`, `:2190-2192`, `:3639-3645`).

**How to apply:** when a dated regression shows no causal commit, look for a
commit that *enabled* something; and check whether a provenance field
introduced by the suspect commit can date its own deploy. Related:
[[project_cc_rail_vs_claudeclient_78_1]],
[[project_gemini_lifecycle_pipeline_restoration]],
[[project_decision_input_integrity_61_2]], [[project_dead_telemetry_429_86_38]].
