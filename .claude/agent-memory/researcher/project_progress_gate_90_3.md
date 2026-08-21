---
name: progress-gate-90-3
description: Step 90.3 progress-gated retry -- the proposed digest file set contains the gate's OWN output (self-vacuous); 90.1's backfill already ran while 90.1 is still pending; the digest is the WEAK signal per measured literature; three denominators drifted again
metadata:
  type: project
---

Research for masterplan step 90.3 (progress-gated retry: exempt rail drops, never read a
digest as convergence), measured 2026-08-21.

**SELF-REFERENCE HAZARD -- the finding 90.3's own audit_basis does not name.** Criterion 1
derives the digest file set from `git diff --name-only HEAD` union
`git ls-files --others --exclude-standard`. Run live, that set was exactly
`handoff/audit/attempt_budget_audit.jsonl` (written BY the gate, and TRACKED -- confirmed with
`git ls-files --error-unmatch`), `handoff/audit/pre_tool_use_audit.jsonl` (written by the
PreToolUse danger guard on EVERY tool call), and the brief itself. Unless the "checked-in root
allowlist" excludes `handoff/audit/`, the gate's own append mutates its own next input and the
digest advances by construction -- 89.1's exact defect through a different door. Corollary: after
Main commits a cycle's fixes the diff can be EMPTY, and an empty set hashes identically for two
different evidence states. No masterplan step has a `files`/`paths` key (checked the step-level
key union across the whole plan), so criterion 1's "declared masterplan paths" has no existing
source.

**Why: ** a gate whose input includes its own output is the Kubernetes self-referential webhook
deadlock; it is satisfiable on paper and vacuous in fact.
**How to apply:** any future work on `scripts/harness/attempt_gate.py` that hashes evidence must
state its allowlist explicitly and exclude hook-written audit streams.

**90.1's backfill HAS ALREADY RUN while step 90.1 is `status: pending`.** The live
`handoff/audit/attempt_budget_audit.jsonl` carries `outcome` + `total_tokens` keys (92 of 118 rows
have an int token count), and `attempt_gate.py:87-89` imports `resolved_rows`, called at `:243`.
So 90.3's "LAND AFTER 90.1" prerequisite is satisfied on disk even though the step is parked.
Do not infer landed-ness from `status`. See [[project_phase90_accounting_and_the_relocating_seam]].

**The measured literature INVERTS the naive intuition about hashing.** CUDABeaver
(arxiv.org/html/2605.08455) ships four stagnation signals with per-model firing rates: the two
SHA-256 signals `duplicate_code` (0.0-50.8%) and `code_cycle` (0.7-3.8%) versus the SEMANTIC
`no_progress` -- "(category, primary error signature) unchanged across three consecutive
iterations" -- at 44.6-84.6%, dominant. The hash catches only the degenerate case. Optimal-stopping
theory (arxiv.org/html/2608.10729v1) stops on an ABSOLUTE score and explicitly declines
inter-iteration change. Anthropic's harness-design article says NOTHING about when to stop
retrying -- never cite it for a ceiling.

**Three denominators drifted AGAIN.** verdict_ledger.jsonl: 89.1 said 134, 90.3's audit_basis says
138, live is **146** (NO_VERDICT still 16 = 11.0%). attempt ledger: 90.1 says 92, CLAUDE.md says
93, live is **118** (114 attempts + 4 extensions; 20 NO_VERDICT = 17.5% of attempts). Also 89.1's
"perfect 1:1" commit claim reproduces exactly for 86.118 only (4/4); 86.116 is 5/6 and 86.108 is
2/3 -- the direction holds, the pairing does not. See
[[feedback_url_count_must_be_re_derived]] for the same re-derive discipline.

**Criterion 5's grep test is ~all noise as specified:** `grep -rIn 'digest' .claude/ scripts/`
returns 1086 lines across 111 files, overwhelmingly `.claude/.masterplan.json.bak.*` snapshots
containing 89.1's and 90.3's own criteria text -- the checker matches its own specification.
Same over-report class as [[project_phase_tag_in_ui_91_9]].

**The exemption should key on the REASON, not the value.** `attempt_outcomes.py:227-241` already
splits NO_VERDICT into `killed` / `args_unparseable` / `structured_output_drop` /
`not_an_evaluation` / `completed_without_result` / `no_verdict_other`. Temporal's definition --
"permanent failures, by definition, require you to make some change to your logic or your input"
-- puts only the genuine transients in the exemption; `not_an_evaluation` is NOT a rail drop.
Brief: `handoff/current/research_brief_90.3.md`.
