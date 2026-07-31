# Contract — phase-81.0 — Harness continuity repair

**Written:** 2026-07-31, BEFORE any GENERATE work (per `feedback_contract_before_generate`).
**Status at write time:** masterplan `81.0` = `pending`.

---

## 1. Research gate summary

Researcher run via the Workflow rail (operator directive 2026-07-27: both dev-MAS agents on
Workflows). Brief: `handoff/current/research_brief_81.0.md` (25,787 B, written incrementally).

Envelope returned: `tier=complex`, `external_sources_read_in_full=9`,
`snippet_only_sources=26`, `urls_collected=35`, `recency_scan_performed=true`,
`internal_files_inspected=20`, **`gate_passed=true`**.

**The gate passed on research quality and then returned a NOT-SAFE-AS-WRITTEN verdict with ten
blockers.** Every measurement in the original 4-change plan reproduced (132,714-char render;
92-of-262 shown; 24 verdict-gate hits; a `Next actionable` naming a step from a phase the render
never printed) — but six of the *fixes* were wrong or incomplete, two of them silent no-ops.

Blockers and their disposition:

| # | Blocker | Disposition in this contract |
|---|---|---|
| B1 | A1 is a silent no-op: all three gate dispatches end in `proceed\|*)`. Researcher **executed** the case block with 6 tokens — `noinput`/`warn`/`""` all fall through silently. | FIXED — A1 is a two-file change; a `no_input)` arm is added **before** the catch-all. Criterion 2. |
| B2 | Root cause was wrong. Six `evaluator_critique_<id>.json` exist, written **2026-07-26 — after** the 07-24 sweep. Convention drifted rolling → per-step; hook hardcodes rolling at `:200`. | FIXED — hook resolves per-step first, rolling as fallback. Criterion 3. |
| B3 | `ROLLING_KEEP` omits `.json` in **both** housekeeping scripts; backfill is documented idempotent, so it re-sweeps the restored file. | FIXED — `.json` added to both. Criterion 4. |
| B4 | WARN lands in `handoff/logs/` (gitignored `.gitignore:76`) and PostToolUse stdout is not shown in transcript — the alarm is itself silent. | FIXED — emit `systemMessage`. Part of criterion 2's intent. |
| B5 | Adding a 4th token makes pending **75.5.9**'s immutable "contract identical to verdict_gate.py" criterion ambiguous. | **OPERATOR DECISION 2026-07-31 (AskUserQuestion):** add the token; 75.5.9 **inherits** the 4-token contract. Recorded in `harness_log.md`. No criteria edited. |
| B6 | A4 enable is owner-gated by done step **38.4** (`owner_approval_recorded_before_enabling_the_gate`). | **OPERATOR DECISION 2026-07-31:** build a warn token, **ship OFF**. `HARNESS_LOG_GATE_ENABLED` is NOT set, so 38.4's criterion is untouched. |
| B7 | "WARN-mode first" is not expressible — gate emits only `proceed\|passed\|skip`; enabling jumps straight to HOLD, and `skip` exits before `git add -A`. | FIXED — a real `warn` token + shell arm that logs and CONTINUES. Criterion 5. |
| B8 | 200-line tail holds only ~2 cycle blocks of a 29,823-line file → false `skip`. | FIXED — widen the tail. |
| B9 | Leak is not `deferred`-specific. **Three** phases leak: phase-5 (11 open), phase-36 (16 open), **phase-77 — `status` key absent → `None`** (3 open). | FIXED — exhaustive partition (`ACTIVE = complement of DONE`), not a string special-case. Criteria 6–7. |
| B10 | Deleting `sessions/` orphans the boot directive at `known-blockers.md:56`; that file has a **constructed-path** consumer (`incident_log_p0_test.py:23`) parsing `## RESOLVED` / `## STILL ACTIVE`. | FIXED — directive repaired in the same change; heading structure preserved; no new `P0` string introduced. Criterion 8. |

**Framing correction accepted from research (not a blocker):** Anthropic's `verify-gate.sh` fails
*closed* (`if [ ! -s "$log" ]` → `{"decision":"block"}`, read verbatim) because it is a **PreToolUse**
hook on the agent's *claim*. Ours is **PostToolUse** on *delivery*, which the official hooks
reference states "Can block? No — the tool already ran." Our fail-open is **forced by the event
type, not chosen.** This contract therefore does **not** attempt to make the PostToolUse hook
stricter; it makes its silence audible. A fail-closed PreToolUse gate on the `status:"done"` Write
is a separate, larger idea and is explicitly **out of scope**.

**Scope guards from research:** do **not** touch `live_check_gate.py` (owned by pending 75.5.10);
do **not** set `HARNESS_LOG_GATE_ENABLED`.

**Verified-safe, do not over-engineer:** `settings.json:171` `Write(.claude/context/sessions/**)` is
preceded by a bare `"Write"` at `:170`, so removing it is a no-op on effective permissions;
`push-credential-diagnosis.md` has zero consumers.

---

## 2. Hypothesis

The harness does not lack checks — it has at least seven controls that return the same answer on
every branch. The failure mode is **controls whose silence is indistinguishable from success**.
Two of those silences are load-bearing (a verdict gate that has had no input since 2026-07-20; a
log gate that has never been armed), and the boot command every cold session runs reports a
`Next actionable` step it never printed.

If a control's *no-input* and *not-armed* states become distinguishable in the log **and** visible
to the operator, and the boot render accounts for every status exhaustively, then a future session
can tell "this was checked and passed" from "nothing checked this" — which is the precondition for
every other repair in the queue being trustworthy.

**This step ships no new gate that can hold a push.** Every behaviour change is either
observability (A1, A4) or correctness of a report (A2) or deletion (A3).

---

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json` 81.0)

1. `verdict_gate.gate_decision returns 'no_input' when the JSON is ABSENT, and still returns 'proceed' for unreadable / non-dict / step-mismatch / no-verdict-field, and 'hold' for a step-matched non-PASS`
2. `auto-commit-and-push.sh contains a 'no_input)' case arm positioned BEFORE the 'proceed|*)' catch-all in the verdict-gate dispatch, and a 'warn)' arm before the catch-all in the harness_log dispatch`
3. `the verdict-gate dispatch resolves a per-step handoff/current/evaluator_critique_<step_id>.json when present, falling back to the rolling evaluator_critique.json`
4. `both scripts/housekeeping/backfill_handoff_archive.py and scripts/housekeeping/verify_handoff_layout.py list the evaluator_critique .json name in ROLLING_KEEP`
5. `harness_log_gate.gate_decision can return a 'warn' token and HARNESS_LOG_GATE_ENABLED remains unset in .claude/settings.json env`
6. `the /masterplan render names no next-actionable step from a phase it did not print, and emits an explicit count of open steps not shown`
7. `phase-77 (status key absent) and phase-36 (deferred) are both accounted for by the render's status partition -- not silently dropped`
8. `.claude/context/sessions/ and .claude/context/push-credential-diagnosis.md are gone, known-blockers.md carries no dangling reference to the deleted session-log path, and incident_log_p0_test.py still reports 6/6`
9. `MUTATION EVIDENCE: for each new token, a fixture with the defect PRESENT is shown going red, and reverting the fix is shown restoring the old silent behaviour`

**Verification command (immutable):**
```
source .venv/bin/activate && python -m pytest backend/tests/test_phase_71_3_verdict_gate.py backend/tests/test_phase_38_4_hook_gate.py -q && bash .claude/hooks/lib/harness_log_gate_test.sh && python3 scripts/go_live_drills/incident_log_p0_test.py && python3 scripts/housekeeping/verify_handoff_layout.py
```

---

## 4. Plan

1. **A1a** `verdict_gate.py` — split the file-absent branch out of the fail-open cluster; return
   `no_input`. All other current `proceed` paths unchanged.
2. **A1b** `backend/tests/test_phase_71_3_verdict_gate.py` — `test_missing_json_fails_open_proceed`
   asserts `== "proceed"` today. This is a **deliberate consumer-contract change**, so the test
   changes *with* the behaviour and is renamed to say what it now pins. Every other test unchanged.
3. **A1c** `auto-commit-and-push.sh` — resolve per-step JSON then rolling; add `no_input)` arm
   before the catch-all; emit `systemMessage`.
4. **A1d** both housekeeping scripts — add the `.json` name to `ROLLING_KEEP`.
5. **A4** `harness_log_gate.py` — add `warn`; widen the tail; shell `warn)` arm. Do **not** enable.
6. **A2** `.claude/skills/masterplan/SKILL.md` — exhaustive partition, denominators, no
   next-actionable from a suppressed phase, strike the two `TaskCompleted` references.
7. **A3** delete `push-credential-diagnosis.md` + `sessions/`; repair `known-blockers.md:56`;
   drop the dangling `settings.json:171` permission.
8. **Mutation matrix** (criterion 9) — run AFTER the code is final, per
   `feedback_executor_sees_mutation_transients`.

**Out of scope, stated so the Q/A does not ask for it:** `live_check_gate.py` (75.5.10);
enabling the log gate (38.4 owner gate); the `masterplan.json` name-field schema migration; a
fail-closed PreToolUse gate on the status flip; the archive misattribution sweep (75.11.4).

---

## 5. References

- `handoff/current/research_brief_81.0.md` — the gate brief (9 sources read in full).
- `https://github.com/anthropics/cwc-long-running-agents` + `verify-gate.sh` — default-FAIL
  contract; read verbatim; the PreToolUse-vs-PostToolUse distinction above.
- `https://code.claude.com/docs/en/hooks` — "PostToolUse | Can block? No"; `systemMessage` is a
  documented PostToolUse output.
- `https://www.anthropic.com/engineering/harness-design-long-running-apps` — hard-threshold-or-fail;
  partial extraction, disclosed by the researcher.
- IEC 61511 proof-test material — a silently-failed control is a Dangerous-Undetected failure;
  motivates the follow-up proof-test step noted below.
- Operator decisions 2026-07-31 (AskUserQuestion): warn-token-first for A4; 4-token inheritance
  for 75.5.9.

**Queued follow-up (research: "MISSING FROM THE PLAN ENTIRELY"):** there is no periodic proof test
asserting each gate still receives real input. Without one this repair has the same half-life as
the last. To be filed as its own research-gated step per
`feedback_queue_discovered_defects_in_masterplan` — NOT absorbed into 81.0.
