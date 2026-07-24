# Contract — phase-75.4.2: skill_optimizer post-write delivery invariant

- **Step id:** 75.4.2 (phase-75 follow-up queue, **P1**; executor: **sonnet-tagged → delegated Sonnet executor GENERATE** per the operator tiering directive + the overnight delegated-executor model; Main reviews + runs the mutation matrix; gates opus/max via Workflow)
- **Date:** 2026-07-24
- **Boundary:** exactly ONE code file (`backend/agents/skill_optimizer.py`) + ONE new test file. Do NOT touch `backend/config/prompts.py` (the 75.4-fixed loader), any `skills/*.md`, or the phase-71.4 flag-gated review. The invariant is UNCONDITIONAL (no flag gate) and $0 (deterministic, no LLM).

## Research-gate summary (gate PASSED — wf_91b00dc2-3ea)

Envelope: `tier=moderate, external_sources_read_in_full=6, snippet_only_sources=11, urls_collected=17, recency_scan_performed=true, internal_files_inspected=8, gate_passed=true`. Brief: `handoff/current/research_brief_75.4.2.md` (contains VERBATIM implementation code in §4a/§4b — the executor follows it).

Load-bearing findings:

1. **The gap:** `apply_modification` (skill_optimizer.py:406-479) rewrites skills/*.md with only two guards: occurs-exactly-once (:424) and post-write `load_skill()` doesn't raise (:463-470). A `###`→`##` heading promotion loads fine while silently truncating delivery (the loader's region regex stops at the first `## ` — the exact 75.4 regression, 7532→190 chars, re-openable by the optimizer).
2. **Silent-degradation path today:** broken skill loads → returns True → `_git` commits → `reload_skills` makes it live → every subsequent analysis delivers the truncated prompt; metric delta usually 0 → PENDING (:813) → persists, never auto-reverts.
3. **Fix:** deterministic, ALWAYS-ON, fail-CLOSED postcondition: snapshot `delivered_before = load_skill(agent)` pre-write; post-write compare via `_delivery_invariant_ok`: (a) placeholder-set subset guard (`re.findall(r'\{\{(\w+)\}\}')` — no delivered placeholder may vanish), (b) length-retention guard (`DELIVERY_MIN_RETAIN_RATIO = 0.80`). Two independent guards so each is mutation-killable. Revert uses the in-function byte-exact `skill_path.write_text(content)` (:467 pattern) — NEVER `revert_modification()` (that does `git checkout HEAD~1`). `import re` must be added at module top (currently only function-local).
4. **Vacuous-test trap (measured):** `{{quant_model_data}}` occurs TWICE in quant_model_agent.md (prose :27 + template :79) — a bare placeholder old_text is rejected by the PRE-EXISTING occurs-once guard, so that fixture would stay green under mutation. T2 must use the unique 2-line old_text `'### Quant Model Data\n{{quant_model_data}}'`; every fixture asserts `content.count(old_text) == 1`.
5. **Canonical fixture:** quant_model_agent.md — `## Prompt Template` (:72), placeholders :73/:74/:79, `### Quant Model Data` (:78), region ends at `## Experiment Log` (:123). Promoting :78 to `## ` trips BOTH guards.
6. **Test construction:** `SkillOptimizer.__new__(SkillOptimizer)` (skips __init__/BQ creds; apply_modification uses no self state); monkeypatch `SKILLS_DIR` in BOTH `backend.agents.skill_optimizer` AND `backend.config.prompts`, plus `prompts._SKILL_FILE_ID_CACHE_PATH` → tmp and `skill_optimizer._git` → no-op. Stubbing _git/cache is fine — the criterion's "not a stub" refers to the SKILL FILE (real temp copy) + REAL apply_modification/load_skill.
7. **External canon (6 read in full):** Meyer DbC (postcondition violation = fail-fast), DSPy Assert fail-closed vs Suggest fail-open (arXiv:2312.13382), APE/OPRO score-before-adopt (2211.01910/2309.03409), VeriGuard verify-before-execute (2510.05156), Governed Capability Evolution 2026 — a self-evolved artifact is a CANDIDATE that must pass or ROLL BACK, "zero unsafe activations" (2604.08059).

## Hypothesis

An unconditional post-write delivery postcondition (placeholder-subset + 80% length-retention, byte-exact revert on violation, checked BEFORE the git commit) closes the optimizer's silent-truncation hole fail-closed, while the negative control proves it does not blanket-refuse legitimate prose edits.

## Execution model

Sonnet executor implements EXACTLY the brief's §4a/§4b code + the 4-test suite (T1 heading-promotion revert + sha unchanged; T2 unique-2-line placeholder drop revert + byte-identical; T3 negative control prose edit accepted + placeholders intact; T4 length-guard-only trip). Main re-derives lint scope, runs the mutation matrix (M1 remove invariant call; M2 weaken helper to `return True,'ok'`; M3 drop placeholder guard; M4 drop length guard; + a fixture mutation M5: break T2's unique old_text back to the bare placeholder → its own count==1 assert must fail), assembles live_check, spawns Q/A.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.4.2)

> command: `cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_4_2_optimizer_invariant.py -q`

1. "New backend/tests/test_phase_75_4_2_optimizer_invariant.py passes offline and calls the REAL apply_modification against a temp copy of a skill file (not a stub): a modification that promotes a '### ' heading back to '## ' inside the Prompt Template region is REVERTED and apply_modification returns False"
2. "A modification that drops a {{placeholder}} from the delivered template is REVERTED; the file on disk is byte-identical to its pre-call content"
3. "A legitimate modification that changes only body prose is ACCEPTED and written -- proving the guard is not blanket-refusing (negative control)"
4. "Mutation matrix recorded in experiment_results.md: removing the invariant check, and weakening it to a load_skill()-succeeds check only, each fail at least one test"

live_check spec (verbatim): "handoff/current/live_check_75.4.2.md: verbatim verification command output (exit 0) + git diff --stat + evidence the guard fails CLOSED (a rejected write leaves the file byte-identical, shown by hash before/after)."

## References

- `handoff/current/research_brief_75.4.2.md` (verbatim implementation §4a/§4b; 6 read-in-full incl. DSPy/APE/OPRO/VeriGuard/DbC)
- skill_optimizer.py:406-479 (:416 content snapshot, :424 occurs-once, :431-450 dark 71.4 review, :463-470 load-check, :467 revert pattern), prompts.py:196/:209, quant_model_agent.md:72-123
- 75.4 doctrine: test_phase_75_skill_delivery.py:22-27 (real-loader assertions, per-case functions, paired negative controls)
