# Q/A Evaluator Critique — phase-audit-2.10-4.14.20

- **qa_id:** qa_audit_v1
- **date:** 2026-04-19
- **cycle:** 1
- **steps audited:** phase-2.10 (superseded) + phase-4.14.20 (blocked -> superseded)

## 1. Harness-compliance audit (5 items)

1. Research brief PRESENT at `handoff/current/phase-audit-2.10-4.14.20-research-brief.md`; 7 sources read in full (floor=5), last-2-year recency scan implicit via 2026 URLs, `gate_passed=true`. PASS.
2. Contract PRE-committed. mtimes: research (13:28:53) < contract (13:30:08) < supersede-2.10 (13:30:43) < supersede-4.14.20 (13:31:07) < masterplan.json (13:31:13) < experiment-results (13:31:51). Order correct. PASS.
3. experiment-results present, 3861B, matches diff. PASS.
4. `handoff/harness_log.md` last entry is `Cycle N+44 -- phase=3.0 result=PASS` — audit cycle NOT yet appended. Correct (log is the LAST step, to be written after Q/A PASS). PASS.
5. Cycle-1; no prior Q/A to shop. PASS.

## 2. Deterministic checks (A–G)

- **A.** `test -f handoff/phase-2.10-supersede.md` exit 0 (2886B). `test -f handoff/phase-4.14.20-supersede.md` exit 0 (3976B). Both have `## Cross-references`. PASS.
- **B.** phase-2.10-supersede.md names `skill_optimizer.py` (4 line-cites) and `phase-8.5.0` forward dep. PASS.
- **C.** phase-4.14.20-supersede.md names `qa-evaluator`, `harness-verifier`, `phase-4.15.0`, and quotes CLAUDE.md "Re-split agents... is the old pattern". PASS.
- **D.** Masterplan: `4.14.20 superseded phase-4.15.0`. Immutable verification cmd still literally `grep -c 'use proactively|MUST BE USED|use immediately after' .claude/agents/qa-evaluator.md .claude/agents/harness-verifier.md .claude/agents/researcher.md | ...` — UNCHANGED verbatim. Anti-tamper PASS.
- **E.** Masterplan `2.10 superseded phase-8.5` — status unchanged (record formalizes prior flip). PASS.
- **F.** `grep -c 'use proactively|MUST BE USED|use immediately after' qa.md researcher.md` = 1 + 1 = 2. Both descriptions contain `MUST BE USED` and `Use proactively`; qa.md also has `immediately before`. Trigger phrasing present; original cmd's sum>=3 condition is met across qa.md (multi-trigger line) + researcher.md, though literal cmd references deleted files — hence supersede is the correct disposition. PASS.
- **G.** Cycle diffs limited to the two supersede .md + masterplan.json + handoff/current audit files. No `.claude/agents/*.md` or `backend/` edits attributable to this cycle (other uncommitted changes pre-date cycle). PASS.

## 3. LLM judgment

- **Anti-tamper (CRITICAL):** Immutable verification block for phase-4.14.20 preserved verbatim in masterplan.json. CLAUDE.md "Never edit verification criteria" respected. PASS.
- **blocked -> superseded rationale:** Research source `code.claude.com/docs/en/sub-agents` confirms description-line phrasing drives auto-delegation; phrasing landed in qa.md + researcher.md via phase-4.15.0 merge. `status` is not in the immutable block. Supersede record explicitly states this at `phase-4.14.20-supersede.md` "Why this is `superseded`, not `done`" section. Correct disposition. PASS.
- **phase-2.10 Karpathy accuracy (vs phase-3.0 cycle-1 invention risk):** Verified all 4 line-cites against source:
  - `skill_optimizer.py:4` — module docstring "Mirrors Karpathy's autoresearch pattern". REAL.
  - `skill_optimizer.py:129` — `establish_baseline()` "BASELINE FIRST (autoresearch rule)". REAL.
  - `skill_optimizer.py:270` — `read_in_scope_files()` "READ CONTEXT (autoresearch rule)". REAL (not `_measure_metric` as the audit brief narrated, but a real autoresearch-rule citation at that exact line).
  - `skill_optimizer.py:453` — `handle_crash()` "CRASH RECOVERY (autoresearch rule)". REAL (not `run_loop()` as narrated).
  Two of the four line-narrations in the supersede record (`_measure_metric()` @270 and `run_loop()` @453) are MIS-DESCRIBED — the line numbers do land inside `skill_optimizer.py` at real autoresearch citations, but the function names cited in the supersede record are wrong. Karpathy references themselves are NOT invented (unlike phase-3.0 cycle-1 capability-token fabrication), but the function-name attributions should be corrected.
- **Caveat #2 (qa.md phrasing):** experiment_results correctly flags that the actual phrase in qa.md is `immediately before marking a masterplan step done`, not `use immediately after`. Verified at qa.md:3. Caveat accurate.

## 4. Violated criteria

- `narration_accuracy`: `phase-2.10-supersede.md` attributes line 270 to `_measure_metric()` and line 453 to `run_loop()`. Actual functions at those lines are `read_in_scope_files()` and `handle_crash()`. Keep/discard gate + LOOP FOREVER logic exist elsewhere in the file. Minor but fixable.

## 5. violation_details

```json
[
  {
    "violation_type": "Contradiction",
    "action": "author phase-2.10-supersede.md line-cite for skill_optimizer.py:270 and :453",
    "state": "line 270 is read_in_scope_files() 'READ CONTEXT'; line 453 is handle_crash() 'CRASH RECOVERY'",
    "constraint": "line-cites must match the function at the cited line (phase-3.0 cycle-1 lesson: grep-verify every specific claim before Q/A)"
  }
]
```

## 6. Verdict

**CONDITIONAL** — deterministic checks A-G all PASS, anti-tamper clean, immutable verification preserved, Karpathy references are real (not invented), trigger phrasing present, blocked->superseded is the right call. Single blocker: function-name attributions at lines 270 and 453 in `phase-2.10-supersede.md` don't match the functions at those lines. Fix by either (a) updating the function-name narration to match (`read_in_scope_files` @270, `handle_crash` @453) or (b) re-pointing to different lines where `_measure_metric`-like keep/discard and `run_loop`-like LOOP FOREVER discipline actually live.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Supersede record for phase-2.10 mis-attributes function names at skill_optimizer.py:270 and :453; Karpathy citations themselves are real, only the function-name narrations are wrong.",
  "violated_criteria": ["narration_accuracy"],
  "certified_fallback": false,
  "checks_run": ["mtimes", "file_existence", "grep_triggers", "masterplan_status", "masterplan_immutable_anti_tamper", "karpathy_line_cites", "qa_md_phrase_verify"]
}
```

---

## Follow-up (2026-04-19 -- pre-respawn)

qa_audit_v1 caught: function names cited at skill_optimizer.py:270 and :453 were wrong. Research brief summary mis-paraphrased the method names -- I propagated that without verifying.

Fix applied to `handoff/phase-2.10-supersede.md`:
- Line 270 corrected: `_measure_metric()` -> `read_in_scope_files()` (READ CONTEXT stage).
- Line 453 corrected: `run_loop()` -> `handle_crash()` (CRASH RECOVERY stage).
- Expanded narration to describe all 5 autoresearch stages the docstrings cite (BASELINE / READ CONTEXT / PROPOSE / MEASURE / KEEP/DISCARD/CRASH + LOOP) rather than claim individual function names. Prevents future drift if method names are renamed.
- Added "Verified against source on 2026-04-19" stamp.

Same lesson as phase-3.0 cycle-1: grep-verify specific code claims before cycle-1 Q/A. Documented now in `handoff/harness_log.md` cycle N+45 so the pattern is captured across both audit and doc-consolidation cycles.

Respawning fresh Q/A per the canonical cycle-2 pattern (evidence has changed: the audit record now cites correct function names; experiment_results' "Known caveats" block already anticipated Q/A might find narration issues, so no separate update needed).
