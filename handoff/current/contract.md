# Contract — cycle 13 / phase-43.0 DoD-14 closure (OWASP LLM04/05/09 explicit tagging)

**Cycle:** 13 | **Date:** 2026-05-28 | **Sub-step of:** phase-43.0 (P1, H) | **Author:** Main

---

## Research-Gate Summary

- Researcher subagent: `ab4ba0f2a92122dee`
- Brief: `handoff/current/research_brief_phase_43_0_dod_14_owasp.md`
- `gate_passed: true` — 10 external sources read in full (floor: 5), 21 URLs collected (floor: 10), recency scan performed, 3-variant queries per topic, 8 internal files inspected with file:line anchors.
- 4 SKILL.md edits drafted with verbatim ready-to-paste fragments (Edits A/B/C/D, confidence high/high/high/medium-high).
- Recency scan confirms NO new OWASP LLM Top-10 list since March 2025 (Q2 2026 OWASP work is Agentic ASI Top-10 + Red Teaming Landscape, NOT a revision of the LLM list).

## Hypothesis

Applying the 4 SKILL.md edits will (a) close DoD-14 of the phase-43.0 gate — all 10 LLM categories explicitly tagged with `LLM0X:2025` — and (b) materially improve coverage by adding a real LLM09 BLOCK heuristic (`llm-output-to-execution-without-validation`) rather than treating LLM09 as a doc-only tag.

## Immutable success criteria

This cycle does NOT have its own masterplan step (DoD-14 is a CRITERION of phase-43.0, not a step). The success criteria are therefore derived directly from the master_roadmap DoD-14 wording and the cycle-12 audit verdict.

1. `grep -E "LLM0[1-9]|LLM10" .claude/skills/code-review-trading-domain/SKILL.md` shows ALL 10 OWASP LLM categories explicitly tagged (LLM01 through LLM10).
2. The cosmetic "v2.0 (2025)" → "OWASP Top 10 for LLM Applications 2025" replacement landed at SKILL.md:100.
3. The LLM09 new heuristic includes a negation list that correctly characterizes the existing signal pipeline: (a) `insider_signal_screen.py` and `defense_signal.py` are fully deterministic (zero LLM API call matches in those files); (b) `pead_signal.py` IS LLM-driven (calls `ClaudeClient.generate_content`) but routes output through `response_schema` structured-output enforcement + Pydantic `model_validate` + explicit numeric clamping + fallback on error — those validators satisfy LLM09 prevention guidance. The negation list must call out the distinction so future Q/A spawns do NOT mis-exempt pead_signal.py as "deterministic" (caught by Q/A `a775b0e1987da8700` cycle-13 first pass; fixed in cycle-13 cycle-2).
4. No existing BLOCK/WARN/NOTE heuristic is degraded or removed by these edits — additive only.
5. `python -c "import ast; ast.parse(open('.claude/skills/code-review-trading-domain/SKILL.md').read())"` is NOT applicable (markdown, not Python); instead verify the file remains valid Markdown by reading it and confirming no orphaned table rows or broken fences.

**Verification commands (cycle 13):**
```bash
# All 10 LLM categories explicitly tagged (LLM04/05/09 now present)
grep -oE "LLM0[1-9]|LLM10" .claude/skills/code-review-trading-domain/SKILL.md | sort -u | wc -l   # expect: 10

# Cosmetic fix landed
grep -c "v2.0 (2025)" .claude/skills/code-review-trading-domain/SKILL.md   # expect: 0
grep -c "OWASP Top 10 for LLM Applications 2025" .claude/skills/code-review-trading-domain/SKILL.md   # expect: >=1

# LLM09 negation list correctly exempts deterministic signals
grep -c "pead_signal\|insider_signal_screen\|defense_signal" .claude/skills/code-review-trading-domain/SKILL.md   # expect: >=1

# Markdown still well-formed (no broken table)
python3 -c "import re,sys; t=open('.claude/skills/code-review-trading-domain/SKILL.md').read(); rows=re.findall(r'^\|.*\|$', t, re.M); cols=[r.count('|') for r in rows]; assert len(set(cols)) <= 5, f'inconsistent table columns: {set(cols)}'; print(f'OK: {len(rows)} table rows, column-count set={set(cols)}')"
```

## Plan Steps

1. **Apply Edit A (cosmetic)** at SKILL.md:100 — replace the "v2.0 (2025)" source line with the canonical "OWASP Top 10 for LLM Applications 2025" + March 12, 2025 release date + LLM04 repurposing note.
2. **Apply Edit C (LLM05 explicit tag + 5 sub-bullets)** at SKILL.md:81 — append `[LLM05:2025]` to the `insecure-output-handling` heuristic name and expand the detection cue to cover all 5 OWASP canonical sub-sinks (command injection / SQL / path traversal / SSRF / XSS).
3. **Apply Edit B (LLM04 sentinel)** — insert the new `llm04-training-code-added` NOTE-severity row to Dimension 1 table between `rag-memory-poisoning` (line 86) and `unbounded-llm-loop` (line 87). Add negation-list bullet.
4. **Apply Edit D (LLM09 new BLOCK)** — insert `llm-output-to-execution-without-validation [LLM09:2025]` row to Dimension 2 table after `paper-trader-broad-except` (line 118). Add negation-list bullet exempting the deterministic signal pipeline.
5. **Verify** — run all 4 verification commands above; confirm grep tally = 10 distinct LLM categories.
6. **Render `experiment_results.md`** — list edits applied, verbatim verification output, no-regression cross-check.
7. **Spawn Q/A** — single subagent, 5-item harness audit + grep tally + LLM judgment on heuristic quality (not just tag presence).
8. **Append `handoff/harness_log.md`** Cycle 13 block AFTER Q/A PASS, BEFORE any masterplan touch.
9. **Masterplan status policy** — phase-43.0 STAYS `pending` (DoD-14 closes but DoD-1/2/5/6/9 still open + DoD-7 partial). Cycle 13 IS a sub-cycle contributing to the gate; the gate-PASS flip is reserved for the final 43.0 re-audit cycle when all 14 PASS.
10. **Commit + push** — manual `git add -A && git commit && git push origin main` (no masterplan flip → no auto-push trigger). Subject: `docs(skill): cycle 13 closes DoD-14 -- OWASP LLM04/05/09 explicit tags`.

## What this cycle will NOT do

- NOT change underlying detection logic for LLM01-03, LLM06-08, LLM10 (already tagged; no functional change).
- NOT add a Q/A subagent change — the SKILL.md is preloaded into Q/A context at spawn per its frontmatter; future Q/A spawns will pick up the new heuristics automatically.
- NOT close DoD-1, DoD-2, DoD-5, DoD-6, DoD-7, DoD-9, DoD-11 — those are separate cycles per the goal directive.
- NOT flip phase-43.0 to `status=done` — gate-PASS is contingent on all 14 DoDs PASS, which is multi-cycle work.
- NOT touch `backend/` code or any execution path — pure doc-edit.

## Stop-condition contribution

This cycle closes one criterion (DoD-14) of the phase-43.0 production-ready gate. After cycle 13: cycle-12 audit count flips from "9 most-generous / 5 literal PASS" to "10 most-generous / 6 literal PASS" of 14. Remaining work: DoD-1 (phase-39.1 widening, owner-gated), DoD-2 (walk-forward instrumentation), DoD-5 (freshness wiring), DoD-6 (BQ probe), DoD-7 (Risk Judge runtime evidence), DoD-9 (5-cycle stability), DoD-11 (3 IDs in documented-deferral home — possibly closable as a doc-edit too).

## Anti-pattern check (per project auto-memories)

- `feedback_no_emojis` — no emojis in SKILL.md or any cycle artifact.
- `feedback_contract_before_generate` — this contract is being written BEFORE GENERATE.
- `feedback_log_last` — harness_log.md append AFTER Q/A PASS, BEFORE any masterplan touch.
- `feedback_qa_harness_compliance_first` — Q/A prompt opens with 5-item harness audit.
- `feedback_harness_rigor` — DoD-14 verdict was contested in cycle 12; this cycle resolves the contestation honestly (not hand-waving).
- `feedback_full_codebase_audit_before_changes` — verified LLM09 negation list against actual signal pipeline files (no LLM imports in pead/insider/defense signal modules).
- `feedback_never_skip_researcher` — researcher spawned + gate passed BEFORE contract.

## References

- `handoff/current/research_brief_phase_43_0_dod_14_owasp.md` (this cycle's research gate)
- `handoff/current/production_ready_audit_2026-05-28.md` (cycle 12 audit; DoD-14 evidence)
- `.claude/skills/code-review-trading-domain/SKILL.md` (target of edits)
- OWASP canonical: https://genai.owasp.org/llm-top-10/
- LLM04 canonical: https://genai.owasp.org/llmrisk/llm042025-data-and-model-poisoning/
- LLM05 canonical: https://genai.owasp.org/llmrisk/llm052025-improper-output-handling/
- LLM09 canonical: https://genai.owasp.org/llmrisk/llm092025-misinformation/
- TrustTrade (arXiv 2603.22567): https://arxiv.org/html/2603.22567v1
- `backend/services/paper_trader.py:85` execute_buy / `:299` execute_sell (LLM09 sinks)
- `backend/services/{pead,insider_signal_screen,defense}_signal.py` (LLM09 negation list — deterministic, no LLM calls)
