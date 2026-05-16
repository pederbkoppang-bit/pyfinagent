# Contract: phase-16.59 — Uplift Q/A with code-reviewer capabilities

**Date:** 2026-05-16
**Step:** 16.59 (phase-16)
**Priority:** P0
**Blocks:** 16.15 (Go/No-Go verdict)
**Research brief:** `handoff/current/research_brief_16_59.md` (gate_passed=true; 7 sources read in full; tier=complex)

## Research-gate summary

Max-tier research brief completed. 7 Tier-1/2 sources fetched in full:
1. [Anthropic Code Review docs](https://code.claude.com/docs/en/code-review) — multi-agent severity model (Important/Nit/Pre-existing), REVIEW.md customization, verification step against actual code behavior
2. [arXiv 2509.16533 EMNLP 2025 — Sycophancy Under Rebuttal](https://arxiv.org/abs/2509.16533) — LLMs flip verdicts under detailed-but-wrong rebuttals; simultaneous presentation mitigates
3. [arXiv 2404.18496 — AI-Powered Code Review](https://arxiv.org/html/2404.18496v2) — multi-agent specialization beats single-LLM; alert-fatigue is a real anti-pattern
4. [SurePrompts LLM-as-Judge Guide](https://sureprompts.com/blog/llm-as-judge-prompting-guide) — four structural biases (position/verbosity/self-preference/authority) + five mitigations; RCAF prompt structure; chain-of-thought grounding
5. [OWASP LLM Top-10 2025](https://www.invicti.com/blog/web-security/owasp-top-10-risks-llm-security-2025) — 4 new entries vs 2023; LLM02 (sensitive info) promoted to #2; LLM06 (excessive agency) + LLM07 (system prompt leakage) new
6. [Cloudflare AI code review at scale](https://blog.cloudflare.com/ai-code-review/) — explicit negation lists (what NOT to flag) are highest-leverage; coordinator-level reasonableness filter; security-sensitive paths get unconditional full review
7. [OWASP Top-10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — baseline v1.1 list

Internal code inventory: 7 files inspected (kill_switch.py, risk_engine.py, paper_trader.py, security.md, backend-services.md, risk_debate.py, limits_schema.py) with file:line citations.

## Hypothesis

Adding 5 dimensions of code-review heuristics + REVIEW.md-style explicit negation lists to `.claude/agents/qa.md` will make Q/A reject sycophantic-PASS, catch security/trading-correctness regressions in the actual diff (not just contract claims), and resist the empirically-measured "second-opinion-shopping" failure mode (58% sycophancy rate per SycEval 2025).

The upgrade is purely **additive** — no existing Q/A behavior is removed. The current 5-item harness-compliance audit (researcher, contract pre-commit, results, log-last, no-verdict-shopping per `feedback_qa_harness_compliance_first.md`) remains intact and runs FIRST.

## Immutable success criteria (verbatim from .claude/masterplan.json 16.59.verification.success_criteria)

1. `qa_md_contains_code_review_heuristics_section`
2. `section_covers_security_owasp_secrets_injection`
3. `section_covers_trading_domain_correctness_stops_sizing_risk_guard_kill_switch`
4. `section_covers_code_quality_idiomatic_python_types_test_coverage`
5. `section_covers_anti_rubber_stamp_on_financial_logic`
6. `additional_research_gate_findings_documented_in_qa_md`
7. `fresh_qa_subagent_post_session_restart_self_discloses_new_section`
8. `separation_of_duties_note_appended_to_harness_log_md_for_peder_review`

**Verification command:**
```
source .venv/bin/activate && bash scripts/qa/verify_qa_roster_live.sh && grep -nE 'code.review|owasp|secret|risk.guard|stop.loss|anti.rubber.stamp' .claude/agents/qa.md
```

**Live check:** Fresh qa subagent dispatched after session restart self-discloses the new code-review heuristics section by name in its first response.

## Plan steps

1. **GENERATE 1** — Edit `.claude/agents/qa.md` to add a new top-level section "Code review heuristics (phase-16.59)" with sub-sections per dimension. The section MUST:
   - Reference the brief at `handoff/archive/phase-16.59/research_brief_16_59.md` (path after step archive)
   - List the top-15 heuristics from the executive summary verbatim
   - Add 5 dimension sub-sections (Security, Trading-domain, Code quality, Anti-rubber-stamp, LLM-evaluator anti-patterns) with the heuristic tables from the brief
   - Add an explicit **"What NOT to flag" negation list** (Cloudflare pattern) per dimension — reduces false positives
   - Add a **severity dispatch rule**: BLOCK heuristics auto-FAIL; WARN heuristics force CONDITIONAL; NOTE heuristics PASS-with-flag
   - Add a **simultaneous-presentation rule** for cycle-2 spawns (per arXiv 2509.16533) — fresh Q/A reads BOTH the updated handoff files AND the previous verdict in one shot, not sequentially
   - Add a **3rd-CONDITIONAL auto-FAIL counter** check as one of the heuristics (already in CLAUDE.md but Q/A must explicitly enforce it)
   - Cite each dimension to at least one Tier-1 source URL

2. **GENERATE 2** — Append separation-of-duties note to `handoff/harness_log.md` per CLAUDE.md "Separation of duties on agent edits". Note must:
   - State that THIS session both authored the qa.md edit
   - Require Peder review of the diff BEFORE 16.15 starts
   - Require session restart + `scripts/qa/verify_qa_roster_live.sh` before next cycle
   - Be visible in the next `/masterplan` view (cross-link)

3. **VERIFY** — Run the verification command. Capture verbatim output in `handoff/current/experiment_results.md`.

4. **Q/A SPAWN** — Spawn fresh Q/A to evaluate this GENERATE. Q/A scope: did the edit match the research-gate findings and the 8 success criteria? (NOT whether the new Q/A is "better" — that is tested by the live_check after session restart, which is outside this step's session.)

5. **LOG** — Append cycle entry to `handoff/harness_log.md` BEFORE the masterplan status flip to `done`.

## References

- Research brief: `handoff/current/research_brief_16_59.md`
- CLAUDE.md: harness protocol, separation-of-duties on agent edits, session-restart requirement, 3rd-CONDITIONAL auto-FAIL rule
- `.claude/rules/research-gate.md`: gate floor (≥5 sources read in full)
- `.claude/agents/qa.md`: current Q/A definition (to be edited)
- `scripts/qa/verify_qa_roster_live.sh`: live-check script for post-restart roster verification
- Auto-memory: `feedback_harness_rigor.md`, `feedback_qa_harness_compliance_first.md`

## Out of scope (not changed in this step)

- Q/A's existing 5-item harness-compliance audit (researcher / contract-pre-commit / results / log-last / no-verdict-shopping) — preserved verbatim
- Researcher.md or Main.md — only qa.md is edited
- Any actual code review of historical PRs — the upgraded Q/A will be tested on the NEXT step's verdict (16.15)
- Frontend or backend code — no application code changes
