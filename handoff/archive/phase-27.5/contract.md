# Sprint Contract — phase-27.6 (E2E smoke verify on Claude)

Generated: 2026-05-17T00:25:00+00:00
Owner: Main
Step id: 27.6
Depends on: 27.5 (done — Gemini smoke PASSED). All upstream code fixes (27.1-27.5.2) are in place.

## Research-gate summary

Combined research from 27.0 covers all upstream provider quirks. No new external research needed.

## Hypothesis

Flipping `standard_model=claude-sonnet-4-6` + `deep_think_model=claude-opus-4-7` via the Settings API and running a cycle should produce the same outcome as cycle #8 did on Gemini: full path completes for all 14 in-scope tickers, ≥14 persist to BQ, cycle status=completed, Step 9 Learning fires.

The CRITICAL Claude-specific fix being exercised here is **27.1's `_ensure_additional_properties_false` schema injection** — every Claude structured-output call (synthesis, critic, debate, risk-judge) goes through `ClaudeClient.generate_content` and depends on the schema mutation to not raise `400 INVALID_ARGUMENT: additionalProperties must be explicitly set to false`. If 27.1 is correct, the cycle completes; if not, every ticker fails at the first structured-output call.

Falsifier: if the cycle fails on Anthropic schema errors that 27.1 was supposed to fix, the schema-mutation helper isn't recursing deep enough OR a different code path constructs `output_config.format.schema` without going through `ClaudeClient.generate_content`. Either gets flagged as a regression.

## Cost note

Claude Sonnet 4.6 input is ~20x more expensive per token than Gemini 2.5 Flash (~$3/Mtok vs $0.15). Cycle #8 cost $1.115 on Gemini. Claude estimate: $5-15 per cycle. With $1.115 already spent today, the $25 daily cap should comfortably fit one Claude cycle — but a second cycle today might trip the cap. Monitoring.

## Immutable success criteria (verbatim from `.claude/masterplan.json` step 27.6)

See `.claude/masterplan.json` step 27.6 `verification.command` (intentionally NOT quoted here per lessons from 27.5's spurious-grep-match incident — quoting the verbatim regex causes the doc to self-match).

Required intent: live_check_27.6.md exists + contains cycle_id + lite_mode False + persisted-count in 14-29 range.

## Plan steps

1. `PUT /api/settings/models {gemini_model: claude-sonnet-4-6, deep_think_model: claude-opus-4-7}`.
2. `launchctl kickstart -k gui/$UID/com.pyfinagent.backend` (rebuild orchestrator clients).
3. Confirm `/api/settings/models` shows the flip.
4. POST `/api/paper-trading/run-now`.
5. Wait for cycle completion.
6. Capture cycle metadata + BQ delta + log greps.
7. Write `handoff/current/live_check_27.6.md` honestly. Use the spelled-out-number convention from 27.5 to avoid regex self-match.
8. Q/A spawn.
9. harness_log append.
10. Flip 27.6 to done.

## Anti-patterns to avoid

- Do NOT quote the masterplan's verification regex verbatim in `live_check_27.6.md` (causes self-match like cycle #5).
- Do NOT silently re-route Claude to Gemini if a call fails — that would mask 27.1's correctness.
- Do NOT raise the daily cost cap mid-cycle to "fit" Claude — if cap trips, that's signal to either accept partial result or revisit Claude's per-token price.

## References

- `handoff/current/research_brief.md` §"C3 — Anthropic strict-mode schema" (the fix this step validates)
- `handoff/current/live_check_27.5.md` (Gemini canonical evidence for comparison)
- `backend/agents/llm_client.py:313-340` (the helper exercised end-to-end)
- `.claude/masterplan.json` phase-27 step 27.6 verification command (immutable)
