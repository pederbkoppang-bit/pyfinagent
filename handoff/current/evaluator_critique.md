# Claude as default LLM provider -- Evaluator Critique

**Cycle:** task #49 -- 2026-04-24
**Verdict:** PASS (single cycle, no respawn)
**Q/A agent:** qa

## Harness-compliance audit (5-item, all PASS)

1. Researcher before contract -- PASS. `claude-default-research-brief.md` gate_passed=true, 5 sources read-in-full, recency scan present.
2. Contract before code -- PASS. contract.md 19:42:31 predates settings.py 19:42:38, autonomous_loop.py 19:42:59, page.tsx 19:43:11.
3. experiment_results.md with verbatim output -- PASS.
4. Log-last -- PASS (harness_log.md not yet appended for task #49).
5. First-cycle Q/A -- PASS.

## Deterministic checks (all PASS)

| # | Check | Result |
|---|---|---|
| 1 | `gemini_model` default = `claude-sonnet-4-6` | PASS (grep 1) |
| 2 | `deep_think_model` default = `claude-opus-4-6` | PASS (grep 1) |
| 3 | `_run_claude_analysis` no hardcoded `model="claude-sonnet-4-6"` | PASS (grep 0) |
| 4 | Settings loads with Claude defaults | `std=claude-sonnet-4-6 deep=claude-opus-4-6` |
| 5 | autonomous_loop imports clean | OK |
| 6 | page.tsx contains "Claude is the default" banner | PASS (grep 1) |
| 7 | zero_orders drill PASSes | PASS |
| 8 | Frontend `npm run build` exits 0 | PASS |

## Mutation-resistance (both fired correctly)

A) **Tampered `gemini_model` default back to `gemini-2.0-flash`** --
   criterion 1 grep returned 0, FAIL detected as expected. Restored.

B) **Deleted "Claude is the default" banner from page.tsx** --
   criterion 6 grep returned 0, FAIL detected as expected. Restored.

## LLM judgment

- **Fallback-to-Gemini path verified.** `_run_single_analysis` lines
  359-362: `try: _run_claude_analysis` -> `except Exception` -> logs
  warning -> `AnalysisOrchestrator.run_full_analysis` (Gemini). A 401
  raises `anthropic.AuthenticationError` which inherits `Exception`,
  so the fallback engages. Monday's cycle will produce trades via
  Gemini even if the Anthropic OAuth-token 401 persists.
- **Banner accurate.** "Claude is the default" + Gemini switchable +
  3 Gemini-only features (RAG, grounding, structured-output schemas)
  matches the actual code behavior.
- **Scope honesty.** Deferring the OAuth-token-vs-API-key rotation to
  a manual Peder action is defensible because the Gemini fallback
  keeps Monday's cycle alive. Exactly the user's explicit request
  ("Monday's cycle should work via Gemini").

## Minor observation (not blocker)

The field name `gemini_model` now holds a Claude model string by
default. Harmless at runtime (routing is via `make_client`
model-prefix dispatch) but would confuse a future reader. A follow-up
cycle can rename the field to `standard_model`; not urgent.

## Violated criteria

None.

## Verdict

PASS. Main appends harness_log.md, commits + pushes, flips task #49 done.
