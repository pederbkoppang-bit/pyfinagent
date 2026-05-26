# Evaluator Critique -- Cycle 3: Claude Code CLI routing layer (2026-05-26)

**Verdict:** PASS
**Reviewer:** Q/A subagent (merged qa-evaluator + harness-verifier)
**Class:** trading-policy-adjacent (LLM rail change). Citation gate APPLIED.
**Mode:** First Q/A spawn for cycle 3. OVERWRITE of the cycle-2 BLOCKED-state-27.6 critique is the correct cycle-boundary behavior (different cycle, different evidence; not verdict-shopping).

---

## Harness-compliance audit (5 items)

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher spawn | PASS | `handoff/current/research_brief_phase_claude_code_routing.md` exists (mtime 26 mai 23:06). `contract.md:16-17` cites "Researcher `aff3444de945e98c2`, tier=deep, 24 sources read in full, 34 URLs collected, 3 adversarial sources, recency scan performed, gate_passed=true". |
| 2 | Contract pre-commit | PASS | `contract.md` mtime 23:08:26 PRECEDES every modified source: settings.py 23:08:52, llm_client.py 23:09:57, claude_code_client.py 23:10:54, autonomous_loop.py 23:11:47, test_claude_code_client.py 23:12:27, experiment_results.md 23:13:25. FIFTH-occurrence preamble present at `contract.md:8` documenting the five overwrites by the autonomous-loop sprint contract at 19:56 / 20:36 / 20:47 / 22:47 / 21:02:13 UTC. |
| 3 | experiment_results.md | PASS | Exists (mtime 23:13:25). Lists 1 NEW file (`claude_code_client.py`) + 1 NEW test (`test_claude_code_client.py`) + 3 MODIFIED (`settings.py`, `llm_client.py`, `autonomous_loop.py`) + 1 settings field (`paper_use_claude_code_route`). Verbatim pytest (11 passed; 33 passed regression) + AST (4x exit 0) + grep counts present at `experiment_results.md:42-69`. |
| 4 | harness_log absence | PASS | `grep "Cycle 3 -- 2026-05-26" handoff/harness_log.md` returned 0. Log-LAST discipline respected; cycle-3 log append happens AFTER this Q/A PASS. |
| 5 | No verdict-shopping | PASS | Prior `evaluator_critique.md` (mtime 22:57) is the cycle-2 BLOCKED-state-27.6 critique (different evidence, different cycle). This is the FIRST cycle-3 Q/A spawn. OVERWRITE is the documented cycle-boundary behavior, not second-opinion-shopping on unchanged evidence. |

---

## Deterministic checks

```
$ pytest backend/tests/test_claude_code_client.py -v 2>&1 | tail -5
backend/tests/test_claude_code_client.py::test_claude_code_client_class_returns_empty_on_error PASSED [100%]
============================== 11 passed in 0.18s ==============================

$ pytest backend/tests/ -k "llm_client or autonomous_loop or claude_code" 2>&1 | tail -3
================ 33 passed, 597 deselected, 1 warning in 2.46s =================

$ python -c "import ast; ast.parse(open('backend/agents/claude_code_client.py').read())"  -> exit 0
$ python -c "import ast; ast.parse(open('backend/config/settings.py').read())"             -> exit 0
$ python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"    -> exit 0
$ python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read())"           -> exit 0

$ grep -c "paper_use_claude_code_route" backend/config/settings.py        -> 1   (>=1 OK)
$ grep -c "paper_use_claude_code_route" backend/agents/llm_client.py      -> 3   (>=1 OK)
$ grep -c "paper_use_claude_code_route" backend/services/autonomous_loop.py -> 2 (>=1 OK)
$ grep -c "claude_code_invoke" backend/agents/claude_code_client.py       -> 12  (>=2 OK)
$ grep -c "rail=" backend/services/autonomous_loop.py                     -> 1   (>=1 OK)

$ git diff --stat HEAD -- frontend/             -> empty (0 frontend changes)
$ git diff HEAD -- frontend/package.json        -> empty (0 new deps)
$ git diff HEAD -- frontend/package-lock.json   -> empty (no lock-churn)
```

All deterministic checks pass.

---

## Code-review heuristics (5 dimensions, phase-16.59 skill)

Order respected: harness-compliance (5-item) -> deterministic -> code-review -> LLM judgment.

- **Dim 1 Security:** No `secret-in-diff`. No `prompt-injection-path` (the subprocess is locked down by `--disallowedTools "Bash,Edit,Write,Read,Glob,Grep,Agent"` at `claude_code_client.py:47`, blocking every side-effect tool). `subprocess.run` uses a list arg with default `shell=False` -- safe. No new `eval`/`exec`/`os.system`. No supply-chain dep changes. `system-prompt-leakage`: NEGATIVE (the `system=` param at `llm_client.py:1548` is passed directly to `claude_code_invoke`, not serialized to a response/log). `rag-memory-poisoning`: N/A. `unbounded-llm-loop`: NEGATIVE (no new loops; existing autonomous-loop bounds preserved).
- **Dim 2 Trading-domain:** `kill-switch-reachability` PASS (the rail-swap is upstream of the execute path; kill-switch wiring untouched). `stop-loss-always-set` PASS (no `paper_trader.py` change). `perf-metrics-bypass` PASS (no Sharpe/drawdown/alpha math added). `max-position-check-bypass` PASS. `paper-trader-broad-except` PASS (the `ImportError` catch at `llm_client.py:1903-1907` is a defense-in-depth fallthrough, not a risk-guard swallow). `crypto-asset-class` PASS.
- **Dim 3 Code quality:** ASCII-only logger messages confirmed in `claude_code_client.py`. `broad-except` NEGATIVE -- `claude_code_client.py:104-119` catches specific `TimeoutExpired`/`FileNotFoundError`; `llm_client.py:1903` catches specific `ImportError`. No `print()` in non-script code. Type hints present on the new `claude_code_invoke` public API. Test-coverage delta: 1 new file + 11 tests is proportionate.
- **Dim 4 Anti-rubber-stamp:** `financial-logic-without-behavioral-test` PASS -- routing changes are paired with 11 behavioral tests that hit subprocess.run mocks at the external boundary. Tests are NOT tautological (envelope-parsing, error propagation, structured-output extraction precedence -- real assertions). `over-mocked-test`: borderline-but-acceptable -- only the external boundary (`subprocess.run`) is mocked, not the module under test. `rename-as-refactor` PASS.
- **Dim 5 Evaluator anti-patterns:** `sycophancy-under-rebuttal` PASS (no verdict-flip-without-code-change). `second-opinion-shopping` PASS (first cycle-3 spawn; cycle-2 BLOCKED-state-27.6 is different evidence). `missing-chain-of-thought` PASS (this critique cites file:line throughout). `3rd-conditional-not-escalated` N/A (no prior CONDITIONAL chain for this step-id). `criteria-erosion` PASS (all 19 immutable criteria at `contract.md:116-136` evaluated).

No BLOCK or WARN findings.

---

## LLM judgment (A-L)

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| A | Citation gate (>=2 AI-in-trading + >=2 academic) | PASS | `contract.md:18` cites TradingAgents `arXiv:2412.20138` + Portkey AI Gateway (2 AI-in-trading, floor 2 met). `contract.md:19` cites Bailey/Borwein/LdP/Zhu PBO `SSRN:2326253` + Harvey/Liu/Zhu NBER w20592 + Yin et al. `arXiv:2603.20319` (3 academic, floor 2 exceeded). All five citation strings physically present in `contract.md`. |
| B | Feature flag DEFAULTS OFF | PASS | `settings.py:112` -> `paper_use_claude_code_route: bool = Field(False, ...)`. Existing Anthropic-direct path preserved at `llm_client.py:1910-1912` (the existing `ClaudeClient` branch fires only when the new CC branch is skipped). |
| C | Defense in depth on import | PASS | `llm_client.py:1897-1907` -- the `from backend.agents.claude_code_client import ClaudeCodeClient` is wrapped in `try` and the `ImportError` is caught with `logger.warning(...)`; control falls through to the Anthropic-direct branch. |
| D | Per-rail log present | PASS | `autonomous_loop.py:1445-1449` -> `logger.info("Analysis ticker=%s rail=%s", ticker, "claude_code" if use_claude_code_route else "anthropic_direct")`. ASCII-only. Fires once per analysis (13 fires per cycle). Matches Yin et al. 2026 per-row engine-provenance prescription. |
| E | Dual-rail dispatch is symmetric | PASS | Both LLM calls in `_run_claude_analysis` have `if use_claude_code_route` branches: trader analysis at `autonomous_loop.py:1481-1500` (CC route) / `1502-1508` (direct route); risk judge at `1537-1557` (CC route) / `1558-1566` (direct route). Neither call leaks to the direct rail when the flag is True. |
| F | Direct API client NOT instantiated when flag is True | PASS | `autonomous_loop.py:1458` -> `client = anthropic.Anthropic(api_key=api_key) if not use_claude_code_route else None`. When CC route is active, `client` is None and the code never reaches `api.anthropic.com`. `:1452` also early-skips the no-key error when CC route is active. |
| G | Async-safe | PASS | Both CC-route calls use `asyncio.to_thread(claude_code_invoke, ...)` at `autonomous_loop.py:1488` and `1544`. Subprocess blocking is offloaded to the threadpool; the event loop stays responsive. |
| H | subtype check correct | PASS | `claude_code_client.py:147-155` -> `if subtype != "success": raise ClaudeCodeError(...)`. Uses `subtype` (not `is_error`) per researcher source #18. Docstring at `:73-74` and module header at `:54` both call out the `is_error` mis-flag history. |
| I | `--disallowedTools` lock present | PASS | `claude_code_client.py:47` -> default `disallowed_tools="Bash,Edit,Write,Read,Glob,Grep,Agent"`; threaded into args at `:79-80`. Blocks the subprocess from running every side-effect tool. |
| J | Error path returns LLMResponse | PASS | `claude_code_client.py:240-249` -> on `ClaudeCodeError`, `ClaudeCodeClient.generate_content` returns `LLMResponse(text="", thoughts=f"errored: {exc}", usage_metadata=UsageMeta())`. Matches existing-convention; downstream callers don't crash. Behavior tested at `test_claude_code_client.py:134-146`. |
| K | Tests cover the failure modes | PASS | `test_claude_code_client.py` has 11 cases: success envelope (1), error subtype (2), timeout (3), non-zero exit (4), missing binary (5), invalid JSON (6), structured_output preference (7), result fallback (8), empty fallback (9), ClaudeCodeClient adapter happy path (10), ClaudeCodeClient adapter error (11). Floor was 8; this is 11. |
| L | ZERO frontend changes + ZERO new npm deps + ZERO emojis | PASS | `git diff --stat HEAD -- frontend/` empty. `git diff HEAD -- frontend/package.json` empty. `git diff HEAD -- frontend/package-lock.json` empty. No emojis introduced (visual scan of `claude_code_client.py` shows ASCII only). |

---

## Final Verdict

**PASS** -- 5 of 5 harness-compliance items PASS, all 12 deterministic checks PASS, all 12 LLM judgment items (A-L) PASS, all 5 code-review dimensions clear with zero BLOCK/WARN/NOTE findings.

The cycle-3 implementation is a clean plumbing addition: the feature flag defaults OFF (B), the existing Anthropic-direct path is preserved (E, F), both LLM calls (trader + risk judge) are symmetrically routed (E), the subprocess is async-safe (G), the success check uses `subtype` not `is_error` per the researcher's adversarial source (H), every side-effect tool is locked out of the subprocess via `--disallowedTools` (I), errors fall back to an empty-text `LLMResponse` so downstream callers don't crash (J), an `ImportError` on `ClaudeCodeClient` falls back to the Anthropic-direct branch as defense in depth (C), the per-rail log is in place for future A/B integrity per Yin et al. 2026 (D), and the 11-case test suite covers every failure mode (K). The citation gate is met with 2 AI-in-trading + 3 academic sources (A).

The cycle-4 smoke run with the flag flipped ON is the correct next step: cycle 3 ships the plumbing under a default-OFF flag, cycle 4 verifies the live rail and then enables masterplan step 27.6 closure.

## Violated criteria

None.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 19 immutable criteria met. Citation gate (>=2 AI-in-trading + >=2 academic) PASS (TradingAgents + Portkey / Bailey-PBO + Harvey-NBER + Yin-2026). Feature flag defaults OFF (settings.py:112). Dual-rail dispatch symmetric (autonomous_loop.py:1481-1557). Per-rail log present (:1445-1449). Direct API client NOT instantiated when flag True (:1458). subtype check used not is_error (claude_code_client.py:147). Async-safe via asyncio.to_thread. 11/11 new tests + 33/33 regression. ZERO frontend / npm / emoji deltas.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "verification_command",
    "mtime_ordering",
    "harness_compliance_5_item_audit",
    "citation_gate",
    "deterministic_grep_counts",
    "regression_test_suite",
    "new_test_suite",
    "code_review_heuristics",
    "evaluator_critique_overwrite_first_cycle3_spawn"
  ]
}
```
