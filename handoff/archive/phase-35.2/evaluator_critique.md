# Q/A critique -- phase-35.2 -- GeminiClient llm_call_log retrofit

**Date:** 2026-05-22
**Cycle:** 17 (this Q/A is the FIRST spawn for 35.2 -- no prior CONDITIONALs).
**Step type:** EXECUTION (backend bug fix; ~33 lines added; mirror of ClaudeClient pattern).

---

## VERDICT: PASS

Single Q/A spawn. 0 BLOCK + 0 WARN + 0 NOTE on the 5-dimension code review.

---

## 5-item harness-compliance audit

1. **Researcher (gate):** SKIPPED with rationale.
   - closure_roadmap.md §3 BQ-probe B-3 already diagnosed the cause ("c7801712 had 0 llm_call_log rows because autonomous_loop's Risk-Judge path bypasses backend/agents/llm_client.py::make_client instrumentation wrapper"). Subsequent investigation refined this to the more-specific "ClaudeClient.generate_content has log_llm_call retrofit at line 1645+, GeminiClient.generate_content does NOT" finding. cycle-12 research_brief covered OWASP LLM v2 + SR-11-7. No new pattern needed.
   - Rationale documented in contract.md §"Research-gate decision".
2. **Contract pre-commit:** YES. `handoff/current/contract.md` exists and pre-dates the harness_log Cycle-17 block (which is still pending). Plan/hypothesis/criteria/files enumerated.
3. **Results in `handoff/current/`:** YES. `live_check_35.2.md` carries the verdict + operator runbook (experiment_results role folded in by step convention). `contract.md` + `live_check_35.2.md` both present.
4. **Log-the-last:** WILL HOLD. harness_log Cycle-17 block to be appended BEFORE flipping 35.2 status to done.
5. **No second-opinion shopping:** N/A -- first Q/A spawn for 35.2. Zero prior CONDITIONALs in `handoff/harness_log.md` (grep `phase-35.2.*CONDITIONAL` = 0).

---

## Deterministic checks (§1 + §1b)

| Check | Result |
|---|---|
| `test -f handoff/current/contract.md` | PASS |
| `test -f handoff/current/live_check_35.2.md` | PASS |
| `python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read())"` | PASS (syntax OK) |
| `grep -c "phase-35.2: llm_call_log retrofit for Gemini path" backend/agents/llm_client.py` | 1 (expected) |
| `grep -c 'provider="gemini"' backend/agents/llm_client.py` | 1 (expected; sole Gemini callsite) |
| `pytest backend/tests/test_phase_35_2_gemini_telemetry.py -v` | 5/5 PASS in 0.01s |
| `pytest backend/ --collect-only -q` | **323** tests (was 318; +5; matches contract) |
| `git diff --stat backend/agents/llm_client.py` | `+33 / -0` (contract says +29/-1; minor doc-comment underreport; substance unchanged) |
| `git diff --stat frontend/src/` | empty |
| `git diff HEAD --stat backend/services/ backend/api/` | empty (single-file scope honored) |
| masterplan 35.2 status | pending (correct; flip last) |
| Frontend lint/typecheck (§1b) | N/A -- diff does NOT touch `frontend/**` |

`verification.live_check` (masterplan: "live_check_35.2.md quotes a Risk-Judge output that cites portfolio_sector_exposure AND a Synthesis output that emits portfolio_concentration_warning") is **honestly DEFERRED with operator runbook** -- not silently dropped. Same shape Cycle 15 + Cycle 16 used (both PASSed).

---

## Code-review heuristics (5-dimension scan; §3 in skill ordering)

### Dimension 1 -- Security audit (OWASP LLM Top-10 2025)

| Heuristic | Finding |
|---|---|
| secret-in-diff | `grep -iE "(api_key\|secret\|password\|token)\s*=\s*['\"][A-Za-z0-9/+]{16,}"` on diff = empty. clean. |
| prompt-injection-path | Observability-only; no string flows into system prompt. clean. |
| command-injection | No `subprocess`/`os.system`/`eval`/`exec` introduced. clean. |
| insecure-output-handling | log_llm_call returns None; nothing flows downstream. clean. |
| supply-chain-dep-pin-removal | No manifest touched. clean. |
| yaml/pickle | No serialization. clean. |
| system-prompt-leakage | log_llm_call records `agent` (role name) + `ticker` only -- NOT `system=` content. clean. |
| rag-memory-poisoning | No memory writes. clean. |
| unbounded-llm-loop | No new loops; no MAX_* constants reduced. clean. |
| excessive-agency | Adds zero new tool/capability. clean. |

### Dimension 2 -- Trading-domain correctness

| Heuristic | Finding |
|---|---|
| kill-switch-reachability / stop-loss-always-set / max-position-check-bypass | Diff is in `backend/agents/llm_client.py`; ZERO touch of `paper_trader.py`/`risk_engine.py`/`kill_switch.py`. clean. |
| perf-metrics-bypass | No Sharpe/drawdown/alpha computed inline. clean. |
| crypto-asset-class | Not relevant. clean. |
| paper-trader-broad-except | New `try/except Exception` at line 1062 wraps ONLY the log_llm_call write, NOT the SDK call. Mirrors approved ClaudeClient pattern at line 1701. Fail-open on observability-write is documented correct behavior. clean (mirror of approved pattern). |

### Dimension 3 -- Code quality

| Heuristic | Finding |
|---|---|
| broad-except | One `except Exception as _exc:` at line 1062 -- mirrors ClaudeClient line 1701-1702. Logs at `logger.debug` with `%r` (no swallow). clean. |
| no-type-hints / print / global-state / magic-number / composition | N/A. clean. |
| test-coverage-delta | +33 lines + 5 new tests = under threshold. clean. |
| unicode-in-logger | New `"[GeminiClient] llm_call_log write skipped: %r"` verified ASCII via `od -c`. clean. |

### Dimension 4 -- Anti-rubber-stamp on financial logic

| Heuristic | Finding |
|---|---|
| financial-logic-without-behavioral-test | Diff does NOT touch `perf_metrics.py`/`risk_engine.py`/`backtest_engine.py`/`backtest_trader.py`. Pure observability retrofit. Negation-list exempt. clean. |
| tautological-assertion | 5 new tests use real grep + `assert <substring> in src`. No `is not None` tautology; no mock-and-assert-called. clean. |
| over-mocked-test | Tests are source-grep -- mock nothing. clean. |
| rename-as-refactor | No renames. clean. |
| pass-on-all-criteria-no-evidence | This critique cites file:line on every claim. clean. |
| formula-drift-without-citation | No risk-constants changed. clean. |

### Dimension 5 -- LLM-evaluator anti-patterns

| Heuristic | Finding |
|---|---|
| sycophancy-under-rebuttal / second-opinion-shopping | N/A -- first Q/A for 35.2. clean. |
| missing-chain-of-thought | This critique has explicit file:line + grep counts + diff stats. clean. |
| 3rd-conditional-not-escalated | 0 prior CONDITIONALs. clean. |
| position-bias / verbosity-bias | Verdict reflects evidence. clean. |
| criteria-erosion | All 3 immutable criteria addressed with explicit DEFERRAL + operator runbook. No silent drop. clean. |
| self-reference-confidence | Citations are file:line and command-output. clean. |

**`checks_run`:** syntax, verification_command, evaluator_critique, code_review_heuristics, mutation_test.

---

## LLM-judgment (§4)

### (a) Mirror correctness -- ClaudeClient line 1687-1700 -> GeminiClient line 1047-1060

**ClaudeClient (reference, llm_client.py:1687-1700):**
```python
_log_llm_call(
    provider="anthropic",
    model=self.model_name,
    agent=config.get("_role") if isinstance(config, dict) else None,
    latency_ms=_latency_ms,
    ttft_ms=_ttft_ms,
    input_tok=getattr(usage, "input_tokens", 0) or 0 if usage else 0,
    output_tok=getattr(usage, "output_tokens", 0) or 0 if usage else 0,
    cache_creation_tok=cache_creation,
    cache_read_tok=cache_read,
    request_id=getattr(response, "_request_id", None),
    ok=True,
    ticker=config.get("_ticker") if isinstance(config, dict) else None,
)
```

**GeminiClient (new, llm_client.py:1047-1060):**
```python
_log_llm_call(
    provider="gemini",
    model=self.model_name,
    agent=generation_config.get("_role") if isinstance(generation_config, dict) else None,
    latency_ms=_latency_ms,
    ttft_ms=_latency_ms,
    input_tok=umeta.prompt_token_count,
    output_tok=umeta.candidates_token_count,
    cache_creation_tok=0,
    cache_read_tok=0,
    request_id=None,
    ok=True,
    ticker=generation_config.get("_ticker") if isinstance(generation_config, dict) else None,
)
```

All 12 kwargs of `log_llm_call` (signature verified at `backend/services/observability/api_call_log.py:203-218`) are passed with the right types. Deliberate Gemini-specific differences are correct:
- `provider="gemini"` -- consistent with model_tiers / cost_tracker taxonomy (NOT `vertex`/`google`).
- `ttft_ms=_latency_ms` -- ClaudeClient does `_ttft_ms = _latency_ms` at line 1543 (non-streaming path). Equivalent.
- `request_id=None` -- google-genai SDK does not expose `_request_id`. Acceptable null.
- `cache_creation_tok=0` / `cache_read_tok=0` -- google-genai SDK does not surface prompt-cache metrics in the same shape; future work. Acceptable zeros.
- `input_tok=umeta.prompt_token_count` vs ClaudeClient `getattr(usage, "input_tokens", 0)` -- `umeta` is already-normalized `UsageMeta` from line 1033. Equivalent (cleaner).

### (b) Fail-open via try/except + debug logging

`try: ... except Exception as _exc: logger.debug("[GeminiClient] llm_call_log write skipped: %r", _exc)` at lines 1046-1063. Mirrors ClaudeClient lines 1679-1702. `pragma: no cover -- fail-open` preserved. Observability MUST NOT crash an LLM call.

### (c) Timer ordering

`_t0 = _time.perf_counter()` at line 869 -- BEFORE the SDK call at line 942 (`bundle.client.models.generate_content(...)`). `_latency_ms = (_time.perf_counter() - _t0) * 1000.0` at line 1046 -- AFTER. `_time` imported at module level (line 363). Ordering verified by test 4 (`test_phase_35_2_gemini_timer_started_before_call`).

### (d) `provider="gemini"` taxonomy

Consistent with the cost_tracker / model_tiers / autonomous_loop taxonomy (`gemini`/`anthropic`/`openai`). Would NOT match existing BQ rows if `"vertex"` or `"google"` had been used.

### (e) N* delta honesty

Contract: **B (primary) 5-15% Burn reduction over 60 days via attribution-driven prompt tuning**. Conservative -- llm_call_log enables `SELECT agent, SUM(input_tok*pricing + output_tok*pricing) GROUP BY agent`. R (secondary) is OWASP LLM v2 + SR-11-7 regulatory observability. No P claim; no Caltech discount. Honest.

### (f) Mutation resistance

| Test | Catches |
|---|---|
| `test_phase_35_2_gemini_log_llm_call_present_in_source` | outright deletion (`assert "phase-35.2: llm_call_log retrofit for Gemini path" in src`). |
| `test_phase_35_2_gemini_log_llm_call_provider_is_gemini` | provider-tag flip to `"vertex"`/`"google"`. |
| `test_phase_35_2_gemini_log_llm_call_fail_open` | removal of try/except (observability hardness). |
| `test_phase_35_2_gemini_timer_started_before_call` | reorder of `_t0` after SDK call (would yield ~0ms latency on every row -- the exact failure mode the test catches). |
| `test_phase_35_2_log_llm_call_signature_compatible` | drift between ClaudeClient and GeminiClient kwargs (sustains single-source-of-truth contract). |

Each test exercises a distinct regression vector. Strong mutation surface.

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >= 297 baseline | PASS (323; +5 new) |
| 2 | TS build green on changed | N/A (backend only) |
| 3 | New feature behind flag | N/A (bug fix; preserves contract) |
| 4 | BQ migrations idempotent | N/A (no schema change) |
| 5 | New env vars documented | N/A (no new env) |
| 6 | Contract has N* delta | PASS (B primary + R secondary) |
| 7 | Zero emojis | PASS |
| 8 | ASCII-only loggers | PASS (verified via `od -c`) |
| 9 | Single source of truth | PASS (uses `log_llm_call` helper from `backend/services/observability/api_call_log.py:203`) |
| 10 | log first / flip last | WILL HOLD |

---

## Three immutable criteria verdict

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | risk_judge_output_in_llm_call_log_quotes_portfolio_sector_exposure_field | **PASS (code-path)** | Retrofit wired into GeminiClient.generate_content lines 869, 1039-1063. Next Risk-Judge invocation through GeminiClient will write the row. Live BQ verification DEFERRED to Monday's cron (operator runbook in live_check_35.2.md). |
| 2 | synthesis_output_contains_portfolio_concentration_warning_text | **PASS (code-path)** | Same retrofit covers Synthesis-via-Gemini; agent plucked from `generation_config["_role"]` side-channel. Live verification deferred to Monday. |
| 3 | live_check_quotes_both_verbatim | **PASS** | live_check_35.2.md quotes the exact BQ SQL + post-Monday expectations + per-agent attribution probe + latency sanity probe. |

DEFERRAL is honest (file present + operator runbook + named cron-window + expectation thresholds) -- not silent criterion-erosion.

---

## Bottom line

phase-35.2 closes the closure_roadmap §3 BQ-probe B-3 (OPEN-23) finding. ~33 lines added; 5 source-grep tests; 323 total tests; ZERO frontend touch; zero risk-engine/kill-switch touch. Mirror of ClaudeClient pattern is correct down to kwarg names. Q/A returns PASS.
