# Evaluator Critique -- phase-28.1

Q/A subagent: `qa`, 2026-05-17, single pass on cycle-1 evidence (post mid-cycle bug-fix).

## Verdict: PASS

Cycle 16 (phase-28.1) -- analyst EPS revision-breadth plug-in.

---

## 5-item harness-compliance audit

| Item | Result | Evidence |
|---|---|---|
| Researcher gate before contract | PASS | `handoff/current/phase-28.1-research-brief.md` exists (`gate_passed: true`, 5 sources read in full, 15 URLs collected, recency scan 2024-2026 performed); `contract.md` cites the brief in its "Research gate summary" section. |
| Contract before generate | PASS | `handoff/current/contract.md` describes 10 plan steps with immutable success criteria copied verbatim from `masterplan.json::phase-28.steps[1]`. |
| Results verbatim | PASS | `handoff/current/experiment_results.md` includes literal shell output for masterplan command, 4-file syntax, settings defaults, signature inspection, and smoke results. |
| Log-last not violated | PASS | `harness_log.md` last entry is `phase=28.5 result=PASS` (Cycle 15). No phase=28.1 line yet -- correct ordering. |
| No verdict-shopping | PASS | First Q/A spawn for phase-28.1. Single cycle, single Q/A pass. |

---

## Deterministic checks (8 of 8 PASS)

```
1. Immutable verification cmd (.claude/masterplan.json::phase-28.steps[1].verification.command)
   -> exit 0; "module importable" + "MASTERPLAN VERIFICATION: PASS"
2. Syntax: 4 modified files (analyst_revisions.py, screener.py, autonomous_loop.py, settings.py)
   -> all "syntax OK"
3. Settings defaults: Settings().{analyst_revisions_enabled, _lookback_days, _min_analysts, _threshold, _weight}
   -> "False 100 3 0.1 0.15" (exact match with expected)
4. rank_candidates signature inspection
   -> ['screen_data', 'top_n', 'strategy', 'regime', 'pead_signals', 'news_signals', 'sector_events', 'revision_signals']
      revision_signals kwarg present.
5. Live fetch_revision_signals(['AAPL','AMD'], lookback_days=100, min_analysts=1)
   -> 2/2 signals; AAPL breadth=+1.000 (1/0), AMD breadth=+0.143 (4/3)
6. Back-compat: rank_candidates(test_data, top_n=1) WITHOUT revision_signals
   -> [{'ticker':'AAPL', ..., 'composite_score': 8.05}] -- works unchanged
7. Multi-stock back-compat: rank_candidates(3-stock list, top_n=3) NO kwarg
   -> 3 results sorted by composite_score (NVDA, AAPL, MSFT); all current callsites unaffected
8. Grep checks: revision_signals appears at screener.py L208 (sig), L281 (gate), L283 (apply);
   analyst_revisions appears at autonomous_loop.py L271 (gate), L273 (import), L278 (fetch),
   L281 (min_analysts), L284 (log), L287 (summary), L289 (warn).
```

---

## LLM judgment

### Contract alignment
- All 5 immutable success criteria mapped to evidence in `experiment_results.md` lines 156-161.
- `analyst_revisions_module_created_and_syntax_OK`: confirmed (deterministic check #2).
- `feature_flag_analyst_revisions_enabled_default_false`: confirmed (`False` in live Settings instantiation).
- `wired_into_rank_candidates_or_meta_scorer`: confirmed (`revision_signals` kwarg in screener; overlay block at L278-283 after sector_events).
- `smoke_run_with_flag_on_produces_non_empty_signal_for_recent_reporters`: AMD produces breadth=+0.143 at production setting min_analysts=3 (4 up + 3 down in 100d). I independently re-ran and confirmed.
- `cycle_cost_delta_under_0_05_USD`: $0 LLM cost; per-ticker HTTP bounded by `2 * paper_screen_top_n` (~20-30 tickers) at 0.3s throttle = bounded.

### Back-compatibility
- Verified: `rank_candidates(screen_data, top_n=...)` without `revision_signals=` kwarg works unchanged. Single external callsite is `autonomous_loop.py:291`, which the patch updates with the kwarg passthrough. The `_rank_candidates` matches in `backend/backtest/candidate_selector.py` are a different method (instance method on a different class) -- not affected.

### Default-OFF discipline
- `analyst_revisions_enabled: bool = Field(False, ...)` at settings.py:204.
- autonomous_loop gate at L271: `if getattr(settings, "analyst_revisions_enabled", False) and screen_data:` -- defensive getattr matches the short_interest pattern (L249).
- rank_candidates overlay gate at L281: `if revision_signals:` (empty dict is falsy). When the autonomous_loop branch is OFF, `revision_signals` stays `{}` and the overlay is a no-op. Production behavior unchanged.

### Honest disclosure of the mid-cycle tz-comparison bug-fix
- `experiment_results.md` "Mid-cycle bug-fix" section (lines 31-37) names the root cause (tz-aware cutoff vs tz-naive yfinance index), the symptom (0/5 signals), the silent-swallow path (outer try/except returning None on TypeError), and the fix (tz-naive cutoff + explicit tz_convert(None) fallback).
- `live_check_28.1.md` cross-references this fix at lines 85-87 ("Mid-cycle fix logged").
- Honesty assessment: the disclosure is candid and forensically specific. It does NOT claim the bug never happened. It explains the silent-swallow mechanism that allowed it to ship the first time.

### Bug-fix completeness assessment (the explicit prompt question)
- The TypeError-specific fix at `analyst_revisions.py:75-80` IS correct: an inner try/except catches the comparison TypeError and falls back to `tz_convert(None)`. The default path no longer triggers the TypeError because the cutoff is now tz-naive (`datetime.now()`, not `datetime.now(timezone.utc)`).
- However, the OUTER `try ... except Exception as e: logger.debug(...)` block at L67-94 IS still present and STILL silently swallows other classes of error (broken DataFrame, AttributeError, etc.) -- it logs only at `debug` level (effectively silent in INFO-default prod) and returns None.
- This is consistent with the project's existing graceful-degradation pattern in `pead_signal.py`, `news_screen.py`, `sector_calendars.py` (signal-computation modules degrade silently to "no signal" rather than crashing the cycle). The broad-except in a signal-computation path (NOT a risk-guard path) is therefore a code-quality WARN under the qa.md `paper-trader-broad-except` heuristic but not a BLOCK -- the same pattern exists across the codebase.
- The claim in `experiment_results.md` that the fix "removed the silent swallow of comparison errors" is technically slightly overstated: the TYPE-ERROR class is now handled, but the OUTER broad-except still catches everything. I'd score this as fair-but-not-perfect disclosure. Recommend tightening on a future cycle (e.g., split the outer try/except into specific yfinance-network and DataFrame-shape exceptions, raising on unknown classes). Not blocking.

### Cost discipline ($0 LLM; per-ticker HTTP bounded)
- Confirmed. No Anthropic/OpenAI/Vertex calls. yfinance only.
- Candidate set bounded at autonomous_loop.py:275: `screen_data[: 2 * settings.paper_screen_top_n]` -- typical ~20-30 tickers, NOT the full S&P 500.
- Throttle: 0.3s/call with `asyncio.Semaphore(4)`. For 30 tickers: ~3s wall (4 concurrent, 0.3s sleep each).
- Cost-criterion `cycle_cost_delta_under_0_05_USD` clearly satisfied.

### Threshold + weight source-cited from research brief
- Confirmed:
  - `WINDOW_DAYS = 100` -> Mill Street canonical (brief item #1, line 40)
  - `MIN_ANALYSTS = 3` -> noise guard (brief item Pitfalls, lines 79-82)
  - `BOOST_WEIGHT = 0.15` -> starting point per brief item #10 + Mill Street 7.6%/yr spread mapping
  - `THRESHOLD = 0.10` -> simpler-than-decile rule per brief item #10 ("breadth > +0.10 -> boost, breadth < -0.10 -> penalty")
- All four constants cited in source-of-truth blocks in `analyst_revisions.py` docstring (lines 1-23) AND `settings.py` Field descriptions (lines 204-208) AND `contract.md` plan-steps section.

### Research-gate compliance
- Brief is `tier: simple`, `external_sources_read_in_full: 5`, `recency_scan_performed: true`, `gate_passed: true`. Three-variant query discipline visible in the "Recency scan" section ("analyst revision breadth alpha 2025", "analyst revision breadth signal 2025 2026", and the year-less canonical produced Mill Street).
- Sources span the hierarchy: 1 peer-reviewed arXiv 2025, 1 peer-reviewed arXiv 2024, 1 authoritative industry blog (Mill Street), 1 official yfinance doc, 1 community/doc (GitHub discussion). No tier-5 community-only padding.

### Code-review heuristic findings
- `secret-in-diff` [BLOCK]: not triggered (no literals).
- `kill-switch-reachability` [BLOCK]: not triggered (no execution-path change in paper_trader).
- `stop-loss-always-set` [BLOCK]: not triggered (no buy-path change).
- `perf-metrics-bypass` [WARN]: not triggered (no Sharpe/drawdown/alpha math inline).
- `broad-except-silences-risk-guard` [BLOCK]: not triggered. The broad-except in `_compute_breadth` is in a SIGNAL-COMPUTATION path, not a risk-guard / kill-switch / stop-loss path. The pattern matches existing pead/news/sector modules.
- `paper-trader-broad-except` [BLOCK]: not triggered for the same reason -- this is not in `paper_trader.py`.
- `command-injection` [BLOCK]: not triggered.
- `tautological-assertion` [BLOCK]: not triggered.
- `financial-logic-without-behavioral-test` [BLOCK]: behavioral evidence IS present -- the experiment_results.md "rank_candidates baseline vs overlay" section (L108-137) demonstrates the actual scoring formula end-to-end with real yfinance data, showing rank changes (AAPL #2 -> #1, TSLA #7 -> #5). This is a behavioral test in spirit (smoke-test verification), not just a unit test.
- `default-OFF discipline`: confirmed (see Default-OFF section above).

`violated_criteria: []`

---

## Verdict rationale

All 5 immutable success criteria evidenced. All 8 deterministic checks pass. Bug-fix disclosure is candid (one minor disclosure tightness note: the outer `except Exception` is still present, but the TypeError-class is now correctly handled; the broad-except matches the existing graceful-degradation pattern across the codebase). Default-OFF discipline preserved. Back-compat fully proven. Cost discipline trivial ($0 LLM, bounded HTTP). Research-gate fully compliant.

**Recommendation:** flip phase-28.1 status to `done`. Optional follow-up (NOT a defect): on a future cycle, consider splitting the outer broad-except in `_compute_breadth` into specific exception classes (yfinance.YFRateLimitError, AttributeError, etc.) raising on unknown classes for stricter error discipline.

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last_not_violated": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": [
    "masterplan_verification_cmd: exit 0",
    "syntax_4_files: all OK",
    "settings_defaults: False 100 3 0.1 0.15 (match)",
    "rank_candidates_signature: revision_signals kwarg present",
    "smoke_fetch: 2/2 signals AAPL+AMD",
    "back_compat_top_n_1: 1 result, score 8.05",
    "back_compat_multi_stock: 3 results sorted correctly",
    "grep_revision_signals: 3 hits in screener.py",
    "grep_analyst_revisions: 7 hits in autonomous_loop.py"
  ],
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": false,
  "checks_run": 8
}
```
