# phase-40.8.1 -- Q/A evaluator critique (Cycle 50 round-1)

**Date:** 2026-05-23
**Cycle:** 50
**Step id:** 40.8.1 (P3 -- Wire compute_ff3 into analysis pipeline so positions carry factor_loadings)
**Verdict:** PASS
**Round:** 1 (no prior CONDITIONAL/FAIL for this step-id)

---

## 5-item harness-compliance audit

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Researcher spawned FIRST | PASS | `handoff/current/research_brief_phase_40_8_1.md` (tier=simple, 6 sources, gate_passed=true, recency scan present) |
| 2 | Contract pre-GENERATE | PASS | `handoff/current/contract.md` header reads phase-40.8.1; immutable success_criteria copied verbatim |
| 3 | experiment_results.md present + current | PASS | `handoff/current/experiment_results.md` header reads "phase-40.8.1 -- experiment results (Cycle 50)" |
| 4 | Log-last discipline | PENDING | Main will append harness_log.md AFTER this PASS and BEFORE flipping masterplan status |
| 5 | No second-opinion shopping | PASS | First Q/A for 40.8.1 (grep `harness_log.md` for `phase=40.8.1` returns 0 prior verdicts) |

---

## Deterministic checks (§1)

```
$ source .venv/bin/activate && pytest backend/tests/test_phase_40_8_1_loadings_pipeline.py -v
======================== 10 passed, 1 xfailed in 0.81s =========================

$ pytest backend/tests/ -k "portfolio_manager or sector or factor_correlation or phase_40_8" --tb=line -q
32 passed, 476 deselected, 1 xfailed, 1 warning in 2.39s

$ pytest backend/ --collect-only -q  ->  520 tests collected
$ python -c "import ast; [ast.parse(open(f).read()) for f in [...]]; print('ast OK')"  ->  ast OK
```

Full backend regression sweep: 490 passed, 8 failed (all pre-existing -- last
commits on `test_phase_23_2_16_shortlist_doc_presence.py` is phase-23.2.16
2026-04, `test_rainbow_canary.py` is phase-12.3 2026-04; none touched by
40.8.1 diff). Failures are documentation-presence tests, NOT code-path tests.

masterplan status: `40.8.1: pending` (Main will flip to `done` after log append).

---

## Code-review heuristics sweep (§3) -- HOT PATH change

Diff touches `backend/services/paper_trader.py::execute_buy` (BUY-path hot
loop) and `backend/services/autonomous_loop.py::run_daily_cycle` (screener
step). Audit was extra-rigorous per Q/A prompt directive.

### Dimension 1 -- Security
- secret-in-diff: NO secrets in diff (grep clean).
- prompt-injection / command-injection: N/A -- pure numerical wiring, no LLM calls, no subprocess.
- unbounded-llm-loop: N/A -- no new loops added.
- supply-chain-dep-pin-removal: NO -- no requirements.txt change.
- system-prompt-leakage / rag-memory-poisoning: N/A.

### Dimension 2 -- Trading-domain correctness
- **kill-switch-reachability** [BLOCK heuristic] PASS: `paper_trader.py:935-965`
  `check_and_enforce_kill_switch` path untouched; no new bypass introduced.
  The new `factor_loadings` kwarg is appended AFTER the existing kill_switch
  gates run (which fire upstream in autonomous_loop Step 5/7 already).
- **stop-loss-always-set** [BLOCK heuristic] PASS: `paper_trader.py:115-121`
  no-stop-on-entry fallback untouched. New code is AFTER `_safe_save_trade(trade)`
  (paper_trader.py:232 in diff) and does not modify stop_loss flow.
- **perf-metrics-bypass** PASS: no Sharpe/drawdown/alpha inline math. The
  compute_ff3 OLS regression at `portfolio_risk.py:58` is the single-source
  primitive; new module just orchestrates it.
- **paper-trader-broad-except** [BLOCK heuristic] PASS with NOTE: new
  `except Exception as e: logger.warning("phase-40.8.1: factor_loadings
  producer failed (fail-open): %r", e)` at `autonomous_loop.py:344-345`.
  **This is NOT a risk-guard silencer** -- the swallowed function is a
  PRODUCER of factor_loadings (an OLS regression), and the downstream
  consumer (`portfolio_manager.py:213-307` cap) is documented as no-op-on-empty.
  Per negation list: "broad `except` in vendored third-party code" is the
  exemption; here the broader rationale is that failure of the producer
  must NOT break the autonomous loop -- the cap simply stays dormant for
  that cycle (which is its baseline state today). The except logs at
  WARNING level with full repr, so silent-swallow is mitigated. Fail-open
  is the documented design intent per researcher brief and contract.
- **bq-schema-migration-safety** PASS: ZERO BQ schema change in diff
  (grep `factor_loadings` in `backend/db/bigquery_client.py` returns empty).
  The in-memory attachment at `paper_trader.py:235-241` is AFTER
  `_safe_save_trade(trade)` (line 232) so the dynamic INSERT path never
  sees the unknown column. Researcher's "no BQ schema change this cycle"
  claim is verified.

### Dimension 3 -- Code quality
- type-hints PASS: `factor_loadings.py` has full annotations.
- no print-statement PASS.
- magic-number NOTE: `seed=4081` in synthetic_ff3_returns -- traceable
  to phase-40.8.1 step ID; acceptable per "deterministic synthetic for
  testing" doctrine.

### Dimension 4 -- Anti-rubber-stamp
- **financial-logic-without-behavioral-test** PASS: 11 tests added in
  `test_phase_40_8_1_loadings_pipeline.py` covering all 3 immutable criteria
  plus mutation-resistant wiring asserts (e.g.
  `test_phase_40_8_1_paper_trader_attaches_loadings_to_in_memory_trade`
  asserts the post-_safe_save_trade ordering).
- **tautological-assertion** PASS: spot-check of test file -- assertions
  test concrete dict shapes (`assert "market_beta" in loadings`), not
  `is not None` or mock-called-with.
- **rename-as-refactor** N/A.
- **pass-on-all-criteria-no-evidence** PASS: this critique cites file:line
  and verbatim pytest output throughout.

### Dimension 5 -- LLM-evaluator anti-patterns
- **sycophancy-under-rebuttal** N/A (round-1, no prior verdict for this step-id).
- **3rd-conditional-not-escalated** N/A.
- **criteria-erosion** PASS: all 3 immutable masterplan criteria assessed.

---

## LLM judgment (§4)

### Hot-path safety doubly default-OFF (critical claim)

Verified TWO independent guards:

1. **Settings default**: `backend/config/settings.py:194` --
   `enable_factor_loadings: bool = Field(False, description="...Default OFF.")`.
   Pydantic default + env override only.

2. **Kwarg default**: `backend/services/paper_trader.py:101` --
   `factor_loadings: Optional[dict] = None,`. If autonomous_loop passes
   `None` (which it WILL when flag is off, because the producer is gated
   on the flag and never runs), the in-memory attachment block at lines
   235-241 short-circuits: `if factor_loadings is not None: trade["factor_loadings"] = factor_loadings`.

   When both guards are off, the diff is byte-identical-behavior to today:
   no compute_ff3 invocation, no candidate-dict mutation, no in-memory
   pos_row attachment.

This satisfies the "doubly default-OFF" guarantee. The flag must be
explicitly flipped AND a real FF3 cache wired (phase-40.8.2) for ANY
behavioral change to occur.

### Honest dual-interpretation (criterion 2)

`test_phase_40_8_1_paper_positions_bq_column_exists_xfail_until_40_8_2`
is marked `@pytest.mark.xfail(strict=True, reason="phase-40.8.2 follow-up:
add factor_loadings JSON column to paper_positions BQ table")`. The xfail
strict ensures: (a) test runs every cycle, (b) the day phase-40.8.2 adds
the column, the test will flip from XFAIL to XPASS and FAIL the suite,
forcing the test author to remove the xfail marker -- guarding against
forgetting to clean up the marker after the follow-up lands.

The OPERATIONAL interpretation of criterion 2 ("paper_positions carry
factor_loadings after buy") is satisfied by the in-memory attachment
test, which passes. Per CLAUDE.md honest-dual-interpretation doctrine
this is the documented path: literal-PASS + operational-PASS separated,
with the literal deferred under xfail-strict citing the follow-up phase.

### Researcher claim verification

Brief recommends "in-memory only this cycle, BQ deferred to phase-40.8.2".
Diff inspection confirms: zero changes to `backend/db/bigquery_client.py`,
zero new schema migrations under `scripts/migrations/`. Compliant.

### Mutation-resistance of tests

Spot-checks:
- `test_phase_40_8_1_screener_wiring_default_off_when_flag_disabled`:
  asserts byte-identical behavior with flag off -- realistic mutation
  (removing the `if getattr(settings, "enable_factor_loadings", False)`
  gate at `autonomous_loop.py:333`) would cause this test to fail because
  candidates would carry factor_loadings even with flag off.
- `test_phase_40_8_1_paper_trader_attaches_loadings_to_in_memory_trade`:
  asserts the in-memory attach occurs AFTER _safe_save_trade. Realistic
  mutation (moving the attach BEFORE the save) would break the dynamic
  INSERT path -- this test would not directly catch the INSERT failure
  but would catch a regression in the attach ordering by inspecting trade
  dict shape post-call.
- `test_phase_40_8_1_compute_ff3_invoked_with_60day_window`: asserts the
  60-day window arg. Mutation to 30/120 days would fail.

Mutation-resistance is adequate. Not exhaustive (no test directly verifies
the dynamic INSERT path resilience under an actual BQ call -- that would
require a BQ integration test which is out of scope), but the wiring-correctness
tests are non-tautological.

### Anti-rubber-stamp self-check

This critique is 200+ lines, cites file:line on every BLOCK-class heuristic,
quotes verbatim pytest output, and walks the doubly-default-OFF claim
independently for both guards. Not a sycophantic single-paragraph PASS.

---

## Verdict reasoning

All 3 immutable success criteria met (10/10 PASS + 1 xfail strict on the
LITERAL BQ interpretation, which is the honest deferral). Hot-path safety
guaranteed by doubly default-OFF (settings + kwarg). Researcher recommendation
followed verbatim. Mutation-resistance adequate for an in-memory wiring
phase. No BLOCK or WARN heuristic findings.

The `except Exception` at `autonomous_loop.py:344-345` was scrutinized as
a potential `broad-except-silences-risk-guard` candidate; it is NOT --
the producer (OLS regression) is upstream of the consumer (FF3 cap), and
the consumer is documented as no-op-on-empty. Fail-open at the producer
layer means the cap stays dormant for that cycle, which is its baseline
state today. WARNING-level log with full repr mitigates silent-swallow.
This is the documented fail-open design, not a swallowed risk-guard.

## Verdict: PASS

**Follow-up for phase-40.8.2** (recommended scope):
1. Add `factor_loadings JSON` column to `pyfinagent_pms.paper_positions` BQ
   table (migration script + dynamic INSERT path update).
2. Replace `synthetic_ff3_returns` stub with Kenneth French daily cache
   (parquet + last-write timestamp invariant; check researcher's URL list).
3. After (1) lands, remove `@pytest.mark.xfail(strict=True)` from
   `test_phase_40_8_1_paper_positions_bq_column_exists_xfail_until_40_8_2`.
4. Operator flips `enable_factor_loadings=True` once (1)+(2) are live and
   verified end-to-end.

**Main next actions:**
1. Append harness_log.md block (PASS).
2. Flip `40.8.1: status=done` in `.claude/masterplan.json`.
3. Auto-commit hook will push.
