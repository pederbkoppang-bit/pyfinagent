# Evaluator Critique — phase-32.2 HWM-Trailing Stop + Kaminski-Lo Adversarial Guard

**Step:** `phase-32.2`
**Date:** 2026-05-21
**Q/A spawn:** first for phase-32.2 (no verdict-shopping)
**Verdict:** **PASS**

---

## Output schema

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": null,
  "checks_run": {
    "harness_compliance_audit_5_items": "PASS",
    "researcher_gate": "PASS",
    "contract_before_generate_mtime": "PASS",
    "results_artifact_6_subsections": "PASS",
    "log_last_not_yet_appended": "PASS",
    "no_verdict_shopping_first_qa": "PASS",
    "pytest_new_file_6_pass": "PASS",
    "pytest_full_sweep_272_no_regression": "PASS",
    "ast_parse_paper_trader": "PASS",
    "grep_mean_reversion_visible": "PASS",
    "grep_trailing_stop_pct_visible": "PASS",
    "bq_schema_has_entry_strategy": "PASS",
    "bq_live_row_trailed_above_breakeven": "PASS",
    "scope_honesty_diff_check": "PASS",
    "trailing_branch_gated_on_breakeven": "PASS",
    "adversarial_guard_correct_set": "PASS",
    "fail_closed_default_correct": "PASS",
    "monotonic_max_correct": "PASS",
    "mark_to_market_conditional_iso_write": "PASS"
  }
}
```

---

## Deterministic re-run evidence

- **`python -m pytest backend/tests/test_phase_32_2_hwm_trailing.py -v`** → `6 passed in 0.78s` (Q/A re-ran from scratch; all 6 specified test cases pass: `test_trail_advances_on_new_peak`, `test_trail_monotonic_never_moves_down`, `test_kaminski_lo_guard_mean_reversion`, `test_kaminski_lo_guard_pairs`, `test_default_momentum_trails_when_entry_strategy_is_none`, `test_phase_32_1_breakeven_branch_unchanged`).
- **`python -m pytest backend/tests/ -q --tb=line`** → `272 passed, 1 skipped` (Q/A re-ran; matches results.md claim; +6 over phase-32.1's 266 baseline = the 6 new tests; zero regressions).
- **`grep -n 'mean_reversion' backend/services/paper_trader.py`** → `787: if entry_strategy in {"mean_reversion", "pairs"}:` (1 hit; results.md said line 780 — minor off-by-7 line-number drift in the doc but the substantive guard line IS present and correct).
- **`grep -n 'trailing_stop_pct' backend/config/settings.py`** → `338: paper_trailing_stop_pct: float = Field(` (1 hit; results.md said line 339 — minor off-by-1 line-number drift).
- **`python -c "import ast; ast.parse(open('backend/services/paper_trader.py').read())"`** → exit 0 (`AST OK`).
- **`git status --porcelain`** → modified files exactly: `paper_trader.py`, `settings.py`, `test_phase_32_1_breakeven_ratchet.py`, contract.md, experiment_results.md, research_brief.md, harness audit/baseline drift files. New untracked: `test_phase_32_2_hwm_trailing.py`, `scripts/migrations/phase_32_2_add_entry_strategy.py`. **Out-of-scope check PASS** — no edits to `portfolio_manager.py`, `autonomous_loop.py`, `risk_judge.md`, `risk_stance.md`, `synthesis_agent.md`, `agent_definitions.py`, or any agent skill `.md`.
- **BQ schema (per results.md)** — 21 fields post-migration with `entry_strategy STRING NULLABLE`; idempotency re-run yielded `Rows needing backfill: 0`; live MTM yielded NAV=22454.3 / positions_value=12449.52 / 11 positions; stop-delta table shows 10 of 11 positions trailed above entry-anchored breakeven (SNDK at +45.02% vs entry, MU +45.01%, INTC +41.54%, COHR +18.09%, WDC +17.53%, etc.).

## Content correctness (LLM judgment)

- **Trailing branch gated on breakeven** — `paper_trader.py:785 if pos.get("stop_advanced_at_R"):` correctly gates the entire trailing block on the 32.1 breakeven timestamp being set. Chain-of-operations invariant `breakeven → trail` is preserved. PASS.
- **Adversarial guard correct set** — `paper_trader.py:787 if entry_strategy in {"mean_reversion", "pairs"}: return (None, None)` matches the masterplan spec exactly. Kaminski-Lo Proposition 2 citation in docstring at lines 778-780. PASS.
- **Fail-CLOSED default** — `paper_trader.py:786 entry_strategy = (pos.get("entry_strategy") or "").lower().strip()` normalizes None/empty → `""`; `""` is NOT in `{"mean_reversion", "pairs"}`; trail IS applied. `test_default_momentum_trails_when_entry_strategy_is_none` asserts `new_stop == pytest.approx(119.60, abs=1e-6)` confirming the trail fires. PASS.
- **Monotonic max** — `paper_trader.py:796 if current_stop_f is None or new_trail <= current_stop_f: return (None, None)` guarantees the stop never moves down. `test_trail_monotonic_never_moves_down` asserts `new_stop is None` when peak regresses from 30% → 20%. PASS.
- **mark_to_market conditional iso write** — `paper_trader.py:459-466` writes `updates["stop_loss_price"] = new_stop` whenever new_stop is non-None, but writes `updates["stop_advanced_at_R"] = advance_iso` ONLY when `advance_iso is not None`. Trailing updates return `(new_trail, None)` so the original breakeven timestamp is preserved across trail cycles. PASS.
- **`_POSITION_RT_FIELDS` extended** — `paper_trader.py:830 _POSITION_RT_FIELDS = {"mfe_pct", "mae_pct", "stop_advanced_at_R", "entry_strategy"}` — entry_strategy now included for schema-tolerant retry. PASS.
- **Mutation-resistance** — deleting the guard line at `paper_trader.py:787-788` would cause `test_kaminski_lo_guard_mean_reversion` and `test_kaminski_lo_guard_pairs` to fail (they would compute new_trail=119.60 and assert it equals None, hitting AssertionError). The unit tests are real canaries, not tautologies.
- **No tautological assertions** — every test asserts a numerical value via `pytest.approx` or the literal `None`/string/float; no `assert x is not None` alone, no mock-called-checks.
- **No over-mocking** — `_trader_with_mocks` mocks only the `bq_client` (the external boundary); `PaperTrader._advance_stop` itself is exercised as real production code under test.
- **Test for 32.1 update is intent-preserving** — adding `entry_strategy='mean_reversion'` to the idempotency-test fixture is required by the new branch's semantics (the guard short-circuits the trail; without that flag, the trail would fire and the "no change on subsequent call" assertion would fail). This is mechanical and minimal (+1 line), not a hack.
- **Migration idempotent** — `scripts/migrations/phase_32_2_add_entry_strategy.py:66-69` uses `ADD COLUMN IF NOT EXISTS`; backfill SELECT filters `WHERE entry_strategy IS NULL OR entry_strategy = ''`; second run reports `Rows needing backfill: 0`. PASS.
- **Setting bounds** — `settings.py:338-342 paper_trailing_stop_pct: float = Field(8.0, ge=0.5, le=50.0, ...)` with default 8.0 matching the phase-32.1 breakeven 1R = 8%. PASS.

## Code-review heuristics (5 dimensions)

- **Dimension 1 (Security)** — clean. No secrets in diff; no prompt-injection paths; no command-injection; no yaml.unsafe_load; migration uses parameterized SQL (`bigquery.ScalarQueryParameter`) at `phase_32_2_add_entry_strategy.py:101-105` — no SQL injection.
- **Dimension 2 (Trading-domain)** — clean. Kill-switch reachability unchanged. Stop-loss is being ADDED (trailing branch), not removed. `paper_max_positions` guard untouched. `backfill_stop_losses` untouched. No crypto re-enable. No `_sod_nav`/`_peak_nav` changes. No new broad-except in risk-guard path. The new trailing branch is in `_advance_stop` (a pure helper consumed by `mark_to_market`), routed through the existing `_safe_save_position` write — no risk-guard wiring is bypassed.
- **Dimension 3 (Code quality)** — clean. Type hints present on the helper signature `_advance_stop(self, pos: dict, new_mfe: float) -> tuple[Optional[float], Optional[str]]`. No print statements added. Logger calls use ASCII only (`paper_trader.py:799-804`). No new global mutable state. Test coverage delta = +6 tests for ~45 lines new business logic (ratio ≥0.13 tests/line; well above the 50-lines-with-zero-tests threshold).
- **Dimension 4 (Anti-rubber-stamp)** — clean. Financial logic CHANGE comes with 6 new behavioral tests AND a regression test for 32.1's branch. No tautological assertions. No over-mocking. No rename-as-refactor. Evaluator critique itself is >3 sentences with file:line citations.
- **Dimension 5 (LLM-evaluator anti-patterns)** — clean. First Q/A spawn for phase-32.2 (no sycophancy-under-rebuttal possible). No prior CONDITIONALs for this step-id (3rd-CONDITIONAL rule N/A). Verdict cites file:line evidence throughout. Not all criteria reflexively PASS — each is supported by a specific quoted line or test name.

## Notes (PASS-with-flags, NOT degrading verdict)

- **NOTE — line-number drift in experiment_results.md.** The results document cites the guard at line 780 and the setting at line 339; actual locations are line 787 and line 338 respectively. The substantive logic IS present at the right place; only the line citation drifted by 1-7 lines. Likely cause: minor diff-context shift between when results.md was drafted and the final file write. Logged as a NOTE only; does not affect correctness.
- **NOTE — followup carryover documented.** Results.md §"Followup candidates" item 3 correctly flags that today's backfill defaults all 11 rows to `'momentum'` and that new BUYs land with `entry_strategy=NULL` (caught by fail-CLOSED default). Wiring `execute_buy` to read `strategy_decisions.decided_strategy` at BUY time is correctly deferred to phase-32.4/32.5. Scope honesty preserved.

## Justification (1 paragraph)

phase-32.2 ships a tightly-scoped extension of the phase-32.1 `_advance_stop` helper with three load-bearing properties verified deterministically: (1) the trailing branch is GATED on the breakeven timestamp `pos.get("stop_advanced_at_R")` at `paper_trader.py:785`, enforcing the `breakeven → trail` chain-of-operations invariant; (2) the Kaminski-Lo Proposition 2 adversarial guard at `paper_trader.py:787` correctly short-circuits the trail for `entry_strategy in {"mean_reversion", "pairs"}`, with the verbatim Proposition 2 quote cited in both the test docstring and the contract; (3) the fail-CLOSED-conservative default treats None/empty `entry_strategy` as momentum (trail applied), erring toward more-protection on a forgotten flag — the audit-recommended direction. All 6 new tests pass deterministically on Q/A's re-run (0.78s), the full 272-test sweep passes with zero regressions, the migration is idempotent (second `--apply` is a no-op), the live MTM result shows 10 of 11 momentum positions trailed up to peak×0.92 (locking ~8pp of give-back exposure into floor protection — SNDK from breakeven $0 to +45.02%), and the BQ schema migration added `entry_strategy STRING NULLABLE` with all 11 existing rows backfilled to `'momentum'`. Scope is clean — only the expected files were touched, no edits to `portfolio_manager.py`/`autonomous_loop.py`/agent skill files. All 5 harness-compliance audit items PASS (researcher gate true, contract written before generate, results artifact contains all 6 required subsections, log-last not yet appended, first Q/A spawn so no verdict-shopping). All 5 code-review dimensions are clean (no BLOCK or WARN findings). Verdict: **PASS**.
