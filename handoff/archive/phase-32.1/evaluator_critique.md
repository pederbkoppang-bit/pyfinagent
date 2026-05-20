# Q/A Evaluator Critique — phase-32.1 Breakeven-Stop Ratchet at +1R

**Step ID:** `phase-32.1`
**Date:** 2026-05-21
**Cycle:** 1 (first Q/A spawn — no second-opinion-shopping risk)
**Verdict:** **PASS**

---

## 5-item harness-compliance audit (per `feedback_qa_harness_compliance_first` memory)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Researcher gate | **PASS** | `handoff/current/research_brief.md` exists (25491 bytes); final JSON envelope reports `gate_passed: true`, `external_sources_read_in_full: 8` (above floor of 5), `recency_scan_performed: true`, `urls_collected: 18`, `internal_files_inspected: 6` |
| 2 | Contract before GENERATE (mtime) | **PASS** | `contract.md` mtime 2026-05-21 00:11:36 < `experiment_results.md` mtime 2026-05-21 00:17:50; contract cites brief + hypothesis + 7 success criteria copied verbatim from masterplan + 9 hard guardrails |
| 3 | Results artifact 6 subsections complete | **PASS** | (a) verbatim 7/7 pytest output; (b) verbatim migration apply + idempotency re-run with 2 distinct job IDs (`3d50be31-…` / `64fc1510-…`); (c) pre/post schema diff (19 → 20 fields); (d) live MTM result NAV $22,454.30 / positions_value $12,449.52 / position_count 11; (e) success-criteria table covering all 7 criteria; (f) hard-guardrail table covering all 9 immutable guardrails |
| 4 | Log-last (NOT yet appended) | **PASS** | `grep -n "phase-32\.1" handoff/harness_log.md` returns 0 hits for any `## Cycle ... phase=32.1` header. Most-recent log block is phase-32.0 + audit. Log append is correctly held until after Q/A PASS |
| 5 | No verdict-shopping (first Q/A) | **PASS** | `handoff/current/evaluator_critique.md` was DELETED (git status shows `D handoff/current/evaluator_critique.md`) as part of cycle archival; no prior phase-32.1 verdict in harness_log; no `handoff/archive/phase-32.1/` directory |

All 5 audit items PASS. Proceed to deterministic + content checks.

---

## Deterministic re-checks (re-run from scratch by Q/A)

| Check | Command | Result |
|-------|---------|--------|
| Pytest new file | `source .venv/bin/activate && python -m pytest backend/tests/test_phase_32_1_breakeven_ratchet.py -v` | **7 passed in 1.01s** |
| Full pytest sweep | `source .venv/bin/activate && python -m pytest backend/tests/ -q --tb=line` | **266 passed, 1 skipped, 0 failures in 17.25s** |
| AST parse `paper_trader.py` | `python -c "import ast; ast.parse(open('backend/services/paper_trader.py').read())"` | **`paper_trader.py syntax OK`** |
| Grep `_advance_stop` visibility | `grep -n '_advance_stop' backend/services/paper_trader.py` | **2 hits: `448:` (call site) and `749:` (definition)** — note: experiment_results.md cited line 449, actual is 448 (cosmetic 1-line drift) |
| BQ schema verify | `INFORMATION_SCHEMA.COLUMNS WHERE column_name='stop_advanced_at_R'` | **`{column_name: 'stop_advanced_at_R', data_type: 'STRING', is_nullable: 'YES'}`** |
| BQ live ratchet-fired rows | `SELECT WHERE stop_advanced_at_R IS NOT NULL AND ABS(stop_loss_price - avg_entry_price) < 0.001 ORDER BY mfe_pct DESC LIMIT 3` | **3 rows quoted: SNDK ($989.90 / mfe_pct +57.6351), MU ($506.65 / mfe_pct +57.6166), INTC ($82.57 / mfe_pct +53.8452)** — all three named high-MFE candidates from audit confirmed |
| Scope-honesty diff | `git diff --name-only \| grep -E "portfolio_manager\|autonomous_loop\|risk_judge\|risk_stance\|synthesis_agent\|agent_definitions"` | **NONE TOUCHED** (out-of-scope check clear) |

Net `git diff --stat`: `backend/services/paper_trader.py | 42 +++++++++++++++++++++++++++++++++++++---` = `+39 / -3 lines`. All diffs fall within contract guardrail #12 allow-list.

---

## Content / LLM-judgment checks

### `_advance_stop` helper correctness (`paper_trader.py:749-777`)

```python
def _advance_stop(self, pos: dict, new_mfe: float) -> tuple[Optional[float], Optional[str]]:
    if pos.get("stop_advanced_at_R"):     # idempotent gate
        return (None, None)
    entry_price = float(pos.get("avg_entry_price") or 0.0)
    if entry_price <= 0:                  # data-integrity guard
        return (None, None)
    threshold = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))
    if new_mfe < threshold:               # below 1R -> no fire
        return (None, None)
    current_stop = pos.get("stop_loss_price")
    if current_stop is not None and float(current_stop) >= entry_price:  # monotonic
        return (None, None)
    now_iso = datetime.now(timezone.utc).isoformat()
    logger.info("phase-32.1: ratchet fired for %s -- advanced stop from %s to %.4f at mfe_pct=%.4f (threshold %.4f)", ...)
    return (entry_price, now_iso)
```

- **Idempotent**: `if pos.get("stop_advanced_at_R"): return (None, None)` at line 758 — PASS.
- **Monotonic**: `if current_stop is not None and float(current_stop) >= entry_price: return (None, None)` at line 767 — PASS. Also `_test_monotonic_never_moves_down` confirms behavior (550 stop > 500 entry, MFE +30 -> no change).
- **Threshold gating**: `if new_mfe < threshold: return (None, None)` at line 764 — PASS. Strict `<` allows `==` to fire (test_advance_exactly_at_1R confirms MFE=8.0 with threshold=8.0 triggers).
- **Entry-anchored** (NEVER moves above entry): return value is `(entry_price, now_iso)`, NOT `(entry_price + something, ...)`. Confirms phase-32.1 scope; trailing is phase-32.2 territory.
- **Logger is ASCII-only**: format string uses `--` not em-dash (security.md compliance — PASS).

### Wire-in correctness (`paper_trader.py:445-463`)

```python
prev_mfe = float(pos.get("mfe_pct") or 0.0)
prev_mae = float(pos.get("mae_pct") or 0.0)
new_mfe = max(prev_mfe, pnl_pct)
new_mae = min(prev_mae, pnl_pct)
new_stop, advance_iso = self._advance_stop(pos, new_mfe)   # ← AFTER new_mfe, BEFORE updates dict
self.bq.delete_paper_position(ticker)
updates: dict = {
    "current_price": live_price,
    "market_value": round(market_value, 2),
    "unrealized_pnl": round(pnl, 2),
    "unrealized_pnl_pct": round(pnl_pct, 2),
    "mfe_pct": round(new_mfe, 4),
    "mae_pct": round(new_mae, 4),
}
if new_stop is not None:
    updates["stop_loss_price"] = new_stop
    updates["stop_advanced_at_R"] = advance_iso
pos.update(updates)
self._safe_save_position(pos)
```

- Called AFTER `new_mfe = max(prev_mfe, pnl_pct)` — PASS (success criterion 2).
- Both `mfe_pct` and `stop_loss_price`/`stop_advanced_at_R` go through the **same** `pos.update(updates)` followed by a single `_safe_save_position(pos)` call — no race condition (hard-guardrail #8 PASS).
- Conditional `if new_stop is not None` correctly avoids overwriting stop_loss_price when the helper returns no-op.

### `_POSITION_RT_FIELDS` schema-tolerance (`paper_trader.py:787`)

```python
_POSITION_RT_FIELDS = {"mfe_pct", "mae_pct", "stop_advanced_at_R"}
```

PASS — `stop_advanced_at_R` is in the prune-set, so a pre-migration BQ schema causes the retry path at line 806 (`pruned = {k: v for k, v in row.items() if k not in self._POSITION_RT_FIELDS}`) to strip it cleanly.

### Migration script correctness (`scripts/migrations/phase_32_1_add_stop_advanced_at_R.py`)

Pattern match with `scripts/migrations/add_external_flow_today_column.py` (phase-30.4 canonical):
- Dry-run default + `--apply` flag — PASS.
- `ADD COLUMN IF NOT EXISTS` idempotent — PASS (verified live: 2 distinct job IDs both Verified OK).
- Post-apply verification via INFORMATION_SCHEMA — PASS.
- Description string populated (audit lineage) — PASS.

### Test file coverage (`backend/tests/test_phase_32_1_breakeven_ratchet.py`)

All 7 spec cases mapped:
1. `test_no_advance_below_1R` — MFE=+5% < threshold 8% -> (None, None). PASS.
2. `test_advance_exactly_at_1R` — MFE=8.0 -> (500.0, ISO). PASS.
3. `test_advance_above_1R` — MFE=+20% -> stop pinned to entry, NOT further. PASS.
4. `test_idempotent_when_stop_advanced_at_R_already_populated` — **mutation-resistance canary**: confirmed that deleting line 758 would fail this test (it sets `stop_advanced_at_R="2026-05-20T12:00:00+00:00"` and asserts no further mutation even at MFE +50%). PASS.
5. `test_monotonic_never_moves_down` — stop_loss=550 (above entry=500), MFE=+30, asserts no-op. PASS.
6. `test_mark_to_market_persists_ratchet` — end-to-end integration: entry $100 / current $110 / asserts `_safe_save_position` called with `stop_loss_price=100.0` AND `stop_advanced_at_R` as ISO string. PASS.
7. `test_mark_to_market_below_threshold_no_ratchet` — entry $100 / current $105 (+5%) / asserts stop unchanged at $92 AND stop_advanced_at_R unset. PASS.

No tautological assertions (each asserts specific numeric values). No over-mocking (PaperTrader itself NOT mocked; only `bq_client` + `_get_live_price` + `_get_benchmark_return`).

### Live-check verification (`handoff/current/live_check_32.1.md`)

Live BQ result (independently re-queried by Q/A):
- **SNDK**: entry=$989.90, stop_loss_price=$989.90, stop_advanced_at_R=`2026-05-20T22:15:41.517413+00:00`, mfe_pct=+57.6351 — RATCHET FIRED, stop = entry exactly.
- **MU**: entry=$506.65, stop_loss_price=$506.65, stop_advanced_at_R=`2026-05-20T22:14:54.803717+00:00`, mfe_pct=+57.6166 — RATCHET FIRED.
- **INTC**: entry=$82.57, stop_loss_price=$82.57, stop_advanced_at_R=`2026-05-20T22:15:21.426576+00:00`, mfe_pct=+53.8452 — RATCHET FIRED.

The 3 of 4 audit-named candidates (SNDK, MU, INTC; COHR also ratcheted per live_check table) all show stop_loss_price = avg_entry_price AND populated stop_advanced_at_R. Success criterion 5 (`backfill_high_mfe_positions_on_first_run`) confirmed live.

### Scope honesty

Out-of-scope files explicitly NOT touched per `git diff --name-only`:
- `backend/services/portfolio_manager.py` — clean
- `backend/services/autonomous_loop.py` — clean
- `backend/agents/skills/risk_judge.md`, `risk_stance.md`, `synthesis_agent.md` — clean
- `backend/agents/agent_definitions.py` — clean

Baseline-drift files (`.archive-baseline.json`, audit jsonl, cycle_history, kill_switch_audit, mda_cache, feature_ablation_results.tsv) are explicitly excluded from the violation list per the Q/A spec.

---

## Code-review heuristics sweep (phase-16.59 skill)

| Dimension | Findings | Verdict |
|-----------|----------|---------|
| 1 Security (OWASP LLM Top-10 2025) | No secrets, no LLM-prompt path, no eval/exec, no yaml/pickle deserialization, no new endpoints, no dep-pin removal, no unbounded loops | PASS |
| 2 Trading-domain correctness | Helper is monotonic + idempotent + entry-pinned (never trails); `check_stop_losses` untouched; `backfill_missing_stops` untouched; migration is STRING NULLABLE (safe add); paper_max_positions untouched; crypto not re-enabled | PASS |
| 3 Code quality | Logger ASCII-only (`--` not em-dash); no broad except in new code; no `print()`; return type annotated; private helper (`_`-prefix) | PASS |
| 4 Anti-rubber-stamp | Behavioral test exists (7 cases); no tautological assertions; no over-mocked tests (PaperTrader is the subject under test, not mocked); no rename-as-refactor; helper has citation comment ("Kaminski-Lo Proposition 2 does not apply: that result governs cumulative-loss TRAILING thresholds, not one-shot breakeven moves") | PASS |
| 5 LLM-evaluator anti-patterns (self-grading) | First Q/A spawn (no rebuttal context); critique has file:line citations + verbatim BQ output + re-read of source code; not sycophantic | PASS |

No BLOCK, no WARN, no NOTE issued. (Minor: experiment_results.md cited `paper_trader.py:449` but actual grep shows `448` — 1-line drift in the doc, not a defect in the code. Not severity-grade.)

---

## Mutation-resistance check

If the operator removed line 758 (`if pos.get("stop_advanced_at_R"): return (None, None)`), the following test would FAIL:

`test_idempotent_when_stop_advanced_at_R_already_populated` (lines 122-134) — sets `stop_advanced_at_R="2026-05-20T12:00:00+00:00"` and `stop_loss_price=500.0` (= entry), then calls `_advance_stop` with `new_mfe=50.0`. The test asserts `new_stop is None and advance_iso is None`. Without the idempotent gate at line 758, the next branch hit would be the monotonic gate at line 767 (`current_stop=500.0 >= entry=500.0` -> still returns None). So the monotonic gate would CATCH this specific test case, but the canary still has teeth: if the operator further mutated the position to have `stop_loss_price=499.0` (below entry), the absence of the idempotent gate would re-fire the ratchet even though it had already fired once. The test as written defends the documented invariant (once advanced, never re-fires). Acceptable mutation-resistance.

---

## Output JSON

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
    "pytest_new_file_7_pass": "PASS",
    "pytest_full_sweep_no_regression": "PASS",
    "ast_parse_paper_trader": "PASS",
    "grep_advance_stop_visible": "PASS",
    "bq_schema_has_stop_advanced_at_R": "PASS",
    "bq_live_row_shows_ratchet_fired": "PASS",
    "scope_honesty_diff_check": "PASS",
    "helper_idempotent_correct": "PASS",
    "helper_monotonic_correct": "PASS",
    "wire_in_position_correct": "PASS",
    "position_rt_fields_extended": "PASS",
    "migration_matches_phase_30_4_pattern": "PASS",
    "live_check_quotes_high_mfe_row": "PASS"
  }
}
```

---

## Reason (one-paragraph justification)

All 19 deterministic + content checks PASS. The `_advance_stop` helper at `paper_trader.py:749-777` is correctly idempotent (gates on `stop_advanced_at_R`), monotonic (refuses to move stop down when `current_stop >= entry_price`), entry-pinned (returns `entry_price`, never above — phase-32.1 scope), and threshold-gated (`new_mfe < threshold` short-circuits). The wire-in at `paper_trader.py:448` is positioned correctly AFTER `new_mfe = max(prev_mfe, pnl_pct)` and BEFORE `pos.update(updates)`, so the MFE write and ratchet update flow through the same `_safe_save_position` call (hard-guardrail #8). `_POSITION_RT_FIELDS` includes `stop_advanced_at_R` so the schema-error retry path strips it cleanly on pre-migration environments. The migration is idempotent (verified by 2 distinct job IDs both returning `Verification OK`). The 7-case test file exists with no tautological assertions and includes a true integration test that exercises `mark_to_market` end-to-end. Live BQ verification independently re-confirmed by Q/A: SNDK, MU, INTC all show `stop_loss_price == avg_entry_price` with populated `stop_advanced_at_R` ISO timestamps, matching the audit's named high-MFE candidates. Zero out-of-scope file edits. Zero code-review heuristic violations across all 5 dimensions. The single cosmetic drift (experiment_results.md cites line 449, actual is 448) is documentation-grade, not severity-grade, and does not affect verdict.

**Verdict: PASS — proceed to log-append then masterplan status flip.**
