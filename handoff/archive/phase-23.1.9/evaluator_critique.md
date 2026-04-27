---
step: phase-23.1.9
verdict: PASS
cycle_date: 2026-04-27
qa_pass: 1
---

# Q/A Critique — phase-23.1.9

## 5-item harness-compliance audit

1. Researcher brief on disk (`handoff/current/phase-23.1.9-research-brief.md`)
   with `external_sources_read_in_full: 5`, `recency_scan_performed: true`,
   `gate_passed: true` — PASS.
2. Contract front-matter `step: phase-23.1.9` matches step id;
   `verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_9.py'`
   is the immutable command — PASS.
3. `experiment_results.md` includes verbatim verification output
   (`ok 10 paper fields wired + DepositRequest validates`, exit 0) — PASS.
4. `handoff/harness_log.md` not yet appended for `phase=23.1.9`
   (grep count = 0) — PASS (correct ordering: log is the LAST step,
   after PASS).
5. First Q/A spawn for this step — confirmed.

## Deterministic checks

| Check | Command | Result |
|---|---|---|
| A | `PYTHONPATH=. python tests/verify_phase_23_1_9.py` | exit 0; `ok 10 paper fields wired + DepositRequest validates` |
| B | `pytest tests/api/test_settings_api_signal_stack.py tests/api/test_paper_trading_deposit.py tests/services/ -q` | **137 passed** in 2.00s |
| C | `python -c ast.parse(...)` on 4 files | `all syntax ok` |
| D | `cd frontend && npx tsc --noEmit` | exit 0 (silent, 0 errors) |
| J | `git status --short` scope | matches expected file list |

## Code-correctness checks

E. **DepositRequest** (paper_trading.py:48-52): `amount: float = Field(..., gt=0, le=1_000_000)`.
   Endpoint (line 640-693): increments `current_cash`, `starting_capital`, AND
   `total_nav` by the same amount; recomputes
   `total_pnl_pct = ((nav - starting) / starting) * 100`; calls
   `bq.upsert_paper_portfolio(updated)`; calls
   `get_api_cache().invalidate("paper:*")`; logs via `logger.info(...)`. PASS.

F. **`paper_starting_capital` NOT writable** — confirmed absent from
   `SettingsUpdate` (settings_api.py:142-151 contains exactly 9 fields;
   `paper_starting_capital` is only in `FullSettings` line 100 as
   informational read-only). PASS.

G. **9 writable paper fields have ge/le validators** — confirmed in
   settings_api.py:143-151. All 9 use `Field(None, ge=..., le=...)` with
   sensible bounds (e.g., `paper_max_positions` ge=1 le=50;
   `paper_min_cash_reserve_pct` ge=0.0 le=50.0). PASS.

H. **Frontend Manage tab structure** (page.tsx):
   - `{ id: "manage", label: "Manage" }` in TABS (line 338)
   - State vars (line 361+): `manageSettings`, `manageDirty`,
     `depositAmount`, `depositLoading`, etc.
   - `useEffect` at line 489 lazy-loads `getFullSettings()` when
     `tab === "manage" && !manageSettings`
   - `handleDeposit` (line 503) parses amount and validates client-side
   - `handleSettingsSave` early-returns when `manageDirty` empty (line 532)
   - `<PaperSettingNum>` (line 102) only sets dirty on diff
   - Tab content (line 942) correctly placed
   PASS.

I. **Deposit P&L-anchor invariant** — line 659:
   `new_starting = round(starting_before + req.amount, 2)`. Both `current_cash`
   AND `starting_capital` increment by `amount`, so `total_pnl_pct` denominator
   stays anchored — deposit is P&L-neutral by definition. Matches research
   brief Part 2 (Alpaca/Robinhood convention). PASS.

## LLM judgment

- **Cycle accomplishes user ask**: YES. New "Manage" tab provides deposit form
  + 9 paper-trading setting controls; backend POST `/deposit` correctly bumps
  cash + starting_capital + nav atomically; types flow end-to-end.
- **Mutation-resistance**: Verification asserts on field presence + Pydantic
  constraints + DepositRequest boundary validation. 12 unit tests cover
  field-removal, constraint-relaxation, and read-only-discipline mutations.
  Removing any of the 10 paper fields, loosening a validator, or making
  `paper_starting_capital` writable would fail the suite.
- **Anti-rubber-stamp**: experiment_results explicitly defers `paper_deposits`
  BQ audit table + deposit history UI + withdraw button to Phase 2 and
  discloses that deposit currently logs to stdout (no durable audit trail).
  Honest scope-bounding.
- **Scope honesty**: BQ migration / multi-portfolio / idempotency keys / deposit
  rate-limiting all explicitly Phase 2 in the contract.
- **P&L correctness**: Endpoint increments BOTH cash AND starting_capital;
  matches Alpaca/Robinhood convention cited in research brief.
- **Backwards compat**: Manage tab is purely additive (6th tab); existing
  positions/cycle code paths untouched; new settings have safe defaults at
  every layer.
- **Research-gate compliance**: Contract front-matter cites
  `handoff/current/phase-23.1.9-research-brief.md`; envelope shows
  `gate_passed: true`.

## Verdict JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable artifacts present; verification command exits 0 with expected stdout; 137 unit tests pass; frontend tsc clean; DepositRequest correctly increments cash + starting_capital + nav atomically (P&L-anchor invariant); 9 writable paper fields have ge/le validators; paper_starting_capital is genuinely read-only post-init; Manage tab additively wired without disturbing existing tabs; Phase-2 scope honestly disclosed.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "verification_command", "unit_tests", "syntax", "frontend_tsc", "git_diff_scope", "deposit_endpoint_correctness", "settings_update_field_audit", "paper_starting_capital_readonly", "frontend_manage_tab_structure", "pnl_anchor_invariant", "research_gate_envelope"]
}
```
