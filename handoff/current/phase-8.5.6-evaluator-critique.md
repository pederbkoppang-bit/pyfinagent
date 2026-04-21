# Q/A Critique — phase-8.5 / 8.5.6 REMEDIATION (full-breach)

**Verdict:** PASS
**Q/A id:** `qa_856_remediation_v1`
**Agent:** fresh Q/A subagent (supersedes inline qa_856_v1)
**Date:** 2026-04-20 17:3x UTC

## 1. Harness-compliance audit (5/5 PASS)

| # | Check | Result |
|---|-------|--------|
| 1 | Researcher brief >=5 sources read in full | PASS — 5 fetched-in-full (Alpaca guide, Balaena DSR, Wikipedia DSR, 3commas 2025, QuantStart Kelly); 7 snippet-only; 12 URLs; recency scan performed; `gate_passed: true` |
| 2 | Contract mtime < results mtime | PASS — contract 17:31:29 < results 17:31:44 (research brief 17:30:20 precedes both) |
| 3 | Results surface the 3 advisories | PASS — experiment-results.md L5-8 enumerates all three advisories verbatim |
| 4 | Log-last tail = 8.5.5 remediation 04:28 UTC | PASS — `handoff/harness_log.md` tail is `REMEDIATION -- 2026-04-20 04:28 UTC -- phase=8.5.5` |
| 5 | First Q/A on new evidence (not verdict-shopping) | PASS — inline qa_856_v1 was authored without a dedicated researcher spawn; this fresh Q/A is running against freshly-authored 12305-byte brief and remediation-v1 contract/results. Evidence materially new. |

## 2. Deterministic checks A-E (5/5 PASS)

### A. `python scripts/harness/autoresearch_promotion_test.py` → exit 0 + 3/3 PASS
```
PASS: shadow_5_trading_days_minimum -- shadow_trading_days >= 5 enforced
PASS: position_size_tied_to_realized_dsr -- position_size scales with realized DSR
PASS: kill_switch_auto_triggers_on_dd_breach -- kill_switch auto-triggers on |dd| > 0.1
---
PASS
EXIT=0
```
PASS.

### B. `promoter.py` structural requirements
- `@dataclass(frozen=True)` — line 19. PASS.
- `SHADOW_MIN_DAYS=5` — line 14. PASS.
- `DD_TRIGGER=0.10` — line 15. PASS.
- `DSR_MIN_FOR_PROMOTION=0.95` — line 16. PASS.
- position_size formula `max(0.0, min(1.0, (dsr - 0.5) * 2.0))` — line 37. PASS.

### C. Regression
103 tests collected (6 collection errors are pre-existing Slack-bot import issues, present on 8.5.5 baseline and unchanged). Delta from 8.5.5 = 0. Non-regression PASS.

### D. Shadow-days boundary semantics
- `shadow_trading_days=5` → `promoted: True` (NOT strict >). PASS.
- `shadow_trading_days=4` → `promoted: False`, reason `shadow_days_below_min:4<5`. PASS.
Test harness case_shadow_min covers both directions.

### E. Position-size spot-check
| DSR | Expected | Observed |
|-----|----------|----------|
| 0.30 | 0.0 | 0.0 |
| 0.50 | 0.0 | 0.0 |
| 0.60 | 0.20 (lenient advisory) | 0.199999... |
| 0.75 | 0.50 | 0.50 |
| 1.00 | 1.00 | 1.00 |
PASS. Formula matches spec exactly; DSR=0.6 → 20% notional confirms advisory #2 quantitatively.

### DD boundary probe (beyond the listed A-E)
- `current_dd=-0.10` → NOT fired (strict `>`, not `>=`). Confirmed.
- `current_dd=-0.1001` → fired. Confirmed.
Matches docstring "abs(dd) exceeds dd_trigger" and test case.

## 3. LLM judgment

### Contract alignment
Contract lists 3 success_criteria implicit in "Immutable: test exit 0 + 3/3 PASS". All literally met. Advisories are correctly labelled CARRY-FORWARD, not criterion violations.

### Advisory substantiveness (all 3 honest, NOT overclaiming)
1. **5-day shadow window vs 30-90 day industry floor** — brief cites Alpaca (median 30 days) and 3commas 2025 (30-90 days for AI bots) as authoritative, with arXiv:2402.05272v2 noting 5-day windows stressed in volatility regimes. Recommendation (add `min_trades>=N` gate OR document as harness-CI only) is well-scoped.
2. **DSR=0.6 → 20% notional is lenient** — arithmetic verified in deterministic check E. Raising floor to 0.7 would zero DSR in [0.5, 0.7); legitimate configurability ask.
3. **`current_dd` caller-contract ambiguous** — docstring at promoter.py:45 says "abs(dd) exceeds dd_trigger" but does not specify rolling peak-to-trough vs single-bar. Substantive documentation gap, correctly flagged.

### Mutation-resistance
The test suite uses exact `==` assertions (e.g. `position_size({"dsr":0.75}, 10000) == 5000.0`) and boundary cases at day=3 (reject) and day=5 (accept). Flipping `<` to `<=` on line 28 would promote at day=4, caught by case_shadow_min. Changing fraction formula would fail the three equality assertions. Mutation-resistant.

### Scope honesty
Experiment-results discloses all 3 advisories as carry-forward, not as PASS criteria met. Brief's "Debate" section is candid about the 5-day vs 30-90 day disagreement. No overclaiming observed.

### Research-gate compliance
Contract references research findings. Brief emits `gate_passed: true` envelope with 5 read-in-full + recency scan + three-variant query discipline. Compliant.

## 4. Violated criteria
None.

## 5. Disposition — carry-forward to hardening cycle
Open a follow-up step (phase-8.5.7 or phase-9.x hardening) that:
1. **Scopes the 5-day gate** — add module docstring clarifying SHADOW_MIN_DAYS=5 is a harness-CI floor, NOT live-capital floor; or introduce `min_trades>=N` as a second shadow condition.
2. **Makes DSR floor configurable** — promote the currently-hardcoded 0.5 in `(dsr - 0.5) * 2.0` to a dataclass field (e.g. `dsr_size_floor: float = 0.5`), default unchanged, so operators can tighten to 0.7 without forking.
3. **Clarifies `on_dd_breach` caller-contract** — docstring amendment at promoter.py:45 specifying `current_dd` must be running peak-to-trough drawdown, not single-bar return; add assertion or comment.

None of these block 8.5.6 PASS; they are explicit carry-forward items.

## 6. JSON envelope
```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Immutable test exit=0 + 3/3 PASS; 5/5 harness audit; 5/5 deterministic A-E; 3 advisories surfaced honestly as carry-forward; fresh Q/A on new evidence, not verdict-shopping.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_audit_5",
    "verification_command",
    "promoter_structural",
    "regression_delta",
    "shadow_boundary",
    "position_size_spot_check",
    "dd_boundary_probe",
    "mutation_resistance",
    "research_gate_envelope"
  ]
}
```
