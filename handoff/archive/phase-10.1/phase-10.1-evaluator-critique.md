# Q/A Critique -- phase-10 / 10.1 (Sprint calendar config)

**Q/A id:** qa_101_v1
**Date:** 2026-04-20
**Step:** 10.1 -- Sprint calendar config (closure cycle)
**Cycle:** 1 (first Q/A on 10.1, no prior CONDITIONAL/FAIL)
**Verdict:** PASS

## 5-item harness-compliance audit (FIRST)

| # | Check | Result |
|---|---|---|
| 1 | Researcher spawned before contract | PASS -- `handoff/current/phase-10.1-research-brief.md` present; tier=simple (closure-audit); `gate_passed: true`; closure envelope valid (no external literature required for pure-YAML-audit closure; internal_files_inspected=2). |
| 2 | Contract mtime < experiment-results mtime | PASS -- contract 1776659850 < experiment-results 1776659880. Research brief 1776659833 precedes both. Order: research -> contract -> generate. |
| 3 | Results verbatim | PASS -- `phase-10.1-experiment-results.md` carries the exact immutable command output, re-verified in this Q/A run (exit 0, all 4 criteria echoed). |
| 4 | Log-last discipline | PASS (pre-flip state) -- `handoff/harness_log.md` last block is `phase=10.0 result=PASS`. No 10.1 block yet, consistent with Q/A-first-then-log-then-flip sequencing. |
| 5 | No second-opinion shopping | PASS -- first Q/A on this step; no prior qa_101_* verdict exists. |

## Deterministic checks A-D

**A. Immutable command verbatim re-run:**

```
$ test -f backend/autoresearch/sprint_calendar.yaml && python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/sprint_calendar.yaml')); assert d['new_weekly_slots'] == 2 and 'thursday' in d['days'] and 'friday' in d['days']"
EXIT=0
```

PASS -- exit 0 matches experiment_results claim.

**B. Independent YAML parse (beyond the immutable assertion):**

```
new_weekly_slots: 2 ==2? True
thursday in days: True
friday in days: True
monthly_anchor.rule: last_trading_friday == last_trading_friday? True
monthly_anchor.hitl: True == True? True
min_challenger_days: 20
```

PASS -- all four `success_criteria` from `.claude/masterplan.json` phase-10.1 are independently confirmed:
- `calendar_config_committed`: file exists on disk.
- `new_weekly_slots_equals_2`: literal 2.
- `thursday_and_friday_defined`: both keys under `days`.
- `monthly_anchor_defined`: section present with `rule=last_trading_friday`, `hitl=True`, `min_challenger_days=20`.

**C. ASCII decode:**

```
ASCII: OK
```

PASS -- `backend/autoresearch/sprint_calendar.yaml` decodes cleanly as ASCII. Compliant with `.claude/rules/security.md` ASCII-only discipline (even though it is a static config, not a logger call).

**D. Scope discipline:**

PASS -- only four new handoff artifacts appeared for this cycle:
- `phase-10.1-research-brief.md`
- `phase-10.1-contract.md`
- `phase-10.1-experiment-results.md`
- `phase-10.1-evaluator-critique.md` (this file)

No code or YAML edit beyond the already-authored `sprint_calendar.yaml` (authored 2026-04-20 01:25 earlier in session, well before this closure cycle at 06:37). mtime of the YAML (1776659516) predates all handoff files for this cycle -- it was NOT touched in this cycle.

## LLM judgment

- **Content sanity (cadence):** Two weekly slots are Thursday `thu_batch` (22:00 UTC, batch-trigger on ~100 candidates) and Friday `fri_promotion` (21:00 UTC, promotion-gate writing ledger row). The Thursday-batch / Friday-promote split matches a disciplined "propose then gate" cadence. `new_weekly_slots: 2` correctly reflects two slots per week.
- **Content sanity (monthly anchor):** `last_trading_friday` rule + `champion_challenger_sortino_gate` + `hitl: true` + `min_challenger_days: 20` are all consistent with a Champion/Challenger promotion flow that requires human approval and a minimum 20-day challenger track record before any swap -- a sound statistical-validity guard.
- **Cross-references:** YAML references phase-8.5.2 `budget.py`, phase-8.5.5 `gate.py`, phase-10.0 supersede doc, and `harness_log.md` -- consistent with phase-10.0 PASS (qa_100_v1) which retired 8.5.7 and pointed at this config.
- **Schema clarity:** `new_weekly_slots`, `days.<day>.{slot_id, role, time_utc, notes}`, and `monthly_anchor.{rule, role, hitl, min_challenger_days, notes}` -- unambiguous, no hidden keys.
- **Crypto defense-in-depth:** No hits for `crypto|bitcoin|btc|eth|coinbase|binance` in the YAML. Consistent with phase-5 crypto-removal directive.
- **Scope honesty:** experiment_results correctly discloses this is a closure cycle -- YAML was authored earlier in session, not in this cycle. No overclaim.

## Violated criteria

None.

## Violation details

(empty)

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_101_v1",
  "reason": "All 4 success_criteria + immutable command verified independently. YAML parses cleanly, ASCII-clean, cadence + monthly-anchor semantics coherent, cross-references align with phase-10.0 (qa_100_v1) and phase-8.5.2/8.5.5. 5/5 protocol audit, 4/4 deterministic A-D. No scope creep; no crypto contamination.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "independent_yaml_parse",
    "ascii_decode",
    "scope_discipline",
    "crypto_defense_in_depth",
    "log_last_ordering",
    "contract_mtime_before_generate"
  ]
}
```

## Decision guidance for Main

PASS. Proceed with log-last (append `phase=10.1 result=PASS` block to `handoff/harness_log.md`) THEN flip `.claude/masterplan.json` phase-10 / step 10.1 status to `done`. Do not bundle log-append with status-flip.
