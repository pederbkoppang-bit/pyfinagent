# Research Brief: phase-10.7 — Rollback Kill-Switch Wiring

**Tier:** moderate  **Accessed:** 2026-04-20

## Queries run (three-variant)

1. Current-year: "kill switch circuit breaker live trading drawdown auto-demote 2026"
2. Last-2-year: "automated strategy demotion champion challenger rollback 2025"
3. Year-less canonical: "asymmetric approval HITL promotion automatic demotion trading algorithmic"

## Read in full (7 sources; gate floor ≥5)

| URL | Kind | Key finding |
|-----|------|-------------|
| https://www.mql5.com/en/blogs/post/767321 | Practitioner blog (2026-02) | Auto-freeze BEFORE rule violation; stopping is fully automatic, no human gate |
| https://www.modelop.com/ai-governance/glossary/champion-challenger-testing | Vendor doc | Champion/challenger quality gate "reduces risk of performance regression" |
| https://validmind.com/blog/sr-11-7-model-risk-management-compliance/ | Authoritative blog | SR 11-7 requires "clear audit trail"; JSONL log sufficient |
| https://arxiv.org/html/2512.02227v1 | arXiv Dec 2025 | UUID/SHA256 + time-stamped JSONL for replay/audit; binary pass/fail gates |
| https://www.rulematch.com/trading/kill-switch/ | Exchange doc | Asymmetric pattern: auto unilateral block vs manual for judgment calls |
| https://www.finra.org/rules-guidance/key-topics/algorithmic-trading | Regulatory | Rule 3110: post-trade anomaly detection; automated controls must be independently logged |
| https://rngstrategyconsulting.com/insights/industry/financial-services/algorithmic-trading-strategies-regulation-risk-governance/ | Industry | HITL for promotion-level decisions; automated safety actions acceptable |

## Recency scan (2024-2026)

Most relevant recent work: arXiv 2512.02227 (Dec 2025) confirms JSONL audit + binary pass/fail gates. MQL5 Feb 2026 confirms "auto-freeze BEFORE rule violation" pattern. No 2024-2026 work supersedes the existing `DD_TRIGGER = 0.10` from `promoter.py:15`.

## Key findings

1. **`DD_TRIGGER = 0.10` already exists** at `promoter.py:15`. Don't re-derive — import from `backend.autoresearch.promoter`.
2. **`kill_switch.py` is a SYSTEM-WIDE pause** — `KillSwitchState.pause()` halts ALL trading. Do NOT use for per-challenger demotion.
3. **`compliance_logger.write_rationale()` requires `approver_id` + `decision in {approve,reject,modify}`** — wrong schema for an automated event with no human approver. Will raise `ValueError`.
4. **Correct audit pattern:** JSONL append to `handoff/demotion_audit.jsonl`, mirroring the `kill_switch_audit.jsonl` pattern at `kill_switch.py:77-87`.
5. **Ledger sink:** `weekly_ledger.py` `notes` column (free-text, no schema change) accepts `"auto_demoted:challenger_id:dd=X.XX"`.
6. **State integration:** `monthly_champion_challenger.py` state file supports a new terminal status `"auto_demoted"` alongside existing `approved/rejected/expired`.
7. **Asymmetric approval is canonical:** demotion is automatic (safety); promotion is HITL. RULEMATCH + MQL5 + FIA/FINRA all confirm.

## Internal inventory

| File | Role | Status |
|------|------|--------|
| `backend/services/kill_switch.py` | System-wide pause; JSONL audit at kill_switch.py:77-87 | Active; audit pattern reusable |
| `backend/autoresearch/promoter.py:15` | `DD_TRIGGER = 0.10` | Import as single source of truth |
| `backend/autoresearch/promoter.py:40-49` | `on_dd_breach(current_dd, kill_fn)` | Active; wire kill_fn to rollback |
| `backend/autoresearch/monthly_champion_challenger.py:172-183` | State file supports new terminal statuses | Add "auto_demoted" |
| `backend/autoresearch/weekly_ledger.py:21-30` | `notes` column free-text | Append demotion marker |
| `backend/services/compliance_logger.py:152-194` | WORM `write_rationale`; requires approver | NOT for automated events |

## Recommended design (Option A)

**Module:** `backend/autoresearch/rollback.py` (new).

**Public function:**
```python
def auto_demote_on_dd_breach(
    *,
    challenger_id: str,
    challenger_current_dd: float,  # negative float like -0.11
    dd_threshold: float = DD_TRIGGER,  # imported from promoter.py
    state_path: Path | None = None,
    audit_path: Path | None = None,
    ledger_path: Path | None = None,
    week_iso: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Returns {demoted, decision: "auto_demoted"|"no_breach", challenger_id, dd, threshold, ts}.
    Writes to three sinks: (1) state JSON status="auto_demoted", (2) JSONL audit append,
    (3) weekly_ledger notes. No human-approval code path."""
```

**Idempotency:** If state already shows `status="auto_demoted"`, short-circuit as no-op.

**Test script (`scripts/harness/phase10_rollback_test.py`)** — 3 cases matching masterplan verbatim:
1. `challenger_dd_breach_auto_demotes` — dd=-0.11 → demoted=True, decision="auto_demoted"
2. `demotion_logged_with_auto_demoted_decision` — JSONL contains the decision record
3. `no_human_approval_required_for_demotion` — function signature has no `slack_fn` / approver kwargs; call completes without any HITL gate

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/phase-10.7-research-brief.md",
  "gate_passed": true
}
```
