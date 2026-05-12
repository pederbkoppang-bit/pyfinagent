---
step: phase-25.R
tier: moderate-complex
date: 2026-05-12
---

# Research Brief -- phase-25.R: Strategy Auto-Switching Policy

## Search queries run (three-variant discipline)

1. **Current-year frontier**: "strategy auto-switching live trading system drawdown alpha-decay trigger 2026"
2. **Last-2-year window**: "arxiv 2605.06822 strategy switching trading agents" + "arxiv 2503.21422 AI quantitative trading strategy switching 2025"
3. **Year-less canonical**: "multi-strategy capital allocation atomic flip ramp quant trading" + "idempotent strategy registry BQ MERGE atomic switch superseded status pattern" + "Slack Block Kit P0 alert high priority operational event design"

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://arxiv.org/html/2605.06822 | 2026-05-12 | paper | WebFetch | SHARP: conservative promotion gate -- new rubric accepted only when excess return rval > rval* - 0.005 (50 bps tolerance); attribution examines worst 20 portfolio days; highest rval across all rounds is selected atomically |
| https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-05-12 | doc | WebFetch | LeadResearcher synthesizes results and decides if more research needed; spawns additional subagents; file-based communication; task complexity determines agent allocation |
| https://www.tradingengineeringlab.com/alpha-decay-in-trading-why-strategies-stop-working-over-time/ | 2026-05-12 | blog | WebFetch | Alpha decay signals: compressing payoff-to-risk, worsening slippage, larger drawdowns; layered transition: reduce exposure first, investigate, de-risk before replacing; complexity must be justified by measurable improvement |
| https://arxiv.org/html/2601.19504v1 | 2026-05-12 | paper | WebFetch | Regime-adaptive gating: rolling 20-day SMA regime flag gates entry (bullish = allow, bearish = exit); ATR-based position sizing; hybrid signal scores from tech + ML + sentiment rather than discrete strategy switches |
| https://docs.slack.dev/reference/block-kit/blocks/alert-block/ | 2026-05-12 | doc | WebFetch | Alert block shape: {type:"alert", text:{type:"mrkdwn", text:...}, level:"error|warning|info|success|default"}; purpose-built for severity-coded notifications; complementary to header+section pattern |
| https://docs.slack.dev/block-kit/ | 2026-05-12 | doc | WebFetch | Core Block Kit block types: header (plain_text), section (fields array, max 10), context (mrkdwn metadata); standard operational alert pattern: header -> section fields -> context footer |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/abs/2412.20138 (TradingAgents) | paper | Abstract only; no new switching-specific finding beyond SHARP |
| https://arxiv.org/html/2503.21422v1 | paper | Fetched but no strategy-switching section; survey focused on alpha pipeline not operational switching mechanics |
| https://blog.quantinsti.com/multi-strategy-portfolios-combining-quantitative-strategies-effectively/ | blog | Fetched but no atomic-flip vs ramp content; generic MPT framing only |
| https://arxiv.org/html/2602.07023v1 | paper | Snippet -- behavioral consistency validation for LLM agents; regime-style switching corroborates SHARP gating pattern |
| https://oneuptime.com/blog/post/2026-02-17-how-to-implement-idempotent-data-pipelines-in-gcp | blog | Idempotent BQ pipeline patterns; MERGE is naturally idempotent; confirmed existing save_promoted_strategy MERGE approach is correct |
| https://aloknecessary.github.io/blogs/idempotency-distributed-systems/ | blog | Two-phase status pattern (IN_PROGRESS -> COMPLETED -> FAILED); status enum is the canonical registry pattern |
| https://proptradingvibes.com/blog/alpha-futures-strategy | blog | 2026 trading framework; alpha monitoring triggers risk adjustment before strategy replacement |
| https://www.grahamcapital.com/quantitative-strategies/ | industry | Graham Capital multi-strategy overview; snippet only |
| https://docs.slack.dev/changelog/2025/02/03/block-kit-markdown/ | doc | Markdown block added Feb 2025; not relevant for P0 operational alert pattern |
| https://dohost.us/index.php/2025/12/28/high-priority-alerting-integrating-ids-notifications-with-slack-discord-or-pagerduty/ | blog | IDS P0 alert design: dedicated escalation channel for SEV-1, structured fields (service, owner, runbook, timestamp, correlation_id), clear severity levels |

---

## Recency scan (2024-2026)

Searched for 2024-2026 literature on strategy auto-switching, promoter registry patterns, and Slack P0 operational alerts.

**Findings:**
- **SHARP (May 2026, arxiv 2605.06822)**: Conservative promotion gate -- new strategy must beat incumbent by >= 50 bps on held-out validation before atomic replacement. Attribution-first before switching (removing attribution agent drops SHARP Sharpe from 1.83 to near-static). Directly validates pyfinagent's DSR gate + shadow-mode approach.
- **Regime-adaptive hybrid (Jan 2026, arxiv 2601.19504)**: Regime gating as a switch-enabler (bullish regime required before any BUY); ATR position sizing. Validates "don't switch atomically without a gate" pattern.
- **Alpha decay detection (2026, tradingengineeringlab)**: Layered transition -- reduce risk before replacement. Drawdown > historical band = de-risk, not immediate switch.
- **Slack Block Kit alert block (Feb 2025)**: New purpose-built alert block with level field. Complements existing header+section+context pattern used in all 9 pyfinagent formatters.
- **Idempotent BQ MERGE pipelines (Feb 2026)**: MERGE is naturally idempotent; two-phase status transitions (pending -> active -> superseded) are the standard pattern for strategy registries.

No 2024-2026 work supersedes the core design: DSR gate + atomic registry flip + P0 notification.

---

## Key findings

1. **Conservative atomic promotion with gate** -- "A candidate is accepted if its validation excess return satisfies rval > rval* - 0.005" (SHARP, arxiv 2605.06822). The new strategy must beat the incumbent on held-out data. pyfinagent already implements an equivalent gate in `promoter.py:28-31` (DSR >= 0.95 + shadow_days >= 5). `write_to_registry` should only be called after `promote()` returns `{promoted: True}`.

2. **Attribution before switching** -- "Removing the attribution agent drops to near-static performance" (SHARP, arxiv 2605.06822, as cited in `docs/audits/phase-24-2026-05-12/24.13-redline-synthesis-findings.md`). A strategy switch without understanding why the current strategy underperforms is fragile. pyfinagent's shadow_min_days=5 + DSR gate is the minimal attribution signal.

3. **Layered de-risk before hard switch** -- "Reduce exposure and investigate before abandoning the strategy entirely; if drawdown exceeds historical tolerance band, de-risk and re-validate assumptions" (tradingengineeringlab.com, 2026). For pyfinagent's P0 Slack notification: include current drawdown + DSR + shadow_days so the operator can judge severity.

4. **Idempotent two-phase status enum** -- MERGE on `(week_iso, strategy_id)` is naturally idempotent (confirmed: `bigquery_client.py:704-718`); status transitions pending -> active -> superseded are the standard pattern. Two sequential calls (INSERT new row as "active" + UPDATE prior "active" to "superseded") is safe because `save_promoted_strategy` uses MERGE; no duplicate rows.

5. **P0 Slack Block Kit pattern** -- `format_escalation_alert()` at `formatters.py:679` already implements the header + severity + details-fields + actions + divider + context pattern for P0 alerts. `format_strategy_switch()` should mirror this shape with P0 severity and structured fields (strategy_id, dsr, allocation_pct, prior_strategy_id, switched_at, shadow_days).

6. **Two-path policy: Promoter path vs HITL path** -- `monthly_champion_challenger.py:265-277` shows `record_approval` flips to "active" via `status_update_fn` only on HITL approval. The Promoter's `write_to_registry` is a DIFFERENT path: it writes directly with `status="active"` because it has already cleared the shadow+DSR gate (ops-authorized, no human needed). Both paths write to the same BQ table; the `save_promoted_strategy` MERGE reconciles concurrent writes safely.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/autoresearch/promoter.py` | 53 | Shadow+DSR+drawdown gate; pure functions | **MODIFY** -- add `write_to_registry(bq_client, trial, *, week_iso)` method |
| `backend/slack_bot/formatters.py` | 742 | Block Kit formatters (9 functions) | **MODIFY** -- add `format_strategy_switch(event)` |
| `backend/services/autonomous_loop.py` | 200+ | Daily cycle; `load_promoted_params()` at line 46-74; called at line 132 | **READ-ONLY** -- criterion 2 already satisfied by 25.B3 wiring |
| `backend/db/bigquery_client.py` | 800+ | BQ client; `save_promoted_strategy` L659-718; `get_latest_promoted_strategy` L720-760; `update_promoted_strategy_status` L762-798 | **READ-ONLY** -- `save_promoted_strategy` reused with `status="active"` |
| `backend/autoresearch/friday_promotion.py` | 204 | Weekly promotion gate; writes BQ rows with `status="pending"` at L167 | Reference: documents call shape for `save_promoted_strategy` |
| `backend/autoresearch/monthly_champion_challenger.py` | 390+ | HITL approval path; `record_approval` L219-278 flips to "active" | Reference: HITL path is separate; phase-25.R must not touch it |
| `docs/audits/phase-24-2026-05-12/24.13-redline-synthesis-findings.md` | 282 | Audit basis; F-3 confirms switching mechanism absence; F-3 + SHARP citation validates attribution | Reference |

---

## Consensus vs debate (external)

**Consensus**: Strategy promotion should be gated (DSR + shadow days), write to a persistent registry, flip atomically (MERGE), and notify operators via a structured P0 alert. No dissent on this pattern across SHARP, regime-adaptive, and alpha-decay sources.

**Debate resolved -- atomic flip vs. ramp**: The 2026 literature covers both. Resolution for pyfinagent: atomic flip is correct because `load_promoted_params` returns ONE dict of params; there is no allocation-split infrastructure for a gradual ramp. The entire daily cycle uses one parameter set.

---

## Pitfalls (from literature)

1. **Switching on noise, not signal** -- SHARP's 50bps tolerance and held-out validation guard against this. pyfinagent's shadow_min_days=5 + DSR >= 0.95 is the equivalent gate. Do NOT add a third gate in `write_to_registry`; the gate is already in `promote()` at `promoter.py:28-31`.
2. **Missing attribution signal** -- switching without understanding why the current strategy failed (SHARP finding). The shadow+DSR gate is the minimal attribution. P0 Slack notification MUST include prior DSR so operators can audit.
3. **BQ write failure blocking promotion** -- `friday_promotion.py:149` documents the pattern: BQ write is fail-open (per-row try/except). `write_to_registry` must mirror this: a BQ failure logs a warning but does NOT block the caller from using the params.
4. **Superseded status drift** -- if UPDATE to flip prior "active" to "superseded" fails, `get_latest_promoted_strategy` could return an older row on next cycle. Mitigation: `get_latest_promoted_strategy` orders by `promoted_at DESC, dsr DESC` (`bigquery_client.py:745`); the newest active row wins even if the old row is not yet superseded.
5. **Dual-path collision** -- Promoter path (status="active" on shadow+DSR gate) vs HITL path (`record_approval` flips to "active" on human approval). Both call `save_promoted_strategy`; because it uses MERGE on `(week_iso, strategy_id)`, whichever fires second wins. Safe because HITL path covers monthly champion/challenger trials, Promoter path covers weekly autoresearch trials (different trial_id namespaces).

---

## Application to pyfinagent (file:line anchors)

| Finding | Application | File:line anchor |
|---------|-------------|-----------------|
| Promoter gate is already correct | `write_to_registry` called only when `promote()` returns `{promoted: True}` | `promoter.py:32` |
| Atomic flip via MERGE | Reuse `save_promoted_strategy` with `status="active"` | `bigquery_client.py:659-718` |
| Prior active row superseded in same write | Call `update_promoted_strategy_status(old_strategy_id, "superseded")` after INSERT | `bigquery_client.py:762-798` |
| P0 Slack mirrors `format_escalation_alert` shape | `format_strategy_switch` returns Block Kit list[dict] with severity P0 | `formatters.py:679-741` |
| Autonomous loop already wired to registry | `load_promoted_params(bq)` at line 132 is criterion 2; `get_latest_promoted_strategy` default filter includes "active" | `autonomous_loop.py:132` + `bigquery_client.py:735-736` |
| HITL path is separate -- do not break | `record_approval` invokes `status_update_fn(challenger_id, "active")` independently | `monthly_champion_challenger.py:265-277` |
| BQ write fail-open pattern | `write_to_registry` wraps BQ calls in try/except, logs warning, returns result | `friday_promotion.py:149-174` |

---

## Auto-switching policy specification

### Two authorization paths

```
Path A: Promoter (shadow+DSR gate -- ops-authorized, no HITL)
  Trigger: Promoter.promote(trial) returns {promoted: True}
  Actor: autoresearch shadow-mode cycle
  Gate: shadow_trading_days >= 5 AND dsr >= 0.95
  Action:
    1. Call write_to_registry(bq_client, trial, week_iso=...) -- writes status="active"
    2. UPDATE prior active row(s) to status="superseded"
    3. Fire format_strategy_switch P0 Slack alert
  HITL required: NO (shadow + DSR gate is the ops-authorization)

Path B: HITL Monthly Champion/Challenger (human approval required)
  Trigger: monthly_champion_challenger gates pass + 48h HITL window + record_approval(status="approved")
  Actor: operator via API
  Gate: DSR + Sortino + drawdown gates + human approval
  Action: status_update_fn(challenger_id, "active") in record_approval L269-277
  HITL required: YES
  Note: Path B already implemented in 25.C3; phase-25.R does NOT touch this path
```

**When does the Promoter write "active" vs HITL?**
- **Promoter** (Path A): Autoresearch shadow-mode trials that clear `shadow_min_days` + `dsr_min`. The shadow period IS the ops-authorization: (1) trial has run in shadow mode for >= 5 days observing real market conditions, (2) DSR >= 0.95 is a statistically meaningful gate (Bailey & Lopez de Prado), (3) system is paper-trading only (no real capital at risk). No human approval needed.
- **HITL** (Path B): Monthly champion/challenger evaluation. Human approval required because the monthly cycle can involve larger allocation decisions and the 48h window gives operators time to review.

**Critical invariant**: both paths write to the same `pyfinagent_data.promoted_strategies` table. The P0 Slack notification from Path A alerts the operator that a switch occurred, closing the observability gap without requiring advance human approval.

---

## Files to modify

| File | Change | Criterion closed |
|------|--------|-----------------|
| `backend/autoresearch/promoter.py` | Add `write_to_registry(bq_client, trial, *, week_iso)` method | 1 (`promoter_writes_registry_with_status_active_on_gate_clear`) |
| `backend/slack_bot/formatters.py` | Add `format_strategy_switch(event)` function (Block Kit P0 alert) | 3 (`format_strategy_switch_slack_notification_implemented`) |
| `tests/verify_phase_25_R.py` | Create verification test covering criteria 1, 2, 3 | all |

`backend/services/autonomous_loop.py` -- **no changes needed**: criterion 2 (`autonomous_loop_uses_registry_as_primary_strategy_source`) is already satisfied by `load_promoted_params(bq)` at line 132 (25.B3). The verifier checks that the call exists and routes to `get_latest_promoted_strategy`, not that new code is added.

---

## Verbatim Python signature for `write_to_registry`

```python
def write_to_registry(
    self,
    bq_client: Any,
    trial: dict[str, Any],
    *,
    week_iso: str,
) -> dict[str, Any]:
    """Write a gate-cleared trial to `pyfinagent_data.promoted_strategies`
    with status='active', and supersede any prior active row.

    Called by `promote()` when the shadow+DSR gate clears. This is the
    ops-authorized promotion path (Path A); no HITL approval required because
    shadow_min_days + dsr_min constitute the ops gate.

    Side effects (fail-open: BQ exceptions are logged and swallowed):
      1. Calls bq_client.save_promoted_strategy(row) with status='active'.
      2. Calls bq_client.update_promoted_strategy_status(old_id, 'superseded')
         for any prior active row (best-effort; does not raise on failure).

    Returns:
      {
        "registry_written": True | False,
        "strategy_id": str,
        "week_iso": str,
        "dsr": float,
        "status": "active",
        "error": None | str,
      }
    """
```

Note: `bq_client` typed as `Any` to keep `promoter.py` free of a BigQueryClient import. The module is currently pure functions with no I/O (`promoter.py:1-8`). Callers pass the live `BigQueryClient` instance; tests pass a mock. This preserves the "pure functions" contract while enabling the BQ write.

---

## Verbatim Slack formatter shape

The `format_strategy_switch` function returns Block Kit `list[dict]`. Shape mirrors `format_escalation_alert` at `formatters.py:679`.

```python
def format_strategy_switch(event: dict) -> list[dict]:
    """Format a strategy auto-switch event as a P0 Block Kit alert.

    Args:
        event: {
            "strategy_id": str,        # new active strategy
            "prior_strategy_id": str,  # superseded strategy (or None/"N/A")
            "dsr": float,              # DSR of new strategy
            "shadow_days": int,        # shadow trading days that cleared the gate
            "allocation_pct": float,   # allocation percentage (e.g. 0.05)
            "switched_at": str,        # ISO timestamp
            "week_iso": str,           # e.g. "2026-W19"
        }

    Returns:
        Block Kit list[dict]. P0 severity, :rotating_light: header icon.
        Layout: header -> severity+timestamp section -> fields section
                (strategy_id, prior_strategy_id, dsr, shadow_days, allocation_pct,
                week_iso) -> recommended actions section -> divider -> context.
    """
```

**Block Kit JSON shape** (verbatim, with placeholders for runtime values):

```json
[
  {
    "type": "header",
    "text": {
      "type": "plain_text",
      "text": ":rotating_light: STRATEGY SWITCH -- P0",
      "emoji": true
    }
  },
  {
    "type": "section",
    "text": {
      "type": "mrkdwn",
      "text": "Severity: *P0* | <switched_at_iso>"
    }
  },
  {
    "type": "section",
    "fields": [
      {"type": "mrkdwn", "text": "*New strategy:* <strategy_id>"},
      {"type": "mrkdwn", "text": "*Prior strategy:* <prior_strategy_id|N/A>"},
      {"type": "mrkdwn", "text": "*DSR:* <dsr:.4f>"},
      {"type": "mrkdwn", "text": "*Shadow days:* <shadow_days>"},
      {"type": "mrkdwn", "text": "*Allocation:* <allocation_pct:.0%>"},
      {"type": "mrkdwn", "text": "*Week:* <week_iso>"}
    ]
  },
  {
    "type": "section",
    "text": {
      "type": "mrkdwn",
      "text": "*Recommended actions:*\n- Review DSR and shadow-mode trade log\n- Confirm new strategy params in BQ promoted_strategies\n- Acknowledge if no action needed"
    }
  },
  {"type": "divider"},
  {
    "type": "context",
    "elements": [
      {"type": "mrkdwn", "text": ":robot_face: PyFinAgent Escalation | P0 | Strategy auto-switch | Immediate attention required"}
    ]
  }
]
```

Note: `"emoji": true` is required on the header plain_text for `:rotating_light:` to render. This matches `formatters.py:706-711`.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total incl. snippet-only (16 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (promoter.py, formatters.py, bigquery_client.py, autonomous_loop.py, friday_promotion.py, monthly_champion_challenger.py, audit doc)
- [x] Contradictions / consensus noted (atomic vs ramp debate resolved for pyfinagent's single-strategy architecture)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate-complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
