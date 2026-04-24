---
title: "BLOCKER-3 Research Brief — HITL Champion/Challenger Promotion Gate End-to-End Exercise"
step: blocker-3
tier: moderate
date: 2026-04-24
---

## Research: HITL Champion/Challenger Promotion Gate End-to-End Exercise

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|------------|----------------------|
| https://brightlume.ai/blog/shadow-mode-rollouts-ai-agents-pilot-production | 2026-04-24 | Blog (AI deployment) | WebFetch full | "the agent processes the same inputs, generates outputs, and logs decisions—but humans remain the final decision-makers"; agreement rate >= 85% + no critical errors in 500 decisions = readiness bar |
| https://annpastushko.substack.com/p/step-functions-for-human-in-the-loop | 2026-04-24 | Technical blog | WebFetch full | HITL state machine: task token embedded in Slack message; human clicks Approve/Reject -> SendTaskSuccess/Failure; configurable TimeoutSeconds on waiting state; up to 1-year wait supported |
| https://aws.amazon.com/blogs/machine-learning/dynamic-a-b-testing-for-machine-learning-models-with-amazon-sagemaker-mlops-projects/ | 2026-04-24 | AWS official docs/blog | WebFetch full | Approval is explicit manual data-scientist sign-off in model registry; offline holdout evaluation precedes approval; no automatic timeout specified -- approval window is open-ended per SageMaker pattern |
| https://medium.com/@fraidoonomarzai99/deployment-evaluation-strategies-in-mlops-c208585aa3bd | 2026-04-24 | Technical blog (2026) | WebFetch full | Gates auto-reject if "must beat current production model / must not degrade fairness / must meet latency SLA" fail; shadow mode generates: prediction distribution, feature drift, P95 latency, confidence calibration as telemetry |
| https://www.snowflake.com/en/developers/guides/ml-champion-challenger-model-deployment/ | 2026-04-24 | Official vendor guide | WebFetch full | Automated champion/challenger via alias swap ("CHAMPION" / "CHALLENGER") + AUC comparison; no explicit human approval gate required in their pattern; confirms LIVE_VERSION tag as observable state artifact |
| https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/ | 2026-04-24 | Industry blog (GA March 2025) | WebFetch full | "strict approval workflows and audit trails...only authorized personnel can propose and analyze challenger models or replace the current champion"; shadow mode before promotion recommended |
| https://docs.h2o.ai/mlops/v0.64.0/py-client/py-client-examples/champion-challenger-deployment | 2026-04-24 | Official docs | WebFetch full | Shadow deployment element wraps both champion and challenger; each is a "DeployShadowElement"; deployment composition is the observable artifact |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.modelop.com/ai-governance/glossary/champion-challenger-testing | Glossary | Fetched but shallow -- no promotion gate mechanics beyond brief definition |
| https://www.sparklinglogic.com/champion-challenger-for-rolling-out-deployments/ | Blog | Snippet only; no additional depth beyond canary split percentages |
| https://www.mlops.community/public/blogs/the-what-why-and-how-of-a-b-testing-in-ml | Community blog | Snippet only; covered by other sources |
| https://docs.relay.app/human-in-the-loop/human-in-the-loop-steps | Vendor docs | Snippet only; Relay.app specific, not applicable |
| https://help.zapier.com/hc/en-us/articles/38731463206029 | Vendor docs | Snippet only; Zapier-specific |
| https://github.com/Sai-Lalith-Sistla/My-MLOps-expertise/blob/main/01_Champion%20vs%20challenger%20strategy.md | GitHub | Snippet only; lightweight summary |
| https://altstreet.investments/blog/quant-2-architecture-modern-trading-stack-ai-mlops | Quant blog | Snippet only; canary at 5% for 2 weeks before full ramp confirmed |
| https://www.fico.com/en/latest-thinking/white-paper/champion-challenger-strategy-design-and-deployment | Industry white paper | Not fetched; URL referenced in prior phase-10.6 brief, available as prior art |
| https://alldaystech.com/guides/devops-sysadmin/ci-cd-pipeline-explained | DevOps guide | Snippet only; generic pipeline stages |
| https://expertcisco.com/what-is-mlops/ | Generic MLOps | Snippet only; too broad |

### Recency scan (2024-2026)

Searched explicitly:
1. Current-year frontier: "champion challenger HITL approval MLOps 2026", "MLOps deployment gate drill synthetic test 2026"
2. Last-2-year window: "champion challenger promotion gate human approval 2025", "HITL Slack approval MLOps 2025"
3. Year-less canonical: "champion challenger strategy promotion human-in-the-loop", "quant algo promotion gate evidence"

Findings: DataRobot champion/challenger went GA with human approval workflow in March 2025 -- confirms HITL approval is established MLOps practice. H2O MLOps v0.64 (2024-2025) documents shadow deployment element pattern. Medium article from Feb 2026 documents shadow + canary + gate auto-rejection pattern. No new literature supersedes the 48h window or HITL approval state-machine design shipped in phase-10.6. The 48h expiry pyfinagent uses is not explicitly an industry standard but is within normal range (AWS/SageMaker uses open-ended windows; Step Functions supports up to 1 year).

---

### Key findings

1. **48h expiry is defensible but conservative.** Industry patterns (Step Functions, SageMaker) do not prescribe a fixed timeout -- they allow indefinite waits or configurable timeouts. The 48h hard expiry in pyfinagent's `_APPROVAL_WINDOW_HOURS=48` is reasonable for a personal trading system where Peder is the sole approver. (Source: annpastushko.substack.com, aws.amazon.com/blogs/machine-learning)

2. **Minimum evidence bar.** Brightlume recommends 5,000-10,000 decisions in shadow mode before cutover, plus >= 85% agreement rate, no critical errors in last 500. For pyfinagent, the equivalent is the Sortino delta >= 0.3, PBO < 0.2, DD ratio <= 1.2 check -- these are the pre-approval gates already coded. (Source: brightlume.ai)

3. **Telemetry that proves a gate fired vs. bypassed.** Observable: (a) state JSON write at `monthly_approval_state.json`, (b) BQ row insert to `strategy_deployments_log`, (c) Slack message sent with approval metadata. All three must be independently verifiable. (Source: fraidoonomarzai99 Medium, snowflake guide)

4. **Approval gate drill pattern (industry).** Snowflake and DataRobot both support parallel shadow runs with synthetic data; no explicit "drill" mode documented, but the standard approach is to inject synthetic champion/challenger metrics that satisfy quality gates, trigger the gate, approve/reject, and verify state artifacts. (Source: snowflake.com guide, datarobot.com blog)

5. **No BQ write is wired yet.** `monthly_champion_challenger.py` writes ONLY to `monthly_approval_state.json`. There is NO code path that writes a row to `pyfinagent_pms.strategy_deployments_log` when a promotion is approved. This is the primary gap. (Source: internal code audit, `monthly_champion_challenger.py:167-196`)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/autoresearch/monthly_champion_challenger.py` | 311 | `run_monthly_sortino_gate()` + `record_approval()` + `is_last_trading_friday()`; state at `handoff/logs/monthly_approval_state.json`; `_APPROVAL_WINDOW_HOURS=48`; `actual_replacement=False` hardcoded | Active |
| `backend/api/monthly_approval_api.py` | 167 | REST endpoints: `GET /api/harness/monthly-approval/status` + `POST /api/harness/monthly-approval/{month_key}` with `{"action":"approved"|"rejected"}`; delegates to `record_approval()` | Active |
| `backend/main.py` | -- | `monthly_approval_router` included at line 321-322 | Active |
| `scripts/migrations/create_strategy_deployments_view.py` | 196 | Creates `pyfinagent_pms.strategy_deployments_log` (append-only base table) + `pyfinagent_pms.strategy_deployments` view (UNION ALL with synthetic seed `seed_0000`) | Shipped (phase-10.5.1) |
| `backend/api/sovereign_api.py` | -- | `_fetch_strategy_deployments()` queries the view at line 201-207; note at line 331 says "fallback: pyfinagent_pms.strategy_deployments view not yet shipped (10.5.1)" -- stale comment, view exists | Active (stale comment) |
| `scripts/harness/phase10_monthly_sortino_test.py` | -- | Verification CLI for monthly gate; exercises `run_monthly_sortino_gate()` directly with injected dates | Test/CLI |
| `tests/autoresearch/test_monthly_champion_challenger.py` | 12 cases | Unit tests: gate pass/fail, 48h expiry, approval/rejection state transitions | All pass |
| `backend/autoresearch/rollback.py` | -- | `_DEFAULT_STATE_PATH = Path("handoff/logs/monthly_approval_state.json")` at line 28 -- shared state path | Active |
| `handoff/logs/monthly_approval_state.json` | N/A | **Does not exist** -- no drill has ever been run | Missing |

**Critical gaps found:**

1. **No BQ write on approval.** `record_approval()` (line 201-231) writes ONLY to the JSON state file. There is no `INSERT INTO pyfinagent_pms.strategy_deployments_log` anywhere in `monthly_champion_challenger.py`. The contract says "BQ log row" must exist -- this code path is absent.

2. **No real Slack send wired.** `run_monthly_sortino_gate()` accepts `slack_fn` as an injectable parameter (line 57) but no caller in the production path (cron, API, harness) passes a real Slack client. The function is a pure library. Nothing in `autoresearch_weekly_packet.py` or the harness invokes `run_monthly_sortino_gate` with a real `slack_fn`.

3. **`is_last_trading_friday()` gate blocks synthetic triggers.** To fire the gate on an arbitrary date (for a drill), the caller must either pass a date that IS the last Friday of its month, OR the drill must call `run_monthly_sortino_gate` directly with a matching date (bypassing the calendar check).

4. **`monthly_approval_state.json` absent.** File at `handoff/logs/monthly_approval_state.json` does not exist. First call to `_load_state()` will fail-open to `{}` -- this is safe, but means no prior state to inspect.

5. **No manual trigger endpoint or CLI.** There is no `/api/harness/monthly-approval/seed` endpoint, no admin button, and no script that can create a `pending` state row directly without calling `run_monthly_sortino_gate`. The only path to a PENDING state is through `run_monthly_sortino_gate()` with a date that passes `is_last_trading_friday()` and metrics that pass the Sortino/PBO/DD gates.

---

### Consensus vs debate (external)

Consensus: HITL approval gates belong in ML promotion pipelines; shadow-before-promotion is universal; Slack-based approval buttons are an established pattern (Step Functions, n8n, Zapier).

Debate: timeout semantics vary widely -- SageMaker leaves windows open indefinitely; Step Functions allows TimeoutSeconds; pyfinagent's 48h is a fixed local convention with no external mandate. Auto-reject on expiry (vs. auto-extend) is a conservative choice that matches the pyfinagent design.

### Pitfalls (from literature)

- **Bypassing the gate silently.** Snowflake's pattern auto-promotes; if Slack message fails to deliver and expiry is 48h, the gate can auto-expire with no human review. Telemetry must confirm the message was sent, not just attempted (fail-open logging hides this). In pyfinagent, `slack_fn` fail-open at line 193-196 means a Slack failure does not prevent the PENDING state -- the window opens even if Peder never sees the message.
- **Stale state masking a fresh gate.** If a prior month's PENDING row is in the JSON, `run_monthly_sortino_gate` returns `prior_pending_not_expired` and exits early (line 106-112) without re-evaluating metrics. Drill must clear prior state before re-running.
- **BQ log row missing.** Currently `approved` state never writes to `strategy_deployments_log`. A PASS verdict without that row is incomplete per the contract.

### Application to pyfinagent (mapping findings to file:line)

| Finding | File:line | Action required |
|---------|-----------|-----------------|
| No BQ write on approval | `monthly_champion_challenger.py:201-231` | Add `INSERT INTO pyfinagent_pms.strategy_deployments_log` inside `record_approval()` when status == "approved" |
| No real Slack send | `monthly_champion_challenger.py:57, 185-196` | Wire real `slack_fn` from Slack bot client at the cron/harness call site |
| Calendar gate blocks drill | `monthly_champion_challenger.py:84-87` | Inject a real last-Friday date for the drill (2026-04-24 is NOT the last Friday of April; 2026-04-24 is a Friday but not the LAST one -- 2026-04-24 is actually the last Friday of April 2026: April has 30 days, April 25 is Saturday, so April 24 is the last Friday) |
| No manual trigger | N/A | Drill calls `run_monthly_sortino_gate()` directly in Python with injected date + synthetic metrics |
| Stale state | `monthly_champion_challenger.py:91-112` | Clear or move `monthly_approval_state.json` before drill if prior key exists |

**Date check: 2026-04-24.** April 2026 has 30 days. April 30 = Thursday. Last Friday = April 24. Today IS the last trading Friday of April 2026. The calendar gate will fire for today's date.

---

## Ranked test strategy options

### Option A: Mock Slack send + synthetic approval (no real Slack)
Call `run_monthly_sortino_gate()` from a CLI script with today's date + synthetic champion/challenger metrics that pass all three gates. Pass a `slack_fn=lambda msg, data: print(msg)` stub. Observe JSON state file write. Then call `record_approval("2026-04", status="approved")` directly. Observe state transition. Add BQ insert manually.

Cheapest. Zero Slack noise. Exercises the state machine completely. Weakness: does NOT prove the real Slack channel receives the message or that a human can click a button.

### Option B: Real Slack ping to Peder with TEST-labelled promotion + his real approval (RECOMMENDED)
Call `run_monthly_sortino_gate()` with today's real last-Friday date + synthetic metrics (challenger_id="DRILL-2026-04") + a real `slack_fn` wired to the Slack bot client. This sends an actual Slack message to Peder clearly marked "[DRILL]". Peder approves via `POST /api/harness/monthly-approval/2026-04 {"action":"approved"}`. State file transitions to `approved`. BQ row inserted (after the missing BQ write is added). All three observables are confirmed by a human who actually clicked a button.

This is the cheapest end-to-end proof with real human interaction. No capital risk (paper-only enforced by `actual_replacement=False`). The approval is explicitly synthetic (challenger_id="DRILL-*") so no real strategy is promoted.

### Option C: Full handler path drill without state flip
Same as B but Peder rejects instead of approving, or the state is immediately expired after the drill. Proves the message delivery and handler path without leaving a permanent `approved` row. Useful if the team is concerned about the BQ row being a false positive in future reports.

**Recommendation: Option B.** It is the only option that proves: (a) real Slack message delivered, (b) real human clicked approve, (c) REST endpoint transitioned the state, (d) BQ row written, (e) JSON file reflects `approved`. That is the full set of observables the contract requires. The drill label (`DRILL-2026-04`) ensures operators can distinguish it from a real promotion.

---

## Test plan: state transitions + observables for PASS

1. **Pre-condition check:** `handoff/logs/monthly_approval_state.json` either absent or has no `"2026-04"` key (or key has status != "pending").
2. **BQ log table exists:** `SELECT COUNT(*) FROM pyfinagent_pms.strategy_deployments_log` returns 0 rows (or any number -- table must exist).
3. **Gate fire:** call `run_monthly_sortino_gate(date(2026, 4, 24), champion_returns=[...], challenger_returns=[...], champion_max_dd=0.10, challenger_max_dd=0.08, challenger_pbo=0.10, challenger_id="DRILL-2026-04", slack_fn=<real_slack_client>)`. Assert return dict has `fired=True`, `gate_pass=True`, `approval_pending=True`, `reason="opened_hitl_window"`.
4. **JSON state written:** `handoff/logs/monthly_approval_state.json` key `"2026-04"` present with `status="pending"`, `expires_at_iso` set to ~48h from now.
5. **Slack message received:** Peder confirms receipt in Slack channel of message containing "Monthly Champion/Challenger gate fired for 2026-04" and "[DRILL]" label.
6. **Approval action:** `POST /api/harness/monthly-approval/2026-04 {"action":"approved"}` -> HTTP 200, response `{"status":"approved", "month":"2026-04", "resolved_at_iso":"..."}`.
7. **JSON state updated:** key `"2026-04"` has `status="approved"`, `resolved_at_iso` populated.
8. **BQ row inserted:** `SELECT * FROM pyfinagent_pms.strategy_deployments_log WHERE notes LIKE '%DRILL-2026-04%'` returns 1 row with `status="challenger"` (or `"champion"` if design promotes the challenger).
9. **Sovereign API reflects deployment:** `GET /api/sovereign` returns leaderboard with the drill strategy visible (or the seed row still visible -- either proves the view is live).

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total (10 snippet-only + 7 full = 17)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/blocker-3-research-brief.md",
  "gate_passed": true
}
```
