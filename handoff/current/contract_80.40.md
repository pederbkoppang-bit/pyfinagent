# Contract — phase-80.40

**Step id:** `80.40` (phase-80, priority **P0**, `harness_required: true`)
**Title:** *The kill-switch indicator has never had data, on any backend state* —
`/api/paper-trading/performance` never returned `max_drawdown_pct`.

## TIER (assigned before GENERATE)

| field | value |
|---|---|
| Tier | **T3** |
| Model | Opus 5, effort `max` |
| Rationale | Safety-critical P0, indicator half of the kill switch. |

## Research

Same Workflow as `36.7` (`wf_b2205517-994`), run in parallel with it as an independent research
task, then implemented sequentially (80.40 first, since 36.7's implementation referenced it) and
verified together by the same four adversarial lenses.

> **PROTOCOL BREACH, DISCLOSED — same two findings as `36.7`, not repeated in full here.** No
> qualifying external research-gate artifact existed until `research_brief_36.7_80.40.md` was
> commissioned after adversarial Q/A flagged the gap; this contract's own artifact order also
> postdated GENERATE evidence. See `contract_36.7.md`'s disclosure for the full account — both
> steps share one Workflow run and one correction.

## Immutable success criteria (verbatim)

1. `GET /api/paper-trading/performance returns a numeric max_drawdown_pct on a healthy backend -- verbatim curl output recorded showing the key present alongside sharpe_ratio`
2. `The value is computed in backend/services/perf_metrics.py, not inline in the API layer, per the backend-services single-source-of-truth rule`
3. `A test pins the computation against a fixed NAV series with a KNOWN drawdown, and it must FAIL against a stub returning 0 -- record that failing output verbatim`
4. `The -15% label on the cockpit row and the 10% trailing-DD limit in backend/services/kill_switch.py are reconciled EXPLICITLY: state which is authoritative and make them agree, or document why they legitimately differ`
5. `phase-80.36's frontend behaviour is UNCHANGED: with the field present the row renders a real verdict, with perf absent it still renders NO DATA. Assert both.`
6. `MUTATION-TEST: reverting the perf_metrics computation must fail the new test`

**Verification command (immutable):**
```
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/perf_metrics.py').read())" && python -m pytest backend/tests/ -q -k 'perf_metrics or drawdown'
```

**live_check (immutable):** *Verbatim curl of `/api/paper-trading/performance` showing
`max_drawdown_pct` present and numeric, plus a Playwright capture of `/paper-trading/positions`
showing the kill-switch row rendering a REAL verdict instead of NO DATA.*

Satisfied against the isolated rig (`:8001`/`:3100`) over the real `handoff/kill_switch_audit.jsonl`
and archives — see `handoff/current/live_check_80.40.md`.

## Criterion 4 — threshold reconciliation, decided

Two ladders exist, both correct, on different concepts:

| Ladder | Source | Meaning |
|---|---|---|
| A — halt trading | `backend/config/settings.py` (4.0 daily / 10.0 trailing) | positive magnitude, **current** DD vs persisted peak — this is what actually trips the kill switch |
| B — derisk/block new buys | `backend/agents/mcp_servers/signals_server.py` (-15.0/-10.0/-5.0) | negative percent, a different gate entirely |

The cockpit's kill-switch row was rendering **Ladder B's numbers under Ladder A's name**, on an
**all-time max** field while the switch trips on **current** DD. Live `:8000` confirms Ladder A is
the operative one (`thresholds: {daily_loss_limit_pct: 4.0, trailing_dd_limit_pct: 10.0}`).

**Decision:** label-only fix. `"Kill switch (-15%)"` → `"Max drawdown (-15%)"`. **No threshold value
changed** — verified via `git diff --numstat` returning empty on every threshold-bearing file
(`settings.py`, `signals_server.py`, `analytics.py`, `paper_go_live_gate.py`, `drawdown_alarm.py`).
The remaining cross-tab inconsistency (this row still uses a `-10/-13` band while
`PaperVsBacktestCard` uses `-15` on the identical field) is real and queued as `36.11`.

## Do-no-harm

Frontend + backend, paper only. No `.env`, no flag flips. No threshold value changed anywhere.

## References

See `handoff/current/experiment_results_80.40.md` and `handoff/current/live_check_80.40.md`.
Follow-up defects queued as `36.10`, `36.11`, `80.43`, `80.44`.
