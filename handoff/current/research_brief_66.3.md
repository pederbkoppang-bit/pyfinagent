# Research Brief — phase-66.3 Cost-truth restoration (phantom $0.50 failure-cost writer)

Status: COMPLETE | Tier: moderate | Date: 2026-07-07 | Agent: researcher
(Length exceeds the moderate 700w knob because the caller scoped a 6-part internal
audit with verbatim quotes; tool calls ~27 vs 18 budget, driven by the mandated
3x3 search-variant matrix. Disclosed, not hidden.)

## Question
Pin the writer stamping session_cost_usd=0.50 on FAILED cc_rail rows in
`pyfinagent_data.llm_call_log`; make the sentinel metered figure mean dollars actually
billed; make failure counts their own signal.

## 1. Writer PINNED — not a hardcoded $0.50; it is the phase-26.1 cumulative-gauge auto-stamp

The chain (every link read in source):

1. `backend/agents/claude_code_client.py:474-497` `_log_cc_call` — calls
   `log_llm_call(...)` and **omits** `session_cost_usd` + `cycle_id` (docstring :478-480
   claims "session_cost_usd delta 0" — a false belief, see next link). Failure site
   :550-554 (`ok=False`, `envelope=None` → 0 tokens); success site :569-573. Agent
   string :488: `f"cc_rail:{agent}" if agent else "cc_rail"`.
2. `backend/services/observability/api_call_log.py:211-285` `log_llm_call` — signature
   :225 `session_cost_usd: float | None = None`; **auto-population** :245-254: omitted
   kwarg → lazy import of `autonomous_loop.get_session_cost_usd` → row stamped with the
   module gauge. No default/estimate constant exists in this file.
3. `backend/services/autonomous_loop.py:119-121` `get_session_cost_usd` returns
   `_session_cost` — a **running cumulative per-cycle accumulator**: reset to 0.0 at
   cycle START (:193), fed at ONE site :893-895
   (`cost = analysis.get("total_cost_usd", 0.1); _add_session_cost(cost)` — note the
   **$0.10 default per cost-less analysis**). Column semantics confirmed by the DDL:
   "running cumulative LLM cost in USD at the moment this row was logged"
   (`scripts/migrations/add_session_budget_to_llm_call_log.py:47`; documented read
   pattern is `MAX(session_cost_usd)` per cycle, :15).
4. Stale-gauge leak: the cycle `finally` clears `_current_cycle_id = None`
   (`autonomous_loop.py:1401-1403`) but **never resets `_session_cost`** — post-cycle
   rows carry cycle_id=NULL with the dead cycle's last gauge value.

So "$0.50" = the gauge after ~5 analyses accrued ~$0.10 each; every failed cc_rail row
logged while the gauge sat at 0.50 got stamped 0.50. Rows early in a cycle get 0.0.
It is neither per-call cost nor billed dollars.

**Exonerated suspects** (all read):
- `~/Library/LaunchAgents/com.pyfinagent.claude-code-proxy.plist` → runs
  `/opt/homebrew/bin/node /Users/ford/.openclaw/claude-code-proxy.js` (192 lines): a
  localhost HTTPS→`claude -p` SSE shim. **Zero BigQuery code, writes no cost anywhere**;
  it only fabricates `usage.input_tokens = ceil(prompt.length/4)` inside its SSE
  `message_start` (:96), never persisted. Not a llm_call_log writer.
- `backend/agents/cost_tracker.py:189` `cost *= 0.5` — Batch-API 50% discount
  (phase-25.C9), unrelated.
- Tree-wide grep (`0\.50?` near cost fields; `session_cost_usd` writers): the ONLY
  llm_call_log INSERT path is `api_call_log.py` (buffered flush :288-331); callers
  `claude_code_client.py:485`, `llm_client.py:1086/1756/2154`, `orchestrator.py:826`,
  `autonomous_loop.py:1910` — **none passes `session_cost_usd` explicitly**, so ALL
  rows get the gauge auto-stamp.

## 2. BQ fingerprint (read-only ADC, queried 2026-07-07)

Q1 — 2026-06-18, `agent LIKE 'cc_rail%'`:

| ok | cost | cycle_id NULL | n | avg lat ms | tokens |
|----|------|---------------|---|-----------|--------|
| false | 0.0 | no | 97 | 120113 | 0 |
| false | 0.3 | no | 52 | 50865 | 0 |
| false | 0.5 | no | 44 | 29135 | 0 |
| false | 0.4 | no | 6 | 3749 | 0 |
| false | 0.1 | no | 3 | 120097 | 0 |
| false | 0.5 | **yes** | 3 | 120070 | 0 |
| false | 0.2 | no | 2 | 120104 | 0 |

A **staircase** (0.0→0.1→…→0.5) — a cumulative gauge sampled over time, impossible for
a flat hardcoded estimate. All rows: agent='cc_rail' exactly (never labeled),
request_id NULL, 0 tokens. The 3 cycle-NULL rows at 0.5 = the stale-gauge leak (§1.4).
Latency bands 5s/29s/50s/120s = distinct failure modes (fast auth/spawn fail vs 120s
subprocess timeout).

Q2 — daily attribution: 06-17 rail SUM=$16.30 (137 rows, 137 fails) vs non-rail $0.21;
06-18 rail SUM=$42.20 (207/207 fails) vs non-rail $0.80. **The entire sentinel breach
was gauge-stamps on $0-billed flat-fee rail failures.**

Q3 — 2026-07-03 18:02-18:05 block: cost=0.0 with cycle ACTIVE (gauge not yet fed);
the 18:14 block's 0.50 = same writer later in the cycle after the gauge stepped up.

## 3. sentinel.sh — exact metered leg + where fixes go

`scripts/away_ops/sentinel.sh` (155 lines, read in full):
- Metered query :67-71 verbatim:
  `SELECT COALESCE(SUM(COALESCE(session_cost_usd, 0)), 0) AS usd FROM llm_call_log
  WHERE DATE(ts) = CURRENT_DATE()` — **sums a cumulative gauge across rows**. Comment
  :63-66 asserts "Flat-fee claude_code-rail rows log session_cost_usd=0 by design
  (60.4 writer)" — false premise (§1.2 auto-populate).
- Baseline: `BASELINE_USD = 8.00` **pinned constant at :40** (rationale header :17-21);
  NOT from `flag_baseline.json` (that file holds only `grandfathered` +
  `ops_flags_exempt`, consumed at :86-88 for the flag gate). Budget gate :79-81 →
  `gates_failed=["metered_budget"]` → exit 1 (:150-153; exit 2 = infra only).
- Report JSON :41-49: `metered_llm_usd_today, baseline_usd, kill_switch_paused,
  flags_match_tokens, ok, gates_failed[], warnings[]`. `warnings[]` is the natural home
  for a failure-count signal; a new top-level key (`rail_failures_today`) is additive
  and non-breaking for `run_away_session.sh`.
- Test overrides :52-58 (`SENTINEL_TEST_METERED_USD`, `SENTINEL_ENV_FILE`,
  `SENTINEL_TEST_BQ_FAIL`), inflate-only doctrine :23-30. A `SENTINEL_DATE` override
  slots into the same block; the SQL is a Python heredoc, trivially parameterizable.

## 4. Replay design (no data mutation)

Date-parameterize the metered SQL: `DATE(ts) = @d`, default `CURRENT_DATE()`,
override via env `SENTINEL_DATE=2026-06-18` (mirrors 62.4 `SENTINEL_TEST_*`
convention; plain SELECT on historical partitions; streaming buffer untouched).
Replay = run sentinel with `SENTINEL_DATE=2026-06-18`, assert
`metered_llm_usd_today <= 8.0` and `"metered_budget" not in gates_failed`. Fits
`backend/tests/test_phase_62_4_sentinel.py` conventions exactly: `run_sentinel(env)`
subprocess helper (:22-28), env-override tamper tests, `@pytest.mark.requires_live`
for BQ-touching legs (:98-102). Safety asymmetry holds: a date override changes WHICH
day is measured; it cannot fabricate a pass for today (run_away_session.sh never sets
it).

## 5. Schema: billing_class column vs filter-only — RECOMMEND aggregation+filter, NO migration

Columns verified (row dict `api_call_log.py:258-274`): ts, provider, model, agent,
latency_ms, ttft_ms, input_tok, output_tok, cache_creation_tok, cache_read_tok,
request_id, ok, ticker, cycle_id, session_cost_usd. No billing_class.

- The rail is already fingerprintable: `provider='anthropic' AND agent LIKE 'cc_rail%'`
  (only :488 writes the prefix).
- BUT a rail filter alone is INSUFFICIENT: every writer omits the kwarg, so non-rail
  rows carry the same cumulative gauge — SUM over any row subset still gauge-sums. The
  semantically correct read is the migration's own documented pattern: **per-cycle MAX,
  summed across cycles** (`SUM` of `MAX(session_cost_usd) GROUP BY cycle_id`).
  Replayed on 06-18 that yields ~$0.5-1.3 (gauge peak ~0.5/cycle) vs $43.00 — no breach.
- A `billing_class` column would be a 4th independent cost surface (cost_tracker
  MODEL_PRICING canonical + settings display list + governance estimate already exist
  and drift) and requires a migration script per CLAUDE.md; it buys nothing the
  (provider, agent-prefix, per-cycle-MAX) triple doesn't encode. **Defer.**
- Residual honesty gap to state in the contract: the gauge is NOMINAL (cost_tracker
  API-rate pricing; `analysis.get("total_cost_usd", 0.1)` pads $0.10 per cost-less
  analysis, `autonomous_loop.py:893`). Per-cycle-MAX is an order-of-magnitude-honest
  nominal ceiling, not invoice truth; exact billed-per-call needs cost-at-write (out of
  66.3 scope). Cheap adjacent fixes worth contracting: (a) reset `_session_cost = 0.0`
  in the finally block (kills the stale-gauge leak); (b) change the :893 default
  0.1 → 0.0 so failures stop padding the gauge (weigh: slightly weakens
  `_SESSION_BUDGET_USD` trip paranoia). Do NOT make `_log_cc_call` pass
  `session_cost_usd=0.0` — that would falsify the gauge-column semantics; fix the READ.

## 6. Failure counts as their own signal

Second read-only aggregate in the sentinel over the same table/date:
`COUNTIF(ok = false AND agent LIKE 'cc_rail%')` → new report key
`rail_failures_today` + `warnings[]` entry above a threshold — a SIGNAL, not a new
exit-1 gate (66.1's rail guard already latches rail health in-loop; the sentinel's job
is visibility). Immutable test command:
`python -m pytest backend/tests/test_phase_66_3_cost_truth.py -q`.

## External research

Query discipline: 3 topics x 3 variants each (2026 frontier / 2025 window / year-less
canonical) = 9 searches, ~75 unique URLs collected.

### Read in full (6; counts toward the gate)
| URL | Accessed | Kind | Fetched | Key finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/agent-sdk/cost-tracking | 2026-07-07 | official doc | WebFetch full | "`total_cost_usd` and `costUSD` fields are client-side estimates, not authoritative billing data… Do not bill end users or trigger financial decisions from these fields." Also: "Both success and error result messages include `usage` and `total_cost_usd`… read cost data from the result message regardless of its `subtype`." |
| https://code.claude.com/docs/en/costs | 2026-07-07 | official doc | WebFetch full | "Claude Max and Pro subscribers have usage included in their subscription, so the session cost figure isn't relevant for billing purposes." "The dollar figure is an estimate computed locally from token counts and may differ from your actual bill." |
| https://langfuse.com/docs/observability/features/token-and-cost-tracking | 2026-07-07 | official doc | WebFetch full | "Ingested usage and cost are prioritized over inferred usage and cost." When usage cannot be inferred, **cost inference fails entirely** — the tool records nothing rather than fabricating a figure. |
| https://learn.microsoft.com/en-us/cloud-computing/finops/focus/what-is-focus | 2026-07-07 | official doc | WebFetch full | FOCUS separates cost bases into distinct columns: ListCost, ContractedCost, EffectiveCost (amortized), and "**BilledCost** that was or will be invoiced". Billed vs estimated/nominal must never share a field. |
| https://www.finops.org/framework/capabilities/invoicing-chargeback/ | 2026-07-07 | official framework | WebFetch full | "Finance must be able to rely on cost data consistency, timeliness and accuracy to allocate expenses in a transparent and accountable way." Baseline = monthly macro invoice reconciliation; estimates require true-up. |
| https://docs.datadoghq.com/llm_observability/monitoring/cost/ | 2026-07-07 | official doc | WebFetch full | Costs computed from public pricing are consistently labeled **estimated** down to attribute names (`@metrics.estimated_total_cost`); failed-call costing is not defined (no fabrication). |

### Identified, snippet-only (~69; context, not gate-counted) — representative
| URL | Kind | Why not fetched |
|---|---|---|
| https://www.braintrust.dev/articles/best-tools-tracking-llm-costs-2026 | vendor blog | snippet sufficient: "observability tools generally do not reconcile estimated costs against the actual provider invoice" (corroborates 62.4 header's Braintrust cite) |
| https://www.buildthisnow.com/blog/guide/mechanics/claude-billing-change-june-2026 | blog | recency item captured via search summary (see scan below) |
| https://github.com/anthropics/claude-code/issues/68501 | issue tracker | adversarial-adjacent: CLI+Desktop co-install silently misroutes subscription usage to metered Extra Usage — "rail bills $0" is an assumption that can silently break |
| https://focus.finops.org/focus-specification/v1-2/ | spec | Microsoft column summary read in full instead (same definitions, renders better) |
| https://www.cloudzero.com/blog/chargeback-vs-showback/ | vendor blog | snippet: ~90% allocation accuracy before chargeback; inaccurate allocation = escalated budget disputes (= our two false P1 pages) |
| https://code.claude.com/docs/en/headless | official doc | `--output-format json` envelope incl. total_cost_usd already covered by cost-tracking doc |
| plus ~63 more from the 9 searches (Langfuse FAQ/discussion #9739, Helicone cost cookbook, Traceloop, OpenObserve, Uptrace, Comet, Maxim, FOCUS v1.0/v1.1/v1.4 pages, finops.org chargeback-previous, nOps, morphllm x4, sitepoint x2, support.claude.com, platform.claude.com pricing, …) | mixed | tier budget; lower authority or duplicative |

### Recency scan (2024-2026) — performed, findings present
1. **FOCUS 1.4 ratified 2026-06-04**: adds Invoice Detail + Billing Period datasets so
   "AP, finance, and FinOps teams reconcile from the same data" — the standard keeps
   moving TOWARD strict billed-vs-estimated separation (focus.finops.org; Amnic 2026 guide).
2. **Claude Code billing change 2026-06-15** (snippet-level, verify before relying):
   Agent SDK / `claude -p` usage moves off plan limits onto a fixed monthly Agent SDK
   credit; overflow goes to usage credits at API rates only if enabled, else requests
   stop. Direct 66.3 impact: "rail = flat-fee, bills $0" is now a DATED assumption —
   the rail's billing class must be an explicit, documented, revisitable filter, not a
   hardcoded belief (support.claude.com; buildthisnow.com; matches the 59.1 memory:
   Fable-5 June-22 credit cliff).
3. **June 2026 `--attribution` flag**: `.claude/attribution.json` per-session cost
   breakdown — a future authoritative-ish per-session source for rail volume
   (sitepoint.com June-2026 features).
4. LLM-observability tooling (2025-2026 Braintrust/Datadog): consensus hardened that
   trace-level cost is an estimate and invoice reconciliation is a separate, explicit
   step (Datadog CCM pulls real invoices next to estimates).

### Consensus vs debate
- **Consensus**: (a) per-call cost computed from price tables is an ESTIMATE and must be
  labeled as such (Datadog attribute naming; Agent SDK warning); (b) budget/allocation
  gates should run on billed/cash-basis figures (FOCUS BilledCost; FinOps
  invoicing-chargeback); (c) never fabricate cost when usage is absent — record nothing
  or 0 with a failure marker (Langfuse inference-fails-closed behavior).
- **Debate**: whether observability tools should reconcile to invoices themselves
  (Datadog CCM does; Langfuse explicitly does not; Braintrust: most don't) — for
  pyfinagent the 58.1 ledger remains the reconciliation instrument; the sentinel is
  showback, so a documented lower-bound proxy is acceptable IF its class filter is
  explicit.

### Pitfalls (from literature -> this step)
1. Estimate-as-billed conflation: the SDK's own doc forbids "financial decisions" on
   total_cost_usd-class fields — the away sentinel's exit-1 gate is exactly such a
   decision; keep the gate but feed it a billed-truth-shaped figure.
2. Gauge-vs-delta (internal analogue of FOCUS's amortized-vs-billed row semantics):
   never SUM a cumulative column; use the DDL's documented MAX-per-cycle read.
3. Failure-cost fabrication: a failed call that consumed 0 tokens bills $0 — count it
   as a FAILURE (own signal), not as cost (Langfuse/Datadog behavior).
4. Billing-class drift: the rail's $0 premise has a shelf life (2026-06-15 change);
   document the filter + revisit trigger in the sentinel header.

## Application to pyfinagent (external -> file:line)
- Sentinel metered leg (`sentinel.sh:67-71`): replace row-SUM with per-cycle
  MAX-sum + explicit comment that session_cost_usd is a cumulative NOMINAL gauge
  (DDL `add_session_budget_to_llm_call_log.py:47`), citing FOCUS billed-vs-effective
  separation; correct the false :63-66 premise.
- Failure signal: new aggregate + `rail_failures_today` key + `warnings[]` append
  (report dict `sentinel.sh:41-49`).
- Replay: `SENTINEL_DATE` env in the override block (`sentinel.sh:52-58`) +
  `test_phase_66_3_cost_truth.py` with a `requires_live` replay test asserting the
  2026-06-18 window does not breach ($42.20 → ~$1 class), per test conventions in
  `test_phase_62_4_sentinel.py:22-28,98-102`.
- Optional writer-side hygiene (contract decision): reset `_session_cost` in the
  finally block (`autonomous_loop.py:1401`); default 0.1→0.0 at :893.
- Leave `_log_cc_call` (`claude_code_client.py:474-497`) untouched except docstring
  correction ("delta 0" → "column is the cycle gauge; billed rail cost is $0 by class,
  enforced at READ time").

## Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6, all official)
- [x] 10+ unique URLs total (~75)
- [x] Recency scan (last 2 years) performed + reported (4 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
Soft checks:
- [x] Internal exploration covered every relevant module (writer chain, proxy, sentinel,
      tests, migrations, gauge feed) — proxy exonerated with evidence
- [x] Contradictions/consensus noted (reconcile-or-not debate; :63-66 false premise)
- [x] Claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 69,
  "urls_collected": 75,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/research_brief_66.3.md",
  "gate_passed": true
}
```
