---
name: cost-truth-66-3
description: session_cost_usd is a CUMULATIVE per-cycle gauge (not per-call cost); phantom $0.50 = 26.1 auto-stamp on rail rows; sentinel SUM bug; proxy exonerated
metadata:
  type: project
---

phase-66.3 research (2026-07-07): the "phantom $0.50 failure-cost writer" is NOT a
hardcoded estimate — `llm_call_log.session_cost_usd` is a **running cumulative
per-cycle NOMINAL gauge**, auto-stamped on EVERY row whose writer omits the kwarg
(ALL writers omit it).

**Why:** chain = `claude_code_client.py:474-497` `_log_cc_call` omits kwarg →
`api_call_log.py:245-254` auto-populates from `autonomous_loop.get_session_cost_usd()`
(gauge; reset at cycle START :193 only, NOT in the finally block :1401-1403 → stale-gauge
rows with cycle_id=NULL; fed at ONE site :893-895 with `analysis.get("total_cost_usd", 0.1)`
— $0.10 default per cost-less analysis produces the 0.0→0.5 staircase). DDL confirms gauge
semantics + MAX-per-cycle read pattern (`scripts/migrations/add_session_budget_to_llm_call_log.py:47,:15`).
BQ fingerprint 06-18: rail rows SUM $42.20 = 207 failed $0-billed rows carrying gauge
stamps (staircase 0.0/0.1/0.2/0.3/0.4/0.5 — impossible for a flat constant).
`sentinel.sh:67-71` SUMs the gauge across rows (gauge-vs-delta bug); its :63-66 comment
("rail rows log 0 by design") is a false premise. `BASELINE_USD=8.00` is a pinned
constant at sentinel.sh:40, NOT in flag_baseline.json (that file = flag grandfathering
only). claude-code-proxy.js (~/.openclaw/, launchd node shim) EXONERATED: zero BQ code.
cost_tracker.py:189 `cost *= 0.5` = batch discount, unrelated.

**How to apply:** (a) never treat session_cost_usd as per-call cost — aggregate as
SUM of MAX per cycle_id; (b) rail filter = `provider='anthropic' AND agent LIKE
'cc_rail%'` (only claude_code_client.py:488 writes the prefix; failed rail rows:
request_id NULL, 0 tokens, agent='cc_rail' unlabeled); (c) filter alone is
insufficient — non-rail rows carry the gauge too; (d) recency: Claude Code
2026-06-15 billing change moves `claude -p` onto a monthly Agent SDK credit
(overflow = metered usage credits), so "rail bills $0" is a dated assumption —
any billing-class filter needs a documented revisit trigger; (e) Agent SDK doc:
error results DO carry total_cost_usd but it is a client-side estimate — "do not
trigger financial decisions from these fields". Related: [[cc-rail-guard-66-1]],
[[cost-pricing-tables-inventory]].
