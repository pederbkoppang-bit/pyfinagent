# live_check 66.3 -- Cost-truth restoration (2026-07-07)

Required shape: "live_check_66.3.md with writer file:line, replay output, and the
live-day hand-audit comparison."

## 1. Writer, pinned to file:line (criterion 1)

There is NO flat-$0.50 constant. The nonzero cost on failed cc_rail rows is the
CUMULATIVE per-cycle session-cost gauge, stamped by the lazy-fill in
`backend/services/observability/api_call_log.py:245-254` (pre-fix numbering) from
`backend/services/autonomous_loop.py:119-121` (`get_session_cost_usd` -> module
accumulator `_session_cost`), which grows by `analysis.get("total_cost_usd", 0.1)`
at `autonomous_loop.py:893-895` and resets only at cycle end (:1401-1403). BQ
fingerprint: failed-rail costs form a 0.0 -> 0.5 staircase within each cycle
(2026-06-18; 2026-07-03 18:02Z block=0.0 early-cycle vs 18:14Z block=0.5
late-cycle). claude-code-proxy exonerated (no BQ writes).

FIX (in `log_llm_call`): failed calls with zero token movement never receive the
gauge -- they log `session_cost_usd=0.0` (explicit caller values still win;
failures WITH tokens keep the gauge since mid-stream work may bill).

## 2. Sentinel metered figure = dollars actually billed (criterion 2)

Documented choice: provider/agent FILTERING (no schema migration). Metered =
token-derived (tokens x pinned price table, `scripts/away_ops/metered_spend.py`)
over metered providers; EXCLUDED as flat-fee/credit: `agent LIKE 'cc_rail%'`
(Max flat fee; since 2026-06-15 headless draws the Agent SDK credit -- still not
per-token API billing) and `provider='claude-code'`. Unpriced metered models
count $0 + FAIL-VISIBLE warning. Consecutive-failure signal: sentinel JSON gains
`rail_failures_today` + a warnings[] entry at >=20 (mirrors the 66.1 breaker).

## 3. Replay + live-day hand audit (criterion 3) -- verbatim

```
$ python scripts/away_ops/metered_spend.py --date 2026-06-18
{"metered_llm_usd": 0.0043, "rail_failures": 207, "warnings": [], "unpriced_models": []}
$ python scripts/away_ops/metered_spend.py --date 2026-06-17
{"metered_llm_usd": 0.0026, "rail_failures": 137, ...}
```
Old logic reported $43.00 / $16.51 -> NEW figures $0.0043 / $0.0026: NO breach vs
the $8.00 baseline (a ~10,000x over-count eliminated). The real story of those
days -- 207/137 failed rail calls -- now surfaces as `rail_failures`.

Live day (2026-07-07), module vs INDEPENDENT hand audit (separate BQ aggregation
+ manual arithmetic):
```
module:      {"metered_llm_usd": 0.0013, "rail_failures": 0}
hand audit:  anthropic/claude-haiku-4-5-20251001: n=1 in=1000 out=50 -> $0.0013
             HAND-AUDIT TOTAL: $0.0013
```
MATCH to the 4th decimal. (06-18 hand audit likewise: gemini 1829 in/1488 out ->
$0.0043 == module.)

Full sentinel run (live, post-change):
```
{"metered_llm_usd_today": 0.0013, "rail_failures_today": 0, "baseline_usd": 8.0,
 "kill_switch_paused": false, "flags_match_tokens": true, "ok": true,
 "gates_failed": [], "warnings": []}
```

## 4. Tests (immutable command)

```
$ python -m pytest backend/tests/test_phase_66_3_cost_truth.py -q
10 passed
```
(within the combined run: 66_3 + 62_4 sentinel + observability = 32 passed.)
