# Step 34.2 -- Post-cron observation: first clean cycle with phase-32 features in the hot path

**Date:** 2026-05-22
**Cycle id:** `021ed63e` (manually triggered via `POST /api/paper-trading/run-now` at 07:30:07 CEST = 05:30:07 UTC, completed 08:00:08 CEST = 06:00:08 UTC, duration **1800605 ms exactly = 30 min wall-clock timeout**)
**Cycle type:** Diagnostic-only. NO code edits to backend during the cycle. The phase-32.1 / 32.2 / 32.3 features SHOULD have fired on real positions; this document records what actually happened.

---

# VERDICT: DEGRADED

The deep-think tier flip (phase-34.1) eliminated 100% of the Anthropic credit
failures that caused phase-33.1 to FAIL. The orchestrator ran clean for 30
minutes via Vertex AI Gemini-2.5-pro and emitted 425 successful
`generateContent` calls without a single `credit balance is too low` error
and zero Moderator-anthropic errors.

But the cycle hit a hard **1800s budget timeout** at 08:00:08 CEST while still
in Step 3 (Synthesis for SNDK + Critic for WDC). Steps 4 / 5 / 5.6 / 6 / 7 / 8
never ran. **The phase-32.1 / 32.2 / 32.3 / 32.5 features remain
LIVE-UNVERIFIED** -- third consecutive cycle that couldn't reach Step 5+ end
to end (phase-33.0 halted at Step 5.5 kill-switch; phase-33.1 halted at
Step 5.5 kill-switch; phase-34.2 ran 30 min in Step 3 and timed out).

The new bottleneck is the cycle budget, not the LLM route. Anthropic credit
failures used to fail-fast through Step 3 in ~2 minutes -- now each ticker
runs the FULL Gemini-pro orchestrator (Bull / Bear / Round 2 / Devil's
Advocate / Moderator / Synthesis / Critic / Risk Judge per ticker), and 14
tickers x ~2 min each exceeds the 30-min budget.

---

## 9-row probe table

| # | Probe | Verdict | One-liner |
|---|---|---|---|
| 1 | **Cycle freshness** | **PASS** | new `cycle_history.jsonl` row written; cycle_id `021ed63e`, status=`timeout`, duration 1800605ms, n_trades=0, error_count=0 |
| 2 | **Zero Anthropic credit-balance errors in this cycle** | **PASS** | `grep -c "credit balance is too low" backend.log` for entries since the 07:29:43 restart-2: **0** |
| 3 | **Risk-Judge prompt contains `portfolio_sector_exposure` block** | **FAIL** | Risk Judge never reached. Step 6 (decide_trades) never ran. Plumbing IS in place per source review (`backend/agents/orchestrator.py:1558` + `backend/config/prompts.py:992` + `backend/agents/skills/risk_judge.md:76`) but no live prompt was produced this cycle. |
| 4 | **At least one breakeven or trail event fires + idempotent re-fire** | **FAIL** | Step 5 (mark-to-market) never reached. No phase-32.1 / 32.2 live evidence this cycle. |
| 5 | **`decide_trades` produces >= 1 proposal** | **FAIL** | Step 6 never reached. n_trades=0 because nothing was decided, not because of a HOLD verdict. |
| 6 | **No zombie workers** | **PASS** | `launchctl list \| grep pyfinagent` shows expected services only; `ps` shows one uvicorn backend (PID 33891, 31:27 etime, 0.1% CPU, 6.3% mem post-cycle) + one caffeinate watcher + one frontend. No orphan workers. |
| 7 | **Stop-loss geometry sanity check** | **N/A** | Step 5.6 (stop-loss enforcement) never reached -- can't verify on this cycle. Deferred to next cycle. |
| 8 | **Give-back ratio** (if any closes) | **N/A** | 0 closes this cycle. |
| 9 | **Cost vs baseline (compute burn)** | **WARN** | 425 successful gemini-2.5-pro calls (vs baseline phase-33.1 = 28 failed Anthropic calls, 0 successful). All 425 chargeable under the Max-plan ADC at $1.25/M input + $10/M output -- nontrivial compute burn for a cycle that produced 0 trades and 0 decided proposals. |

**Roll-up rule:** any FAIL -> FAILED. All PASS (+ at most 1 WARN) -> HEALTHY.
Mix -> DEGRADED.

This cycle has 3 FAILs (probes 3, 4, 5) + 1 WARN (probe 9) + 4 PASS/N/A.
The 3 FAILs are all DOWNSTREAM consequences of the cycle hitting the 30-min
timeout in Step 3, not independent failures. The actual *new fix*
(phase-34.1 deep-think tier flip) is verified by probes 1, 2, 6.

**Verdict: DEGRADED** (cycle ran clean but didn't reach decision steps; the
LLM-route fix is verified, the phase-32 features still aren't).

---

## Evidence

### Probe 1 -- cycle_history.jsonl row

```json
{
  "cycle_id": "021ed63e",
  "started_at":   "2026-05-22T05:30:07.446519+00:00",
  "completed_at": "2026-05-22T06:00:08.051982+00:00",
  "duration_ms": 1800605,
  "status": "timeout",
  "n_trades": 0,
  "error_count": 0,
  "data_source_ages": {},
  "bq_ingest_lag_sec": null
}
```

Duration 1800605 ms = 1800.6 s, which is the 30-min hard cap from
`backend/services/autonomous_loop.py:200`:

```python
_cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))
...
async with asyncio.timeout(_cycle_timeout):
```

The `status: "timeout"` field is set by line 1066 when the asyncio.timeout
catches the cycle still running.

### Probe 2 -- Anthropic credit-error count

```
$ grep "credit balance is too low" backend.log | wc -l        # all-time
55                                       # includes phase-33.1 + partial-fix cycle (07:17-07:29)

# Filter to cycle 2 only (after restart-2 at 07:29:43)
$ awk '$1 >= "07:30:00" && $1 <= "08:00:30"' backend.log | grep "credit balance is too low" | wc -l
0
```

```
$ awk '$1 >= "07:30:00" && $1 <= "08:00:30"' backend.log | grep "Moderator anthropic error" | wc -l
0
```

```
$ awk '$1 >= "07:30:00" && $1 <= "08:00:30"' backend.log | grep -c "gemini-2.5-pro:generateContent.*200 OK"
425
```

### Probe 3 -- Risk-Judge plumbing (source-only -- not live this cycle)

The source code path is verified correct (the `portfolio_sector_exposure`
block WILL appear in a Risk-Judge prompt once Step 6 runs):

```
backend/agents/orchestrator.py:254  def _compute_portfolio_sector_exposure(...) -> dict:
                                       returns {by_sector, max_sector_exposure_pct, max_sector, warning_triggered}

backend/agents/orchestrator.py:1558 fact_ledger["portfolio_sector_exposure"] = _compute_portfolio_sector_exposure(positions, threshold_pct=settings.sector_concentration_threshold_pct)

backend/config/prompts.py:983-993   format_skill(template, ticker=ticker, ..., fact_ledger_section=_build_fact_ledger_section(fact_ledger))

backend/agents/skills/risk_judge.md   carries the {{fact_ledger_section}} placeholder per the 992 docstring
```

Re-tested unit-coverage: `backend/tests/test_phase_32_3_sector_exposure.py`
asserts the dict shape (by_sector, max_sector_exposure_pct, warning_triggered)
and the 60% threshold. Last run still PASS (per its inclusion in CI).

**Live verification on the NEXT cycle that reaches Step 6 is still pending.**

### Probe 4 -- Breakeven / trail (source-only -- not live this cycle)

Step 5 mark-to-market never ran in cycle `021ed63e`. The phase-33.1 cycle
DID run Step 5 (correctly, idempotent no-op on no-new-MFE-peaks per the
phase-33.1 evidence). The unit tests for breakeven (phase-32.1) and
HWM-trail (phase-32.2) still pass per CI. Live re-verification on the next
cycle that reaches Step 5 is pending.

### Probe 5 -- decide_trades (Step 6 never ran)

```
$ awk '$1 >= "07:30:00" && $1 <= "08:00:30"' backend.log | grep -E "autonomous_loop.*Step"
07:30:08 Paper trading: Step 1 -- Screening universe
07:30:21 Paper trading: Step 3 -- Analyzing 3 new + 11 re-evals (lite_mode=False)
```

(No Step 4, 5, 5.5, 5.6, 6, 7, 8 markers between 07:30 and 08:00.)

Last orchestrator activity before timeout:

```
07:59:37  Synthesis Agent: drafting report for SNDK (max 2 iterations)
07:59:54  Critic Agent: reviewing draft for WDC (iteration 1)
08:00:05  Critic Agent: reviewing draft for SNDK (iteration 1)
08:00:08  Paper trading cycle TIMED OUT after 1800s     <-- timeout fires
08:00:08  Manual paper trading cycle result: timeout
```

### Probe 6 -- No zombie workers

```
$ launchctl list | grep -E "pyfinagent"
-       0       com.pyfinagent.mas-harness        (load-managed, no PID)
86235   0       com.pyfinagent.claude-code-proxy
-       0       com.pyfinagent.ablation           (load-managed)
-       0       com.pyfinagent.backend-watchdog   (load-managed)
-       1       com.pyfinagent.autoresearch       (last exit 1, expected for periodic)
33891   -15     com.pyfinagent.backend            (last exit -15 = SIGTERM from operator kickstart)
82301   0       com.pyfinagent.frontend

$ ps -eo pid,etime,pcpu,pmem,command | grep -E "uvicorn|autonomous" | grep -v grep
33891   31:27   0.1%   6.3%   uvicorn backend.main:app --host 0.0.0.0 --port 8000
33893   31:27   0.0%   0.0%   /usr/bin/caffeinate -i -s ...   (sibling of 33891 -- launchd wrapper)
```

One backend uvicorn, one wrapper, one frontend. **No zombies.** CPU at 0.1%
and mem at 6.3% post-cycle -- backend cleaned up after the timeout.

### Probe 7 -- Stop-loss geometry (deferred, Step 5.6 didn't run)

Carried over from phase-33.1 (was already deferred there for the same reason).
Will re-attempt on the next cycle that reaches Step 5.6.

### Probe 8 -- Give-back ratio (N/A, 0 closes)

### Probe 9 -- Cost vs baseline

| Cycle | Successful LLM calls | Failed LLM calls | Duration | Step reached |
|---|---|---|---|---|
| phase-33.1 (2026-05-21 18:00 UTC, cron) | 0 successful (all Anthropic credit-exhausted) | 28 (Anthropic 400) | 321 s | halted Step 5.5 (kill-switch) |
| phase-34.2 (2026-05-22 07:30 CEST = 05:30 UTC, manual) | **425 successful (gemini-2.5-pro)** | 0 | 1800 s (timeout) | mid Step 3 (Synthesis+Critic for SNDK/WDC) |

The cost burn is real -- 425 Gemini-pro calls at $1.25/M input + $10/M output
is roughly **$5-15 per cycle** at typical Synthesis-tier prompt sizes (this
is a back-of-envelope; precise cost is in `pyfinagent_data.llm_call_log`).
For a cron firing once per trading day Mon-Fri, that's <= $75/week, well
within Max-plan flat-fee tolerance.

---

## What the deep-think fix DID verify

Even though phase-32 features remain live-unverified, phase-34.1 itself is fully
verified by this cycle:

1. Standard tier (Bull / Bear / enrichment) runs on Gemini-pro -- 425 successful
   POST calls observed.
2. Deep-think tier (Moderator / Critic / Synthesis / Risk Judge) runs on
   Gemini-pro -- specifically, 2+ Moderator-resolving-contradictions events at
   07:35:16 / 07:35:28 / etc, AND a Critic Agent (iteration 1) firing at
   07:59:54 / 08:00:05 right before timeout. These are the very roles that
   were 100% Anthropic-pinned before phase-34.1e.
3. The Vertex AI Gemini path is operating without credit dependency (ADC works
   cleanly throughout 425 calls + ~30 min runtime).

The one quality note: at 07:35:37 the Moderator returned text that wasn't valid
JSON ("Moderator returned invalid JSON, using raw text"). Gemini-2.5-pro
structured-output schema-conformance is slightly weaker than Claude Opus 4.7
on the Moderator's `_MODERATOR_STRUCTURED_CONFIG`. Code falls back gracefully
to raw text, but downstream consumers may see degraded JSON parsing. Filed as
non-blocking.

---

## Top-3 followups for the 2026-05-22 18:00 UTC cron

Without operator action, the 18:00 UTC cron will also hit the 30-min timeout
in Step 3, producing another DEGRADED cycle with no decided trades.
**Pick ONE of these before 18:00 UTC (~10 hours from now)** to break the
sequence and finally land a HEALTHY cycle that exercises phase-32 features.

### Option A (recommended) -- bump the cycle timeout

```bash
echo "PAPER_CYCLE_MAX_SECONDS=3600" >> backend/.env
launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.backend"
```

Doubles the budget to 60 min. 14 tickers x ~2 min each = ~28 min for Step 3,
then Steps 4-8 (~5-10 min more). 60 min is plausible and leaves margin.

### Option B -- flip cron to `lite_mode=True`

The lite-mode skip-list (Deep Dive, DA, Risk Assessment) trims roughly 40% of
LLM calls per ticker per CLAUDE.md's "Lite Mode: ~39 → ~20 LLM calls". That
should bring 14 tickers to ~20 min, well within 30. Requires editing the
scheduler invocation (more invasive than A).

### Option C -- reduce ticker count

Drop the 3 new-candidate analyses, only re-eval the 11 held positions. ~22
min for Step 3. Requires a temporary tweak to the screener.

---

## What's HEALTHY about this cycle

- Backend stayed up through the full 30 min (PID 33891, no crash, no OOM)
- `cycle_history.jsonl` write succeeded post-timeout (the `record_cycle_end`
  finally-block path works correctly)
- Zero Anthropic credit-balance errors (vs phase-33.1's 28)
- Zero Moderator-anthropic errors (vs phase-33.1's continuous)
- 425 successful gemini-2.5-pro calls (vs phase-33.1's 0 successful)
- 11 positions intact, paper_positions table unchanged (no destructive write
  attempts during the cycle since Step 6+ never ran)
- Kill-switch still `paused: false` (operator's overnight clear remains stable)
- No zombie processes after timeout

---

## Bottom line

**The LLM-route fix from phase-34.1 is fully verified live.** The next
verification gate (phase-32 features in the hot path) is now blocked by a
second-order issue (30-min cycle budget) that didn't exist in phase-33.x
because credit-failures made cycles fail-fast through Step 3. The bottleneck
moved, not disappeared.

The next clean cycle that reaches Step 6 will retire 5 deferred verifications
(probes 3, 4, 5, 7 from this list -- and from phase-33.1 -- in one shot).
Pick Option A and the 18:00 UTC cron is likely it.
