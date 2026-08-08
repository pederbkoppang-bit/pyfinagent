# Contract — phase-85.4 (P0 ENGINE HEALTH)

**Step id:** 85.4 · **Cycle:** 182 — 2026-08-08
**Status at write time:** research DONE, contract WRITTEN, **GENERATE NOT STARTED.**
This contract exists so the next session begins at GENERATE with the research
gate already satisfied and the order research → contract → generate intact.

---

## 1. Research gate

`handoff/current/research_brief_85.4.md` — `gate_passed: true`, **9** sources
read in full, **25** URLs, recency scan performed, 11 internal files inspected,
every internal claim file:line-anchored.

**The gate refuted two of the step's own premises. Do not build against the
audit basis; build against these measurements.**

| Step premise | Measured reality |
|---|---|
| "a terminal row is not always written" | **False.** A `finally` at `autonomous_loop.py:1746` wrote `status:timeout` rows for all three timeouts. The real defect is status **fidelity** — the kill-switch halt returns at `:1327`, leaving the `:362` initializer's placeholder `"running"` as the terminal status, and firing a P1 titled "Autonomous trading cycle **running**". |
| "the failure was invisible" | **False.** P1 "Autonomous trading cycle timeout" was `delivered=True` on 08-04/06/07. It was **buried** under ~24 hourly freshness P1s. |

Independently corroborated by my own re-derivation: `cycle_history.jsonl` shows
08-06 and 08-07 *did* write `timeout` terminal rows, while **08-05 ended on
`running`** — exactly the fidelity defect, not a missing-row defect.

## 2. Measured baseline (re-derived this cycle)

- Last **completed** cycle `2026-07-31T18:00Z`, **7 days ago**, 1 trade.
- All-time terminal: completed 74, timeout 13, error 2. Last 10: completed 7, timeout 3.
- Verification command output: `Counter({'completed': 7, 'timeout': 3})`.
  **Note: this command only reports; it cannot fail. The criteria are the gate.**

## 3. Immutable success criteria — VERBATIM from `.claude/masterplan.json`

> 1. the analysis-phase duration is MEASURED per cycle (tickers analysed x per-ticker wall-clock, from the orchestrator's own logs) and the timeout is judged against that measurement -- the step states whether 7200s is too short for the current ticker count or whether the phase genuinely hangs, with the evidence either way
>
> 2. the root cause of the non-completion is identified to file:line with a reproduction, distinguishing (a) legitimate slowness vs (b) a hang/deadlock vs (c) an unhandled per-ticker failure that stalls the gather
>
> 3. a non-completing cycle becomes LOUD: a terminal row is always written to cycle_history.jsonl (including on timeout/crash) AND an alert fires naming the phase it died in -- proven by a fault-injected cycle that dies mid-analysis, not by inspection
>
> 4. the last-completed-cycle age is exposed as a health signal the existing watchdog can page on, so 'no completed cycle in N days' cannot again be discoverable only by hand-reading a jsonl
>
> 5. no change to order/sizing/risk logic; any behavioural change is flag-gated dark and the flag is NOT promoted by this step
>
> 6. fresh Q/A PASS

**Verification command (immutable):**
`bash -c 'python3 -c "import json,collections;rows=[json.loads(l) for l in open(\"handoff/cycle_history.jsonl\") if l.strip()];term=[r for r in rows if r.get(\"status\") in (\"completed\",\"timeout\",\"error\")];print(collections.Counter(r[\"status\"] for r in term[-10:]))"'`

## 4. Criterion-by-criterion plan

**C1 + C2 — answer is (a) legitimate slowness, and the arithmetic is the evidence.**
Gate measured: 6 tickers at concurrency 3, 176 rail calls, 17.7% timeout rate,
median 91s, **18,158 serial subprocess-seconds at 2.52 effective parallelism →
~7,500–8,100s required against a 7,200s budget.** The budget is arithmetically
short. Reinforcing detail: successes max out at **145s against a 150s cap**, so
the distribution is *truncated by the cap* — most "timeouts" are slow successes,
burning ~26% of rail time for nothing.
C2 must still distinguish (b): `_run_single_analysis` (`:1859-2520`) has **zero
inner timeouts**, so a genuine hang is possible and unbounded. Exceptions are
already safe (`return_exceptions=True` + inner try/except), which rules out (c).
Use `python -m asyncio ps <backend-pid>` (new in 3.14) for a live read-only
hang-vs-slowness verdict **without a restart**.

**C3 — re-aim at status fidelity, since "always writes a row" is already true.**
Fix the `:1327` kill-switch halt path so the terminal row carries a real terminal
status and the alert names the phase. `orphan_rows()` exists but nothing pages on
it — the SIGKILL orphan case is the second half. **Proof must be fault-injected,
not inspected**, per the criterion's own words.

**C4 — the genuine watchdog gap, precisely located.**
`cycle_health.py:193` `cycle_heartbeat_alarm` skips only `started` rows, so a
`timeout` row's `completed_at` resets age to ~0 and the alarm can *never*
observe "nothing COMPLETED in 8 days". Extend this — do not rebuild it.

**C5 — the constraint that shapes everything.**
Order/sizing/risk logic untouched; any behavioural change flag-gated **dark** and
**not promoted** by this step. **This binds the config recommendations:**
raising `timeout_s` 150→210 (`claude_code_client.py:593`) and
`paper_cycle_max_seconds` 7200→10800 are behavioural, so they must be dark/
operator-gated, not applied inline. The standing goal also forbids
`backend/.env` writes and flag promotions.

## 5. Explicit non-goals / hazards

- **Do NOT lower `paper_analyze_top_n`** to fit the budget — it narrows the
  funnel feeding trade selection (the phase-82 cash-drag cause).
- **Do NOT migrate the gathers to `TaskGroup`** — it cancels siblings on first
  failure, the opposite of what a per-ticker fan-out wants.
- **Do NOT raise the timeout before C1's measurement is recorded** — the audit
  basis's own scope warning; a longer timeout on a hung cycle just moves the silence.
- **Do not start, trigger or interrupt a cycle; do not restart any service.** A
  backend restart is already owed to the operator (85.5 / ask #18 correction).

## 6. Blocking reality that 85.4 cannot fix — separate step required

**The kill switch has been latched `paused` since 2026-08-04T11:43:31Z** (last
resume 2026-07-27). On 08-05 — the one day all 6 tickers finished — the cycle
logged `kill-switch active (paused) -- skipping decide/execute` and traded
nothing. **Even a perfect 85.4 ships zero trades until the operator resumes.**
This is out of 85.4's scope and gets its own step rather than a prose mention.

Compounding, and disclosed as ask #21: my own test runs wrote 12 `manual` pause
rows to the live audit journal today, so `paused_at` now reads
`2026-08-08T08:35:16Z` instead of the real 08-04 pause. All 12 are `pause`;
zero resumes; the switch never left the fail-safe state.

## 7. References

- `handoff/current/research_brief_85.4.md` (gate output)
- `autonomous_loop.py:362, :1327, :1746, :1859-2520`; `cycle_health.py:193`;
  `claude_code_client.py:593`; `settings.py:33`
- Operator asks #18 (backend restart owed), #20 (malformed token, de-escalated
  as a root cause), #21 (kill-switch provenance)
