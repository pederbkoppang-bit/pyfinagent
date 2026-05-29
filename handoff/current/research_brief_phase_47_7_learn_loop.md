# Research Brief — phase-47.7: Learn-loop outcome-write fix

**Step**: phase-47.7 — "Learn-loop outcome-write fix (sell-closes write outcome_tracking + reflections)"
**Cycle**: 8 (priority 6 / DoD-6)
**Tier**: moderate-complex
**Date**: 2026-05-28
**Researcher**: research-gate (external + internal Explore merged)

---

## STATUS: COMPLETE — gate_passed: true

**TL;DR:** The learn-loop is BROKEN (both outcome_tracking and agent_memories are EMPTY/0 rows).
The cycle_block_summary hypothesis (swap-SELL bypasses the path) is WRONG — the swap-SELL DOES
reach the handler. Root causes: (1) `paper_learn_loop_enabled=False` makes the per-ticker loop
`continue` before any write; (2) a field-name bug — the fallback reads `trade["return_pct"]` but
the SELL row only has `realized_pnl_pct`, so even with the flag ON it would write `0.0`. Minimal
fix = read `realized_pnl_pct`, carry `exit_reason` metadata, + operator flag flip (deferred to
live). Unit-testable with a swap-SELL fixture. DoD-6 probe `WHERE cycle_id IS NOT NULL` is broken
(no such column).

---

## A. TRUE LEARN-LOOP STATE (Critical First Question)

### A.1 — agent_memories REAL location + state
- **Location:** `sunny-might-477607-p8.financial_reports.agent_memories`, region **us-central1**.
  Confirmed via `bq.get_table()` live query (2026-05-29).
- **Wiring:** `backend/db/bigquery_client.py:487,494` — built from
  `settings.bq_dataset_reports` = `"financial_reports"` (settings.py:43). This is the SAME
  dataset as paper_* tables and outcome_tracking. It is NOT in `pyfinagent_data`.
- **Why the earlier probe got NotFound:** the probe queried `pyfinagent_data` and/or
  the wrong location (US). agent_memories is in `financial_reports` @ **us-central1**, so a
  US-location query returns NotFound (BQ is location-scoped).
- **Schema (5 cols):** `agent_type, ticker, situation, lesson, created_at`. **NO cycle_id.**
- **Row count: 0 (EMPTY, ever).** No rows from the 2026-05-29 KEYS sell-close or any prior cycle.
- **VERDICT:** agent_memories' PRIMARY artifact is NOT populated. The learn-loop's primary
  output has never written a row. The DoD-6 gap is NOT narrower — both artifacts are empty.

### A.2 — outcome_tracking: is it the right table?
- **Location:** `sunny-might-477607-p8.financial_reports.outcome_tracking`, region **us-central1**.
- **Wiring:** `bq.save_outcome` (bigquery_client.py:375-392) writes to `self.outcomes_table`
  = `{project}.{bq_dataset_outcomes}.{bq_table_outcomes}` = `financial_reports.outcome_tracking`
  (settings.py:46-47). **YES — this is exactly the table the per-sell-close writer targets.**
- **Schema (9 cols):** `ticker, analysis_date, recommendation, price_at_recommendation,
  current_price, return_pct, holding_days, beat_benchmark, evaluated_at`. **NO cycle_id.**
- **Row count: 0 (EMPTY, ever).**
- **save_outcome is NOT an UPSERT.** bigquery_client.py:390 uses plain `insert_rows_json` =
  streaming APPEND. The autonomous_loop.py:1977 comment ("bq.save_outcome is an UPSERT in the
  existing implementation") is **FACTUALLY WRONG** — it is append-only. (Minor: idempotency
  claim in the code comment is false; not the blocker, but should be corrected.)

### A.3 — DoD-6 probe `WHERE cycle_id IS NOT NULL` is BROKEN
- The cycle_block_summary DoD-6 probe queries `outcome_tracking WHERE cycle_id IS NOT NULL`.
- **outcome_tracking has NO `cycle_id` column** (neither does agent_memories). The probe would
  raise a BQ "Unrecognized name: cycle_id" error, OR (if written defensively) always return 0.
- **There is NO separate outcome_tracking table with cycle_id.** `grep` finds only the one
  table ref. The DoD-6 criterion references a column that does not exist in the schema.
- **FIX for the probe:** DoD-6 verification must query `outcome_tracking` filtering on
  `evaluated_at >= <cycle_start>` (or just `COUNT(*) > 0` after the first enabled sell-close),
  NOT `cycle_id`. Same for agent_memories: filter on `created_at >=`.

---

## B. ROOT CAUSE OF EMPTY outcome_tracking (validated)

### B.1 — The cycle_block_summary hypothesis is WRONG
cycle_block_summary.md:103-105 claims the swap-rotation SELL
"doesn't flow through the evaluate_recommendation -> fallback save_outcome ->
_generate_and_persist_reflections path." **This is incorrect.** Proof:

- `autonomous_loop.py:962-975` — the Step-7 sell-execution loop iterates ALL `orders` where
  `action == "SELL"` (swap-SELLs included; they carry `reason="swap_for_higher_conviction"`
  per portfolio_manager.py:527) and appends EVERY executed sell ticker to `closed_tickers`
  (line 975). There is NO reason-based filter.
- `autonomous_loop.py:1035-1038` — `if closed_tickers: await _learn_from_closed_trades(...)`.
  So the swap-SELL KEYS WAS passed into the learn-loop handler.
- Log confirms the handler ran: `02:54:16 OutcomeTracker reflection-model constructed`
  (the phase-31.1 line at autonomous_loop.py:1915, which executes at the TOP of
  `_learn_from_closed_trades`, before the per-ticker loop).

### B.2 — The ACTUAL root cause: the feature flag is OFF
- `autonomous_loop.py:1936` reads `learn_loop_enabled = bool(settings.paper_learn_loop_enabled)`.
- **Live value: `paper_learn_loop_enabled = False`** (settings.py:32 default False; not set in
  backend/.env). Confirmed via `get_settings()` at runtime (2026-05-29).
- `autonomous_loop.py:1970-1971`:
  ```python
  if not learn_loop_enabled:
      continue
  ```
  This `continue` fires for EVERY ticker (including KEYS) BEFORE the `save_outcome` fallback
  (line 1983) and BEFORE `_generate_and_persist_reflections` (line 2035). So NO write of
  either artifact occurs.
- **The "reflection-model constructed" log is a red herring** — it logs at line 1915 (top of
  handler), which runs regardless of the flag. It does NOT indicate any write happened.

### B.3 — Log evidence (definitive)
- `grep -ac "fallback outcome_tracking row written|reflections fan-out fired|reflection-model
  constructed"` over the full 309MB backend.log returns **4 hits, ALL "reflection-model
  constructed"**. ZERO "fallback outcome_tracking row written" and ZERO "reflections fan-out
  fired" lines have EVER been emitted. This is exactly the signature of: handler entered, model
  built, then `continue` (flag off) — no writes.
- KEYS timeline (UTC): `00:53:14 SELL 4.2297 x KEYS @ $339.13` (swap fired 00:53:01 SELL KEYS
  score=5 -> BUY STX score=7) -> `00:54:16 reflection-model constructed` -> [flag-off continue]
  -> no outcome row.

### B.4 — A SECONDARY latent bug (would bite even with flag ON)
Even if the flag were flipped ON, the PRIMARY path would early-return None and force the
fallback, AND the fallback itself has a data-shape gap for swap-SELLs:

1. **evaluate_recommendation early-returns None for swap-SELLs.** outcome_tracker.py:42-45
   re-fetches `get_comprehensive_financials(ticker)` for a live `Current Price`; on yfinance
   flake -> None. More importantly, `analysis_date` for a swap-SELL = `trade.get("analysis_id")
   or trade.get("created_at")` (autonomous_loop.py:1944). The swap-SELL TradeOrder
   (portfolio_manager.py:524-529) sets NO `analysis_id` and NO `risk_judge_decision` on the
   SELL leg — only the BUY leg carries those. So `analysis_date` falls back to the sell trade's
   `created_at`, and `datetime.fromisoformat(analysis_date)` at outcome_tracker.py:47 will parse
   the SELL timestamp, giving `holding_days≈0` (cosmetic, not fatal). Net: evaluate_recommendation
   likely returns None (no stored rec at that "analysis_date") -> fallback path.
2. **Fallback save_outcome needs return_pct + holding_days the swap-SELL trade row may lack.**
   autonomous_loop.py:1981-1982 pulls `trade.get("return_pct")` and `trade.get("holding_days")`
   from the paper_trades row. Need to confirm the paper_trades SELL row carries these (the swap
   path emits the SELL via execute_sell — need to verify execute_sell computes return_pct).
   If absent -> defaults to 0.0 (writes a row, but with return_pct=0 — degraded but non-fatal).

So the **minimal fix is two-part**: (1) enable the flag (operator gate), and (2) harden the
swap-SELL data shape so the write is meaningful, with a unit test that does NOT need the flag
or a live cycle.

### B.5 — THIRD bug: field-name mismatch (`return_pct` vs `realized_pnl_pct`)
This is the most important code-level finding for the fix:
- `paper_trader.execute_sell` (paper_trader.py:350-369) writes the SELL paper_trades row with
  the P&L field named **`realized_pnl_pct`** (line 364). It does NOT write a `return_pct` field.
- The learn-loop fallback reads **`trade.get("return_pct")`** (autonomous_loop.py:1981) which
  is ABSENT on the SELL row -> defaults to `0.0`.
- Consequence: even with the flag ON, every fallback outcome_tracking row would record
  `return_pct=0.0` and `beat_benchmark=(0.0 > 0)=False` — a silently WRONG row. The existing
  phase-35.1 test (test_phase_35_1_learn_loop_writer.py:39) masks this because its mock trade
  fixture hand-sets `"return_pct": 17.89` — a field the REAL execute_sell row never has.
- Same for `holding_days`: execute_sell DOES write `holding_days` (line 363) so that one is OK.
- **This is a test-fixture-vs-reality drift** exactly matching the operator's standing feedback
  ("grep all consumers + live-smoke real-env"; see auto-memory feedback_full_codebase_audit).
  The mock fixture diverged from the production row shape and hid a real bug.
- **VERIFIED against live schema (2026-05-29):** `paper_trades` has `realized_pnl_pct=True`,
  `return_pct=False` (22 rows). `get_paper_trades` is `SELECT *` with no remap
  (bigquery_client.py:693-700) -> the trade dict carries `realized_pnl_pct`, never `return_pct`.

---

## EXTERNAL RESEARCH

### Read in full (6; counts toward the gate — floor is 5)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/ | 2026-05-29 | industry/blog | WebFetch full | Triple-barrier labels exits by REASON: +1 profit-take (upper), -1 stop (lower), 0 time/vertical. Outcome attribution must distinguish exit mechanism; combining event-sampling + triple-barrier + meta-labeling "improves the performance of the strategies." |
| https://en.wikipedia.org/wiki/Meta-Labeling | 2026-05-29 | reference | WebFetch full | Meta-model is a binary classifier trained on REALIZED trade outcomes (made/lost money) to filter false positives and size positions. The feedback loop REQUIRES realized-outcome data to be recorded — empty outcome store = no meta-model training signal. |
| https://arxiv.org/abs/2506.06698 | 2026-05-29 | paper (preprint) | WebFetch (abstract+metrics) | Contextual Experience Replay: accumulates past experiences into a dynamic memory buffer; agents retrieve relevant knowledge. +51.0% relative success-rate over GPT-4o baseline on WebArena (36.7%). Populating the buffer is what drives the gain. |
| https://arxiv.org/html/2603.24639 | 2026-05-29 | paper (preprint) | WebFetch full HTML | Experiential Reflective Learning: heuristics "stored in a persistent pool" after outcome reflection. +7.8% over ReAct; pass^3 reliability +8.3–10.6%. Reflective memory stabilizes consistency — but ONLY if reflections are actually persisted. |
| https://arxiv.org/html/2508.17565v1 | 2026-05-29 | paper (preprint) | WebFetch full HTML | TradingGroup self-reflection: labels each decision with ACTUAL market outcome, compiles an "experience summary," injects into context. Ablation: self-reflection lifts cumulative return AMZN 7.26%->40.46%, MSFT 6.36%->20.27%, COIN -22.1%->70.6%. **Does NOT distinguish exit reasons** (rebalance vs stop vs profit) — treats all outcomes uniformly. [ADVERSARIAL on the exit-labeling sub-question — see Consensus vs debate] |
| https://arxiv.org/html/2603.07670v1 | 2026-05-29 | paper (survey, 2026) | WebFetch full HTML | Memory for Autonomous LLM Agents survey. §4.2: dense retrieval default, "often augmented with sparse BM25 and metadata filters." §4.4: "Orchestration failures in hierarchical memory tend to be silent...no exception, no log entry." §7.7: recommends logging "every write, read, update, delete" — production rarely instruments memory writes, "leaving silent failures undetected until behavioral drift becomes severe." |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://arxiv.org/html/2303.11366 (Reflexion, NeurIPS 2023) | paper | Canonical prior-art; covered via search snippet (persistent episodic memory buffer; +22% AlfWorld, +20% HotPotQA). Foundational, not new. |
| https://arxiv.org/html/2508.17565v1 FinMem/FinAgent refs | paper | Layered memory + dual-level reflection in trading agents; snippet sufficient for corroboration. |
| https://arxiv.org/pdf/2512.02261 (TradeTrap) | paper | LLM-trading-agent reliability/faithfulness; adjacent, not core to the write-path bug. |
| https://arxiv.org/html/2605.19337v1 (Agentic Trading) | paper | Survey of LLM agents in markets; snippet only. |
| https://arxiv.org/html/2603.21357 (AgentHER hindsight replay) | paper | Hindsight relabeling of trajectories; adjacent ML angle. |
| https://github.com/jo-cho/meta_labeling_simplified | code | Reference impl of meta-labeling; not needed in full. |
| https://www.quantconnect.com/forum/discussion/14706/ (meta-labeling not a silver bullet) | community | Lower-tier; useful caveat snippet only. |
| https://mem0.ai/blog/state-of-ai-agent-memory-2026 | industry | 2026 memory-benchmark landscape; snippet corroborates §7.7 of the survey. |
| https://www.newsletter.quantreo.com/p/the-triple-barrier-labeling-of-marco | blog | Triple-barrier explainer; redundant with Hudson&Thames. |
| https://arxiv.org/pdf/2510.15949 (ATLAS adaptive trading) | paper | Adaptive LLM trading; adjacent. |
| https://arxiv.org/pdf/2601.15075 (Agentic Attribution) | paper | Internal-driver attribution; adjacent to outcome attribution. |

### Recency scan (last 2 years, 2024–2026)
Searched 2024–2026 literature on trading-agent learn-loops, reflection memory, outcome
attribution, and BM25 memory stores (queries: "...2026", "...2025", and year-less canonical).
**Result: MULTIPLE new findings that COMPLEMENT (do not supersede) the canonical Lopez de Prado
meta-labeling + Reflexion (2023) base:**
- TradingGroup (arXiv 2508.17565, Aug 2025): ablation quantifies that DISABLING self-reflection
  collapses cumulative returns (e.g. AMZN 40.46%->7.26%) — direct evidence that an empty/inert
  reflection write-path is materially costly, not cosmetic.
- ERL (arXiv 2603.24639, 2026) + CER (arXiv 2506.06698, 2025): persistent experience pools yield
  +7.8% / +51% relative gains — gains are CONTINGENT on the store actually being populated.
- Memory survey (arXiv 2603.07670, 2026) §4.4 + §7.7: NEW, directly on-point — "memory write
  failures are SILENT (no exception, no log)" and the fix is per-write logging + write-assertion
  tests. This is the most recent and most relevant finding to phase-47.7's exact failure mode.
No source CONTRADICTS the core claim that recording realized outcomes is necessary; see the
adversarial note below for the one sub-question where sources diverge.

### Consensus vs debate (external)
- **CONSENSUS:** A learn-loop requires realized trade outcomes to be PERSISTED; an empty
  outcome/memory store yields zero learning signal (meta-labeling, Reflexion, CER, ERL,
  TradingGroup all agree). Silent write failures are a known, costly production trap (2026 memory
  survey §4.4/§7.7).
- **DEBATE / [ADVERSARIAL] (exit-reason labeling):** The objective asks whether a ROTATION exit
  ("sold for better", reason=swap_for_higher_conviction) warrants DIFFERENT reflection labeling
  than a conviction/stop exit. Lopez de Prado's triple-barrier (Hudson&Thames) says YES — exit
  mechanism (profit/stop/time) is a first-class label dimension. But TradingGroup (the most
  directly comparable LLM-trading system) EXPLICITLY does NOT distinguish exit reasons — it labels
  every closed decision by its market outcome uniformly and still gets large gains. This is a
  genuine disagreement. **Resolution for pyfinagent:** do NOT block phase-47.7 on exit-reason
  taxonomy. Record the `reason` (already on the trade row + round_trip exit_reason) as a metadata
  tag on the outcome/lesson so future meta-labeling CAN segment by exit type (cheap, aligns with
  triple-barrier), but compute return/beat_benchmark uniformly (aligns with TradingGroup). A swap-
  SELL is a "replaced, not failed" exit — its return_pct can be positive or negative and is still
  a valid learning datum about the SOLD ticker's realized performance at exit.

### Pitfalls (from literature)
1. **Silent write failures** (memory survey §4.4): the #1 trap — a learn-loop can "run" (model
   constructed, log emitted) while writing nothing. EXACTLY pyfinagent's case. Mitigation: assert
   the write in a unit test + log every write (the code already logs "fallback...written" /
   "fan-out fired", but those lines never fire because of the flag + field bug).
2. **Store-everything noise** (memory survey §7.1): not pyfinagent's problem yet (0 rows), but
   when enabled, the reflection should be filtered/canonicalized, not raw.
3. **Backtest-vs-live Sharpe degradation** (2026 search): orthogonal to the write-path but worth
   noting — the learn-loop's value shows up live, not in a short window.

### Application to pyfinagent (external -> internal anchors)
- Survey §7.7 "log every write + assert writes" -> the FIX must include a unit test asserting
  `save_outcome` + `_generate_and_persist_reflections` are called for a swap-SELL
  (autonomous_loop.py:1983, 2035), mirroring test_phase_35_1_learn_loop_writer.py but with a
  swap-SELL fixture and the REAL field name `realized_pnl_pct`.
- Triple-barrier exit-reason labeling -> carry `trade["reason"]` into the lesson/outcome metadata
  (portfolio_manager.py:527 sets reason="swap_for_higher_conviction"; paper_trader.py:358 persists
  it on the trade row and as round_trip exit_reason at line 389).
- TradingGroup uniform-outcome labeling -> compute return_pct from `realized_pnl_pct` regardless
  of exit reason; don't special-case swap-SELLs in the P&L math.

---

## C. MINIMAL FIX (verifiable WITHOUT a ~1h45m live cycle)

The fix has three parts; (1)+(2) are code (unit-testable), (3) is the operator/live gate.

**(1) Fix the field-name mismatch (the real silent-write bug).**
In `autonomous_loop.py:1981`, the fallback reads `trade.get("return_pct")`. Change to read the
field the SELL row actually has, with a safe fallback chain:
```python
pnl_pct = float(
    trade.get("return_pct")            # legacy / hand-set (kept for back-compat)
    or trade.get("realized_pnl_pct")   # the field execute_sell ACTUALLY writes (paper_trader.py:364)
    or 0.0
)
```
This makes the fallback outcome_tracking row carry the TRUE realized P&L for swap-SELLs (and all
sells), not 0.0. `holding_days` already maps correctly (both sides use `holding_days`).

**(2) Carry the exit reason into the outcome/lesson metadata (cheap, triple-barrier-aligned).**
Optional but recommended per the adversarial resolution: thread `trade.get("reason")` into the
synthetic `outcome` dict (autonomous_loop.py:1993-1999) as `"exit_reason"`, so the reflection
prompt and any future meta-labeling can segment rotation exits from stop/conviction exits. Do NOT
gate the write on reason — all sells (including swap-SELLs) write.

**(3) Enable the operator flag for the live confirmation (deferred to cron/manual cycle).**
`paper_learn_loop_enabled = False` is the gate that makes the per-ticker loop `continue` before
any write (autonomous_loop.py:1970-1971). Flipping it to `True` (env `PAPER_LEARN_LOOP_ENABLED=true`
or settings.py default) is REQUIRED for the live row to land. This is an operator decision (per
/goal integration gate 3, the feature is intentionally default-OFF) and its live effect can only
be confirmed on the next sell-close cycle — so it is DEFERRED, not part of the unit-test PASS.

**Also fix the DoD-6 probe** (cycle_block_summary / any verification SQL): replace
`WHERE cycle_id IS NOT NULL` with a `created_at`/`evaluated_at >= <cycle_start>` filter (or
`COUNT(*) > 0`), because neither table has a `cycle_id` column.

**Also correct the stale comment** at autonomous_loop.py:1977 ("bq.save_outcome is an UPSERT") —
it is append-only (`insert_rows_json`).

---

## D. UNIT-TEST SHAPE (asserts the path WITHOUT a live cycle, BQ mocked)

Add to `backend/tests/test_phase_35_1_learn_loop_writer.py` (or a new
`test_phase_47_7_swap_sell_learn_loop.py`) a test that uses a REALISTIC swap-SELL trade row —
i.e. the actual `execute_sell` output shape, NOT a hand-set `return_pct`:

```python
@pytest.fixture
def mock_bq_swap_sell():
    bq = MagicMock()
    bq.get_paper_trades.return_value = [{
        "ticker": "KEYS",
        "action": "SELL",
        "reason": "swap_for_higher_conviction",   # the rotation exit
        "price": 339.13,
        "realized_pnl_pct": -2.41,    # REAL field name from paper_trader.py:364 (NOT return_pct)
        "holding_days": 3,
        "analysis_id": "",            # swap-SELL leg carries no analysis_id (paper_trader.py:359)
        "created_at": "2026-05-29T00:53:14+00:00",
        "risk_judge_decision": "",    # empty -> coerced to HOLD
        # NOTE: deliberately NO "return_pct" key -> proves the fix reads realized_pnl_pct
    }]
    bq.get_report.return_value = None
    bq.save_outcome = MagicMock(return_value=None)
    return bq

def test_phase_47_7_swap_sell_writes_real_pnl(mock_bq_swap_sell, settings_flag_on):
    """A swap_for_higher_conviction SELL (rotation exit) MUST write an
    outcome_tracking row carrying the REAL realized P&L (-2.41), not 0.0,
    and MUST fire the reflections fan-out. Guards the field-name bug:
    the trade row has realized_pnl_pct, never return_pct."""
    with patch("backend.services.outcome_tracker.OutcomeTracker") as MockTracker:
        instance = MockTracker.return_value
        instance.evaluate_recommendation.return_value = None  # swap-SELL has no stored rec -> fallback
        instance._generate_and_persist_reflections = MagicMock()
        _run_learn(mock_bq_swap_sell, settings_flag_on)

        assert mock_bq_swap_sell.save_outcome.called
        saved = mock_bq_swap_sell.save_outcome.call_args.kwargs
        assert saved["ticker"] == "KEYS"
        assert saved["return_pct"] == -2.41          # FAILS today (reads absent return_pct -> 0.0)
        assert saved["beat_benchmark"] is False       # -2.41% is a loss
        assert instance._generate_and_persist_reflections.called  # reflection fan-out fires
```
Key assertions: (a) `save_outcome` called for a swap-SELL, (b) `return_pct == -2.41` (the real
`realized_pnl_pct`, proving the field-name fix), (c) reflections fan-out fired. This test FAILS
on current code (asserts -2.41 but code reads the absent `return_pct` -> 0.0) and PASSES after fix
(1). Run: `pytest backend/tests/test_phase_47_7_swap_sell_learn_loop.py -q` (no BQ, no live cycle,
sub-second). Also keep a regression test that the existing fixture's hand-set return_pct path still
works (back-compat of the `or` chain).

---

## E. WHAT IS DEFERRED (cannot be unit-verified)
1. **Live row landing in BQ.** Requires `paper_learn_loop_enabled=True` AND a real sell-close
   cycle (~1h45m, or the daily cron). The unit test proves the CODE PATH calls the writers; the
   live row confirms BQ accepts the insert. Defer to cron / a manual cycle with the flag on.
2. **Operator flag flip.** Enabling `paper_learn_loop_enabled` is an operator/owner decision
   (intentional default-OFF per /goal gate 3). The fix should NOT silently flip it; document it as
   the activation step.
3. **5-consecutive-clean-cron-cycle streak** (the broader DoD-6 durability bar) is days of
   wall-clock, not in scope for the code fix.
4. **agent_memories reflection content quality** (LLM-generated lessons) — only verifiable once
   rows land + retrieval is exercised; out of scope for the write-path fix.

---

## RESEARCH GATE CHECKLIST

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: Hudson&Thames, Meta-Label
      Wikipedia, CER, ERL, TradingGroup, Memory survey)
- [x] 10+ unique URLs total (6 read-in-full + 11 snippet-only = 17)
- [x] Recency scan (last 2 years) performed + reported (3-variant queries; 2025–2026 findings)
- [x] Full papers / pages read (not abstracts) for the read-in-full set (HTML full reads;
      CER was abstract+published metrics — flagged honestly)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (autonomous_loop, portfolio_manager,
      paper_trader, outcome_tracker, bigquery_client, settings, existing test) + live BQ schema
- [x] Contradictions / consensus noted (exit-reason labeling debate, marked [ADVERSARIAL])
- [x] All claims cited per-claim with file:line or URL

---

## INTERNAL CODE INVENTORY
| File | Lines | Role | Status |
|------|-------|------|--------|
| backend/services/autonomous_loop.py | 962-975 | Step-7 sell loop appends ALL sells (incl swap) to closed_tickers | OK — swap-SELL is NOT bypassed |
| backend/services/autonomous_loop.py | 1035-1038 | Dispatches closed_tickers -> _learn_from_closed_trades | OK |
| backend/services/autonomous_loop.py | 1880-2046 | Learn-loop handler (phase-31.1/35.1) | BLOCKED by flag (1970-71) + field bug (1981) |
| backend/services/autonomous_loop.py | 1915-1919 | "reflection-model constructed" log (pre-flag) | Red herring — fires regardless of flag |
| backend/services/autonomous_loop.py | 1970-1971 | `if not learn_loop_enabled: continue` | ROOT CAUSE #1 (flag OFF) |
| backend/services/autonomous_loop.py | 1981 | `trade.get("return_pct")` | ROOT CAUSE #3 (field absent -> 0.0) |
| backend/services/autonomous_loop.py | 1977 | comment "save_outcome is an UPSERT" | WRONG — append-only |
| backend/services/portfolio_manager.py | 524-529 | swap-SELL TradeOrder reason="swap_for_higher_conviction", no analysis_id | confirms exit reason |
| backend/services/paper_trader.py | 350-369 | execute_sell SELL row: writes realized_pnl_pct, analysis_id="" | source of field mismatch |
| backend/services/outcome_tracker.py | 42-45 | evaluate_recommendation early-returns None on missing price | forces fallback path |
| backend/services/outcome_tracker.py | 147-148 | reflections gated on `if self._model` | model now passed (phase-31.1) |
| backend/db/bigquery_client.py | 375-392 | save_outcome -> financial_reports.outcome_tracking (append) | target table confirmed |
| backend/db/bigquery_client.py | 477-494 | save/get_agent_memory -> financial_reports.agent_memories | target table confirmed |
| backend/db/bigquery_client.py | 674-700 | get_paper_trades SELECT * (no remap) | confirms return_pct never appears |
| backend/config/settings.py | 32 | paper_learn_loop_enabled default False | the gate |
| backend/config/settings.py | 43-47 | bq_dataset_reports/outcomes = financial_reports | both tables in us-central1 |
| backend/tests/test_phase_35_1_learn_loop_writer.py | 33-45 | mock fixture hand-sets return_pct=17.89 | masks the field bug |
| financial_reports.outcome_tracking (live) | — | 0 rows, us-central1, no cycle_id col | EMPTY |
| financial_reports.agent_memories (live) | — | 0 rows, us-central1, no cycle_id col | EMPTY |
| financial_reports.paper_trades (live) | — | 22 rows, has realized_pnl_pct, NOT return_pct | confirms B.5 |

---

## JSON ENVELOPE

```json
{
  "tier": "moderate-complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 11,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```

