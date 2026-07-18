# Research Brief — phase-73.2 "D2b LEARN-LOOP V2 DESIGN"

Tier: **moderate** (caller-specified). NOT audit-class.
Role: D2b design-input research. Deliver `design_inputs` for FIVE components
(crash-fix, reflection-on-close, decay-rerank, retrieval-injection,
evidence-source-attribution) + an EXHAUSTIVE `deadness_causes` list with
file:line, every one re-verified in code (not inherited from the frontier map).

Constraints binding on every design choice:
- 2-person local paper fund on Peder's Mac (no fleet).
- `historical_macro` FROZEN until operator token.
- $0 dev metered budget; Gemini reflections ride the flat-fee rail; Claude on the
  Max rail. No metered embeddings (REJECTED in frontier-map #2).
- Existing BM25 memory + debate/gates STAY — ADAPT decay onto BM25, do NOT rebuild.

## Status: COMPLETE (gate_passed: true — 6 sources read in full; all deadness re-verified in code)

---

## 1. External sources read in full (6; >=5 required; counts toward the gate)

| # | Source | arXiv/URL | Accessed | Fetched how | Kind |
|---|--------|-----------|----------|-------------|------|
| E1 | FinMem (Yu et al., Stevens) — exact decay/importance math Eq 1-6 | 2311.13743 (ar5iv) | 2026-07-18 | `ar5iv/html` OK | memory/decay |
| E2 | Generative Agents (Park et al., Stanford) — recency·importance·relevance retrieval score | 2304.03442 (ar5iv) | 2026-07-18 | `ar5iv/html` OK | canonical memory scoring |
| E3 | LLM Financial-Trading Agent Survey (Ding et al.) — memory taxonomy | 2408.06361 | 2026-07-18 | `/html` OK | survey |
| E4 | FinCon (Yu et al.) — CVRF episodic self-critique on realized P&L | 2407.06567 | 2026-07-18 | `/html` OK | reflection/self-critique |
| E5 | Reflexion (Shinn et al.) — episodic reflection buffer (last-3), delayed-reward tension | 2303.11366 (ar5iv) | 2026-07-18 | `ar5iv/html` OK | reflection quality |
| E6 | Agentic Trading: When LLM Agents Meet Financial Markets (2026) — Outcome-Embargo, reflect-on-close, decay term | 2605.19337 | 2026-07-18 | `/html` OK | RECENCY, on-point |

### External key findings (per-claim)

**E1 FinMem — the exact decay/importance blueprint for component #3.**
- Recency (Eq 2): `S_Recency = e^(-δ/Q)`, `δ = t_now - t_event`. Layer half-lives `Q ∈ {14, 90, 365}` days (shallow news / 10-Q / 10-K).
- Importance (Eq 4-6): base `v ∈ {40,60,80}` (piecewise by layer), degraded `θ = α^δ` (`α ∈ {0.9, 0.967, 0.988}`), `S_Importance = v·θ`; a **pivotal event gets +5 importance and recency reset to 1.0**.
- Combined (Eq 1): `γ = S_Recency + S_Relevancy + S_Importance` — a SIMPLE SUM, each term scaled to [0,1]. Purge when `S_Recency < 0.05` or `S_Importance < 5`.
- Layering justified by SOURCE timeliness (news→days, filings→year). Load-bearing for our Q choice: FinMem's layers are keyed to FILING type, which our closed-trade-reflection corpus does not have.

**E2 Generative Agents — the canonical score the field builds on (Park 2023).**
- `score = α_recency·recency + α_importance·importance + α_relevance·relevance`, all α=1, three terms **min-max normalized to [0,1] then summed**. Recency = exp decay factor 0.995/hour since last access; importance = LLM 1-10 poignancy; relevance = embedding cosine. Confirms the additive-sum canon; our BM25 supplies the relevance leg for $0.

**E3 Survey — memory taxonomy + our-design honesty.** Layered memory (raw→summary→reflection buckets), retrieval by recency+relevance+importance; reflection grounded in cognitive science (interact→feedback→memory→lesson). Flags that most agents "apply LLMs through in-context learning without any fine-tuning" and that token cost of memory is largely unquantified — supports our BM25 (no-embedding) + $0 posture.

**E4 FinCon — reflect on REALIZED outcomes (validates component #2 trigger).** CVRF self-critique fires when `ρ_t < ρ_{t-1} or r_t < 0` (CVaR drop or realized loss); consumes the investment trajectory + realized P&L across consecutive episodes + risk signals; extracts winning-vs-losing "conceptualizations" and back-propagates them to relevant agents. Reflection is per-EPISODE (realized), not per-step. Directly supports: trigger on realized close, feed the reflection the trajectory + realized P&L + risk/exit signal.

**E5 Reflexion — bound the reflection buffer.** Self-reflection from a scalar-or-binary reward + trajectory + persistent memory; **buffer truncated to the last 3 self-reflections** (1 for programming tasks) to fit context; prepended to the next trial. Plateau ~trial 12. Takeaway for #4: inject a SMALL k of lessons (2-3), not the whole store.

**E6 Agentic Trading (2026, RECENCY) — the two decisive nuggets for #2/#3.**
- **Delayed reward:** "Reflexion-style systems assume relatively prompt feedback, which can be in tension with trading settings where outcomes materialize only after meaningful market delay" → reflect on CLOSED positions.
- **Outcome Embargo (Sec 4.2):** "an episode recorded at t cannot expose its outcome field to retrieval until t_now ≥ t+k" — a leakage-clean rule that dovetails with #3; our reflection only exists post-close, so we satisfy it by construction, but injection must not surface a same-cycle just-closed lesson to a still-open decision on the same name.
- **Time-aware retrieval:** relevance score with a decay term `e^(-λ(t_now-t_k))` (validates #3's multiplicative form); regime-conditioned decay is flagged as a FUTURE option with "no quantitative guidance on single vs. layered half-lives" — so a single-Q v1 is defensible, not contradicted.
- Does NOT address evidence-source vs agent-role attribution → confirms component #5 is OUR forward-compatible build.

---

## 2. Internal code inventory (re-verified this session, in full)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/outcome_tracker.py` | full (221) | `evaluate_recommendation` (crash), `evaluate_all_pending` (sibling guards :100-111), `_generate_and_persist_reflections` (:152-197), `evaluate_recent` wrapper (:204-220) | read in full |
| `backend/agents/memory.py` | full (269) | `FinancialSituationMemory` flat BM25, `add_memory` stores timestamp (:92), `get_memories` IGNORES it (:113-128), `format_for_prompt` (:131-143), `generate_reflection` prompt (:213-254) | read in full |
| `backend/services/autonomous_loop.py` | :300-344, :1520-1570, :2874-3050 | LIVE learn-loop dispatcher `_learn_from_closed_trades` (:2874); close seam hoist (:332); Step-9 call (:1547-1552) | read (3 windows) |
| `backend/config/settings.py` | :33 | `paper_learn_loop_enabled=False` default (gates fan-out) | read |
| `backend/db/bigquery_client.py` | :257-282, :379-512 | `get_recent_reports` (native-datetime rows), `save_outcome` (:379), `save_agent_memory` (:481, 5-col schema), `get_agent_memories` (:496) | read |
| `backend/agents/orchestrator.py` | :698-754, :2092-2114, :2255-2269 | memory init (4 instances), `_load_memories_from_bq` (:730), injection into debate (:2102-2114) + risk (:2255-2269) | read |

**agent_memories BQ schema (bigquery_client.py:484-490):** `{agent_type, ticker, situation[:2000], lesson[:1000], created_at}`. NO P&L column, NO evidence-source column — both #3 (importance bump) and #5 (source attribution) need ONE additive nullable-column migration.

---

## 3. Deadness causes (EXHAUSTIVE, re-verified in code — NOT inherited)

The 73.0/frontier-map framing ("~2-line fix, keystone crash") is *directionally* right
but INCOMPLETE and partly STALE. Re-verifying in code this session found the live
close-loop already has model-injection (phase-31.1) and a wired close seam (phase-30.3),
so the true deadness is a STACK of independent causes, three of which survive a perfect
crash-fix. All must be enumerated for the design to actually turn the loop green.

**DC1 — datetime type/tz crash in `evaluate_recommendation` (outcome_tracker.py:47, :50).**
`rec_date = datetime.fromisoformat(analysis_date)` (:47) then `naive_now - rec_date` (:50).
The `.replace(tzinfo=None)` at :50 normalizes ONLY the `now` side; `rec_date` is left
unguarded on BOTH axes. Two live failure legs, one per caller:
- **TYPE axis** (evaluate_all_pending path): `evaluate_all_pending` normalizes its *own*
  `rec_date` at :100-111 but passes the RAW `report["analysis_date"]` — a native
  `datetime` (BQ TIMESTAMP, per its own :96-99 comment) — to the unguarded method at
  :137 → :47 `fromisoformat(<datetime>)` raises `TypeError: argument must be str`.
- **TZ axis** (live `_learn_from_closed_trades` path): :2938-2940 coerces `analysis_date`
  via `created_at.isoformat()` → a tz-AWARE ISO string ("...+00:00"); :47 parses it to a
  tz-aware datetime; :50 `naive - aware` raises `TypeError: can't subtract offset-naive
  and offset-aware`. (If `analysis_id` is a non-ISO string, :47 raises `ValueError`.)
- NOTE the early-return at :44 (`if not current_price...: return None`) means the crash
  only fires when yfinance returns a valid price — i.e. the crash gets WORSE with better
  data, and it fires BEFORE the phase-35.1 fallback/reflection code at :2967+, defeating
  the very crash-resilient fallback that path was built to provide.
- **Fix** = ~4-line normalize inside `evaluate_recommendation`, mirroring the sibling
  :100-111 verbatim: accept datetime-or-str (`isinstance(analysis_date, datetime)`), then
  `if rec_date.tzinfo is not None: rec_date = rec_date.replace(tzinfo=None)`. Zero new
  deps, $0. This is the correct place (the crash is IN the method every caller shares).

**DC2 — `paper_learn_loop_enabled=False` default (settings.py:33) gates the ENTIRE
fan-out (autonomous_loop.py:2930, :2964).** Even with DC1 fixed, `_learn_from_closed_trades`
hits `if not learn_loop_enabled: continue` (:2964) BEFORE both the fallback `save_outcome`
and the `_generate_and_persist_reflections` call (:3039). So with the flag OFF (the live
default, per 73.0 flag-state), ZERO reflections and ZERO agent_memories rows are written
regardless of the crash. This is an INDEPENDENT deadness cause requiring an operator token
to flip — the design must call it out as gate #2, not fold it into "the crash".

**DC3 — the silent DEBUG swallow (autonomous_loop.py:3050).** The per-ticker
`except Exception as e: logger.debug(f"Outcome evaluation failed for {ticker}: {e}")`
catches the DC1 crash at **DEBUG** level. Default `LOG_LEVEL=INFO` (settings.py:21; operator
runs WARNING for quiet terminals) → the crash is INVISIBLE in production. Step 9 still
appends `"learning"` to `summary["steps"]`, so the loop *looks* healthy while every ticker
silently dies. This is why 36+ days of empty tables went unnoticed. A fix must raise this
to WARNING (or count+surface failures) so a future regression is observable.

**DC4 — reflection branch dead in the `evaluate_all_pending` callers (model=None).**
`_generate_and_persist_reflections` is gated `if self._model:` (outcome_tracker.py:147).
The live close-loop passes a model (phase-31.1, autonomous_loop.py:2908/2921) so it is NOT
dead there — but `evaluate_recent()` (:213), `reports.py` (:59/:76), and
`skill_optimizer.py` (:83) all construct `OutcomeTracker(settings)` with NO model, so the
reflection branch at :147-148 is permanently dead on those paths. Scope: keeps the
periodic/manual eval paths from ever reflecting; the live close-loop is unaffected. List it
so the design doesn't "re-fix" a fixed thing on the live path while leaving the shared
method's :147 guard mismatched across callers.

**DC5 — reflections compute P&L from a ROLLING mark, not the realized close, on the
primary path (outcome_tracker.py:42-43).** `evaluate_recommendation` derives `return_pct`
from `get_comprehensive_financials(ticker)` → *current market price* (:42-43), i.e. an
unrealized mark of a possibly-reopened position, not the executed exit. Only the
phase-35.1 *fallback* branch (autonomous_loop.py:2982-2985) reads the true
`realized_pnl_pct` off the SELL trade. So when the primary path succeeds, the reflection
learns from the wrong (rolling) P&L; the phase-47.7 comment (:2977-2981) documents that a
sibling field-name bug once recorded 0.0 realized return for EVERY sell-close. This is a
correctness defect (not strictly "dark"), but it is exactly what component #2
(reflection-on-close) must correct: trigger + P&L source must be the realized round-trip.

**NOT a current deadness cause (verified fixed — do NOT re-fix):**
- Close-event seam: `closed_tickers` hoisted to cycle-top (:332, phase-30.3) and fed to
  `_learn_from_closed_trades` at :1550 (guarded `if closed_tickers:`). WIRED.
- Model injection on the live path: phase-31.1 (:2908/:2921). DONE.
- BQ write path: `save_agent_memory`/`save_outcome` insert rows and log (not raise) on
  error — a soft-fail surface (rows silently dropped on BQ error, :394/:492) but the tables
  exist and no migration is needed for the base write.

---

## 4. Design inputs (FIVE components)

### C1 — crash-fix (outcome_tracker.py)
**Spec:**
- Insert a type+tz normalize INSIDE `evaluate_recommendation` at outcome_tracker.py:47-50,
  mirroring the sibling `evaluate_all_pending` :100-111 verbatim:
  `_ad = analysis_date; rec_date = _ad if isinstance(_ad, datetime) else datetime.fromisoformat(str(_ad)); if rec_date.tzinfo is not None: rec_date = rec_date.replace(tzinfo=None)`.
  Then the existing :50 subtraction works on both axes. ~4 lines, $0, no deps.
- Add a `ValueError` guard: if `str(analysis_date)` is a non-ISO `analysis_id` (live path passes
  `trade["analysis_id"] or created_at` at autonomous_loop.py:2938), fromisoformat raises
  ValueError — catch → `logger.warning` + `return None` (graceful skip), NOT a bare crash. Better:
  have the live dispatcher prefer `created_at` (always ISO) over `analysis_id` for the date arg.
- Raise the swallow (DC3): change autonomous_loop.py:3050 `logger.debug` → `logger.warning`
  (or increment a `summary["learn_failures"]` counter) so a future regression is observable.
**Seams:** outcome_tracker.py:47-50 (fix site) · :100-111 (template to mirror) · callers:
evaluate_all_pending:137 (native-datetime, TYPE axis), autonomous_loop.py:2949 (tz-aware str, TZ axis),
evaluate_recent:214/reports.py:77/skill_optimizer.py:135 (shared) · swallow at autonomous_loop.py:3050.
**Cost:** $0, ~4-6 lines. Zero new deps. No metered spend.

### C2 — reflection-on-close
**Spec:**
- Make the REALIZED exit the PRIMARY reflection input, not the rolling yfinance mark. The close
  event already exists: `closed_tickers` (hoisted autonomous_loop.py:332, phase-30.3) → Step 9
  `_learn_from_closed_trades` (:1550). The realized fields live on the SELL trade row from
  `bq.get_paper_trades` (:2924): `realized_pnl_pct` (written by paper_trader.execute_sell; read at
  :2982), `holding_days` (:2986), `price` (exit), and the exit `reason` (stop_loss / take_profit_2R/3R
  / signal-flip — verify exact column). RECOMMEND: build the outcome dict from these trade fields as
  the canonical path; demote `evaluate_recommendation`'s current-price path to fallback (invert
  today's ordering where evaluate_recommendation is primary and the trade-field build is fallback).
- Reflection prompt (memory.py:213-243 `generate_reflection`) should consume: entry price, exit price,
  exit_reason, realized_pnl_pct, holding_days, and the ORIGINAL thesis (debate consensus +
  contradictions from `full_report`, already available via build_situation_description). This matches
  FinCon (trajectory + realized P&L + risk signal) and Reflexion (trajectory + scalar reward).
- Token bound: today 4 Gemini calls/close (one per REFLECTION_AGENTS, outcome_tracker.py:25/177) on
  the flat-fee rail = $0 marginal. RECOMMEND TRIM to 1 (or 2) reflections/close — not for cost but for
  RETRIEVAL HYGIENE: 4 near-identical lessons on the same (situation, outcome) inject 4 correlated
  rows that then co-dominate BM25 top-k (a 4x duplication bias). Reflexion/FinCon both distil ONE
  lesson per episode. Keep per-agent only if the prompt truly differentiates each agent's stance.
- Idempotency: `save_outcome`/`save_agent_memory` APPEND (not upsert; autonomous_loop.py:2971-2974);
  a retry/re-run duplicates rows. RECOMMEND a dedup key on the SELL trade's unique id — store a
  nullable `source_trade_id` (part of the C5 migration) and skip if a row for that trade_id exists.
- Outcome-Embargo (E6): reflection only materializes post-close, so leakage-safe by construction;
  ensure injection (C4) does not surface a same-cycle just-closed lesson to a still-open decision.
**Seams:** autonomous_loop.py:1547-1552 (trigger) · :2924-2950 (trade lookup + primary) · :2967-3048
(fallback + fan-out) · outcome_tracker.py:152-197 (`_generate_and_persist_reflections`) · memory.py:213-254
(prompt) · bigquery_client.py:481-494 (save_agent_memory) · paper_trader.execute_sell (realized_pnl_pct source).
**Cost:** $0 marginal (flat-fee Gemini); trim 4→1-2 calls/close for corpus hygiene, not dollars.

### C3 — decay-rerank (FinMem-adapted onto BM25)
**Spec:**
- In `get_memories` (memory.py:113-128), after `normalized = scores[idx]/max_score`, apply a
  recency multiplier `recency = exp(-δ_days/Q)` (FinMem Eq 2), `δ_days = (now - ts).days` from the
  ALREADY-STORED metadata timestamp (add_memory:92; load_from_bq_rows:150-155 sets
  metadata["timestamp"]=created_at). Parse ts with the SAME tz/type normalize as C1 (reuse the lesson).
- Importance bump for large-|PnL|: `imp_mult = 1 + k·clip(|realized_pnl_pct|/CAP, 0, 1)` (e.g. k=0.5,
  CAP=10% → a ≥10% move → 1.5x), FinMem's v·θ adapted to our single signal. Requires storing |pnl|
  (agent_memories has NO P&L column — bigquery_client.py:484-490 → additive migration, see C5). Until
  migrated, `imp_mult = 1.0` (neutral) — forward-compatible, no behavior change on legacy rows.
- Composite: RECOMMEND MULTIPLICATIVE `final = bm25_norm · exp(-δ/Q) · imp_mult` (the task's spec form;
  also E6's decay-term shape). Chosen over FinMem's additive sum because multiplicative drives a STALE
  lesson's score → 0 (true fade), whereas the additive floor keeps recency mass even at zero relevance.
  **Code-structure fix:** today :116-118 sorts by RAW bm25 then slices top-k; the rerank must reorder
  by `final` BEFORE the top-k slice.
- **Q recommendation: SINGLE half-life Q ≈ 90 days** (not FinMem's layered 14/90/365). Justification at
  our scale: (a) our corpus is small (loop was dark; grows slowly) and single-provenance
  (closed-trade reflections), so FinMem's filing-typed layers don't map and can't fill 3 layers with
  volume; (b) Q=90 matches the strategy holding horizon (~90-135d triple-barrier labels, frontier-map
  #3) and the CENTER of `EVAL_WINDOWS=[7,30,90,180,365]` (outcome_tracker.py:22); at Q=90 a 90-day
  lesson keeps e^-1=37%, 180-day keeps 14% — a sensible fade for a quarter-scale strategy. Make Q a
  settings knob; the E6 regime-conditioned layered decay is a documented FUTURE option, not v1.
**Seams:** memory.py:113-128 (rerank + reorder-before-slice) · :92, :150-155 (timestamp already in
metadata) · :123 relevance floor (see C4).
**Cost:** $0. O(N) arithmetic (BM25 is already O(N); one exp() + one mult per doc). No embeddings, no API.

### C4 — retrieval-injection
**Spec:**
- Injection points (VERIFIED live): orchestrator.py:2102-2104 (bull/bear/moderator
  `format_for_prompt(situation_desc)`) → run_debate `past_memories` (:2114); :2255 (risk_judge) →
  run_risk_debate (:2264-2269). Startup load: `_load_memories_from_bq` (:730-754,
  get_agent_memories limit=200). Per-ticker situation built at :2096-2101.
- Token budget: `format_for_prompt` default n_matches=2, each ≤500 chars (200 situation + 300 lesson,
  memory.py:141) ≈ 250-350 tok/agent × 4. RECOMMEND keep k=2 (k=3 for the judge) — Reflexion's last-3
  buffer + context discipline. Memory competes with the Debate 1536-tok output budget (backend-agents.md);
  keep injected ≤ ~1000 chars/agent.
- Relevance floor: today `normalized > 0.1` (memory.py:123, 10% of max BM25). With C3, RECOMMEND a
  TWO-STAGE gate: (1) hard relevance floor on the BM25 component FIRST (`bm25_norm > 0.1`) so a
  recent-but-irrelevant lesson can't surface just because it's fresh; (2) rerank the survivors by the
  C3 composite; (3) top-k. Decay/importance reorder WITHIN the relevant set — they never pull in
  off-topic lessons. This is the key injection-safety nuance.
**Seams:** memory.py:103-143 (get_memories/format_for_prompt) · orchestrator.py:2096-2114, :2255-2269,
:730-754.
**Cost:** $0 (local BM25). No metered spend; injected tokens ride the existing flat-fee debate call.

### C5 — evidence-source-attribution (forward-compatible field for 73.5 PiT-RAG)
**Spec:**
- Today reflections attribute to AGENT type (REFLECTION_AGENTS, outcome_tracker.py:25 →
  agent_memories.agent_type). 73.5's PiT-RAG needs per-EVIDENCE-SOURCE-family attribution
  (news / sec_insider / technical / macro / sentiment / fundamentals). Add ALONGSIDE agent_type,
  do NOT replace.
- ONE additive, nullable, BQ `ALTER TABLE ADD COLUMN` migration on agent_memories (idempotent,
  `--verify`, scripts/migrations/ pattern) adding THREE columns that serve C2+C3+C5:
  `evidence_source_family STRING` (C5), `abs_pnl_pct FLOAT64` (C3 importance), `source_trade_id STRING`
  (C2 idempotency). Additive nullable ⇒ zero backfill; legacy rows read NULL and fall back to neutral
  (imp_mult=1.0, source_family="legacy/unknown").
- Populate `evidence_source_family` at write time from the enrichment_signals ALREADY IN HAND in
  `_generate_and_persist_reflections` (outcome_tracker.py:166 `enrichment_signals =
  full_report.get("enrichment_signals", {})`). v1: store `list(enrichment_signals.keys())` (the signal
  names ARE the source families) or the dominant family from build_situation_description's
  bullish/bearish/neutral partition (memory.py:183-198). 73.5 can later index by (source_family,
  event_type, horizon) — event_type/horizon derive from EVAL_WINDOWS with no further migration.
- Thread the new column through `save_agent_memory` (bigquery_client.py:481-494, add params) and
  `load_from_bq_rows` (memory.py:145-156, read into metadata) so retrieval can consume it later.
**Seams:** bigquery_client.py:481-494 (save + 5→8-col schema) · outcome_tracker.py:166/177-194 (write
site has enrichment_signals) · memory.py:145-156 (load into metadata) · scripts/migrations/ (additive migration).
**Cost:** $0 (BQ ADD COLUMN is metadata-only, no table rewrite, no scan cost). No metered spend.

---

## 5. Recency scan (2024-2026) — PERFORMED

Three query variants run: current-year frontier ("...realized outcome decay 2026"), year-less canonical
("...recency importance relevance forgetting curve"), + the anchor arXiv IDs. Findings:
- **The frontier CORROBORATES FinMem's exp-decay as canon and adds the delayed-reward nuance.** New
  2026 work: "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers"
  (2603.07670), "From Storage to Experience: Evolution of LLM Agent Memory" (2605.06716), "Learning
  What to Remember: Multi-Factor Value Model for Agentic Memory" (2606.12945), and E6 Agentic Trading
  (2605.19337, read in full). MemoryBank (Ebbinghaus forgetting curve) is the year-less prior-art
  corroborating exp decay; forgetting is categorized time-/frequency-/importance-based — our
  C3 = time (exp-δ/Q) + importance (|PnL|) driven.
- **Supersession check:** no 2025-26 work overturns the FinMem/Generative-Agents recency·importance·
  relevance canon; the movement is toward EVENT-DRIVEN / regime-conditioned decay (E6) — captured as a
  documented FUTURE option, not v1. "Useful Memories Become Faulty When Continuously Updated by LLMs"
  (2605.12978) is a caution AGAINST rewriting stored memories in place — supports our append-only +
  decay-rerank (never mutate a stored lesson) posture.
- No new finding changes the ADAPT-onto-BM25 / single-Q / $0 recommendations; they strengthen them.

---

## 6. Research Gate Checklist + envelope

Hard blockers (all satisfied):
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — 6 (E1-E6; FinMem+GenAgents+Reflexion
  via ar5iv, survey+FinCon+AgenticTrading via /html).
- [x] 10+ unique URLs total — 6 read-in-full + ~14 recency-scan/snippet hits = 20.
- [x] Recency scan (last 2 years) performed + reported (Section 5).
- [x] Full papers/sections read (not abstracts) for the read-in-full set — Reflexion re-fetched via
  ar5iv after the /abs/ page returned abstract-only.
- [x] file:line anchors for every internal claim (Sections 2-4).

Soft checks:
- [x] Consensus/contradiction noted (additive FinMem sum vs multiplicative task-spec form — resolved
  in favor of multiplicative with rationale; single-Q vs layered — resolved at our scale).
- [x] Internal exploration covered every relevant module (outcome_tracker, memory, autonomous_loop
  dispatcher, settings, bigquery_client, orchestrator injection).

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 14,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "D2b learn-loop V2 design inputs for 5 components, all re-verified in code. DEADNESS is a STACK, not one crash: DC1 datetime type+tz crash in evaluate_recommendation (outcome_tracker.py:47/:50 — the .replace(tzinfo=None) at :50 guards only the now-side; sibling guards both axes at :100-111 but passes the raw native-datetime to the unguarded method at :137; live path hits the TZ axis via created_at.isoformat() at autonomous_loop.py:2949); DC2 paper_learn_loop_enabled=False (settings.py:33) gates the whole fan-out at autonomous_loop.py:2964; DC3 the crash is swallowed at logger.DEBUG (autonomous_loop.py:3050 — invisible at default INFO, why 36+ days went unnoticed); DC4 model=None on the evaluate_all_pending callers kills the reflection branch (outcome_tracker.py:147) on non-live paths; DC5 primary reflection uses a ROLLING yfinance mark not realized close P&L (outcome_tracker.py:42-43). Close seam (phase-30.3) + live model-injection (phase-31.1) are already FIXED — do not re-fix. Design: C1 mirror the sibling normalize + raise the swallow; C2 make the realized SELL-trade fields the primary reflection input, trim 4->1-2 Gemini calls/close for BM25 corpus hygiene; C3 multiplicative bm25*exp(-d/Q)*imp with SINGLE Q=90d (justified at our small single-provenance scale vs FinMem's filing-typed 14/90/365), reorder-before-slice at memory.py:116; C4 two-stage gate (hard BM25 relevance floor 0.1 FIRST, then decay/importance rerank WITHIN survivors), k=2-3; C5 ONE additive nullable migration adding {evidence_source_family, abs_pnl_pct, source_trade_id} serving C5+C3+C2, populated from enrichment_signals already in hand. All $0/flat-fee/local/O(N); no embeddings. FinMem Eq1-6 + Generative-Agents canon + FinCon realized-episode trigger + Reflexion last-3 buffer + 2026 Outcome-Embargo ground the spec.",
  "brief_path": "handoff/current/research_brief_73.2.md",
  "gate_passed": true
}
```
