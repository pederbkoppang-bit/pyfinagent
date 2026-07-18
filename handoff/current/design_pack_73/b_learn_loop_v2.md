# Design D2b — Learn-Loop v2 (phase-73.2, FINAL — specs verbatim from gate `wf_a195f7b3-5d8`, 6 sources read in full)

Frontier-map #2 verdict executed at implementation depth. Key gate corrections to 73.0's framing: the '~2-line keystone crash' is directionally right but the deadness is a STACK (below); the close-event seam (autonomous_loop.py:332→:1550, phase-30.3) and live model-injection (:2908/:2921, phase-31.1) are ALREADY FIXED — **do NOT re-fix**. Literature: FinMem Eq 1-6 exact decay math (ar5iv), Generative-Agents additive canon, FinCon realized-outcome self-critique, Reflexion one-lesson-per-episode, 2026 Agentic-Trading survey's Outcome-Embargo (our post-close reflection satisfies it by construction). Full notes: `research_brief_73.2.md`.

## Deadness stack (exhaustive, re-verified in code — every cause must clear for one reflection row to exist)

1. DC1 (crash) — outcome_tracker.py:47 rec_date=datetime.fromisoformat(analysis_date) + :50 subtraction; the .replace(tzinfo=None) at :50 guards only the NOW side, rec_date is unguarded on BOTH axes. TYPE axis: evaluate_all_pending passes the RAW native-datetime report['analysis_date'] to the unguarded method at outcome_tracker.py:137 -> :47 fromisoformat(datetime) raises TypeError (sibling guards its own copy at :100-111 but not the arg it passes). TZ axis: the live _learn_from_closed_trades coerces via created_at.isoformat() at autonomous_loop.py:2938-2940 -> tz-aware string -> :50 naive-minus-aware raises TypeError. Fires only after the yfinance early-return at :44, BEFORE the phase-35.1 fallback.
2. DC2 (flag default OFF) — settings.py:33 paper_learn_loop_enabled=False gates the ENTIRE writer fan-out: autonomous_loop.py:2930 reads it and :2964 'if not learn_loop_enabled: continue' short-circuits BEFORE both the fallback save_outcome and the _generate_and_persist_reflections call (:3039). Even a perfect crash-fix writes zero reflections/agent_memories rows until an operator flips the flag. Independent of DC1.
3. DC3 (silent DEBUG swallow) — autonomous_loop.py:3050 'except Exception as e: logger.debug(f"Outcome evaluation failed for {ticker}: {e}")' catches the DC1 crash at DEBUG level, invisible at the default LOG_LEVEL=INFO (settings.py:21; operator runs WARNING), while Step 9 still appends 'learning' to summary['steps'] — the loop looks healthy while every ticker silently dies. This is why 36+ days of empty tables went unnoticed.
4. DC4 (model=None reflection branch) — outcome_tracker.py:147 gates reflections on 'if self._model:'. The live close-loop passes a model (phase-31.1, autonomous_loop.py:2908/:2921) so it is NOT dead there, but evaluate_recent (outcome_tracker.py:213), reports.py:59/:76, and skill_optimizer.py:83 all construct OutcomeTracker(settings) with NO model -> the reflection branch at :147-148 is permanently dead on those (periodic/manual eval) paths.
5. DC5 (rolling-mark P&L, not realized close) — outcome_tracker.py:42-43 evaluate_recommendation derives return_pct from get_comprehensive_financials(ticker) (CURRENT market price / unrealized mark), not the executed exit. Only the phase-35.1 fallback branch (autonomous_loop.py:2982-2985) reads the true realized_pnl_pct off the SELL trade; when the primary path succeeds, the reflection learns from the wrong (rolling) P&L. Correctness defect that component #2 must correct (make the realized close the primary input).
6. NOT-DEAD (verified fixed — do NOT re-fix): close-event seam is wired (closed_tickers hoisted to autonomous_loop.py:332 phase-30.3, fed to _learn_from_closed_trades at :1550 under 'if closed_tickers:'); live model-injection done (phase-31.1, autonomous_loop.py:2908/:2921); agent_memories + outcome_tracking BQ tables exist (no base-write migration needed, though save_agent_memory/save_outcome soft-fail — log-not-raise — at bigquery_client.py:394/:492).

## 1. Crash-fix + deadness stack (build step 73.2.1 [executor: sonnet-4.6/high])

Spec (verbatim from the gate):
- Insert a type+tz normalize INSIDE evaluate_recommendation (outcome_tracker.py:47-50) mirroring the sibling evaluate_all_pending :100-111 VERBATIM: _ad=analysis_date; rec_date = _ad if isinstance(_ad, datetime) else datetime.fromisoformat(str(_ad)); if rec_date.tzinfo is not None: rec_date = rec_date.replace(tzinfo=None). The existing :50 subtraction then works on both the native-datetime (TYPE) and tz-aware-string (TZ) legs. ~4 lines, $0, no deps.
- Add a ValueError guard for a non-ISO analysis_id (the live path passes trade['analysis_id'] or created_at at autonomous_loop.py:2938): on unparseable date, logger.warning + return None (graceful skip) rather than a bare crash; better, have the live dispatcher prefer created_at (always ISO) over analysis_id for the date arg.
- Raise the silent swallow (DC3): change autonomous_loop.py:3050 logger.debug -> logger.warning, or increment summary['learn_failures'], so a future regression is observable at the default INFO/WARNING log level.
- This single fix covers all three callers because the crash lives IN the method they share (evaluate_all_pending:137 hits the TYPE axis; _learn_from_closed_trades:2949 hits the TZ axis; evaluate_recent/reports/skill_optimizer inherit it).

Integration seams: outcome_tracker.py:47-50 (fix site) | outcome_tracker.py:100-111 (sibling guard = the template to mirror) | outcome_tracker.py:137 (evaluate_all_pending passes raw native-datetime to the unguarded method) | autonomous_loop.py:2938-2949 (live path str(analysis_date) via created_at.isoformat -> tz-aware) | autonomous_loop.py:3050 (the logger.debug swallow to raise)

Cost: $0. ~4-6 lines, zero new deps, no metered spend. Pure defect fix on the flat-fee/local path.

## 2. Reflection-on-close (build step 73.2.2 [executor: sonnet-4.6/high])

Spec (verbatim from the gate):
- Make the REALIZED exit the PRIMARY reflection input, not the rolling yfinance mark. The close event already exists (closed_tickers hoisted autonomous_loop.py:332 -> Step 9 _learn_from_closed_trades :1550). Build the outcome dict from the SELL trade row fields returned by bq.get_paper_trades (:2924): realized_pnl_pct (paper_trader.execute_sell; read :2982), holding_days (:2986), price (exit), and the exit reason (stop_loss / take_profit_2R/3R / signal-flip — verify exact column). Demote evaluate_recommendation's current-price path to FALLBACK (inverting today's ordering).
- Extend the reflection prompt (memory.py:213-243 generate_reflection) to consume: entry price, exit price, exit_reason, realized_pnl_pct, holding_days, and the ORIGINAL thesis (debate consensus + contradictions from full_report, already available via build_situation_description). Matches FinCon (trajectory + realized P&L + risk signal) and Reflexion (trajectory + scalar reward).
- Token bound: today 4 Gemini calls/close (REFLECTION_AGENTS, outcome_tracker.py:25/177) = $0 marginal on the flat-fee rail. TRIM to 1 (or 2) per close — NOT for cost but for BM25 corpus hygiene: 4 correlated lessons on the same (situation,outcome) co-dominate retrieval top-k. Keep per-agent only if the prompt truly differentiates each stance.
- Idempotency: save_outcome/save_agent_memory APPEND, not upsert (autonomous_loop.py:2971-2974) — a retry duplicates rows. Dedup on the SELL trade's unique id via a nullable source_trade_id (part of the C5 migration); skip if a row for that trade_id already exists.
- Outcome-Embargo (E6): reflection only materializes post-close so it is leakage-safe by construction; ensure C4 injection does not surface a same-cycle just-closed lesson to a still-open decision on the same name.

Integration seams: autonomous_loop.py:1547-1552 (Step 9 trigger, non-fatal wrap) | autonomous_loop.py:2924-2950 (trade lookup + primary evaluate_recommendation call) | autonomous_loop.py:2967-3048 (fallback save_outcome + reflection fan-out) | outcome_tracker.py:152-197 (_generate_and_persist_reflections) | memory.py:213-254 (generate_reflection prompt) | bigquery_client.py:481-494 (save_agent_memory) | paper_trader.execute_sell (realized_pnl_pct source)

Cost: $0 marginal (flat-fee Gemini). Recommend trimming 4->1-2 LLM calls/close for retrieval-corpus hygiene, not dollars. No metered spend.

## 3. Decay re-rank, Q=90d multiplicative (build step 73.2.3 [executor: sonnet-4.6/high])

Spec (verbatim from the gate):
- In get_memories (memory.py:113-128), after normalized = scores[idx]/max_score, apply recency = exp(-delta_days/Q) (FinMem Eq 2), delta_days = (now - ts).days from the ALREADY-STORED metadata timestamp (add_memory:92; load_from_bq_rows:150-155 sets metadata['timestamp']=created_at). Parse ts with the SAME tz/type normalize as C1.
- Importance bump for large-|PnL|: imp_mult = 1 + k*clip(|realized_pnl_pct|/CAP, 0, 1) (e.g. k=0.5, CAP=10% -> a >=10% move gives 1.5x), FinMem's v*theta adapted to our single signal. Requires storing |pnl| (agent_memories has NO P&L column -> additive migration in C5). Until migrated, imp_mult=1.0 (neutral, forward-compatible, no behavior change on legacy rows).
- Composite = MULTIPLICATIVE final = bm25_norm * exp(-delta/Q) * imp_mult (the task's spec form; also E6's decay-term shape). Chosen over FinMem's additive sum because multiplicative drives a stale lesson's score toward 0 (true fade) whereas the additive floor keeps recency mass at zero relevance.
- CODE-STRUCTURE FIX: memory.py:116-118 sorts by RAW bm25 then slices top-k; the rerank must reorder by final BEFORE the top-k slice.
- Q RECOMMENDATION: SINGLE half-life Q~=90 days (not FinMem's layered 14/90/365). Justified at our scale: small single-provenance corpus (closed-trade reflections, not filings) can't fill 3 layers; Q=90 matches the ~90-135d triple-barrier holding horizon and the center of EVAL_WINDOWS=[7,30,90,180,365]. Make Q a settings knob; E6's regime-conditioned/layered decay is a documented FUTURE option, not v1.

Integration seams: memory.py:113-128 (get_memories rerank; reorder-before-slice at :116-118) | memory.py:92 and :150-155 (timestamp already stored in metadata) | memory.py:123 (existing relevance floor; see C4 two-stage gate) | bigquery_client.py:484-490 (agent_memories schema needs abs_pnl_pct for the importance term)

Cost: $0. O(N) arithmetic (BM25 already O(N); one exp() + one multiply per doc). No embeddings, no API calls — REJECTS FinMem's metered ada-002 relevance leg, keeps our BM25.

## 4. Retrieval injection, two-stage gate (build step 73.2.3 — same executor step as §3)

Spec (verbatim from the gate):
- Injection points VERIFIED live: orchestrator.py:2102-2104 (bull/bear/moderator format_for_prompt(situation_desc)) -> run_debate past_memories (:2114); :2255 (risk_judge) -> run_risk_debate (:2264-2269). Startup load: _load_memories_from_bq (:730-754, get_agent_memories limit=200). Per-ticker situation built at :2096-2101 via build_situation_description.
- Token budget: format_for_prompt default n_matches=2, each <=500 chars (200 situation + 300 lesson, memory.py:141) ~= 250-350 tok/agent x4. Keep k=2 (k=3 for the judge) per Reflexion's last-3 buffer + context discipline; memory competes with the Debate 1536-tok output budget (backend-agents.md) so keep injected <= ~1000 chars/agent.
- Relevance floor = TWO-STAGE gate: (1) hard BM25 relevance floor FIRST (bm25_norm > 0.1, memory.py:123) so a recent-but-irrelevant lesson can't surface just because it's fresh; (2) rerank survivors by the C3 composite; (3) top-k. Decay/importance reorder WITHIN the relevant set and never pull in off-topic lessons — the key injection-safety nuance.
- Leakage: honor the E6 Outcome-Embargo at injection — do not inject a lesson from a trade closed in the SAME cycle into a still-open decision on the same ticker.

Integration seams: memory.py:103-143 (get_memories + format_for_prompt) | orchestrator.py:2096-2114 (debate injection) | orchestrator.py:2255-2269 (risk-judge injection) | orchestrator.py:730-754 (_load_memories_from_bq startup load)

Cost: $0 (local BM25 retrieval). Injected lessons ride the existing flat-fee debate/risk LLM calls; no extra API round-trips, no metered spend.

## 5. Evidence-source attribution + BQ migration (build step 73.2.2 — the single additive migration ships with §2)

Spec (verbatim from the gate):
- Today reflections attribute to AGENT type (REFLECTION_AGENTS, outcome_tracker.py:25 -> agent_memories.agent_type). 73.5's PiT-RAG needs per-EVIDENCE-SOURCE-family attribution (news/sec_insider/technical/macro/sentiment/fundamentals). Add ALONGSIDE agent_type, do NOT replace it.
- ONE additive, nullable, idempotent BQ ALTER TABLE ADD COLUMN migration (scripts/migrations/ pattern, --verify) on agent_memories adding THREE columns serving C5+C3+C2: evidence_source_family STRING (C5), abs_pnl_pct FLOAT64 (C3 importance), source_trade_id STRING (C2 idempotency). Additive nullable => zero backfill; legacy rows read NULL and fall back to neutral (imp_mult=1.0, source_family='legacy/unknown').
- Populate evidence_source_family at write time from enrichment_signals ALREADY IN HAND in _generate_and_persist_reflections (outcome_tracker.py:166 enrichment_signals = full_report.get('enrichment_signals', {})). v1: store list(enrichment_signals.keys()) (signal names ARE the source families) or the dominant family from build_situation_description's bullish/bearish/neutral partition (memory.py:183-198). 73.5 later indexes by (source_family, event_type, horizon); event_type/horizon derive from EVAL_WINDOWS with no further migration.
- Thread the new column through save_agent_memory (bigquery_client.py:481-494, add params) and load_from_bq_rows (memory.py:145-156, read into metadata) so retrieval can consume it later without a schema change.

Integration seams: bigquery_client.py:481-494 (save_agent_memory + 5->8-col schema) | outcome_tracker.py:166 and :177-194 (write site already holds enrichment_signals) | memory.py:145-156 (load_from_bq_rows -> metadata) | scripts/migrations/ (the additive nullable migration, idempotent + --verify)

Cost: $0. BQ ADD COLUMN is metadata-only (no table rewrite, no scan cost). Populated from data already in the write path. No metered spend; forward-compatible for 73.5 without a future migration.

## Design decisions of record

- **Q = single 90-day half-life, multiplicative composite** `bm25_norm × exp(-d/90) × importance_mult` — NOT FinMem's layered 14/90/365 (our corpus is small, single-provenance; 90d matches the 90-135d holding horizon and centers EVAL_WINDOWS). memory.py:116-118 must reorder by the composite BEFORE the top-k slice.
- **Trim reflections 4→1-2 per close** — retrieval hygiene, not cost (4 correlated rows co-dominate BM25 top-k; Reflexion/FinCon distil one lesson per episode).
- **One additive nullable BQ migration serves three components**: agent_memories + {evidence_source_family, abs_pnl_pct, source_trade_id} — metadata-only, $0, zero backfill; forward-compatible with 73.5's PiT-RAG weighting.
- **Two-stage injection gate**: BM25 relevance floor (memory.py:123, >0.1) FIRST, decay/importance rerank WITHIN survivors, THEN top-k — recency must never pull an off-topic lesson into a prompt.
- **Realized-exit-primary**: reflection consumes the SELL-trade fields (realized_pnl_pct, holding_days, exit price/reason); the rolling yfinance mark demotes to fallback. Flag remains `paper_learn_loop_enabled` (operator flips per dark-until-token).