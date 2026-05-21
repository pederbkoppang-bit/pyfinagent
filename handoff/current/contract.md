# Sprint Contract — phase-33.1 Post-Cron Observation

**Step ID:** `phase-33.1`
**Date:** 2026-05-21
**Cycle type:** Diagnostic-only. NO code edits, NO mutating BQ/Alpaca/LLM calls. Single traffic-light verdict (HEALTHY / DEGRADED / FAILED) + 9-row checklist confirming phase-32 features fired in live data.

---

## Research-Gate Summary

- **Tier:** simple (internal-only). **gate_passed:** true (per researcher; transitive recency scan from phase-31.0).
- **Brief:** `handoff/current/research_brief.md`.
- **Cron fire confirmed.** `handoff/cycle_history.jsonl` row `cycle_id=8df751b3`, started 2026-05-21T18:00:00.415Z, completed 18:05:21.983Z. Duration 321 sec — a FULL cycle, not the 10-second halt path. The kill switch must have been resumed by the operator. n_trades=0.
- **n_trades=0 root cause identified pre-probe:** `backend.log` line 20:01:48 (and recurring throughout 20:01-20:02:57 CEST = 18:01-18:02:57 UTC) shows every position re-evaluation AND every candidate analysis failed with `"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."` (request_ids `req_011CbG9o3QNf...` through `req_011CbG9t6FfN...`). Both the full-orchestrator and lite-Claude analyzer paths failed for ALL 11 positions and ALL 4 candidates. The deterministic mechanics (mark-to-market, kill-switch check, backfill helpers, sector caps) ran cleanly; the LLM-dependent analysis pipeline produced no output to drive `decide_trades`.

---

## Hypothesis

The phase-32 deterministic features (breakeven ratchet, HWM trail, backfill helpers, sector caps) fired correctly in the live cycle. The LLM-dependent phase-32 features (Risk Judge `portfolio_sector_exposure` consumption, Synthesis `portfolio_concentration_warning` emission) could not be verified because the upstream synthesis pipeline produced zero output (Anthropic credit balance still empty — same blocker as phase-33.0, which the operator partially-cleared by resuming the kill switch but not by funding/swapping the LLM route).

Expected verdict: **FAILED** (≥2 FAILs on the LLM-dependent categories F + G), with a clear root-cause attribution that this is an unresolved operator-state issue (NOT a code regression).

---

## Success Criteria (IMMUTABLE)

1. `9_probes_each_with_PASS_WARN_FAIL` — every row in the table has a traffic-light verdict + ≥1 piece of verbatim evidence.
2. `single_top_level_verdict_HEALTHY_DEGRADED_or_FAILED` — `live_check_33.1.md` carries exactly one of those three labels at the top.
3. `no_code_edits_no_mutating_bq_or_alpaca_or_llm` — `git diff --stat backend/ scripts/` post-cycle is empty.
4. `live_check_quotes_top_3_followups` — `live_check_33.1.md` lists the top-3 followups for tomorrow's cycle.

Verification command:
```bash
test -f handoff/current/experiment_results.md && \
test -f handoff/current/live_check_33.1.md && \
grep -qE 'HEALTHY|DEGRADED|FAILED' handoff/current/live_check_33.1.md
```

---

## Immutable Hard Guardrails

1. NO code edits anywhere in `backend/` or `scripts/`.
2. NO mutating BQ writes.
3. NO Alpaca orders or mutating Alpaca calls.
4. NO LLM API calls.
5. Scope honesty: post-cycle `git diff --stat` touches ONLY `.claude/masterplan.json` + `handoff/*`. Any out-of-scope diff → revert + re-qa.

---

## Plan Steps

1. **RESEARCH** ✅ done.
2. **PLAN** ✅ this file.
3. **GENERATE** — run 9 probes, write `handoff/current/experiment_results.md` with the per-category table + roll-up verdict + verbatim evidence (curl, BQ, log grep, REPL).
4. **EVALUATE** — spawn `qa` ONCE. CONDITIONAL/FAIL → fix + FRESH qa. CIRCUIT BREAKER: max 2 retries → blocked + STOP.
5. **LIVE CHECK** — write `handoff/current/live_check_33.1.md` with the operator-facing one-page summary: verdict at top, 9-row table, top-3 followups.
6. **LOG** — append cycle block to `handoff/harness_log.md` BEFORE the status flip.
7. **FLIP** — `phase-33.1.status: in_progress → done`. Commit `phase-33.1:`. Stage explicit paths.

---

## References

- Research brief at `handoff/current/research_brief.md`.
- Phase-33.0 brief at `handoff/archive/phase-33.0/research_brief.md` (the predecessor that documented the 2 blockers).
- Phase-31.1 commit `3f020337` (the misnomer-fix that made the Anthropic-credit failure log-visible).
- backend.log @ `/Users/ford/.openclaw/workspace/pyfinagent/backend.log` lines 20:01:48-20:02:57 CEST (= 18:01:48-18:02:57 UTC) for the Anthropic-credit error chain.
- `paper_positions` BQ table for stop/strategy/company_name state verification.
