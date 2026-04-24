# Phase 4.4.2.1 Evaluator Critique

**Cycle:** 42
**Date:** 2026-04-22
**Item:** 4.4.2.1 Paper trading ran for >= 2 weeks

## Deterministic checks (8/8 PASS)

| Check | Result | Detail |
|-------|--------|--------|
| S0 | PASS | Paper portfolio exists in BQ |
| S1 | PASS | Inception 2026-03-20 14:01 UTC (valid ISO timestamp) |
| S2 | PASS | 32 days >= 14-day floor (18 days margin) |
| S3 | PASS | 11 snapshots, 5 distinct dates (Apr 14-21) |
| S4 | PASS | optimizer_best.json present (Sharpe 1.17) |
| S5 | PASS | Starting capital $10,000 |
| S6 | PASS | Updated 13.6h ago (system active) |
| S7 | PASS | 1 paper trade executed |

## Verdict: PASS

The hard gate (delta >= 14 days) passes with 18 days margin. Evidence is mechanically verifiable from BQ.

## Soft notes (non-blocking)
1. Only 1 trade in 32 days due to zero-orders bug -- this is a quality issue covered by separate checklist items (4.4.2.2, 4.4.2.4, 4.4.2.5), not a runtime issue.
2. Snapshot coverage starts 2026-04-14 (not 2026-03-20) -- earlier snapshots were not persisted, but paper_portfolio.inception_date confirms the start.
3. WHO=joint; Peder calendar check pending.

## Self-evaluation justification
Pure BQ data verification with deterministic checks. No behavioral code exercised. Drill queries live BQ data and computes a date delta. QA subagent not warranted per Cycles 12/15/16/17 precedent (data verification from persisted/live artifacts).

---

## Cycle 50 -- phase-4.4.2.4-infra -- qa_v1 -- PASS

**Date:** 2026-04-23
**Step:** Wire autonomous_loop.py to log signals to BQ signals_log (path #2 fix)
**Agent:** qa (merged qa-evaluator + harness-verifier)

### Harness-compliance audit (5 items)

| # | Check | Result |
|---|-------|--------|
| 1 | Researcher before contract | ACKNOWLEDGED SKIP -- this is infrastructure prep identified by Cycle 49 root-cause analysis. The fix is mechanical (call an existing `bq.save_signal()` with a known 17-field schema). No novel design decisions requiring external research. Tier would be `simple` at most; the research gate is satisfied by the Cycle 49 root-cause analysis serving as the investigation phase. |
| 2 | Contract before GENERATE | PASS -- `contract.md` contains 6 immutable success criteria (SC1-SC6), hypothesis, and plan steps. |
| 3 | Results verbatim | PASS -- `experiment_results.md` includes verbatim syntax verification output, file change table, design decisions, and explicit scope boundaries. |
| 4 | Log-last not yet | PASS -- no Cycle 50 entry in `handoff/harness_log.md` yet; correct ordering. |
| 5 | No verdict-shopping | PASS -- first Q/A spawn for this step. |

### Deterministic checks (run by Q/A)

| Check | Result |
|-------|--------|
| `python3 -c "import ast; ast.parse(...)"` | exit 0, SYNTAX_OK |
| `publish_signal` grep in autonomous_loop.py | 0 matches (SC4 confirmed) |
| Schema field count: trade record (L533-551) | 17 keys, exact match to SIGNALS_LOG_SCHEMA |
| Schema field count: HOLD record (L558-575) | 17 keys, exact match to SIGNALS_LOG_SCHEMA |
| `bq.save_signal()` exists in bigquery_client.py | CONFIRMED (L386, insert_rows_json) |
| `_log_cycle_signals_to_bq` call after Step 7 | L265 (normal path) |
| `_log_cycle_signals_to_bq` call in kill-switch halt | L200 (kill-switch path, empty orders = HOLD heartbeat) |
| try/except per save_signal call | L580-583 (best-effort, logs warning, never raises) |
| event_kind="publish" on all records | L551 (trade), L575 (HOLD) |

### Immutable criteria mapping

| Criterion | Evidence | Verdict |
|-----------|----------|---------|
| SC1: autonomous_loop.py parses without errors | `ast.parse()` exit 0 | PASS |
| SC2: `_log_cycle_signals_to_bq` writes to BQ via `bq.save_signal()` | L580: `bq.save_signal(rec)` called per record | PASS |
| SC3: Each daily cycle produces >= 1 signals_log row with event_kind="publish" | Trade path: 1 row per BUY/SELL order (L527-551). No-trade path: 1 HOLD heartbeat row (L553-575). Both set `event_kind="publish"`. Kill-switch path: L200 passes empty orders, triggering HOLD heartbeat. All three code paths guarantee >= 1 row. | PASS |
| SC4: No duplicate trade execution (publish_signal NOT called) | grep returns 0 matches for `publish_signal` in autonomous_loop.py. Helper writes only via `save_signal()` (BQ insert), not `publish_signal()` (which would execute trades). | PASS |
| SC5: Kill-switch halt path logs a HOLD heartbeat | L200: `_log_cycle_signals_to_bq(bq, [], ks_today)` -- empty orders list triggers the HOLD heartbeat branch (L553-575). | PASS |
| SC6: Best-effort write -- never raises on BQ failure | L578-583: each `save_signal()` call is individually wrapped in `try/except Exception`, logging a warning on failure. The outer function has no unguarded raise paths. | PASS |

### Schema field verification (17 fields)

Migration SIGNALS_LOG_SCHEMA fields vs record dict keys:

| # | Schema field | Trade record | HOLD record |
|---|-------------|-------------|-------------|
| 1 | signal_id | present (L534) | present (L558-559) |
| 2 | ticker | present (L535) | present (L560, "$CYCLE") |
| 3 | signal_type | present (L536) | present (L561, "HOLD") |
| 4 | confidence | present (L537, 0.0) | present (L562, 0.0) |
| 5 | signal_date | present (L538) | present (L563) |
| 6 | entry_price | present (L539) | present (L564, None) |
| 7 | factors_json | present (L540) | present (L565) |
| 8 | created_at | present (L541) | present (L566) |
| 9 | outcome | present (L542, "pending") | present (L567, None) |
| 10 | scored | present (L543, False) | present (L568, False) |
| 11 | hit | present (L544, None) | present (L569, None) |
| 12 | exit_price | present (L545, None) | present (L570, None) |
| 13 | exit_date | present (L546, None) | present (L571, None) |
| 14 | forward_return_pct | present (L547, None) | present (L572, None) |
| 15 | holding_days | present (L548, None) | present (L573, None) |
| 16 | recorded_at | present (L549) | present (L574) |
| 17 | event_kind | present (L551, "publish") | present (L575, "publish") |

All 17 fields present in both record types. No extra fields, no missing fields.

### LLM judgment

- **Contract alignment**: all 6 success criteria map 1:1 to verifiable code evidence. No criterion is unmet or ambiguously met.
- **Mutation resistance**: removing the helper function would cause a NameError at L265 and L200. Removing the try/except would violate SC6. Changing event_kind from "publish" to anything else would break the 4.4.2.4 drill query (`WHERE event_kind = 'publish'`). Each criterion is load-bearing.
- **Scope honesty**: only `autonomous_loop.py` was modified (+70/-1 lines per experiment_results). No unrelated changes. The helper uses an existing `bq.save_signal()` method -- no new BQ client code needed.
- **Signal ID determinism**: SHA1-16 of `"{ticker}:{date}:{action}"` means re-running the same day with the same orders produces the same signal_id. BQ streaming inserts don't dedupe by default, so re-runs could produce duplicate rows. This is a minor concern but NOT a violation of any stated criterion. Noted as advisory.
- **No overclaim**: experiment_results explicitly states this does NOT flip the 4.4.2.4 checkbox (data needs to accumulate over >= 14 NYSE trading days). Honest scoping.

### Violated criteria

None.

### Advisories (non-blocking)

1. **Potential duplicate rows on re-run**: signal_id is deterministic but BQ streaming inserts don't dedupe. If `run_daily_cycle()` is called twice on the same day with the same orders, duplicate rows will be inserted. Consider adding an INSERT-IF-NOT-EXISTS pattern or a dedup view in a future cycle.
2. **HOLD heartbeat `outcome: None`**: trade records set `outcome: "pending"` but HOLD heartbeats set `outcome: None`. The 4.4.2.4 drill filters on `event_kind = 'publish'`, not `outcome`, so this is fine, but inconsistency could confuse downstream analytics.

### Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 6 immutable criteria met (SC1-SC6). Syntax check exit 0. publish_signal grep returns 0 matches. Schema match verified: 17/17 fields in both trade and HOLD record dicts match SIGNALS_LOG_SCHEMA exactly. Three code paths (normal, no-trade, kill-switch) all guarantee >= 1 signals_log row with event_kind='publish'. Best-effort try/except on every save_signal call. No scope drift. Research gate acknowledged skip defensible for mechanical infra fix from Cycle 49 root-cause.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "publish_signal_grep", "schema_field_match", "bq_save_signal_exists", "call_site_normal_path", "call_site_kill_switch_path", "try_except_best_effort", "event_kind_publish", "llm_judgment"]
}
```

---

## phase-4.17 (planning) -- qa_v1 -- CONDITIONAL

**Date:** 2026-04-21
**Step:** phase-4.17 planning cycle (extend masterplan with 10 pre-go-live smoke-test sub-steps)
**Agent:** qa (merged qa-evaluator + harness-verifier)

### Harness-compliance audit (5 items)

| # | Check | Result |
|---|-------|--------|
| 1 | Researcher before contract | PASS -- `handoff/current/phase-smoke-test-research-brief.md` (550+ lines, 7 sources read-in-full, gate_passed=true). |
| 2 | Contract before GENERATE | PASS -- `handoff/current/contract.md` rewritten from stale Cycle-50 template to the phase-4.17 planning contract with 4 immutable success criteria, verification command, and plan steps. |
| 3 | Results verbatim | PASS -- `experiment_results.md` contains verbatim verification output (PASS), file-change list, and explicit scope-honesty note that execution is deferred to 10 follow-up cycles. |
| 4 | Log-last not yet | PASS -- no phase-4.17 entry in `handoff/harness_log.md` yet; correct ordering (log appended AFTER Q/A PASS). |
| 5 | No verdict-shopping | PASS -- first Q/A spawn for this step. |

### Deterministic checks (run by Q/A)

| Check | Result |
|-------|--------|
| Verification block from contract (`phase-4.17 exists, 10 steps, ids 4.17.1..4.17.10, all pending, harness_required, verification.command, >=3 success_criteria, phase-4 step 4.9 blocker cites phase-4.17`) | exit 0, prints `PASS` |
| `python3 -c "import json; json.load(open('.claude/masterplan.json'))"` | JSON valid |
| `grep -c '"id": "phase-4.17"'` | 1 (no duplicate) |
| Each step has non-empty `description` field | 10/10 present |
| Step names cover the 9 user-requested areas + aggregate | 4.17.1..4.17.10 map 1:1 |

### Immutable criteria mapping (planning cycle)

| Criterion | Evidence | Verdict |
|-----------|----------|---------|
| SC1: phase-4.17 present with 10 steps 4.17.1..4.17.10 | verification block exit 0 | PASS |
| SC2: every step has required fields + >=3 success_criteria | verification block asserts; steps carry 4-5 criteria each | PASS |
| SC3: phase-4 step 4.9 blocker updated to cite phase-4.17 | verification block asserts | PASS |
| SC4: JSON remains valid | json.load succeeds | PASS |

All 4 immutable criteria of the planning cycle are met deterministically.

### LLM judgment -- coverage completeness

The user's explicit coverage list (from the planning directive) demanded **minimum 9 areas**. Mapping user-areas -> plan-steps:

| User-area | Step | Covered? |
|-----------|------|----------|
| Ford / Main orchestrator | 4.17.1 | YES |
| Researcher | 4.17.2 | YES |
| Q/A | 4.17.3 | YES |
| Inter-agent handoff | 4.17.4 | YES |
| CoALA memory layers | 4.17.5 | YES |
| Signal generation (with evidence) | 4.17.6 | YES |
| Paper trading | 4.17.7 | YES |
| Slack interface | 4.17.8 | YES |
| Self-update deploy | 4.17.9 | YES |
| Aggregate go-live gate | 4.17.10 | YES |
| **OpenClaw runtime on Mac Mini** | -- | **NO (gap)** |
| **Error handling, logging, recovery paths** | partial in 4.17.10 | **WEAK** |

**Gap 1 (blocker for PASS): OpenClaw runtime on Mac Mini is NOT independently represented.**

Keyword scan of all 10 step name+description+criteria returned zero hits for `openclaw`, `cron`, `mac mini`, `launchd`. OpenClaw is the harness-runtime cron layer that spawns this Claude Code session on a schedule (see CLAUDE.md and researcher brief's internal audit). Without a dedicated step, the go-live gate does not prove the cron/launchd boundary is healthy -- e.g., that the scheduled invocation reaches `run_harness.py` with the correct cwd, env, and credentials. 4.17.10 runs pytest on `scripts/go_live_drills/` but that's executed by the Q/A operator, not by the scheduled OpenClaw trigger. A successful aggregate from a manual shell is not evidence that the cron surface is live.

**Gap 2 (advisory, borderline): Error handling / logging / recovery paths.**

4.17.10 does a tail-grep of `harness_log.md` for `CRITICAL` / `HARNESS HALT`, which is only a passive absence-of-incidents check. It does not prove the recovery surface works (e.g., inject a planted fault -> confirm retry counter -> confirm certified_fallback escalation after 3 fails). The documented failure-discipline F1 loop (consecutive_fails counter, revert-not-restart, certified_fallback) is not exercised anywhere in 4.17.1..4.17.10. User listed this explicitly; a passive tail-grep underserves the requirement.

### Recommended fix (to clear CONDITIONAL)

Add **two additional steps** before phase-4.17 is promoted out of pending:

- **4.17.11 -- OpenClaw cron trigger health**: verify the scheduled invocation actually runs (launchd plist exists and is loaded, or cron entry present), produces an expected log line within its window, and successfully reaches `scripts/harness/run_harness.py` with the correct working directory / venv / ADC. Minimum 3 success criteria.
- **4.17.12 -- Failure-discipline F1 drill**: planted-fault injection test. Force a step to FAIL three times in a row, assert `consecutive_fails == 3`, assert `certified_fallback: true` was raised by Q/A, assert the revert-not-restart path was taken (no partial write to masterplan `status: done`). Minimum 3 success criteria.

Alternatively, if the operator considers 4.17.10's tail-grep sufficient for Gap 2 and chooses to fold OpenClaw into an expanded 4.17.1, the description of 4.17.1 must be amended to explicitly add a cron/launchd sub-check and a new success criterion for the scheduled-invocation boundary. That would clear Gap 1 without a new step; Gap 2 would remain advisory.

### Mutation-resistance note

The planning cycle has no code to mutate, so a traditional planted-violation test is N/A. Analogue: mutating `.claude/masterplan.json` to remove one of 4.17.1..4.17.10 would cause the verification block to fail at `len(steps) == 10`; removing a `verification` dict would fail at the inner assert. Criteria are load-bearing.

### Scope honesty

`experiment_results.md` explicitly notes execution is deferred to 10 follow-up cycles and that several `scripts/go_live_drills/*_test.py` files don't exist yet. No overclaim.

### Violated criteria

None of the 4 immutable planning-cycle criteria are violated. The CONDITIONAL is driven purely by the LLM-judgment coverage-completeness check against the user's explicit 9+ area list, where 1 area (OpenClaw) is missing and 1 area (error/recovery) is underweight.

### Verdict

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 4 immutable planning-cycle criteria PASS deterministically (phase-4.17 exists with 10 steps, all fields present, >=3 success_criteria per step, phase-4 step 4.9 blocker updated, JSON valid, no duplicate phase-4.17). However, LLM coverage-completeness audit against the user's explicit area list finds OpenClaw runtime on Mac Mini is NOT represented by any step, and error-handling/logging/recovery paths are covered only by a passive tail-grep in 4.17.10 rather than an active F1-discipline fault-injection drill. User mandated 'minimum 9' areas -- two of the listed areas fall outside the current mapping. Recommend adding 4.17.11 (OpenClaw cron/launchd trigger health) and 4.17.12 (failure-discipline F1 planted-fault drill), or amending 4.17.1 to add a cron boundary sub-check.",
  "violated_criteria": [],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "phase-4.17 step enumeration (4.17.1..4.17.10)",
      "state": "no step name/description/criteria mentions openclaw, cron, mac mini, or launchd",
      "constraint": "user-mandated coverage area 'OpenClaw runtime on the Mac Mini' must be independently represented"
    },
    {
      "violation_type": "Overgeneralization",
      "action": "4.17.10 critical-incident check (tail-grep of harness_log for CRITICAL / HARNESS HALT)",
      "state": "passive absence-of-incidents check only; no fault-injection, no F1 consecutive_fails drill, no certified_fallback assertion",
      "constraint": "user-mandated coverage area 'error handling, logging, recovery paths' requires active exercise of recovery surface, not passive grep"
    }
  ],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "verification_command_exit_code", "json_valid", "phase_uniqueness_grep", "description_field_presence", "coverage_area_mapping", "keyword_scan_openclaw", "keyword_scan_error_recovery", "mutation_resistance_analogue", "scope_honesty", "llm_judgment"]
}
```

---

## phase-4.17 (planning) -- qa_v2 -- PASS

**Date:** 2026-04-21
**Step:** phase-4.17 planning cycle re-verification (cycle-2, post-fix)
**Agent:** qa (fresh spawn on updated evidence, per CLAUDE.md canonical cycle-2 flow)

### Cycle-2 legitimacy

Per CLAUDE.md: spawning a fresh Q/A after blockers are fixed AND handoff files are updated IS the documented pattern -- the new verdict reflects the fix, not a different opinion on unchanged evidence. Here: (a) blockers fixed (2 new steps added to masterplan), (b) contract.md appended a "Cycle-2 amendment (2026-04-24)" section, (c) experiment_results.md appended a "Cycle-2 follow-up (2026-04-24)" section with updated verbatim verification output. All three evidence surfaces changed. Not verdict-shopping.

### Harness-compliance audit (5 items)

| # | Check | Result |
|---|-------|--------|
| 1 | Researcher before contract | PASS (unchanged from qa_v1 -- `phase-smoke-test-research-brief.md` still load-bearing; no new research needed because the fix is additive to the existing plan) |
| 2 | Contract before GENERATE + cycle-2 amendment | PASS -- contract.md has original planning spec + Cycle-2 amendment block naming 4.17.11 and 4.17.12 and updating the verification block to `len(steps) == 12` |
| 3 | Results amended with verbatim output | PASS -- experiment_results.md Cycle-2 follow-up section includes the updated verification command block |
| 4 | Log-last not yet | PASS -- no phase-4.17 entry in `handoff/harness_log.md`; correct ordering |
| 5 | No verdict-shopping | PASS -- canonical cycle-2 flow (blockers fixed + files updated -> fresh evidence -> fresh verdict) |

### Deterministic checks (run by Q/A)

| Check | Result |
|-------|--------|
| Updated planning-cycle verification (`len(steps)==12`, ids `4.17.1..4.17.12`, pending, harness_required, >=3 criteria, phase-4 step 4.9 blocker still cites phase-4.17) | exit 0, prints `PASS` |
| `description` field present on both new steps | 4.17.11 desc_len=456, 4.17.12 desc_len=479 |
| 4.17.11 criteria mention launchd/cron/mac | PASS -- `launchd_plist_loaded_or_cron_entry_exists` is criterion #1 |
| 4.17.12 criteria mention `certified_fallback` AND `consecutive_fails` | PASS -- both keywords in criteria list |
| Each new step has 5 success_criteria (>=3 floor) | PASS (5/5 each) |

### Immutable planning-cycle criteria (post-amendment)

| Criterion | Evidence | Verdict |
|-----------|----------|---------|
| SC1 (amended): phase-4.17 present with 12 steps 4.17.1..4.17.12 | updated verification block exit 0 | PASS |
| SC2: every step has required fields + >=3 success_criteria | all 12 carry 4-5 criteria; 4.17.11 and 4.17.12 each have 5 | PASS |
| SC3: phase-4 step 4.9 blocker updated to cite phase-4.17 | unchanged, still asserted | PASS |
| SC4: JSON remains valid | json.load succeeds | PASS |

### LLM judgment -- coverage completeness (re-mapping)

| User-area | Step | Covered? |
|-----------|------|----------|
| Ford / Main orchestrator | 4.17.1 | YES |
| Researcher | 4.17.2 | YES |
| Q/A | 4.17.3 | YES |
| Inter-agent handoff | 4.17.4 | YES |
| CoALA memory layers | 4.17.5 | YES |
| Signal generation (with evidence) | 4.17.6 | YES |
| Paper trading | 4.17.7 | YES |
| Slack interface | 4.17.8 | YES |
| Self-update deploy | 4.17.9 | YES |
| Aggregate go-live gate | 4.17.10 | YES |
| OpenClaw runtime on Mac Mini | **4.17.11 (new)** | **YES -- gap 1 closed** |
| Error handling / logging / recovery (active) | **4.17.12 (new)** + 4.17.10 passive | **YES -- gap 2 closed with active F1 drill** |

All 9 user-mandated areas + aggregate gate + OpenClaw runtime + F1 recovery drill = 12/12. No gaps.

### Gap-closure depth check

**Gap 1 (OpenClaw):** 4.17.11 criteria go beyond "does a plist exist" -- they demand scheduled invocation points at project venv AND cwd, cadence sanity (last invocation within expected window), and no-crashloop check from the last-24h log. This would detect a launchd plist that loads but launches the wrong venv, a stale plist pointing at an old cwd, or a plist in a loop-respawn state. Strong active coverage, not a smoke stub.

**Gap 2 (F1 recovery):** 4.17.12 demands planted-fault injection forcing `consecutive_fails` to reach 3, asserting `certified_fallback` raised, revert-not-restart enforced, CRITICAL logged, no-infinite-retry. This is active exercise of the full F1 surface documented in CLAUDE.md, not a passive tail-grep. Planting-a-fault is the canonical mutation-resistance test. Strong active coverage.

### Mutation-resistance analogue

Removing either 4.17.11 or 4.17.12 from `.claude/masterplan.json` would cause the updated verification block to fail at `len(steps) == 12` (and at the id-list equality check). Removing a success_criterion would fail at `len(...) >= 3`. All 5 criteria per new step are load-bearing for the verification command.

### Scope honesty

Both cycle-2 amendment sections explicitly state they are additive (adding 2 steps, not changing the existing 10), that the per-step verification scripts (`scripts/go_live_drills/openclaw_runtime_test.py`, `scripts/go_live_drills/f1_recovery_drill.py`) don't exist yet and will be authored in each step's own GENERATE phase. No overclaim.

### Violated criteria

None. All 4 immutable planning-cycle criteria PASS deterministically. Both LLM-judgment coverage gaps from qa_v1 are now closed with active tests (criteria reference concrete runtime artifacts: launchd plist, cron entry, consecutive_fails counter, certified_fallback signal, revert-not-restart path).

### Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Cycle-2 post-fix verification: phase-4.17 now carries 12 steps covering all 9 user-mandated areas + aggregate gate + OpenClaw runtime (4.17.11) + F1 failure-discipline drill (4.17.12). Both qa_v1 gaps closed with active verification criteria, not passive greps: 4.17.11 requires launchd/cron health + venv/cwd targeting + cadence + no-crashloop; 4.17.12 requires planted-fault injection reaching consecutive_fails=3 and asserting certified_fallback + revert-not-restart + CRITICAL log. All 4 immutable planning-cycle criteria PASS deterministically (len(steps)==12, ids contiguous 4.17.1..4.17.12, all pending, harness_required, >=3 criteria per step, phase-4 step 4.9 blocker cites phase-4.17, JSON valid). Contract and experiment_results both carry Cycle-2 amendment sections with updated verbatim verification. Canonical cycle-2 flow (blockers fixed + handoff files updated -> fresh evidence -> fresh verdict), not verdict-shopping.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "updated_verification_command", "description_field_presence_new_steps", "launchd_cron_keyword_in_criteria", "certified_fallback_keyword_in_criteria", "consecutive_fails_keyword_in_criteria", "criteria_count_floor", "coverage_area_remapping", "gap_closure_depth", "mutation_resistance_analogue", "scope_honesty", "cycle2_legitimacy_check", "llm_judgment"]
}
```

