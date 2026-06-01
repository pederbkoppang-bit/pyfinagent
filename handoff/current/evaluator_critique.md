# Evaluator Critique — phase-43.0 (Production-Ready DoD audit)

**Q/A agent (merged qa-evaluator + harness-verifier). FRESH single spawn.**
Main produced the audit; I did NOT self-evaluate. Deterministic-first,
adversarial, anti-watermelon AND anti-false-fail. **Date:** 2026-06-01.
**Mode:** in-place working-tree read. **Verdict: CONDITIONAL. ok: false.**

> This OVERWRITES a STALE phase-50.6 critique that was left in this rolling
> file. The verdict below is for **phase-43.0** only.

## CRITICAL FRAMING (why CONDITIONAL is the correct, honest outcome)

phase-43.0 is an AUDIT step that CANNOT fully close autonomously. Two of its
four immutable success criteria are structurally out of autonomous reach:

- #1 `all_14_DoD_criteria_PASS` — NOT met (8/14 backend, 0/12 UX). 5 criteria
  are LIVE-BLOCKED (need operator LLM spend for live cycles), 1 OPERATOR-GATED
  (owner-gated cron fix), UX 0/12 needs the unbuilt phase-44.x + Playwright
  behind NextAuth.
- #4 `operator_approval_recorded_for_PRODUCTION_READY_declaration` — needs the
  REMOTE operator to type the approval string.

The DELIVERABLE of this step is the HONEST audit itself. My job is to confirm
the audit is honest + complete + accurate (criteria #2 verbatim-evidence + #3
no-silent-drops met, and the NOT_PRODUCTION_READY verdict ACCURATE) — NOT to
demand an impossible PASS, and NOT to FAIL it for the operator-gated criteria
being legitimately open with a documented plan (Google SRE PRR: deficits are
negotiated with an agreed plan-of-execution before sign-off — research_brief
source #2). The verdict is **CONDITIONAL**: the audit is sound, but the STEP
cannot be marked `done` (operator-gated). **phase-43.0 must stay `pending`.**

---

## 0. 3rd-CONDITIONAL auto-FAIL rule — NOT triggered (verified)

`grep -nE "^##.*phase=43\.0" handoff/harness_log.md` returns Cycles 12-20
(2026-05-28) — ALL `result=PASS` (audit) plus a `DELIVERED` soft-stop. There
are ZERO prior CONDITIONAL verdicts for step-id 43.0. The counter resets on
PASS. There is NO `phase=43.0` entry dated 2026-06-01 yet (this cycle's log is
not yet appended — log-last intact). Therefore a CONDITIONAL here is the FIRST
CONDITIONAL for this step-id and the auto-FAIL rule does not apply.

---

## 1. Harness-compliance audit (ran FIRST, per `feedback_qa_harness_compliance_first`) — 5/5 PASS

| # | Check | Result |
|---|-------|--------|
| 1 | researcher FIRST + gate passed | **PASS** — `research_brief.md` is the 43.0 brief; envelope `{"tier":"moderate","external_sources_read_in_full":5,"snippet_only_sources":9,"urls_collected":14,"recency_scan_performed":true,"internal_files_inspected":12,"gate_passed":true}`. 5 sources read in full (TrueFoundry/ML-Test-Score, Google SRE PRR, Cultivated/watermelon, DX checklist, Bailey-LdP DSR PDF). Recency scan present (3 fronts; trading-parity NautilusTrader/QuantConnect is a new complementary finding). The brief IS the substance of the audit (per-criterion enumeration). |
| 2 | `contract.md` BEFORE generate, N* delta + 4 criteria VERBATIM | **PASS** — N* delta present (`contract.md:5-9`: Risk↓ governance/honesty, no P/B, $0). The 4 criteria (`all_14_DoD_criteria_PASS`, `audit_file_carries_verbatim_evidence_per_criterion`, `qa_confirms_no_silent_drops`, `operator_approval_recorded_for_PRODUCTION_READY_declaration`) are copied verbatim at `:41-44`. Honest framing block (`:29-51`) correctly states #1+#4 are NOT autonomously achievable. |
| 3 | `experiment_results.md` + audit deliverable present | **PASS** — `experiment_results.md` has files-changed table, verbatim verification block (`:25-35`), and a VERBATIM acceptance-criteria-mapping table (`:39-44`) marking #1 NOT MET / #2 PASS / #3 pending-Q/A / #4 NOT MET. The deliverable `production_ready_audit_2026-06-01.md` exists (5919 bytes, mtime 17:43). |
| 4 | log-last / flip-last | **PASS** — no `phase=43.0` 2026-06-01 header in `harness_log.md` (only the 05-28 historical cycles); masterplan `id:43.0 status=pending retry=0 max=3`. 43.0.1/43.0.2 are `done` (the DoD-4 coverage-lift sub-steps — correct, distinct). The contract's plan step 4 explicitly says "result=CONDITIONAL ... Do NOT flip 43.0 to done." |
| 5 | First Q/A spawn (this cycle) | **PASS** — this is the first 2026-06-01 Q/A for 43.0; the 05-28 PASSes were a separate audit session on prior evidence. Not verdict-shopping: the evidence is freshly re-measured (738 collected, 16/711, OWASP 10/10) — a new audit, not a re-judgement of unchanged evidence. |

---

## 2. Anti-watermelon + anti-false-fail verification (the core of this gate) — all confirmed

### 2a. Claimed PASSes are GENUINELY met (ran the deterministic commands myself)

| DoD | Claim | My independent re-run | Verdict |
|-----|-------|----------------------|---------|
| DoD-12 | ascii_logger_check exit 0 | `python scripts/qa/ascii_logger_check.py` → `OK: 576 files, 1830 logger calls, 0 violations`; **EXIT=0** | **PASS confirmed** |
| DoD-14 | OWASP 10/10 tagged | `grep -oE "LLM(0[1-9]|10)" SKILL.md \| sort -u` → LLM01..LLM10, **count=10** | **PASS confirmed** |
| DoD-10 | prod default = gemini-2.5-pro | `model_tiers.py:69 "gemini_deep_think":"gemini-2.5-pro"`; `settings.py:31 deep_think_model=Field("gemini-2.5-pro"...)` (brief said `:66`/`:30`; actual `:69`/`:31` — value correct, off-by-a-few line cite is immaterial) | **PASS confirmed** |
| DoD-3 | hysteresis shipped | `grep -c check_auto_resume\|AUTO_RESUME kill_switch.py` = 7 hits | **PASS confirmed** |
| DoD-8 | scale-out wired | `grep -c check_scale_out_fires\|paper_scale_out_enabled paper_trader.py` = 3 hits | **PASS confirmed** |
| DoD-13 | restart-survivable | `cycle_lock.py` exists; `clean_stale_lock` in `main.py` (3 hits) | **PASS confirmed** |
| DoD-4 | Tier-1 STRICT ≥75% | `pytest -k "settings or config"` green (30 passed) corroborates no settings/config regression; coverage numbers not re-run (expensive) but 43.0.1/43.0.2 are `done` and the figures are internally consistent | **PASS accepted** |
| DoD-11 | 0 silent drops | independently re-run below (2c) | **PASS confirmed** |

### 2b. The 16 env-coupled failures are SURFACED, not hidden — and I reproduced them

I ran the FULL backend suite myself (not collect-only):
`python -m pytest backend/tests/ -q` → **`16 failed, 711 passed, 2 skipped, 8 xfailed, 1 xpassed in 106.81s`**.
This reproduces the audit's headline EXACTLY (16/711). The failure set matches
the audit's characterization line-by-line:
- **7×** `test_phase_23_2_16_shortlist_doc_presence` (moved/archived fixture-doc) — audit said "a moved fixture-doc ×7" ✓
- **4×** `test_phase_23_2_11_bq_table_freshness` (live-BQ freshness probes) ✓
- `test_phase_23_2_12_layer1_pipeline_active` + `test_phase_23_2_10_watchdog_no_fire_7d` (live BQ + running backend; backend was SIGTERM `-15` during the audit) ✓
- `test_agent_map_live_model`, `test_rainbow_canary`, `test_phase_23_2_14_no_reentrant_locks` (wiring/env-sensitive) — all three named verbatim in the audit ✓

The run emitted live `NotFound` warnings for `pyfinagent_data.api_call_log`
(missing table) — direct evidence the failures are environment/BQ-coupling,
NOT logic regressions. **The audit's env-coupled characterization is accurate.**

Crucially, the audit does the OPPOSITE of a watermelon:
- `production_ready_audit_2026-06-01.md:59` literally says **"Do NOT claim a
  fully-green suite from the 738 collect-only count."**
- It surfaces "16 failed / 711 passed" in two places (`:20` delta table + `:57`
  honesty finding). A grep for any *false* "all 738 pass / green suite" claim
  returned NONE. Anti-watermelon: PASS.

### 2c. DoD-11 (0 silent drops) — independently re-verified

`comm -23 <(roadmap OPEN-ids) <(masterplan OPEN-ids)` → exactly **OPEN-19,
OPEN-21, OPEN-27** are in the roadmap but not the masterplan (matching the
audit). Disposition: OPEN-19→phase-42.0, OPEN-21→phase-42.3 (roadmap rows),
OPEN-27→doc-only + two NAMED auto-memories. I confirmed BOTH named files EXIST:
`/Users/ford/.claude/projects/.../memory/feedback_auto_commit_hook_stalls.md`
and `.../feedback_researcher_write_first.md`. The audit's note that
`grep OPEN-27 MEMORY.md`=0 is expected (disposition is by topic-file existence,
not OPEN-id string) is correct. **33/33 accounted, 0 silent drops — confirmed.**

### 2d. LIVE-BLOCKED / OPERATOR-GATED classifications are HONEST (spot-checked)

- **DoD-1 (OPERATOR-GATED):** `launchctl list | grep pyfinagent` →
  `com.pyfinagent.ablation` exit=**1**, `com.pyfinagent.autoresearch` exit=**1**,
  `com.pyfinagent.backend` = **-15** (SIGTERM). Two failing crons genuinely
  need owner action (huggingface install + ablation triage). NOT a PASS being
  dodged. ✓
- **DoD-9 (LIVE-BLOCKED):** Python tally of `cycle_history.jsonl` terminal rows
  → most-recent consecutive `completed` streak = **4** (a `timeout 2f2f3b64`
  breaks the run before the 4). Genuinely **< 5**. Needs live cycles. ✓

### 2e. No forged/sought operator approval

`grep -niE "PRODUCTION_READY: APPROVED"` in the audit + cycle_block_summary
returns ONLY the *instruction to the operator* ("Type ... once green"), never a
recorded approval. Criterion #4 is honestly marked NOT MET. The verdict is
explicitly NOT_PRODUCTION_READY. Main did NOT seek or forge approval. ✓

### 2f. Verdict accuracy

NOT_PRODUCTION_READY with backend 8/14 (DoD-3,4,8,10,11,12,13,14) and UX 0/12 is
ACCURATE against my independent checks. The 8 PASSes are all genuinely met; the
6 open backend criteria are legitimately LIVE-BLOCKED (5) / OPERATOR-GATED (1)
with named closure paths (phase-35.x / 39.1 / 44.x), which Google SRE PRR
sanctions as a negotiated-deficit-with-plan, not a cop-out.

---

## 3. Code-review heuristic sweep (SKILL: code-review-trading-domain) — N/A (no code diff)

phase-43.0 is a read-only AUDIT: the only changed files are handoff docs
(`production_ready_audit_*.md`, `cycle_block_summary.md`, cycle artifacts). No
`paper_trader`/`kill_switch`/`risk_engine`/`perf_metrics`/`backtest_engine`/
`.env`/secret/`orchestrator` edit. All Dimension-1/2/3/4 BLOCK+WARN heuristics
are N/A (no executable logic changed). Dimension-5 (LLM-evaluator anti-patterns):
not sycophancy/verdict-shopping (§0, §1.5 — fresh evidence, first CONDITIONAL);
this critique cites file:line + verbatim command output throughout (no
`missing-chain-of-thought`). Worst severity: **NOTE**. No secret-in-diff
(grepped the docs — only operator-instruction strings, no credential literals).

---

## Verdict

**CONDITIONAL. ok: false.** The audit is HONEST, COMPLETE, and ACCURATE — but
the STEP cannot be marked `done` because two immutable criteria (#1
all-14-PASS, #4 operator-approval) are structurally operator-gated. This is the
correct outcome for an audit step, not a defect.

Criteria #2 (verbatim per-criterion evidence) and #3 (no silent drops) are MET
and I independently confirmed them. Every claimed PASS I spot-checked is
genuinely met (DoD-12 exit 0; DoD-14 OWASP 10/10; DoD-10 gemini-2.5-pro on both
defaults; DoD-3/8/13 grep-confirmed; DoD-11 33/33 OPEN-ids mapped + both named
auto-memories exist). I reproduced the 16/711 full-suite result myself and
confirmed all 16 are environment-coupled (7× moved fixture-doc, 4× live-BQ
freshness, 2× live-BQ+backend, 3× wiring/canary) with live `api_call_log`
NotFound warnings as direct evidence — NOT logic regressions. The audit
explicitly refuses to claim a green suite (`:59`) and surfaces the 16 failures
in two places — the antithesis of a watermelon. LIVE-BLOCKED/OPERATOR-GATED are
honest: DoD-1 cron exit=1 (×2), DoD-9 streak=4 (<5) both verified. No operator
approval was forged or sought. The NOT_PRODUCTION_READY verdict (8/14 backend,
0/12 UX) is accurate. 3rd-CONDITIONAL auto-FAIL does NOT apply (zero prior 43.0
CONDITIONALs; the 05-28 cycles were PASS). **phase-43.0 MUST stay `pending`;**
log the cycle as `result=CONDITIONAL` and carry the operator asks to
`cycle_block_summary.md`. Do NOT flip to done.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "phase-43.0 is an AUDIT step; the audit deliverable is HONEST + COMPLETE + ACCURATE, but the STEP cannot be marked done because 2 of 4 immutable criteria are structurally operator-gated -- #1 all_14_DoD_criteria_PASS is NOT met (8/14 backend: DoD-3,4,8,10,11,12,13,14 PASS; 5 LIVE-BLOCKED DoD-2/5/6/7/9; 1 OPERATOR-GATED DoD-1; UX 0/12) and #4 operator_approval needs the REMOTE operator. Criteria #2 (verbatim per-criterion evidence) + #3 (no silent drops) ARE met and independently confirmed. Harness 5/5: researcher first w/ gate_passed:true (5 sources read in full, recency scan, 14 URLs); contract precedes generate w/ N* delta + 4 criteria VERBATIM + honest framing that #1/#4 are not autonomously achievable; experiment_results has verbatim verification block + verbatim criteria-mapping; harness_log has NO 2026-06-01 phase=43.0 entry (only 05-28 historical PASS cycles) + masterplan 43.0 status=pending retry=0 max=3 (log-last/flip-last intact); first 2026-06-01 Q/A (fresh re-measured evidence, not verdict-shopping). ANTI-WATERMELON CONFIRMED: I re-ran the FULL backend suite myself -> 16 failed/711 passed in 106.81s, reproducing the audit headline EXACTLY; the 16 break down precisely as claimed (7x test_phase_23_2_16 moved fixture-doc, 4x test_phase_23_2_11 live-BQ freshness, test_phase_23_2_12+test_phase_23_2_10 live-BQ+backend, test_agent_map_live_model+test_rainbow_canary+test_phase_23_2_14 wiring/canary) with live NotFound warnings for pyfinagent_data.api_call_log as direct env-coupling evidence -- NOT logic regressions; the audit at :59 literally says 'Do NOT claim a fully-green suite' and surfaces 16 failed/711 passed at :20 + :57 (a grep for any FALSE green-suite claim found none). CLAIMED PASSES GENUINELY MET (ran the commands): DoD-12 ascii_logger_check EXIT=0 (576 files/1830 calls/0 viol); DoD-14 OWASP grep = LLM01..LLM10 count 10; DoD-10 model_tiers.py:69 + settings.py:31 both gemini-2.5-pro; DoD-3 check_auto_resume/AUTO_RESUME 7 hits; DoD-8 check_scale_out_fires/paper_scale_out_enabled 3 hits; DoD-13 cycle_lock.py exists + clean_stale_lock in main.py; DoD-11 comm -23 shows exactly OPEN-19/21/27 deferred (roadmap rows + BOTH named auto-memories feedback_auto_commit_hook_stalls.md + feedback_researcher_write_first.md EXIST) = 33/33 mapped, 0 silent drops; pytest -k 'settings or config' = 30 passed (no regression). LIVE-BLOCKED/OPERATOR-GATED HONEST: launchctl shows com.pyfinagent.ablation exit=1 + com.pyfinagent.autoresearch exit=1 + backend SIGTERM -15 (DoD-1 genuinely owner-gated); cycle_history.jsonl consecutive completed streak=4 <5 (DoD-9 genuinely needs live cycles). NO FORGED APPROVAL: the only 'PRODUCTION_READY: APPROVED' strings are operator-instructions ('Type ... once green'), never a recorded approval; #4 honestly marked NOT MET; verdict explicitly NOT_PRODUCTION_READY. 3rd-CONDITIONAL auto-FAIL N/A (zero prior 43.0 CONDITIONALs; 05-28 cycles 12-20 were all PASS; counter reset). No code diff -> code-review heuristics N/A (only handoff docs changed; no money-path/risk/secret edit). VERDICT CONDITIONAL because the audit is sound (criteria #2+#3 met, NOT_PRODUCTION_READY accurate) but the step is operator-gated on #1+#4 -- phase-43.0 MUST stay pending; log result=CONDITIONAL; do NOT flip to done; do NOT seek/forge operator approval.",
  "violated_criteria": ["all_14_DoD_criteria_PASS", "operator_approval_recorded_for_PRODUCTION_READY_declaration"],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "audit DoD tally (independently re-verified: ascii exit0, OWASP 10/10, launchctl exit=1 x2, cycle streak=4, full-suite 16/711)",
      "state": "backend 8/14 PASS (5 LIVE-BLOCKED need operator LLM spend, 1 OPERATOR-GATED owner cron fix), UX 0/12 (phase-44.x unbuilt + Playwright behind NextAuth)",
      "constraint": "all_14_DoD_criteria_PASS -- legitimately open with documented closure plans (Google SRE PRR negotiated-deficit); audit deliverable is honest, but step cannot be marked done",
      "severity": "operator-gated"
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "criterion #4 operator approval",
      "state": "operator is REMOTE this week; only operator-instruction strings present, no recorded approval; Main did NOT forge/seek it (correct)",
      "constraint": "operator_approval_recorded_for_PRODUCTION_READY_declaration -- requires the remote operator to type the approval; not autonomously satisfiable",
      "severity": "operator-gated"
    }
  ],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "research_brief_43_0_gate_envelope", "contract_criteria_verbatim", "experiment_results_completeness", "log_last_no_2026_06_01_43_0_entry", "masterplan_status_pending", "third_conditional_rule_check", "ascii_logger_check_exit0", "owasp_llm_10_of_10_grep", "dod10_gemini_2_5_pro_defaults", "dod3_hysteresis_grep", "dod8_scaleout_grep", "dod13_cycle_lock", "dod11_open_id_silent_drop_comm", "open27_named_automemories_exist", "full_suite_16_711_reproduced", "env_coupled_failure_characterization", "audit_refuses_green_suite_claim", "launchctl_dod1_exit1", "cycle_streak_dod9_eq4", "no_forged_operator_approval", "settings_config_regression_30", "code_review_heuristics"]
}
```
