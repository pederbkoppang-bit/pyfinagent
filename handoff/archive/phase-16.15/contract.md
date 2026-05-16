# Contract — phase-16.15 (Go/No-Go verdict)

**Step ID:** 16.15
**Step name:** Go/No-Go verdict (Q/A spawn required — NO self-evaluation)
**Date:** 2026-05-16
**Cycle:** post-restart, post-16.59-Q/A-upgrade
**Author:** Main (this session)
**Status:** GENERATE phase (research-gate cleared; awaiting Q/A verdict)

---

## Research gate summary

Read full brief at `handoff/current/research_brief.md`. Researcher (a777c4e3d9d6ab322) returned `gate_passed=true`:

- **Tier:** complex (max research gate)
- **External sources read in full:** 6 (Cornell/CFR 15c3-5, PMC Raahauge 2022 HRO paper, Deloitte/FCA governance, ASIC CP386, Visual Paradigm scoring guide, Knight Capital case study; ESMA Feb 2026 PDF partial)
- **URLs collected:** 19 (6 in-full + 12 snippet-only + 1 partial PDF)
- **3-variant search:** 9 queries disclosed (current-year 2026 + last-2-year 2024-2025 + year-less canonical)
- **Recency scan:** 5 new 2024-2026 findings (ESMA Feb 2026, ASIC Aug 2025, FCA 2025, FIA Jul 2024, ESMA CSA 2024); all uniformly tighten kill-switch + pre-trade controls

**Top 5 research findings (load-bearing for this contract):**

1. SEC 15c3-5 + MiFID II RTS 6 + ASIC CP386 uniformly require kill-switch reachability, pre-trade controls, and exclusive-control on every execution path. A single BLOCK-severity outstanding item = NO-GO regardless of weighted score (no risk-memo override).
2. CONDITIONAL has two sub-states: *conditional-go* (only WARN items outstanding) vs *conditional-hold* (any BLOCK item outstanding). The 16.23 CONDITIONAL was conditional-hold because condition #1 (Anthropic key swap) was BLOCK-severity on the MAS Layer-3 path.
3. Five anti-patterns to police in this verdict: rubber-stamping, risk-memo conflation, ship-pressure sycophancy, verbosity bias, verdict-shopping. All five are encoded in the upgraded Q/A's Dimension 5 heuristics at `qa.md:368-388`.
4. HRO production-readiness pattern: each sub-criterion must be classified as hard-block or advisory BEFORE the aggregate verdict is computed. No weighted average can override a hard-block.
5. Knight Capital ($460M / 45min) failure modes — no kill interlock, ignored pre-market alerts, no halt procedure — directly map to Q/A heuristics `kill-switch-reachability [BLOCK]` (`qa.md:255`), `broad-except-silences-risk-guard [BLOCK]` (`qa.md:259`), `financial-logic-without-behavioral-test [BLOCK]` (`qa.md:260`).

---

## Hypothesis

After the phase-16.59 upgrade landed (commit `ba18ffab`, qa.md:201-426 — 5 dimensions, severity dispatch, simultaneous-presentation rule), the Q/A subagent has sufficient discernment to render a defensible aggregate Go/No-Go verdict on the 14-step phase-16 UAT bundle (16.1-16.14) PLUS the post-CONDITIONAL re-verification cycle (16.16-16.23 + 16.28 + 16.58).

A verdict produced by the upgraded Q/A is *not* the same as the 2026-04-25 verdict (which used the pre-uplift Q/A) — even on identical evidence, the new heuristics may surface findings the old Q/A missed. The verdict-shopping prohibition (CLAUDE.md, `qa.md:375`) does NOT apply because the evaluator's evaluative substrate has changed (new section added to qa.md). This is a documented exception to second-opinion-shopping per the harness-design doctrine: "fresh-respawn after substantive critique change is the documented cycle-2 flow."

---

## Immutable success criteria (verbatim from `.claude/masterplan.json:5444-5451`)

These criteria are immutable; Main MUST NOT edit them.

1. Main spawned a fresh qa subagent with the 16.1-16.14 evidence bundle
2. qa returned verdict == PASS (not CONDITIONAL, not FAIL)
3. harness_log.md appended with the Go/No-Go verdict row referencing the qa output
4. Peder acknowledged the verdict in-session before status is flipped to done
5. Q/A PASS is immutable — self-evaluation is forbidden
6. no_regressions

**Reading note:** Criterion #2 sets a high bar — CONDITIONAL is not GO. The research brief's Dimension 2 framework (BLOCK-severity outstanding = NO-GO) is consistent: a CONDITIONAL with a BLOCK item is functionally NO-GO. Criterion #4 is the gate against autonomous progression — even after Q/A PASS, no status flip until Peder explicitly ACKs in-session.

---

## What this contract requires Q/A to verify

The upgraded Q/A must apply ALL of the following checks to the 16.1-16.14 bundle. Each is itemized so the verdict can quote specific evidence per criterion.

### Block-1: Regulatory floor (SEC 15c3-5 / MiFID II RTS 6 / ASIC CP386 analogues)

| Check | Target | Heuristic |
|-------|--------|-----------|
| Kill-switch reachable from every execution path | `backend/services/kill_switch.py:12-18` + `paper_trader.py` execution paths | `kill-switch-reachability [BLOCK]` (`qa.md:255`) |
| Paper-lockout assert before live trade | `backend/services/paper_trader.py` + `backend/services/account_router.py`; assert `ALPACA_PAPER_TRADE != 'false'` | 16.4 success_criteria #1 |
| Pre-trade risk controls (capital/position/erroneous-order) | `backend/markets/risk_engine.py` + `backend/services/paper_trader.py:131-132` (max-position guard) | `max-position-check-bypass [BLOCK]` (`qa.md:305`) |
| Stop-loss always set on buy path | `backend/services/paper_trader.py:99-114` | `stop-loss-always-set [BLOCK]` (`qa.md:255`) |
| Capital threshold / drawdown guard | `kill_switch.py:9-11` + `risk_engine.py:33-35` constants | inspect for citation discipline |

### Block-2: Resolution status of the 16.23 four conditions

Per `handoff/harness_log.md:12548-12581` + masterplan notes for 16.28 (line 5688) and 16.58 (line 6043):

| Condition | Original severity | Last-known status | Q/A action |
|-----------|-------------------|-------------------|-----------|
| #1 Anthropic key swap (sk-ant-oat01-* → sk-ant-api03-*) | BLOCK on Layer-3 MAS path | 16.58 closure 2026-04-26: "Operator pasted new sk-ant-api03 key (108 chars)" + 16.58 success command asserts key format. | Re-run 16.58's command; confirm key still passes assertion. If yes → condition #1 cleared. If no → condition #1 still BLOCK → verdict CANNOT be PASS. |
| #2 Cron TZ on APScheduler jobs | WARN | 16.24/16.18 closure: TZ=America/New_York wired on 7 cron jobs | Grep for `timezone=` absence in scheduler files; flag any missing |
| #3 Autoresearch diag | WARN | 16.24 closure | Confirm `handoff/autoresearch/` recent error files don't escalate |
| #4 MAS-Layer-2 out (audit findings) | WARN | 16.23 closure 2026-04-25 | Q/A reads `docs/audits/dev-mas-2026-05-11/` and confirms no new BLOCK items |

### Block-3: Live-system probe (snapshot at verdict time)

| Probe | Command | Pass criterion |
|-------|---------|----------------|
| Backend health | `curl -sS http://127.0.0.1:8000/api/health` | HTTP 200 + JSON shape `{status: "ok", mcp_servers: ...}` |
| OWASP headers | `curl -sI http://127.0.0.1:8000/api/health \| grep -iE 'x-content-type\|x-frame\|referrer-policy\|cache-control\|strict-transport'` | 5 of 5 present |
| Paper-trading status | `curl -sS http://127.0.0.1:8000/api/paper-trading/status` | 200 + `paper_trade=true` |
| Kill-switch state | `curl -sS http://127.0.0.1:8000/api/paper-trading/kill-switch` | 200 + `paused` field defined (need not be `false`) |
| Pytest baseline | `source .venv/bin/activate && python -m pytest backend/tests/ -q` | ≥177/178 PASS (16.23 baseline) |
| Drills | `python scripts/go_live_drills/{alpaca_shadow_drill,kill_switch_test,zero_orders_drill}.py` | all 3 PASS (16.19 baseline) |

### Block-4: Dimension-4 anti-rubber-stamp re-check on 16.1-16.14

For each of the 14 sub-steps, Q/A must verify (citing the archive `evaluator_critique.md` where present):

- File:line citation present in the prior verdict (Dimension 5 `missing-chain-of-thought [BLOCK]`)
- Behavioral test exists for any financial-logic claim (Dimension 4 `financial-logic-without-behavioral-test [BLOCK]`)
- Prior verdict not tautological (Dimension 4 `tautological-assertion [BLOCK]`)
- No `pass-on-all-criteria-no-evidence [BLOCK]` violation

### Block-5: Dimension-5 anti-evaluator-pattern on the aggregate verdict itself

The Q/A doing 16.15 must self-apply Dimension 5 heuristics to its own verdict before returning it:

- `sycophancy-under-rebuttal`: am I about to PASS because the prior verdict was PASS, with no fresh evidence audit?
- `second-opinion-shopping`: was there a code change in qa.md between prior verdict (2026-04-25) and this one? Answer: YES — phase-16.59 added 224 lines. This re-spawn is therefore documented cycle-2, not verdict-shopping.
- `3rd-conditional-not-escalated`: count prior CONDITIONALs on step 16.15 in `handoff/harness_log.md`. If ≥2 prior CONDITIONALs without intervening PASS/FAIL, MUST return FAIL.
- `verbosity-bias`: is the PASS reasoning length-driven or evidence-driven?
- `position-bias`: did I check the first criterion against evidence, or just default-PASS?

### Block-6: Hard-block vs advisory classification

Per research brief Dimension 1 + HRO pattern, Q/A must produce a classification table:

| 16.X sub-step | Block class | Status | Evidence (file:line or archive) |
|---------------|-------------|--------|----------------------------------|
| (14 rows) | HARD-BLOCK / ADVISORY | done / in-progress | path |

Aggregate verdict rule: if ANY HARD-BLOCK row is `in-progress` OR has open conditions, verdict = CONDITIONAL or FAIL (not PASS).

---

## What Main will do after Q/A returns

1. **If verdict = PASS:**
   - Write `handoff/current/experiment_results.md` capturing the Q/A's checks_run + violated_criteria (should be empty) + verdict.
   - Write `handoff/current/evaluator_critique.md` capturing the Q/A's full critique.
   - Append `handoff/harness_log.md` with the cycle entry (date, verdict, references).
   - **STOP. Do NOT flip 16.15 status to done.** Wait for Peder's explicit in-session acknowledgment per immutable criterion #4.

2. **If verdict = CONDITIONAL:**
   - Capture the violated_criteria + violation_details from Q/A.
   - Read each condition; decide if it's BLOCK-severity (forces NO-GO until cleared) or WARN (conditional-go acceptable to Peder).
   - Write the three handoff files; append harness_log with CONDITIONAL.
   - **STOP. Do NOT spawn a fresh Q/A for second opinion.** The cycle-2 flow per CLAUDE.md requires FIXING the flagged blockers first, THEN respawning with updated evidence files. No verdict-shopping.
   - Surface the CONDITIONAL to Peder for direction.

3. **If verdict = FAIL:**
   - Same as CONDITIONAL but with hard-block items requiring fix before any retry.
   - Three handoff files + harness_log append.
   - Surface to Peder. The 16.15 step stays `in-progress`.

In all three branches: the harness_log append happens AFTER Q/A returns and BEFORE any status flip. The flip itself is gated on Peder's in-session ACK.

---

## References

- `handoff/current/research_brief.md` (this cycle's brief; `gate_passed=true`)
- `handoff/archive/phase-16.16/` through `phase-16.23/` (prior re-verification critiques)
- `handoff/archive/phase-16.59/` (Q/A upgrade artifact set)
- `.claude/agents/qa.md:201-426` (new code-review heuristics; Dimensions 1-5)
- `.claude/masterplan.json:5178-5599` (phase-16 description + 16.1-16.23 + 16.28 notes)
- `.claude/masterplan.json:5438-5456` (16.15 step definition; immutable criteria; `depends_on_step: 16.59`)
- `handoff/harness_log.md:11862-11878` (16.15 original planning)
- `handoff/harness_log.md:12334-12382` (16.23 CONDITIONAL ruling, 4 conditions)
- `handoff/harness_log.md:12548-12581` (16.23 condition resolution state at time of log)
- `handoff/harness_log.md:18292-18391` (16.59 dependency wiring + SoD note)
- Cornell LII 15c3-5: https://www.law.cornell.edu/cfr/text/17/240.15c3-5
- ESMA Supervisory Briefing Feb 2026: https://www.esma.europa.eu/sites/default/files/2026-02/ESMA74-1505669079-10311_Supervisory_Briefing_on_Algorithmic_Trading_in_the_EU.pdf
- ASIC CP386: https://www.asic.gov.au/about-asic/news-centre/news-items/asic-moves-to-modernise-trading-system-rules-to-keep-pace-with-technology-and-ai/
- Knight Capital case study: https://www.henricodolfing.ch/en/case-study-4-the-440-million-software-error-at-knight-capital/

---

## Hard discipline (Main self-check before Q/A spawn)

- [x] Researcher invoked BEFORE this contract was written (research-gate compliance; `feedback_research_gate.md`)
- [x] Contract written BEFORE GENERATE / Q/A spawn (`feedback_contract_before_generate.md`)
- [x] Immutable success criteria copied verbatim from masterplan.json (criteria not amended)
- [x] qa.md upgrade live in this session (verified by `live_check_16.59.md`; this is the post-restart session)
- [x] Q/A will be a FRESH spawn (no SendMessage continuation; single-turn evaluator)
- [x] Log append planned BEFORE status flip (`feedback_log_last.md`)
- [x] Status flip gated on Peder ACK per immutable criterion #4

Self-evaluation is FORBIDDEN. The verdict comes from Q/A, not Main.
