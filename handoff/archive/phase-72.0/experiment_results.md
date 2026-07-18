# Experiment Results — phase-72.0: P0 scoring-rail restoration audit

Date: 2026-07-18. Session: Fable 5 + ultracode, AUDIT + RESEARCH ONLY (no product code, no .env, no flags, no optimizer, $0 metered — all subagent work on the Max rail via Workflow structured-output launches). Predecessor 63.3 artifacts preserved at `handoff/archive/phase-63.3-parked/`.

## What was built

1. **Research gate** (Workflow `wf_a3f9906e-095`, researcher role, opus/max, tier=complex): gate_passed=true, 9 external sources read in full, 21 URLs, recency scan, 7 internal files. Brief: `handoff/current/research_brief.md`. Identified the two-surface root-cause frame (Surface A pipeline rail, Surface B meta-scorer direct-API bypass) + 8 restoration seams.
2. **Forensics GENERATE** (Workflow `wf_0542bf62-ffb`, 3 read-only Explore auditors + adversarial verifier at effort max; 4/4 returned): log-onset timeline across all rotations, BQ `llm_call_log` daily provider/error series (bounded SELECTs), away-window disentangle, live host inspection (`claude auth status`, `launchctl print`, `ps`). Verifier independently re-ran the load-bearing queries/greps.
3. **`handoff/current/money_diagnosis_72.md`** — P0 section complete: verified mechanism, onset + attribution (ROOT 2026-05-17 credit-400; Surface-B 2026-05-22; observable freeze 06-11/15; markers = instrumentation lag), away-vs-defect per window, corrected restoration stack-rank. The contract's Surface-A hypothesis (route flag/OAuth) is recorded as PARTIALLY REFUTED — OAuth alive, route effectively ON; the real Surface-A failure is rail-tagged calls regressing to the credit-dead direct API + 120s subprocess timeouts tripping the breaker.
4. **`handoff/current/live_check_72.0.md`** — verbatim onset/provider evidence (log lines with request_ids, BQ SQL + key rows, live host state); explicit note that the operator `.env` grep was NOT provided by step close (criteria escape hatch).
5. **Masterplan restoration steps appended** (pending, executor-tagged, immutable live_checks requiring a live cycle log line showing non-degraded scoring): `72.0.1` meta-scorer rail decoupling [sonnet-4.6/high, flag-gated dark], `72.0.2` standard-tier fail-forward on rail-dead [opus-4.8/xhigh, flag-gated dark], `72.0.3` decision-seam observability [sonnet-4.6/high, always-on log], `72.0.4` degraded-alert paging verification [sonnet-4.6/high]. Verification commands smoke-tested exit 0.
6. Operator-side levers routed to the decision sheet (NOT flipped): Anthropic credit decision (dead since 05-17), write approved synthesis-integrity + RJ-shape flags to `.env` + restart, provide the `.env` grep.

## File list

- `handoff/current/contract.md` (PLAN, written after research gate, before GENERATE)
- `handoff/current/research_brief.md` (researcher, write-first)
- `handoff/current/money_diagnosis_72.md` (P0 complete; P1-P4 placeholders for later steps)
- `handoff/current/live_check_72.0.md`
- `.claude/masterplan.json` (phase-72 installed earlier @403f376c; steps 72.0.1-72.0.4 appended this step; 72.0 in-progress)
- `handoff/archive/phase-63.3-parked/` (snapshot of the parked 63.3 rolling files before overwrite)

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/money_diagnosis_72.md && grep -q "P0" handoff/current/money_diagnosis_72.md && grep -Eqi "onset|since" handoff/current/money_diagnosis_72.md && grep -Eqi "restoration" handoff/current/money_diagnosis_72.md'
VERIFICATION COMMAND EXIT: 0 (PASS)
```

## Artifact shape / headline findings

- ROOT onset **2026-05-17 03:55:44** (HTTP-400 "credit balance too low", req_011Cb7JtX5fXgpryDPiYpSxo; genuine anthropic successes cease after this date — later ok-rows are input=1000/output=50 smoke fixtures).
- Meta-scorer LLM leg failed **every trading day 05-22 → 07-17**, zero llm_call_log rows (rail + telemetry bypass), still 400ing 07-17 20:01.
- Away window = **DEFECT not posture** (rail 4 exempts the live pipeline; cc_rail credential death 06-15 + `alerting.py` missing → 34 silent failures).
- Post-return persistence = approved flags never written to agent-locked `.env` + meta-scorer left degraded + direct-API re-blackouts; the 07-09 AMD+MU BUYs were a one-off Gemini-lite catch.
- Gemini leg: **0 failures all window** — the healthy fail-forward target for R2.
- Scope honesty: no product code touched, no .env touched, no flags flipped; the only writes are handoff/**, masterplan step additions, and this session's harness artifacts.
