# live_check_59.3 — Stress-test doctrine run: evidence

**Step:** 59.3. **Date:** 2026-06-11. **Required shape:** harness-free output excerpt + the 3+-dimension comparison table + per-component verdicts.

## A. The harness-free run (saved verbatim)

- Output: `handoff/current/59.3-harness-free-output.md` (282 lines; written by the bare agent as its single permitted write).
- Telemetry: 1 general-purpose Fable 5 session, **310,037 tokens, 126 tool uses, 35.4 minutes**, zero retries/prompt-iteration.
- Blinding: pinned worktree `70a8242b` (verified fix-free: 0 finding-ID comments, dead button present at governance.py:168, no binding flag, no fix tests), handoff/ quarantined, 2 input docs restored; worktree torn down post-scoring.
- Output excerpt (finding index): AW-1 gateway approve-path dead end-to-end (P0) ... AW-5 churn root cause: sentinel conviction 0.0 for holdings without same-cycle analysis ("Treat as worst") + last_analysis_date=now + 3-day reeval = every fresh buy is swap-out bait (P0) ... AW-8 31% flip rate ... AW-10 cost cap logged-not-enforced.

## B. Main's fabrication spot-test (11/11 verified before scoring)

gateway.err.log:5792-5799 (8 error matches) | openclaw.json C0ANTGNNK8D binding | buildMissingApiKeyFailureText in the bundled runtime | backend.log: 416 "Full orchestrator failed" / 166 "lite_mode=False" / 324 claude exit-1 / 113 "was not found" + verbatim "RAG Agent: fail-open for DELL (404 ... gemini-2.0-flash" | portfolio_manager.py:443-449 "Treat as worst" sentinel verbatim + :495-498 delta math | autonomous_loop.py:1705 hardcoded $0.01 ledger | BQ: session_cost_usd CUMULATIVE (0.02->0.08 ascending on 06-01).

## C. Dimension results (rubrics pre-registered in the research brief)

| Dim | Harness chain (55.2) | Bare Fable 5 | Notes |
|---|---|---|---|
| D1 accuracy /10 | (the anchor source) | **10/10, 0 confident-wrong** | + 2 anchors REVISED in the bare run's favor (GT1 gateway provenance; GT10 cumulative-cost over-count) + GT4 mechanism upgrade |
| D2 premise catches /3 | 3/3 (its signature strength) | **3/3** | P2 reframed DEEPER: full->lite silent fallback (F-H re-diagnosed) |
| D3 evidence rigor | QA-verified | **11/11 spot-test, 0 fabrications, 0 F-ID echoes** | |
| D4 coverage /9 | 9/9 + 4 additive findings | **8.5/9 + 5 additive findings** (gateway, 404/CIK census, churn engine, swap-scale bug, cost-cap + FRED-key hygiene) | tool reruns verified-not-rerun (honest partial) |
| D5 calibration | 4 disclosures | **0 overclaims, 7 honest-bound passages** | |
| D6 overhead | 3 sessions / 7 artifacts / 31-70 min | 1 session / 1 artifact / 35.4 min / 310K tok | token totals comparable; harness overhead = orchestration + Main-attention |

## D. Per-component verdicts (operator-gated; attributed-not-isolated)

Researcher gate: rule output PRUNE-candidate -> **recommendation MODIFY (adaptive tiering; the external-literature half stays mandatory — the bare run did zero external research)**. Contract: rule met at 8.5/9 -> **MODIFY (keep for code/money/multi-cycle steps)**. Separate Q/A: **MODIFY-at-most (risk-tiered; full hostile QA stays on code/money steps)** — verification != generation. Handoff files: **KEEP** (optional artifact slim). Turn caps: **MODIFY (already begun in 59.1)** — 126 coherent tool uses observed; treat cap-hits as raise signals. NOTHING changed in the harness this step.

## E. New bug candidates surfaced (feed the normal masterplan flow as findings)

1. **Sentinel-conviction churn engine** (AW-5; likely the away week's primary money bleed — exits with NO reasoning) — P0 candidate.
2. **Retired `gemini-2.0-flash` 404 kills the full pipeline** + KR tickers structurally excluded (SEC CIK) (AW-4) — P0 candidate; also re-diagnoses F-H (not a checkbox desync).
3. Swap-threshold scale bug (comment assumes [0,1], lite emits 1-10).
4. Cost-budget breach logged-not-enforced (AW-10).
5. FRED API key in plaintext backend.log URLs (hygiene).
6. GT10 revision: llm_call_log session_cost_usd is cumulative — burn reports must use MAX-per-session, not SUM.

## F. Constraint compliance

$0 (Max-session subagent; bounded BQ); repo unmodified by the bare run (single output doc); one run, no retries; leakage residuals disclosed (backend.log reachability — low-impact, no fix-era echoes); worktree torn down post-scoring.
