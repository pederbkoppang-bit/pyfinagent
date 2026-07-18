# Operator Decision Sheet — phase-72 (RECOMMEND-ONLY; the operator flips, the session never does)

Started 2026-07-18 by step 72.1 (P1 token reconciliation). P3 lever rankings and the P4 regime policy are appended by steps 72.3/72.4. Every claim traces to `money_diagnosis_72.md`, `live_check_72.0.md`, or `research_brief_72.1.md`.

## ACT-NOW block (highest expected P&L per keystroke)

| # | Action (exact) | Why (evidence) | Risk / rollback |
|---|---|---|---|
| 1 | **Decide Anthropic direct-API credits**: top up the metered key at console.anthropic.com Plans & Billing, OR reply `ANTHROPIC DIRECT: ABANDON` and the remediation steps (72.0.1/72.0.2) reroute those legs | Credit-dead since **2026-05-17 03:55:44** (HTTP-400 "credit balance is too low", req_011Cb7JtX5fXgpryDPiYpSxo); every claude-* leg regresses to this dead API; meta-scorer failed every trading day since 05-22 | Top-up = metered spend resumes (~$0.2-2/day at recent volumes); abandon = Claude legs move to rail/Vertex via 72.0.1-2, no spend |
| 2 | **Append to `backend/.env`**: `PAPER_SYNTHESIS_INTEGRITY_ENABLED=true` and `PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED=true` | You approved this 2026-07-09 (operator_tokens.jsonl:1) but it was never written — the agent cannot write the agent-locked `.env`. These flags stop the 61.2 fabrication (synthesis failure → synthetic HOLD/0.0) from killing every BUY | Reviewed + approved in the 07-09 brief; rollback = delete the two lines + restart |
| 3 | **Restart the backend after #2** (`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`) | Running process pid 98681 started **07-08 23:24** — it predates the 07-09 approval; any `.env` edit is inert until reload | Standard restart; kill parent AND child workers per CLAUDE.md |
| 4 | **Run the grep** `! grep -nE 'SYNTHESIS_INTEGRITY|RISK_JUDGE|SWAP|SCALE_OUT|SESSION_BUDGET|PAPER_MARKETS|PAPER_TRADING|MODEL|ANTHROPIC|GEMINI' backend/.env` | Upgrades every "inferred/UNCONFIRMED" verdict below to confirmed; settles whether #2 was ever partially done | None (read-only) |

Caveat on #2/#3: **necessary, not sufficient** — they let BUYs survive rail hiccups but do not revive the credit-dead scoring legs. #1 (or the 72.0.1/72.0.2 rerouting steps) is what restores scoring.

## P1 — Token reconciliation (every `operator_tokens.jsonl` line + every owed/derived token; 15 rows)

Live-state basis: documentary + runtime inference only — `backend/.env` is agent-locked and the operator grep is **not yet provided** (requested; see ACT-NOW #4). Rows marked *(inferred)* would be upgraded by the grep.

| Token (verbatim) | Date | Gated flag(s) | Code default | Live state | Verdict |
|---|---|---|---|---|---|
| PROMOTE SYNTHESIS-INTEGRITY (66.2) | 2026-07-09 | `paper_synthesis_integrity_enabled` (settings.py:197) | False | DARK — never written; approval postdates last restart (double-blocked) | **NOT-APPLIED → ACT-NOW #2** |
| PROMOTE RJ-SHAPE (66.2) | 2026-07-09 | `paper_risk_judge_shape_fix_enabled` (settings.py:311) | False | DARK — same double-block (bundled on jsonl line 1) | **NOT-APPLIED → ACT-NOW #2** |
| PROMOTE POSITION-REC (66.2) | 2026-07-09 | `paper_position_recommendation_fix_enabled` (settings.py:201) | False | DARK — reviewer verdict HOLD; operator did not approve | NOT-APPLIED (by design; not owed) |
| 60.2 FLAG: ON | 2026-06-11 | `paper_swap_churn_fix_enabled` (settings.py:345) | False | LIVE=true — 06-11 keystroke, predates restart; corroborated by 70.3 test + 65.3 BQ 0-churn | APPLIED |
| 60.3 FLAG: ON | 2026-06-11 | `paper_data_integrity_enabled` (settings.py:45) | False | LIVE=true *(inferred — same keystroke batch; no in-window KR trade to prove runtime)* | APPLIED (keystroke-confirmed) |
| 57.1 FLAG: ON | 2026-06-11 | `paper_risk_judge_reject_binding` (settings.py:307) | False | LIVE=true *(inferred — live_check_61.1.md:14)* | APPLIED (keystroke-confirmed) |
| KS-PEAK-RESET:APPROVED | owed (69.1) | `kill_switch_peak_reset_enabled` (settings.py:38) | False | DARK by design — token never issued | NOT-APPLIED (owed, correctly dark; ranked in P3) |
| sign_safe_overlays flip | owed (69.3) | `sign_safe_overlays` (settings.py:36) | False | DARK — token never issued | NOT-APPLIED (owed, correctly dark; ranked in P3) |
| regime_net_liquidity flip | owed (69.3) | `regime_net_liquidity` (settings.py:37) | False | DARK — token never issued | NOT-APPLIED (owed, correctly dark; ranked in P3) |
| historical_macro un-freeze | owed (69.2/3) | none — operational posture, not a flag | n/a | Frozen (deliberate doctrine) | NOT-A-FLAG |
| KILL SWITCH: RESUME | n/a | none — reserved process action | n/a | Not owed — never paused since 06-11 manual resume (~1% trailing DD) | NOT-A-FLAG (not owed) |
| FABLE PERMANENT: AUTHORIZE | n/a | none — agent-file model pin | n/a | Correctly reverted to `model: opus` (phase-67.4) | NOT-A-FLAG |
| `paper_use_claude_code_route` (no discrete token) | n/a | settings.py:175 | False | Effectively ON at runtime (72.0: rail invoked post-restart) — failure is credits, not this flag | APPLIED (runtime-inferred; NOT a P1 lever) |
| AUTORESEARCH SPEND: RESUME | 2026-07-07 | none — `--preflight-only` in run_nightly.sh | n/a | Applied 07-07 (flag removed) | NOT-A-FLAG (applied) |
| SETUP TOKEN: ADOPTED | 2026-07-08 | none — `CLAUDE_CODE_OAUTH_TOKEN` in 4 launchd plists | n/a | Applied 07-08 (wired into backend/away-am/away-pm/watchdog) | NOT-A-FLAG (applied) |

## P3 — Earning-capacity lever ranking (step 72.3; researcher `wf_c781c347-3ac`, 15 evidence dossiers, full detail in `research_brief_72.3.md`)

**Standing rule (evidence-backed): flip ONE lever at a time, in this order, watching 3-5 cycles between flips** (single-variable discipline; arXiv 2607.06117: only 2 of 26 candidates survived incremental admission). Everything below is post-P0 capacity — nothing earns until the scoring rail is restored (ACT-NOW #1-3).

### Recommend-ON, in sequence

| Seq | Lever (.env line) | Expected impact (evidence) | Risk | Rollback |
|---|---|---|---|---|
| 1 | `KILL_SWITCH_PEAK_RESET_ENABLED=true` — requires you to record `KS-PEAK-RESET:APPROVED` first | $0 in normal operation; removes the permanent-lockout time-bomb (monotonic peak freezes book to 100% cash forever after a ≥10% pullback+flatten). Insurance, flip first while quiet | Guard-behavior change; concurrency nit noted in phase-69.1 critique | flag=false + restart (no-op reset, byte-identical) |
| 2 | `PAPER_SOFT_SECTOR_DIVERSITY_ENABLED=true` + `PAPER_SOFT_SECTOR_DIVERSITY_W=0.20` + `PAPER_MIN_K_SECTORS_ANALYZED=3` + `PAPER_UNKNOWN_SECTOR_CAP_EXEMPT=false` (min-k is the candidate-generation enabler; flip together as one unit) | **+0.20 ann Sharpe** at w=0.20 — the only large quantified alpha lever (70.2 replay: monotonic +0.176/+0.200/+0.234 at w=0.10/0.20/0.30, breadth +2 sectors, turnover-neutral; hard-neutral −0.117 WORSE; FAJ 2023 corroborates). Directly attacks the monosector top-5 funnel | Precondition: run the DSR≥0.95/PBO≤0.5 clearance on the existing replay grid (`scripts/ablation/sector_neutral_replay.py`, no optimizer, historical_macro untouched) — the replay proved Sharpe uplift but not the promotion-gate stats | flags=false/0.0/0 + restart (byte-identical) |
| 3 | `MOMENTUM_52WH_TILT_ENABLED=true` (k stays 0.5) | **+0.05 ann Sharpe**, turnover-neutral (52.1 paired replay; k=1.0 plateaus — keep 0.5); composes with #2 | Small live ranking change; DSR-deflated per settings note | flag=false + restart |
| 4 | `PAPER_ATOMIC_SWAP_ENABLED=true` + `PAPER_AVG_ENTRY_FX_FIX_ENABLED=true` (safety pair) | Protective: stops the net−1-position swap leak (design-70 finding #9) that degrades the **+$1,103** swap profit stream; correct non-US cost basis (stops/realized on KR adds). No new alpha | Low; swap-path only; US byte-identical for the FX fix | both=false + restart |
| 5 | `PAPER_CROSS_SECTOR_ROTATION_ENABLED=true` | Indirect: enables rotation into NEW sectors at max_positions — compounds #2's diversity alpha ("changeable fund") | HARD dependency: churn-fix ON (it is) + #4 first, never before | flag=false + restart (same-sector-only) |
| 6 | `PAPER_SESSION_BUDGET_RECONCILE_ENABLED=true` | Throughput: full candidate set analyzed per cycle instead of truncating at the hidden $1 ceiling (~2× LLM cost, ~$0.4/day vs $0.2 — needs your LLM-cost approval per CLAUDE.md) | Cost approval required; only matters post-P0 | flag=false + restart ($1 ceiling) |
| 7 | `SIGN_SAFE_OVERLAYS=true` and `REGIME_NET_LIQUIDITY=true` | Correctness/information: sign-safe fixes overlay INVERSION when composites go negative (bites in drawdowns only; 69.3 critique reproduced it); net-liquidity enriches regime classification (free FRED, 24h cache). Neither has $ magnitude | Ranking-behavior change (sign-safe); live FRED dependency (net-liq) | each=false + restart (byte-identical) |

### Recommend-HOLD (do not flip — reasons are evidence, not caution)

| Lever | Why HOLD |
|---|---|
| `paper_scale_out_enabled` (:34) | ZERO internal backtest rows; conflicting external evidence; our own exits are trend-like (trail captured avg **+17.82%** on 14 trips, +$1,941) — a 2R partial would cap exactly those winners. Needs a regime backtest first (blocked by the frozen-macro doctrine) |
| `paper_position_recommendation_fix_enabled` (:201) | UNSAFE until ACT-NOW #2 is live: with synthesis-integrity OFF, a rail-failure synthetic HOLD triggers a wrongful downgrade-SELL of a healthy position (settings.py:203 guard). Revisit AFTER #2 + 3-5 healthy cycles |
| `meta_scorer_enabled` (:402) | Credit-dead by construction until restoration step 72.0.1 reroutes it (direct-API bypass); flipping now erases ranking with flat conviction=10 |
| `paper_price_tolerance_pct` (:560) tuning | No rejection ledger exists yet to know if the 5% gate costs trades; tune only after the 70.4 skip-reason ledger measures it |
| `paper_learn_loop_enabled` (:33) | Writer is crash-dead regardless of the flag (tz-aware minus naive TypeError, outcome_tracker.py:50, phase-69 register #30) — flipping does nothing until that code fix ships; queue with the executor batch |
| Dark alpha-overlay library (:362-521, ~20 flags) | Literature priors only, zero validation on OUR data (except 52wh, ranked #3). Batch-enabling risks the 2-of-26 incremental-admission failure mode; each needs its own DSR/PBO gate (blocked by frozen macro) |

Already applied (not ranked): `paper_swap_churn_fix_enabled`, `paper_data_integrity_enabled`, `paper_risk_judge_reject_binding` (06-11 batch, per P1 reconciliation).

**Evidence-hygiene closure (from 72.2)**: the $137.32 realized-P&L delta is RESOLVED — whole-table sum +$3,057.36 (30 trips) vs since-05-15 sum +$3,194.68 (29 trips) differ by exactly one pre-05-15 trip at −$137.32 (BQ, 2026-07-18). Both figures were correct on their own windows; no ledger inconsistency.

## Recurrence prevention (recommended, queued as masterplan step 72.1.1)

The approval-to-deployment gap is structural: `operator_tokens.py:52-61` KNOWN_TOKEN_ENV_MAP registers only `AWAY DRILL` (no auto-apply path for flag promotions), and `sentinel.sh:102-126` reconciles **one-directionally** — it catches an `.env` flag that is ON without a token, but is blind to a recorded approval whose `.env` line is absent (exactly the 07-09 case). Recommended: a report-only reverse leg that diffs recorded approvals against live state and WARNs on approved-but-unapplied (GitOps bidirectional-reconciliation consensus; sources in research_brief_72.1.md). No auto-writes — `.env` stays operator-only.
