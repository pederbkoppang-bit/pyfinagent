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

## Recurrence prevention (recommended, queued as masterplan step 72.1.1)

The approval-to-deployment gap is structural: `operator_tokens.py:52-61` KNOWN_TOKEN_ENV_MAP registers only `AWAY DRILL` (no auto-apply path for flag promotions), and `sentinel.sh:102-126` reconciles **one-directionally** — it catches an `.env` flag that is ON without a token, but is blind to a recorded approval whose `.env` line is absent (exactly the 07-09 case). Recommended: a report-only reverse leg that diffs recorded approvals against live state and WARNs on approved-but-unapplied (GitOps bidirectional-reconciliation consensus; sources in research_brief_72.1.md). No auto-writes — `.env` stays operator-only.
