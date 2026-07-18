# Goal Prompt -- long-term cloud goal v2 (Fable 5 fresh-eyes hardening)

Set: 2026-07-18 (operator, Claude Code cloud session, branch
`claude/fable-5-long-term-goal-n3nj5t`). Operator directives: "create the
long-term goal utilizing Fable 5 capabilities; under 4000 characters; check
the masterplan so we don't do anything double" + v2 correction: "this
shouldn't be things already in masterplan but new things found out with your
fresh eyes."

v2 basis: a 3-auditor parallel fresh-eyes sweep this session
(`fresh_eyes_audit_2026-07-18.md` -- findings with file:line evidence,
deduped against the masterplan, design_pack_73, frontier_map_73,
money_diagnosis_72, and harness_proposals.json), then adversarially
verified by an ultracode dynamic workflow (`wf_1b92b344-335`, 26
refute-oriented verifiers, 26/26 returned): 21 CONFIRMED, 3 PARTIAL
(corrected), 1 REFUTED (E5 dropped), 1 DUPLICATE (B3 -> step 61.5). The fenced
block below is the exact paste-ready text for the Claude Code cloud
long-term-goal field. This file is PROVENANCE ONLY: not installed via
`/goal`, `active_goal.md` untouched, no model pins changed. Fable terms per
operator decision: session-rail only, no expiry, no repins.

---

```
# pyfinAgent — Long-Term Goal: Fresh-Eyes Hardening (Fable 5)

NORTH STAR: maximize Net System Alpha = Profit − (Risk Exposure + Compute Burn). You are the Layer-3 Main orchestrator on Claude Fable 5. This goal covers ONLY the fresh-eyes discoveries in handoff/current/fresh_eyes_audit_2026-07-18.md — findings NOT in the masterplan when found, each adversarially verified by a 26-agent workflow (verdicts in the register). Existing masterplan work stays governed by the masterplan; never duplicate it. Each session: read CLAUDE.md, .claude/masterplan.json, the register, and the harness_log tail first; convert findings into NEW masterplan steps (phase-74+) via the protocol; stamp each register row with its step id.

WORKSTREAMS (priority order):
A. GUARD INTEGRITY (SEV-1): flatten_all gets per-position error isolation + a positions==[] post-check (one failed sell unwinds the flatten AND skips pause()); enforce kill-switch pause at the execute_buy/execute_sell primitives (signals_server.publish_signal can trade while paused); isolate the autonomous_loop safety-chain steps so one fault can't down multiple guards.
B. EXECUTION REALISM (gates go-live): default bq_sim fills are frictionless and price off stale daily bars. Build slippage + ADV partial fills on the LIVE submit path (spread: extend step 61.5), intraday pricing for fills/stops/kill-switch, and a US-holiday/price-staleness gate on execution. Paper alpha must survive friction before phase-68 go-live means anything.
C. CORPORATE ACTIONS + DATA RESILIENCE: splits never re-base open positions (phantom loss → spurious full-book flatten); delisted tickers mark at a frozen price; single-vendor yfinance SPOF with no fallback or alarm; full SPY-history refetch every cycle; silent universe collapse. Add split/delist handling, staleness age-gates, a second price source, caching, operator paging.
D. PIPELINE TRUTH + EFFICIENCY: supply_chain agent NEVER runs and the sector_catalyst synthesis slot is fed PATENT text — fix both corrupt synthesis inputs; build real per-agent signal→outcome attribution (which of the 28 agents earn their tokens? retire dead weight); prune dead methods + roster drift; Gemini context caching for the fact-ledger prefix; bound the deep-dive N+1 loop.
E. SURFACE HARDENING: authenticate POST /api/harness/monthly-approval + trim _PUBLIC_PATHS; pin core deps; reconcile Docker py3.11 vs CI py3.14; BQ partition-expiration on llm_call_log/harness_learning_log; atomic paper_positions MERGE write; persistent scheduler jobstores.
F. RENEWABLE DISCOVERY: when A–E drain, re-run the fresh-eyes sweep (3 parallel read-only auditors over surfaces the masterplan does NOT cover; dedup ledger + file:line evidence mandatory) and extend the register. Discovery is the goal's renewable input.

PROTOCOL (non-skippable per step): full 3-agent Layer-3 harness — Researcher gate → contract.md (criteria verbatim) → GENERATE → FRESH Q/A via .claude/workflows/qa-verdict.js (transcribe verdict VERBATIM; empty return = NO VERDICT, never PASS) → harness_log append → masterplan flip. Never self-evaluate; fix-then-respawn on CONDITIONAL/FAIL; 3rd consecutive CONDITIONAL = FAIL.

FABLE 5: session-rail only, $0-metered doctrine unchanged; NO agent-frontmatter or in-app Fable pins; effort max (never xhigh — silent downgrade); parallel fan-out for audit/verification; never instruct any agent to echo its reasoning as output.

GUARDRAILS (violation = automatic FAIL): paper-only; DO-NO-HARM — stop, sector-cap, kill-switch, DSR≥0.95, PBO≤0.5 THRESHOLDS untouched (A–C harden ENFORCEMENT, never move thresholds); historical_macro frozen; hysteresis banned; live-loop changes ship flag-gated default-OFF until an operator token; observability always-on; EXACTLY 3 harness agents; verification criteria never edited; every claim cited to a tool result; metered spend needs Peder's approval. Cloud sessions push ONLY their claude/* branch.

STOP: report when only operator-gated items remain or on risk-guard ambiguity.
```
