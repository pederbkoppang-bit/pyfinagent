# Contract — phase-72.0: P0 scoring-rail restoration audit

**Step id:** 72.0 (phase-72 "Make the engine earn again", installed 2026-07-18, commit 403f376c)
**Session role:** Fable 5 + ultracode, AUDIT + RESEARCH ONLY. No product-code edits, no .env changes, no flag flips, no optimizer runs, $0 metered. Note: predecessor step 63.3 is PARKED on an operator Slack-post gate; its rolling handoff files were snapshotted to `handoff/archive/phase-63.3-parked/` before this contract replaced them.

## Research-gate summary (gate_passed: true)

Researcher launched via structured-output Workflow (phase-71.1 first-class path), opus/max, tier=complex. Envelope: 9 external sources read in full, 12 snippet-only, 21 URLs, recency scan performed, 7 internal files inspected. Brief: `handoff/current/research_brief.md`.

Load-bearing findings:
1. **Two independent failure surfaces, not one.** Surface A (pipeline rail): standard tier is `claude-sonnet-4-6` (settings.py:30) and reaches the flat-fee Max rail ONLY when `paper_use_claude_code_route=True` (settings.py:175, **default False**) AND the host OAuth session is alive; otherwise every claude-* call regresses silently to the credit-exhausted direct API. Surface B (meta-scorer): `meta_scorer.py:220-225` constructs ClaudeClient with `anthropic_api_key` directly — bypassing the rail and its env-scrub — so the conviction overlay is credit-dead even when the rail is healthy, collapsing to flat conviction=10 (this IS the open "credit-exhaustion class" follow-up, cycle_block_summary.md:27).
2. **402 billing errors are non-retryable** (SDK auto-retries only connection/429/5xx); at $0 balance Anthropic sometimes returns 429 instead of a billing error, so status codes alone cannot disambiguate credit-death from rate-limiting (platform.claude.com/docs/en/api/errors).
3. **Unattended OAuth expiry** is the documented pyfinagent failure mode for the rail (`claude_code_health_probe` runs `claude auth status`, claude_code_client.py:380-425); the away-week outage post-mortem attributes the stall to exactly this.
4. **External consensus** (Fowler circuit-breaker; 2026 LLM-failover practitioner sources; arXiv 2606.14589 on silent failures living "in seams between declared state and runtime state"): fail-fast on dead dependency, degrade to a quality floor, never fabricate a score; pyfinagent fails CLOSED (lite→HOLD→$0 earned) with no fail-forward provider. Live-host evidence (not code reads) is required precisely because committed defaults vs runtime values is such a seam.
5. **Restoration seams enumerated** (8, file:line) — see research_brief.md; the .env-side levers go to the operator decision sheet, the code-side levers become executor-tagged masterplan steps.

## Hypothesis

The July stall persists because BOTH surfaces are down: the cc_rail is unavailable to the launchd backend (probe-fail: expired OAuth and/or `PAPER_USE_CLAUDE_CODE_ROUTE` unset/false in live env) AND the meta-scorer + direct-API calls hit an exhausted Anthropic credit balance. The away-ops $0 posture (2026-06-12..07-06) explains conservative behavior in-window but NOT the post-return persistence (07-06..07-18) — that requires broken runtime state (route flag / OAuth / credits) that survived the operator's return. Onset is pinnable from backend.log rotations + BQ `llm_call_log`.

## Immutable success criteria (verbatim from .claude/masterplan.json step 72.0)

- "P0 section of money_diagnosis_72.md pins the degraded-scoring onset date and the failing rail/key/tier with verbatim log or BQ llm_call_log evidence, not inference"
- "Explicitly resolves away-ops-posture-leftover vs live-defect for the July stall, or states exactly what evidence is missing and how to get it"
- "Restoration steps installed in masterplan as pending, executor-tagged, with immutable live_checks requiring a cycle log line showing non-degraded scoring; this session edited no product code and no .env"

verification.command: `bash -c 'test -f handoff/current/money_diagnosis_72.md && grep -q "P0" handoff/current/money_diagnosis_72.md && grep -Eqi "onset|since" handoff/current/money_diagnosis_72.md && grep -Eqi "restoration" handoff/current/money_diagnosis_72.md'`

verification.live_check: "handoff/current/live_check_72.0.md with verbatim backend.log/BQ llm_call_log lines pinning the degraded-scoring onset + failing provider/key, plus the operator .env grep output (or an explicit note it was not provided)"

## Plan

1. GENERATE — ultracode forensics Workflow (read-only Explore agents + adversarial verifier): (a) log-onset timeline across backend.log rotations (degraded-guard / no-LLM-fallback / probe-fail / breaker events by date; recovery windows); live host state via `claude auth status` + `launchctl print` env inspection of the backend service (runtime seam evidence without touching .env); (b) BQ `pyfinagent_data.llm_call_log` daily provider/model/status/error-class series since 2026-05-01 — pin onset, attribute failing provider/key, separate Surface A vs B callers; (c) away-ops disentangle — degraded evidence before 06-12, inside 06-12..07-06, after 07-06; (d) adversarial verify of onset + attribution.
2. Live .env confirmation: operator grep requested (pending); if not provided by step close, record that explicitly per the criteria's escape hatch, with `launchctl print` runtime env as best-available substitute.
3. Write `handoff/current/money_diagnosis_72.md` (P0 section) + `handoff/current/live_check_72.0.md` (verbatim evidence).
4. Append restoration masterplan steps (pending, executor-tagged, immutable live_checks); route .env-side levers to `operator_decision_sheet_72.md` (recommend-only).
5. EVALUATE — qa-verdict Workflow (`.claude/workflows/qa-verdict.js`), verdict transcribed verbatim into `evaluator_critique.md`.
6. LOG — append Cycle entry to `handoff/harness_log.md`, then flip 72.0 → done.

## References

- `handoff/current/research_brief.md` (researcher envelope + per-source notes)
- `handoff/current/money_recon_2026-07-18.md` (adversarially verified baseline)
- https://platform.claude.com/docs/en/api/errors ; https://support.claude.com/en/articles/11145838 ; https://martinfowler.com/bliki/CircuitBreaker.html ; https://www.truefoundry.com/blog/llm-failover-load-balancing-provider-outages ; https://futureagi.com/blog/what-is-llm-fallback-strategy-2026/ ; https://arxiv.org/html/2606.14589 ; https://zed.dev/blog/anthropic-subscription-changes
- Anthropic harness-design + multi-agent-research (protocol basis, per CLAUDE.md)
