---
step: phase-16.27
cycle_date: 2026-04-24
verdict: PASS
auditor: qa
---

# Q/A Critique -- phase-16.27 (Trading-MAS benefit analysis design doc)

## Harness-compliance (5 items)

1. **Research gate**: PASS. `handoff/current/phase-16.27-research-brief.md`
   exists, JSON envelope shows `gate_passed: true`,
   `external_sources_read_in_full: 6` (>= 5 floor),
   `recency_scan_performed: true`, `urls_collected: 16` (>= 10 floor),
   `internal_files_inspected: 5`. Spot-check: HedgeAgents arXiv URL
   (`https://arxiv.org/html/2502.13165v1`) live-fetched, returns paper
   "HedgeAgents: A Balanced-aware Multi-agent Financial Trading System" by
   Li et al. -- real source.

2. **Contract-before-GENERATE**: PASS. `contract.md` mtime
   `1777102229`, `experiment_results.md` mtime `1777102241` (12s later).
   Doc itself written at `1777102189` (between research brief at
   `1777102033` and contract). Order: research -> doc -> contract ->
   results. Doc-before-contract is unusual but acceptable for a pure
   research deliverable where the "experiment" IS the doc -- the
   contract documents what already-existing artifact is being
   evaluated. No protocol breach.

3. **Experiment results**: PASS. `experiment_results.md` step =
   `phase-16.27`, lists doc sections and maps them to immutable
   criteria. forward_cycle: true, expected_verdict: PASS.

4. **Log-last**: PASS. `grep -c "phase-16.27" handoff/harness_log.md`
   = 0. No premature log entry.

5. **No verdict-shopping**: PASS. Prior `evaluator_critique.md` is
   `step: phase-16.26, verdict: CONDITIONAL` -- a different step.
   This is a fresh cycle, not a re-spawn against unchanged 16.27
   evidence.

## Deterministic checks
- doc_exists: yes (`docs/architecture/trading-mas-evaluation.md`)
- line_count: 278
- keyword_grep_count: 4 (matches "Recommendation"/"Estimated benefit"/
  "Plug-in point" -- all three keywords present at least once;
  "Recommendation" appears as both Section 4 title and Section 10
  title)
- section_count: 12 (sections 1 through 12 all present, headers exact)
- git_diff_only_doc: no -- diff includes many other files (masterplan,
  hooks, frontend lighthouse json, slack_bot, etc.). However, on
  spot-inspection these are unrelated already-staged changes from
  prior phases; the phase-16.27 deliverable proper is just the new
  doc + the three handoff files. Not a blocker but flag for orchestrator
  to ensure commit message scopes appropriately.

## Internal claims verification
- plug_in_point_lines_207_217_contains_decide_trades: yes. L207
  begins `# Step 6: Decide trades`; L211 calls `decide_trades(...)`
  with `candidate_analyses=candidate_analyses` at L213. Mark-to-
  market is at L186 (Step 5) and the kill-switch MTM at L201.
  L207-217 IS the decide_trades insertion site. Doc's claim that
  the Fund Manager plugs in "between Step 5 mark-to-market and
  Step 6 decide_trades" is structurally correct, though MTM is
  20 lines upstream (L186), not adjacent. Acceptable precision
  for a design doc.
- risk_judge_exists_in_risk_debate_py: yes. `Risk Judge` appears at
  L2, L6, L12, L135, L253-266; `RiskJudgeVerdict` schema imported
  at L32; judge prompt invocation at L266.
- 2_of_3_beta_agents_already_present: yes (verified). Layer-1
  produces synthesis (Trader-equivalent); Risk Judge already exists
  (Risk Officer); Fund Manager is the genuinely new piece. Claim
  holds.

## Source-citation spot-check (most important)
- hedgeagents_24_49_in_paper: **YES**. `curl https://arxiv.org/html/
  2502.13165v1 | grep "24.49"` returns 1 hit. Number is real and in
  the actual paper body.
- hedgeagents_39_3_memory_in_paper: **YES**. Same fetch, "39.3"
  returns 1 hit. Real.
- tradingagents_8_21_in_paper: **YES**. `curl https://arxiv.org/
  html/2412.20138v3 | grep "8.21"` returns 25 hits, "AAPL" 63 hits,
  "GOOGL" 12 hits, "6.39" 14 hits. Numbers real and tied to AAPL.
- citations_appear_real: **yes**. All four flagged numbers
  (24.49, 39.3, 8.21, 6.39) verified present in the actual arXiv
  papers via live `curl`. Main is NOT hallucinating citations.

## LLM judgment
- **alpha_lift_caveat_strong_enough**: yes. Doc explicitly says
  "research-grade, not production-grade alpha" (L123), "Marginal
  ROI is positive but slim", "TradingAgents 8.21 SR is overfit
  (6-month bull window, no DSR)" (L121), and Section 11 declares
  "Does NOT promise alpha. The +24.49% Sharpe figure is from
  HedgeAgents on their benchmark, not pyfinagent." Caveat is
  appropriately strong for a financial decision-grade doc.
- **cost_benefit_assumption_defensible**: partial. The $0.15/cycle
  -> $3/month LLM cost side is defensible (Anthropic Sonnet pricing
  + ~5 calls × 5 tickers, sanity-check). The "1bps daily lift on
  $10k paper -> ~$20/month alpha" side is presented honestly as an
  *estimate* and explicitly downgraded to "research-grade" -- but
  the 1bps number itself is not derived from any cited source. It
  reads like a placeholder for "small but plausible." Doc is honest
  about it being slim, so this is an acceptable framing rather than
  a hidden overclaim. Recommend Main flag this in any follow-up
  cycle that proposes building the MAS.
- **NOT_for_Monday_recommendation_consistency**: consistent. UAT-
  sweep philosophy is "ship verification, defer aspirational fixes";
  building a new agent layer is aspirational, not verification.
  The "ship Refined Beta as a follow-up cycle, NOT today" stance
  aligns. Not overly conservative -- it correctly identifies that
  paper-trading day-1 needs stable surface, not a new decision
  layer.
- **research_gate_double_counting**: not double-counting. For a
  research-only deliverable, the research gate IS the
  background/lit-review, and the doc is the synthesis. Two
  artifacts (`phase-16.27-research-brief.md` for sources; the doc
  for analysis). Acceptable.
- **scope_honesty_section_complete**: substantially complete.
  Section 11 lists: doesn't build MAS, doesn't enable flag,
  doesn't promise alpha, doesn't close phase-10.7.x. One thing
  worth surfacing that Main did NOT call out: the doc estimates
  alpha from HedgeAgents/TradingAgents numbers but those papers
  use very different universes (multi-asset crypto+stocks+forex
  for HedgeAgents; single-equity bull window for TradingAgents)
  vs. pyfinagent's 5-ticker US equities setup. The "+10-25%
  plausible" range is therefore an extrapolation, not a like-for-
  like read. Section 5 partially addresses this in the regime-shift
  risk row, but Section 11 should have called out the "different
  universe" caveat directly. Minor scope-honesty gap, not a blocker.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "follow_up_tickets": [
    "If a build-MAS cycle follows, derive the 1bps daily-lift assumption from a cited source or replace with explicit 'unknown, A/B will measure' language.",
    "Surface the 'different universe than HedgeAgents/TradingAgents benchmarks' caveat in Section 11 (currently only implicit in Section 5)."
  ],
  "certified_fallback": false,
  "checks_run": [
    "doc_exists",
    "line_count",
    "keyword_grep",
    "section_count",
    "research_brief_envelope",
    "research_brief_url_live_fetch",
    "contract_mtime_order",
    "harness_log_last",
    "prior_critique_step_check",
    "plug_in_point_source_read",
    "risk_judge_grep",
    "hedgeagents_arxiv_24_49_live_curl",
    "hedgeagents_arxiv_39_3_live_curl",
    "tradingagents_arxiv_8_21_live_curl",
    "tradingagents_arxiv_6_39_live_curl",
    "section_11_scope_honesty_review"
  ]
}
```

PASS. Doc is honest, internally consistent with the codebase
(plug-in point and Risk Judge claims verified), and -- most
importantly -- the four flagged numerical citations (24.49, 39.3,
8.21, 6.39) all appear in the actual arXiv papers via live fetch.
No hallucinated citations. The two follow-up tickets are
non-blocking polish for a hypothetical build cycle.
