# Contract -- 4000.1: Research gate + measured baselines for the Max-rail E2E smoke

Step id: 4000.1 (phase-4000, P1). Written 2026-08-06 by Main, AFTER the research
gate returned gate_passed=true and BEFORE any step artifact is generated.
Suffixed filename (contract_4000.1.md) because another session's rolling
contract.md is in flight on phase-82 (precedent: contract_82.46.md).

## Research-gate summary (gate PASSED)

Researcher ran on the Workflow rail (run wf_5d34fa5e-635, agentType researcher,
opus/max). Envelope: tier=moderate, external_sources_read_in_full=6,
snippet_only=23, urls_collected=29, recency_scan=true, internal_files=17,
gate_passed=true. Brief (write-first, incremental): handoff/current/research_brief_4000.1.md.

Load-bearing findings that reshape this step:
- F0: Anthropic announced then PAUSED (June 15) moving `claude -p` off flat-fee
  onto capped monthly credits (Max 5x $100 / 20x $200). Flat-fee is true TODAY on
  a paused policy. E6 definition gains a second figure (%-of-$100 credit).
- F1: `--bare` will become the `-p` default in a future CLI release and does NOT
  use subscription login; the rail omits --bare and scrubs ANTHROPIC_API_KEY, so
  that flip would leave the rail credential-less. Baseline must record
  `claude --version` so a future break bisects to a CLI upgrade.
- P1/P2: modelUsage keys can share one canonicalModel; resolve_rail_model's dict
  build collapses them last-wins (claude_code_client.py:274-291) -- cost totals
  from the collapsed map UNDER-COUNT. E6 must sum the RAW map; free assertion:
  total_cost_usd == sum(costUSD over ALL modelUsage entries).
- P3: without --bare, every rail call loads CLAUDE.md/hooks/MCP from the
  inherited cwd (measured: 9-token prompt -> 45,580 cache-creation tokens,
  $0.1537 would-be). E6 is overhead-dominated; baseline records the backend cwd.
- (a) THREE llm_call_log writer shapes; provider alone cannot separate rails;
  the executable rule is the complement of spend.py:228-230 (see brief).
- (c) PUT /api/settings WRITES backend/.env + cache_clear, so a flip SURVIVES
  restart; GET is 300s cached (staleness caveat on "live value" captures).
- Entrypoint: POST /api/analysis/ (analysis.py:350) single-ticker, ASYNC (poll
  GET /api/analysis/{id}); rail-calls-per-analysis is NOT derivable from code --
  4000.2's counter must abort mid-analysis; measure the count in the dry pass.

## Hypothesis

The CC rail is ALREADY the live route (Main recon: flag=True on the running
backend; 615 rail rows vs ZERO anthropic-direct rows in 7d), so the smoke's value
is verification + characterization: health (27% of rail calls fail at latencies
clustering at the 150s timeout ceiling), model truth, metered-darkness, and quota
math. Writing the six baseline sections down NOW, before any 4000.3 window,
converts the later comparison from narrative into evidence.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json 4000.1)

1. "scripts/qa/verify_phase_4000_1_baseline.sh exists and inspects the baseline artifact for the six required sections (a)-(f) by content markers, not mere file existence; it accepts the artifact at handoff/current/cc_rail_baseline_4000_1.md OR handoff/archive/phase-4000.1/cc_rail_baseline_4000_1.md so the gate stays green after the archive hook rotates the step."
2. "Section (a) contains a verbatim fenced GET /api/settings capture showing the live paper_use_claude_code_route value with a timestamp."
3. "Section (b) states the running backend process start time and the phase-78.2 commit timestamp side by side, with an explicit YES/NO conclusion on whether the running process predates the --model change."
4. "Section (c) states the CC-rail row-selection rule with the deriving file:line citations from backend code; the rule must be executable as a WHERE clause a later step can apply verbatim."
5. "Section (d) states the trailing-4-week trades/week and round-trips/week with the date filter and the round-trip pairing rule written out; both counts are computed under that one stated rule and the query text is included verbatim."
6. "Section (f) pre-registers every check E1-E7 with a selection rule and an expected pass shape, and the artifact's git timestamp precedes any 4000.3 flag flip."
7. "The research brief exists with gate_passed true in its JSON envelope and >=5 sources read in full including current Anthropic headless/CLI documentation."
8. "MUTATION COVERAGE: deleting any single required section from a copy of the baseline artifact makes the check script exit non-zero, proving it inspects content rather than existence; the executor demonstrates this once and records the observed exit codes in the handoff."
9. "git diff scope for this step touches only handoff/ and scripts/qa/ -- no backend/, frontend/, or settings changes."

## Plan

1. Assemble handoff/current/cc_rail_baseline_4000_1.md, sections (a)-(f), every
   measurement re-captured verbatim in fenced blocks (fresh GET with timestamp +
   the 300s-cache caveat; process start vs 78.2 commits; the 3-shape E1 rule;
   the 28d cadence queries + results; the POST /api/analysis/ entrypoint with
   the unknown-call-count caveat; E1-E7 pre-registered definitions upgraded per
   F0/P1/P2/P3). Plus: claude --version, backend cwd, paper_round_trips
   population check, and a Discovered-defects register (bare-agent attribution
   loss; 27% timeout rate; P1 collapse; spend.py doc drift; F0/F1 watch items).
2. Build scripts/qa/verify_phase_4000_1_baseline.sh (content markers, current-or-
   archive path, exit non-zero on any missing section).
3. Mutation demo: delete one section from a COPY -> script exits non-zero;
   record both exit codes.
4. experiment_results_4000.1.md with verbatim check-script output.
5. Q/A via qa-verdict Workflow (lean prompt, explicit evidence paths); persist
   verdict to evaluator_critique_4000.1.md the turn it returns.
6. harness_log append, then flip 4000.1 -> done ONLY after `git add -An` shows
   no foreign in-flight changes; queue the two discovered code defects
   (attribution loss, P1 collapse) as steps in the same masterplan edit.

## Non-scope (from the masterplan step, binding)

No production code changes; no flag flips; no rail calls beyond the researcher's
single envelope probe (already done); no BQ writes.

## References

- handoff/current/research_brief_4000.1.md (envelope + 29 URLs; key sources:
  support.claude.com/en/articles/15036540 [F0]; code.claude.com headless +
  CLI-reference docs [F1, envelope shape]; arXiv:2412.20138; spend.py:26-39).
- handoff/current/goal_4000_cc_rail_e2e_smoke_DRAFT.md (phase brief).
- Main recon (pre-contract, re-captured in the artifact): scratchpad
  recon_4000_1_results.md.
