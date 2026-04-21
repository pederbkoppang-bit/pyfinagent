# Sprint Contract — Meta-decision: phase-6.5 path selection

**Step id:** phase-6.5-decision (meta; not a regular masterplan step)
**Date:** 2026-04-19
**Tier:** complex (redirects 9+ steps, touches 3 adjacent phases)
**Cycle:** 1

## Research-gate summary

Researcher fetched **7 sources in full** via WebFetch (Springer Digital Finance on r/WSB risk-adjusted returns, ScienceDirect 2024 on WSB attention decay, McLean-Pontiff 2016 post-publication decay, ICLR 2026 GEPA, ATLAS 2026 agent-wars.com, Anthropic multi-agent system, Anthropic harness-design canonical), 26 URLs collected, recency scan present (2024–2026), three-variant query discipline visible. Internal audit covered phase-7 / 8.5 / 10.7 records in `.claude/masterplan.json` and recent `handoff/harness_log.md`. `gate_passed: true`. Brief at `handoff/current/phase-6.5-decision-research-brief.md`.

## Hypothesis

Main's A/B/C framing is wrong. The right frame is "fastest path to live autonomous alpha." Published evidence contradicts the intuition that retail/WSB feeds are high-signal (they're negative-alpha long-term) and confirms academic feeds are factor-graveyard (50–60% post-publication decay). The highest-ROI redirect is to keep only the generic scaffolding of phase-6.5 (schema, registry, prompt-patch queue, smoketest) and skip the source-specific extractors entirely — phase-7 already covers the remaining high-signal sources organised by data type, and phase-8.5 (Karpathy loop) delivers automated alpha without any intel feed.

## Selected path: **Path D (researcher-recommended)**

1. **Keep and execute 4 of 9 phase-6.5 steps:** {6.5.1 schema, 6.5.2 registry+scanner, 6.5.7 novelty client + prompt-patch queue, 6.5.9 e2e smoketest}.
2. **Drop 5 of 9:** {6.5.3 institutional, 6.5.4 academic, 6.5.5 AI-frontier, 6.5.6 player-driven, 6.5.8 Slack digest}. Status → `dropped` with `superseded_by` pointer as appropriate (6.5.3/6.5.5/6.5.6 → phase-7 where the data-type-organised equivalent lives; 6.5.4 → explicit retirement with decay-evidence link; 6.5.8 → merged into 6.5.9 smoketest).
3. **Queue order after 6.5 reduced:** phase-7 (alt-data) in parallel with/after reduced 6.5, then phase-8.5 (Karpathy) with the prompt-patch table wired as a **soft-seed** input to 8.5.3 LLM proposer (not auto-ratification — preserves the human-in-loop risk cap).
4. **Phase-5 (multi-market) remains deferred at end** per user's instruction.

## Immutable success criteria (authored for this meta-decision)

Because this is a meta-decision (not a regular masterplan step), there's no pre-existing immutable criterion. The criteria below are authored now and committed before generate:

- `decision_applied_to_masterplan` — `.claude/masterplan.json` reflects the 4-keep / 5-drop split with correct `status`, `dropped_reason`, and `superseded_by` fields
- `drop_rationale_documented` — each dropped step has a one-line rationale pointing to research brief evidence
- `q_a_independent_verdict` — fresh Q/A subagent returns PASS after reading this contract + the research brief + the updated masterplan
- `harness_log_appended_last` — closeout cycle block appended AFTER Q/A PASS, BEFORE no further state changes

## Plan steps

1. Apply Path D to `.claude/masterplan.json`:
   - Keep 6.5.1 / 6.5.2 / 6.5.7 / 6.5.9 as `status: pending`
   - Mark 6.5.3 / 6.5.4 / 6.5.5 / 6.5.6 / 6.5.8 as `status: dropped` with rationale + `superseded_by`
   - Add a phase-level `path_decision: "D"` note with a pointer to this contract
   - Preserve all existing immutable verification criteria on kept steps
2. Write `handoff/current/phase-6.5-decision-experiment-results.md` with the exact masterplan diff (old → new for each step).
3. Spawn `qa` subagent for fresh independent review.
4. On Q/A PASS: append cycle block to `handoff/harness_log.md`, then the masterplan is the committed state of the decision.

## Out of scope

- No code changes this cycle. The decision only rewrites plan metadata.
- No handoff folder reorg. No git commits.
- No execution of 6.5.1 yet — that starts as a fresh cycle once the decision is ratified.

## Dropped steps — rationale (will be baked into masterplan.json)

- **6.5.3 institutional extractors** → *dropped, superseded by phase-7*. Public institutional pieces are paywalled, lagged, PR-filtered; real research sits behind subscriptions; ToS/scraping risk. Phase-7.2 (13F holdings) captures the positional signal institutional pieces telegraph anyway.
- **6.5.4 academic extractors** → *dropped, McLean-Pontiff post-publication decay*. Published quant-fin strategies lose 50–60% of edge post-publication; arXiv/SSRN monitoring is a factor graveyard, not a pipeline.
- **6.5.5 AI-frontier extractors** → *dropped, out-of-loop for alpha*. Value is model/harness tuning (compute-burn reduction), not trading. Better captured by phase-14-style MCP hygiene or a dedicated model-upgrades phase.
- **6.5.6 player-driven (WSB/SeekingAlpha/QuantConnect/SWF-EDGAR)** → *dropped, superseded by phase-7.5 (Reddit WSB) and phase-7.2 (13F)*. Springer 2023 + ScienceDirect 2024 evidence: WSB high-attention positions average −8.5% holding-period return; Reddit sentiment Sharpe is ~50% of market. Negative long-term alpha in the peer-reviewed record. Phase-7 covers what's worth keeping.
- **6.5.8 Slack digest** → *dropped, merged into 6.5.9 smoketest*. A digest over an empty 4-table schema is noise; the smoketest already proves the schema is queryable.

## References

- `handoff/current/phase-6.5-decision-research-brief.md`
- `.claude/masterplan.json` → phase-6.5, phase-7, phase-8.5, phase-10.7
- `handoff/harness_log.md` (recent 100 lines audited by researcher)
- Canonical: https://www.anthropic.com/engineering/harness-design-long-running-apps
- Evidence:
  - Springer Digital Finance 2023 — r/WSB risk-adjusted returns
  - ScienceDirect 2024 — WSB attention-weighted return decay
  - McLean & Pontiff 2016 — post-publication decay
  - ICLR 2026 — DSPy GEPA prompt evolution
  - ATLAS 2026 — agent-wars.com live 173-day Sharpe improvement
