---
step: phase-19.0
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - docs/architecture/claude-remote-handoff-feasibility.md (NEW, ~250 lines, decision document)
  - .claude/masterplan.json (NEW phase-19 block + 19.0 sub-step)
---

# Experiment Results -- phase-19.0

## What was done

Authored a feasibility decision document answering the operator's question:
"Can pyfinagent push heavy-lifting work to Claude remote agents using my
Max ($200/mo) flat-fee subscription so it doesn't cost additional cash?"

## Headline finding (rebuts prior turn)

The literal hypothesis -- using `CLAUDE_CODE_OAUTH_TOKEN` (Max OAuth)
from the FastAPI backend -- **violates Anthropic ToS as of April 4, 2026**
(per The Register 2026-02-20 + the Claude Agent SDK overview doc which
explicitly states programmatic use requires `ANTHROPIC_API_KEY`).

This DIRECTLY CORRECTS my prior-turn recommendation, which proposed
exactly this approach. The researcher caught the recent ToS change that
my prior assertion missed.

**However,** the underlying intent (offload long-context jobs) is now
economical via standard API: 1M context is **standard-priced** on Sonnet
4.6 / Opus 4.7 in 2026 (no `extended-context-1m-2025-08-07` beta surcharge).
A 300K-token synthesis call ~$0.90; daily 5-call duty cycle ~$5/day,
fits inside existing budget cap.

## Deliverable

### `docs/architecture/claude-remote-handoff-feasibility.md` (NEW, ~250 lines)

Section structure:
1. **Recommendation** -- REJECT literal Max-OAuth hypothesis (ToS), ACCEPT alternative API-key path
2. **TL;DR table** -- 6 questions, plain-English answers
3. **ToS analysis** -- cites The Register + Agent SDK doc
4. **What Max actually covers** -- 6-row table (covered surfaces vs API)
5. **What 1M context costs in 2026** -- standard-priced, with worked examples
6. **5 jobs that benefit** -- ranked by ROI (synthesis, skill optimizer, directive rewriter, outcome tracker, deep dive)
7. **7 jobs that do NOT benefit** -- explicit anti-recommendations
8. **Recommended architecture** -- extend `make_client()` factory, NOT new module
9. **Recommended budget tracker** -- new `anthropic_long_context` provider in YAML
10. **Engineering cost** -- 0.5 spike + 3.5 days full integration
11. **Risk register** -- 6 risks + mitigations; explicit "NOT a risk: Max ToS violation"
12. **Decision** -- proceed with 0.5-cycle spike, gate full integration on A/B win
13. **Cross-references** -- file:line anchors to existing pyfinagent code

### `.claude/masterplan.json` (new phase-19 block)

```
phase-19 "Claude Remote / Max programmatic handoff (feasibility study)"
  19.0 Feasibility study: Claude Remote / Max programmatic handoff [DONE this cycle]
```

## Verification (verbatim, immutable from masterplan)

```
$ test -f docs/architecture/claude-remote-handoff-feasibility.md && grep -q 'Recommendation' ... && grep -q 'Claude Agent SDK' ... && grep -q 'rate limit' ...
$ echo "exit=$?"
exit=0
```

## Files touched

| Path | Action |
|------|--------|
| `docs/architecture/claude-remote-handoff-feasibility.md` | CREATED |
| `.claude/masterplan.json` | edit (new phase-19 block) |
| `handoff/current/contract.md` | rewrite (rolling) |
| `handoff/current/experiment_results.md` | rewrite (this) |
| `handoff/current/phase-19.0-research-brief.md` | created (researcher; verified on disk 23KB / 283 lines) |

NO code changes. NO new tests. NO implementation.

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | File exists at canonical path | PASS |
| 2 | Contains "Recommendation" section | PASS |
| 3 | Cites "Claude Agent SDK" | PASS |
| 4 | Addresses "rate limit" | PASS |
| 5 | Researcher brief on disk + gate_passed | PASS |
| 6 | Document is honest about ToS finding (rebuts prior-turn answer) | PASS |
| 7 | Recommendation is actionable (next step = 0.5 cycle spike) | PASS |

## Honest disclosures

1. **My prior-turn answer was wrong.** I told the operator they could use Max OAuth via Claude Agent SDK / Claude Code subprocess. The researcher's deep dive surfaced the April 2026 ToS change that prohibits exactly this. The decision doc states this directly in the "Headline finding" + "ToS analysis" sections.

2. **Pure research cycle.** No code, no test, no infrastructure changes. The doc IS the deliverable.

3. **Researcher artifact verified on disk** before this experiment_results was written (per the phase-18.0 cycle-2 lesson: don't claim a file exists without `ls`-checking).

4. **Cycle-2 not needed.** First-pass clean.

5. **Operator's underlying intent is satisfiable** -- just not via the literal mechanism they proposed. The doc lays out the alternative path so the operator can decide: spike or shelf.

6. **No emojis** in the doc (per `feedback_no_emojis.md`).

## Closes

UAT-19.0 (next task slot). Masterplan step phase-19.0.

## Next

Spawn Q/A. After PASS: log + flip + archive + commit + push. Operator decision pending: pursue the spike (phase-19.1) or shelf the work.
