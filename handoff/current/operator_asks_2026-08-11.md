# Operator asks -- 2026-08-11 (session `pyfinagent-51`)

Three asks outstanding. None blocks work already committed; all three are
decisions I cannot take under the standing constraints.

---

## ASK #1 -- ratify phase-86.37's REUSED research gate, or direct a fresh one

**Carried over from the 2026-08-10 goal, still unanswered.**

86.37 fixed the researcher rail. Its own research gate was **REUSED, not re-run**
-- and the awkwardness is structural: *the rail being fixed is the rail that runs
the gate, and it had just dropped.* Re-running the gate on the rail under repair
would have been evidence of nothing until the fix was in; not re-running it
leaves the step's gate older than its code.

**The step cannot close without a ruling.** Either:
- **(a) RATIFY the reuse** -- the artifact exists for the step id, the contract
  cites it, and its claims were re-verified against source; or
- **(b) DIRECT A FRESH GATE** now that the rail is repaired, accepting ~180k
  tokens for it.

I have not chosen, because choosing would be me ruling on the adequacy of my own
step's gate.

---

## ASK #2 -- classify the Vertex 429, or accept lite-on-quota-exhaustion

Full detail in `handoff/current/experiment_results_86.38.md` section 8.

Short version: the 429 body is complete and carries **no discriminator** by
design; classification requires reading
`serviceruntime.googleapis.com/quota/rate/net_usage` in the GCP console, which is
outside this step and adjacent to spend decisions.

**Recommended: option A (read the metric, free, ~5 min), then B (accept the
fallback as designed).** NOT option C (Provisioned Throughput / paid tier) --
it is metered spend, and the per-cycle evidence shows degradation is not causing
the trade drought, so it would be money spent on the wrong subsystem.

---

## ASK #3 -- the subagent token budget, and whether today's rate is acceptable

**I have no read on the weekly Max headroom and cannot get one.** What I can
measure, this session only:

| | |
|---|---|
| Workflow runs launched | **13** (all terminal) |
| returned a verdict/envelope | 8 |
| **dropped, returning nothing** | **5 (38.5%)** |
| tokens in dropped runs | **~887k** |
| cumulative subagent tokens | **~3.1M** |

*(These figures were re-derived from the per-run `journal.jsonl` files. Two
earlier drafts of this table were wrong -- "5 of 11 / 45%" quoted from memory,
then "4 of 12 / 36%" from a tally whose age heuristic mis-classified a run that
had just dropped as still running. The figures above enumerate every run by
whether its journal contains a `result` line, which is the only reliable test.)*

**One step alone accounts for four of the thirteen**: 86.38 took 4 spawns and
~702k tokens for ONE completed verdict, and was parked rather than given a fifth.

The peer session reports >8.7M for 2026-08-10 alone. The standing rule is that
**50% of the weekly Max allowance is a hard ceiling**, past which usage moves to
metered credits and breaks the `$0 metered` constraint.

**What I want:** either a headroom figure I can budget against, or an instruction
to cap spawns per day. I have been self-limiting -- parking steps at the
escalation boundary rather than spending another ~180k on a likely-FAIL, and
declining to start P3 steps that would need a full ~400k cycle -- but that is my
judgement substituting for a number.

**Worth noting on the other side:** every dropped run's write-first record was
recovered, and two of them carried findings I acted on. The drops were not
entirely wasted, but ~700k for partial records is a poor rate.
