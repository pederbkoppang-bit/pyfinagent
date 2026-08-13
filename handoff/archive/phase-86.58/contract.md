# Contract — step 86.58

**Step:** 86.58 — the signal_downgrade SELL rule is structurally dead on held positions, and the 2026-08-11 cycle proved it live
**Priority:** P1  |  **Status at contract time:** pending
**Date:** 2026-08-13

---

## Research gate — PASSED

Run `wf_58d0341b-6cb`, `.claude/workflows/research-gate.js`, tier `moderate`.
`gate_passed: true`, recomputed by the script, **no rail drop**. Checks: 6 sources
read in full (floor 5), 38 URLs collected (floor 10), `urls_collected` corroborated
38 <= 38 distinct URLs in the brief, all 6 claimed sources present in the brief,
recency section present, `brief_status_in_brief: COMPLETE`.
Brief: `handoff/current/research_brief_86.58.md` (31,129 chars).

**The gate changed the design. The single-boundary module ALREADY EXISTS.**
`backend/services/recommendation_vocab.py` (209 lines, phase-86.20) holds the
closed scale plus phase-86.22's shared predicates `BUY_INTENT` / `is_buy_intent`,
and its own docstring says it "is meant to be the ONLY one". So the question is
not whether to build an anti-corruption layer — it is that **the existing one
guards the READ side only**:

- `portfolio_manager.py:128` canonicalises on read — guarded.
- `paper_trader.py:452` assigns `_pos_rec = reason` with **no parse step** — the
  WRITE side, unguarded. This is where the order reason enters the field.
- `bigquery_client.py:626` `save_paper_position` MERGEs any dict — but already
  raises on a missing ticker at `:638-639`, which is **precedent for a boundary
  precondition** at exactly this seam.

And the module predicted its own next failure. `recommendation_vocab.py:95-105`
counts the five prior instances and names the sixth's failure mode: *"A caller
that unwraps them back into a literal set has undone the point."*
**`portfolio_manager` IS that caller** — it imports only `canonical_recommendation`
(`:16`) and hand-writes `_BUY_RECS` at `:60-64`.

External: parse-don't-validate (lexi-lambda), RFC 9413 (protocol robustness),
Microsoft's anti-corruption-layer pattern, Fowler's Tolerant Reader, OWASP input
validation, and arXiv:2607.13206v1 (2026) — 641 of 1,646 multi-patch fixes (38.9%)
were incomplete on the first attempt, which is the empirical case for one boundary
guard over a sixth site patch.

---

## Hypothesis

The `signal_downgrade` SELL rule is dead because `paper_positions.recommendation`
receives a value from the ORDER-REASON vocabulary while its reader assumes the
RECOMMENDATION vocabulary. The defect is the unguarded WRITE seam at
`paper_trader.py:452`, not the reader and not the vocabulary's contents. Adding
`new_buy_signal` to the closed set would be the sixth costume.

---

## Immutable success criteria — copied verbatim from `.claude/masterplan.json`

1. the dead-rule claim is proven by DRIVING decide_trades, not by reading source: construct a held position whose recommendation field carries an order reason and whose fresh analysis is HOLD/SELL, and show no signal_downgrade SELL is produced with the flags OFF
2. the population is DERIVED, not assumed: query paper_positions for how many CURRENTLY held rows carry a reason-shaped value in the recommendation field versus a member of the closed recommendation set, and state the query next to the counts
3. the flag-ON behaviour is measured in a NON-LIVE environment and its blast radius quantified: how many currently-held positions would become signal_downgrade SELL candidates if promoted, given the :208-218 warning that HOLD alone triggers the rule
4. flag promotion is NOT performed by this step -- it is operator-gated (ask 06-8) and the step records the recommendation with its measured blast radius instead
5. any guard added is mutation-tested: revert it and show the check goes red, with the control observed GREEN first and a byte-identical restore
6. the phase-86.20 UNRECOGNISED log line is preserved or strengthened, never quieted -- the loudness is the thing that surfaced this and a fix that silences it is a regression

Immutable verification command:
```
bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/portfolio_manager.py\").read()); print(\"parses\")"'
```
Required live_check: `live_check_86.58.md with the verbatim production log line, the measured flag values read from the RUNNING process, and the derived count of held rows carrying a reason-shaped recommendation`

---

## Plan

1. **(done, see disclosure)** Drive `decide_trades` to prove the dead rule, with a
   positive control that observes the rule firing.
2. **(done)** Derive the population from `paper_positions` and the historical
   exit-reason distribution.
3. Quantify the flag-ON blast radius against the CURRENT book, in-process only.
4. Record the operator recommendation for ask 06-8 with that blast radius. **No
   flag is promoted by this step.**
5. If a guard is added, put it at the WRITE seam and mutation-test it, control
   observed green first.
6. Verify the phase-86.20 UNRECOGNISED log line is preserved, never quieted.

---

## DISCLOSURE — protocol order

**Criteria 1 and 2 were executed BEFORE this contract was written**, on 2026-08-13
between roughly 21:00 and 21:15 local, while the research gate was still running.
The contract therefore post-dates part of its own GENERATE.

I am naming this because a file-mtime check would show contract-before-results and
pass — the blind check would not catch it. The work itself is unaffected (the
driven test is reproducible and its control is green), but the ordering was wrong
and the record should say so rather than let the timestamps imply otherwise.

---

## References

- `handoff/current/research_brief_86.58.md` — research gate, PASSED
- `backend/services/recommendation_vocab.py:95-105` — the five prior instances and the predicted sixth failure mode
- `backend/services/portfolio_manager.py:16,60-64,128,242-246,264` — the unwrapping caller and the dead rule
- `backend/services/paper_trader.py:452` — the unguarded write seam
- `backend/db/bigquery_client.py:626,638-639` — the persistence boundary and its existing precondition precedent
- `handoff/current/q1_binding_constraint_86.59.md` — the 86.69 emptiness defect that makes the blast radius acute
