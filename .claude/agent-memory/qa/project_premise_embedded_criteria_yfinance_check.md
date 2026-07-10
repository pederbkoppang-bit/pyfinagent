---
name: premise-embedded-criteria-yfinance-check
description: Criteria embedding factual premises can be unsatisfiable when the premise is false (68.5 AMD/MU); free yfinance fetch is the no-cost independent corroboration path for price-fact claims
metadata:
  type: project
---

Immutable criteria that EMBED a factual premise (68.5: "root-cause the AMD/MU
price defect", "correct the corrupted rows") become unsatisfiable when research
overturns the premise -- 68.0 proved AMD/MU fills were REAL 2026-07-09 prices
(AMD close $546.72 / MU $991.64; the "~$150/~$110" in live_check_66.2.md:402 was
a stale world-knowledge anchor, AMD's 52wk LOW). Correct handling, PASS-worthy:
surface the overturn headlined + route disposition to the operator with criteria
byte-unchanged (verify via `git diff HEAD -- .claude/masterplan.json` empty).
Silent reinterpretation OR criteria edits = FAIL.

**Why:** the 66.2 closer's un-sourced sanity check against world knowledge
created a false P0 premise that survived two phases; the 68.0 evaluation only
became deterministic because I re-fetched prices myself.

**How to apply:** (1) when a step's criteria assert a market/price fact, run the
free corroboration path yourself: `source .venv/bin/activate && python -c
"import yfinance as yf; print(yf.Ticker('X').history(period='7d')['Close'])"`
-- dated bars, no API money; also grep backend.log for the recorded fill lines
(root `./backend.log`). (2) "Hypothesis tree ranked by evidence" criteria
mandate the TREE, not a predetermined conclusion -- "premise false" ranked #1
with per-branch confirm/kill evidence satisfies the literal words. (3) When
68.5 spawns, check whether the operator re-scoped criteria 1-2/4 first; if not,
expect retry burn. See [[criterion-wording-existence-vs-completion]].
