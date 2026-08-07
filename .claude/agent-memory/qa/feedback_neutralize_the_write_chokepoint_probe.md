---
name: neutralize-the-write-chokepoint-probe
description: To grade a "we threaded the kwarg through" fix, neutralise the single write chokepoint and run the REAL pipeline -- one probe proves the fixed seam and exposes the adjacent unfixed seam
metadata:
  type: feedback
---

When a cycle-2 fix claims "the kwarg is now threaded", do not grade it by reading the
diff or by re-implementing the expression. Replace the ONE function every writer funnels
through (`bq_writer._insert_rows`) with a recorder that returns `len(rows)`, then call the
real entry point (`phase6_e2e._run_pipeline(..., backfill=True)`). Zero external writes
occur, every serializer still runs, and you get the exact rows that WOULD have been
inserted, per table.

**Why:** 83.0 cycle 2. The author fixed the seam the cycle-1 Q/A named
(`phase6_e2e:117` now passes `provenance="backfill" if backfill else "live"`). The probe
returned `news_articles n=3 provenance=['backfill']` -- fix real -- **and**
`news_sentiment n=3 provenance=['live']`. The identical falsehood survived one seam over,
because `ScorerResult` has no `provenance` field, `score_ladder` takes no such kwarg, and
`_serialize_sentiment` both defaults to `"live"` AND coerces anything invalid to `"live"`.
The contract's own decision D1 asserted the opposite in writing. Reading the diff would
have shown a correct fix; only executing the whole path showed the half-delivery.

**How to apply:** on any "threaded through" / "wired up" remediation, (1) find the single
chokepoint downstream of ALL affected producers and neutralise it in-process (attribute
assignment, no repo write, nothing to restore); (2) run the real caller with the flag ON;
(3) print the value per table/branch -- a constant across branches is the finding; (4)
separately grep whether ANY test imports the fixed module at all
(`grep -rln <module> backend/tests tests`) -- an unguarded fix silently regresses. Both
checks are cheap and neither trusts the author's prose. See
[[mutate-without-touching-the-tree]] and [[recheck-prior-remediation-list]].
