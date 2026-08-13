---
name: dead-sell-rule-86-58
description: Step 86.58 field-conflation defect -- the canonical vocabulary module ALREADY exists but is installed on the READ side only; portfolio_manager opted out of its shared predicates; the 5 prior instances are counted in-repo at recommendation_vocab.py:95-105
metadata:
  type: project
---

Step 86.58: `paper_positions.recommendation` stores an ORDER REASON (`'new_buy_signal'`)
where readers assume an ANALYSIS RECOMMENDATION, so `signal_downgrade`'s left conjunct
`old_rec in _BUY_RECS` can never match.

**The non-obvious findings (all verified 2026-08-13):**

1. **The single-boundary module already exists** -- `backend/services/recommendation_vocab.py`
   (209 lines, phase-86.20) with `CANONICAL_RECOMMENDATIONS`, `canonical_recommendation()`,
   and phase-86.22's shared predicates `BUY_INTENT`/`is_buy_intent()`. Its docstring says it
   "is meant to be the ONLY one". So the answer is NOT "build an ACL" -- it is "the ACL is
   installed on the wrong side".
2. **It guards the READ, never the WRITE.** `portfolio_manager.py:128` calls the canonicaliser;
   `paper_trader.py:452` assigns `_pos_rec = reason` with no parse step, and
   `bigquery_client.py:626 save_paper_position` MERGEs whatever dict it gets. That choke point
   DOES already raise on a missing `ticker` (`:638-639`) -- precedent for a boundary
   precondition -- and drops `None` before the MERGE (`:637`), so "write nothing" and
   "write NULL" are indistinguishable.
3. **The "five prior instances" are counted IN-REPO**, not from memory:
   `recommendation_vocab.py:95-105` -- "FIVE consumers, TWO mutually incompatible dialects,
   and the sets written out by hand at each site". The same comment names the exact failure
   mode the sixth instance shows: "A caller that unwraps them back into a literal set has
   undone the point."
4. **`portfolio_manager` IS that caller.** It imports ONLY `canonical_recommendation` (`:16`)
   and re-declares `_SELL_RECS`/`_DOWNGRADE_RECS`/`_BUY_RECS` by hand at `:60/:62/:64`, while
   `is_buy_intent()` sits unused. Fixing that alone does NOT fix the defect -- `'new_buy_signal'`
   is not a buy under any spelling -- so it is hardening, not the fix.
5. **Stale anchor, twice copied:** `portfolio_manager.py:53` AND `settings.py:212` both cite the
   rule at `portfolio_manager.py:127`. It is at **`:264`**. Re-derive before citing.
6. **The `:208-218` blast radius is wider than "downgraded positions".** `_DOWNGRADE_RECS`
   contains `HOLD` (`:62`) and `rec` DEFAULTS to `"HOLD"` (`:242-246`), so with the field fixed,
   ANY re-eval degradation on a held ticker sells a healthy position -- not just genuine
   downgrades.
7. **`GET /api/settings/` cannot report these flags.** 45 keys, zero containing
   fix/vocab/integrity/recommend. Structural: `FullSettings` (`settings_api.py:101-123`) has no
   field for them, yet `_FIELD_TO_ENV` (`:261-266`) DOES carry two of the three -- **writable
   but not readable**. `paper_recommendation_vocab_fix_enabled` is in NEITHER map.

**Why:** the operator forbade adding `'new_buy_signal'` to the recommendation vocabulary --
that entrenches the wrong value (RFC 9413's "pathological feedback cycle"). The field
conflation is the defect.

**How to apply:** for any future instance of this class, check
`recommendation_vocab.py` FIRST -- the module and its predicates probably already cover it, and
the real question is which SIDE of the boundary is unguarded. Strongest external evidence is
arXiv:2607.13206v1 (2026): 641 of 1,646 multi-patch fixes (38.9%) were incomplete first
attempts, 860 were multi-location -- per-site patching is empirically the route to an
incomplete fix. See [[guard-from-instance-not-class]].
