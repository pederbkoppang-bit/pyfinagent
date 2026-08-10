"""phase-86.20 -- ONE canonicaliser for the analyst-recommendation vocabulary.

WHY THIS EXISTS
---------------
`portfolio_manager` tests membership against UNDERSCORE sets after `.upper()`.
`.upper()` folds CASE but never the SEPARATOR, and the full-pipeline producer
emits SPACED title case, so `"Strong Buy"` becomes `"STRONG BUY"` and matches
nothing. Plain `"Buy"` works by accident -- which is why the mismatch survived:
it destroys the HIGHEST-conviction spelling while letting the medium one
through, inverting the conviction ordering that reaches the book.

The repo already contains BOTH dialects as first-class citizens:
`backend/api/models.py`'s `Recommendation` enum has spaced title-case VALUES,
while `backend/agents/schemas.py` pins `consensus` to an underscored `Literal`.
Nothing mapped between them. This module is that mapping, and it is meant to be
the ONLY one -- two canonicalisers that disagree would be this same defect
wearing a different hat.

DESIGN, AND WHY IT IS DELIBERATELY NARROW
-----------------------------------------
This is a FINITE MAPPING ONTO A CLOSED SCALE -- the I/B/E/S pattern, where many
broker spellings are mapped onto one small fixed ladder -- and NOT a pattern
match. Concretely:

  * We fold case and treat space / hyphen / underscore as the SAME separator.
  * We do NOT strip other punctuation, do NOT do substring or prefix matching,
    and do NOT expand synonyms. `"Accumulate"`, `"Overweight"`, `"Strong Buy!"`
    and `"N/A"` all resolve to UNKNOWN (`None`).
  * UNKNOWN is never guessed into an intent. On a money path an over-permissive
    gate is worse than a narrow one, because it moves money.

Substring matching is a real defect class here, not a hypothetical: a sibling
site tests `"STRONG_BUY" in rec_label`, where `"STRONG BUY"` fails that test but
still passes a later `"BUY" in rec_label` and is graded against a weaker
threshold. And `"STRONG_SELL"` contains `"SELL"`. That is filed as phase-86.22;
this module must not repeat the mistake.

Canonicalise BEFORE validating, never after (CWE-180: validating first and
canonicalising second is itself the bug class). Never silently drop an
unrecognised token -- see the caller-side loudness in `portfolio_manager`
(IETF `draft-thomson-postel-was-wrong`; Google SRE on implicit errors).

References: handoff/current/contract_86.20.md, research_brief_86.20.md.
"""
from __future__ import annotations

import re

# The closed scale. Everything canonicalises onto exactly one of these, or to
# UNKNOWN. Adding a member here is a RISK DECISION, not a formatting change:
# every consumer's membership sets are written against this vocabulary.
STRONG_BUY = "STRONG_BUY"
BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"
STRONG_SELL = "STRONG_SELL"

CANONICAL_RECOMMENDATIONS: frozenset[str] = frozenset(
    {STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL}
)

# Space, hyphen, underscore and any run of them are ONE separator. Note this
# does not touch other punctuation on purpose -- see the module docstring.
_SEPARATORS = re.compile(r"[\s\-_]+")


def canonical_recommendation(value: object) -> str | None:
    """Map any spelling of a recommendation onto the closed scale, or None.

    Returns one of `CANONICAL_RECOMMENDATIONS`, or `None` for anything outside
    it -- including None, empty/whitespace-only strings, and non-strings.

    `None` means UNRECOGNISED and must be treated by callers as *not* a buy,
    *not* a sell and *not* a downgrade. It is deliberately distinct from
    `HOLD`: a recognised hold and an unparseable value are different facts, and
    collapsing them is how a producer-side vocabulary drift goes unnoticed.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        # Be explicit rather than str()-ing arbitrary objects into the gate:
        # a dict or an enum member reaching here is a caller bug, and turning
        # it into a plausible-looking token would hide that.
        return None
    folded = _SEPARATORS.sub("_", value.strip().upper())
    return folded if folded in CANONICAL_RECOMMENDATIONS else None


def is_recognised(value: object) -> bool:
    """True iff `value` maps onto the closed scale."""
    return canonical_recommendation(value) is not None


# ── phase-86.22: SHARED INTENT PREDICATES ───────────────────────────────────
# phase-86.20 gave the repo one canonicaliser but left every consumer to decide
# for itself what counts as a buy. Measured then: FIVE consumers, TWO mutually
# incompatible dialects, and the sets written out by hand at each site --
# `("Strong Buy","Buy")` in one file, `("BUY","STRONG_BUY")` in another, and a
# SUBSTRING test in a third. Re-deriving membership per call site is how the two
# dialects drifted apart in the first place, so the sets live HERE, once.
#
# These are DELIBERATELY not "does this token appear in some tuple" helpers for
# callers to re-implement: they take the RAW value, canonicalise it, and answer
# the question. A caller that unwraps them back into a literal set has undone the
# point.
#
# HOLD is in NEITHER. It is a recognised recommendation and a real decision --
# it is simply not a directional call, and collapsing it into "not a buy" is
# what makes an unparseable value indistinguishable from a considered hold.

BUY_INTENT: frozenset[str] = frozenset({STRONG_BUY, BUY})
SELL_INTENT: frozenset[str] = frozenset({STRONG_SELL, SELL})


def is_buy_intent(value: object) -> bool:
    """True iff `value` canonicalises to a BUY or STRONG_BUY.

    Unrecognised values are False -- never guessed into an intent. On a money
    path an over-permissive gate is worse than a narrow one, and on a learning
    path a wrong label is worse than an absent one.
    """
    return canonical_recommendation(value) in BUY_INTENT


def is_sell_intent(value: object) -> bool:
    """True iff `value` canonicalises to a SELL or STRONG_SELL."""
    return canonical_recommendation(value) in SELL_INTENT


def is_directional(value: object) -> bool:
    """True iff `value` expresses a direction at all (buy or sell).

    Exists so a caller can tell "this was a HOLD" apart from "this could not be
    parsed" -- the distinction `directionally_correct` silently destroyed by
    reporting False for both.
    """
    return is_buy_intent(value) or is_sell_intent(value)


# ---------------------------------------------------------------------------
# phase-86.25: the OUTCOME boundary -- what gets persisted when no analyst
# recommendation is known.
# ---------------------------------------------------------------------------
#
# WHY A SENTINEL AND NOT "HOLD". Two seams used to hand this module's consumers
# a value from a DIFFERENT vocabulary and let it be read as a recommendation:
#
#   S1  autonomous_loop._learn_from_closed_trades  -- `risk_judge_decision`,
#       an APPROVAL vocabulary (APPROVE_REDUCED / REJECT / APPROVE_HEDGED),
#       coerced to the literal "HOLD" when empty.
#   S2  nightly_outcome_rebuild._compute_outcomes  -- `risk_judge_decision or
#       action`, i.e. the trade ACTION ("SELL") when the approval was empty.
#       This is the seam that actually wrote the three live 'SELL'-spelled
#       outcome_tracking rows.
#
# "HOLD" as the missing-data marker is PEP 661's named anti-pattern: an
# IN-DOMAIN value used to mean "absent", which no downstream reader can tell
# from a considered hold. `outcome_tracking.recommendation` is REQUIRED at the
# destination, so SQL NULL is unavailable; the marker therefore has to be a
# string that is provably not a member of the scale. UNKNOWN is that string, and
# `is_directional(UNKNOWN_RECOMMENDATION)` is False by construction -- asserted
# in the tests rather than assumed here.
#
# MEASURED 2026-08-10, and it is why this resolver exists rather than a lookup:
# NO PRODUCER EMITS AN ANALYST RECOMMENDATION ONTO A TRADE AT ALL.
# `analyst_recommendation` is not a column of
# `financial_reports.paper_trades` (18 columns), and
# `_production_fns.LEDGER_FETCH_SQL` selects ten named columns without it, so
# the callers' lookup reads a dict key nothing writes. The path is dead BY
# CONSTRUCTION. Building an (A)-style lookup and guarding it with stubs would
# be guarding a path that runs for no real row.
#
# CORRECTED cycle 2 (Q/A finding W1). This paragraph previously gave the cause
# as an unreachable ANCHOR -- "the analyst recommendation is reachable for 0 of
# 32 SELL rows; analysis_id is empty on 32/32 SELLs (BUYs carry it 33/33), and
# round_trip_id is ONE-SIDED". Those measurements are real and are recorded in
# `contract_86.25.md` section 6, but they are NOT the operative cause here, and
# stating them told the executor of the queued round_trip_id step that this
# path would self-heal once the anchor landed. IT WILL NOT: it needs a producer
# change. The "0 of 32 SELL rows" phrasing also implied BUY rows differ; they
# do not -- the key is absent for every row of every action.
#
# A cycle-2 remediation MISSED THIS FILE while the artifact claimed all three
# were corrected. The re-check that caught it was `git diff <prior-sha> HEAD --
# <each file named in the prior critique>`; an empty diff for a file on that
# list is the finding. Re-derive the prior list, never trust a summary of it.
#
# NEVER map an approval onto a direction. "risk approved a reduced size" is not
# "the analyst said buy" -- that is a claim nobody made, and inventing it is the
# leniency that phase-86.22 exists to forbid.

UNKNOWN_RECOMMENDATION = "UNKNOWN"


def resolve_outcome_recommendation(analyst_recommendation: object) -> str:
    """Return the value to persist as an outcome row's `recommendation`.

    Takes ONLY an analyst recommendation candidate. It deliberately does not
    accept a `risk_judge_decision` or an `action` argument: a boundary that
    accepts the wrong vocabulary is a boundary a caller can misuse, and both
    historical defects were exactly that misuse. Callers hand over a real
    recommendation or nothing.

    Returns the canonical recommendation when the candidate parses -- including
    HOLD, which is a real and considered call -- and `UNKNOWN_RECOMMENDATION`
    otherwise. Never guesses a direction.
    """
    canon = canonical_recommendation(analyst_recommendation)
    return canon if canon is not None else UNKNOWN_RECOMMENDATION
