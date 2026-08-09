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
