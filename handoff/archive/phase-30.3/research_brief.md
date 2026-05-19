# Research Brief -- phase-30.5: Sector NAV-Percentage Cap

**Tier:** complex. **Effort:** max. **Date:** 2026-05-19.

**Step:** P2 -- add NAV-percentage sector cap alongside existing count cap in `portfolio_manager.py::decide_trades`.

**Scoped questions (verbatim from caller):**
1. Canonical NAV-percentage cap values (5+ sources). Is 30% NAV / sector the right default for $20-25K paper portfolio with 10-11 positions?
2. NAV-pct check site -- same guard at `portfolio_manager.py:188-262`, pure math.
3. Edge cases: cap=0 disables, missing market_value, candidate self-exceeds cap, sector already-over.
4. Test design (A: NAV-pct blocks where count allows; B: allows when fine; C: 0 disables; D: independence).
5. Live-check criterion: overnight pause -> tests suffice.

---

## Section 1: Read-in-full table

(populated below after fetches)

## Section 2: Snippet-only table

(populated below)

## Section 3: Recency scan (2024-2026)

(populated below)

## Section 4: Key findings on NAV-percentage caps

(populated below)

## Section 5: Internal code inventory

(populated below)

## Section 6: Edge-case analysis

(populated below)

## Section 7: Test design recommendations

(populated below)

## Section 8: Application to pyfinagent (recommendation)

(populated below)

## Section 9: Research Gate Checklist

(populated below)

## Section 10: JSON envelope

(populated below)
