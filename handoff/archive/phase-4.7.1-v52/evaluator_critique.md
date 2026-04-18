# Evaluator Critique -- Cycle 78 / phase-4.8 step 4.8.1

Step: 4.8.1 Survivorship-bias + point-in-time audit

## Dual-evaluator run (parallel; anti-rubber-stamp active)

## qa-evaluator: PASS

Substantive 6-point honesty review:
1. `delisted_at_populated` semantic stretch acknowledged in contract
   + audit notes; not hidden.
2. `pit_kwarg_enforced_100pct` has teeth: body-ref guard rejects
   decorative kwargs (after docstring-strip fix).
3. 4-function scope is reasonable: universe selectors + PIT price/
   fundamental reads are where survivorship bias originates.
4. NotImplementedError is real code; fail-loud correctly preferred.
5. Brown/Goetzmann 1995 + Elton/Gruber/Blake 1996 + AFML ch.14
   cites are real peer-reviewed sources.
6. Immutable verification exits 0 with pit_enforced_pct=1.0.

## harness-verifier: FAIL #1 -> PASS #2 (same agent via SendMessage)

### First pass: FAIL
"Check 6 FAIL: the audit's body-ref guard was satisfied by docstring
mentions of `as_of` after the `raise NotImplementedError` block was
removed. `pit_enforced_pct` stayed 1.0, verdict stayed PASS. audit
missed the regression."

Another legitimate catch of a gameable audit check -- this is the
second cycle in a row (Cycle 75 preventDefault substring, Cycle 78
docstring mentions) where harness-verifier has found a real hole.

### Fix applied (orchestrator, same cycle)
- Added `_DOCSTRING_RE` (triple-quoted) + `_COMMENT_RE` (# line)
  regex strippers in survivorship_audit.py.
- `_strip_docstrings_and_comments(src)` removes both.
- `_check_pit_kwarg` now counts refs only in the stripped body.

### Second pass (same agent via SendMessage, no re-spawn): PASS
Two independent mutations (raise block removed; raise + all
docstring refs stripped) both caught with rc=1. File restored
verbatim. Immutable verification PASS.

## Decision: PASS (evaluator-owned, FAIL->PASS arc recorded)

Cycle 78 is the third in a row where a first-pass FAIL/CONDITIONAL
was legitimately earned, fixed in-cycle, and re-verdicted by the
same evaluator via SendMessage -- not second-opinion shopped.
