# Contract -- phase-86.20

**Step:** 86.20 (P1) -- the trade gate and the analyzer speak different
recommendation vocabularies.
**Date:** 2026-08-09. **Cycle:** 196.
**Research gate:** PASSED -- `handoff/current/research_brief_86.20.md`
(run `wf_66bcd575-e9a`; 8 sources read in full >= floor 5, 42 URLs >= floor 10,
recency scan performed, 12 internal files inspected). The rail RECOMPUTED
`gate_passed` and an independent read-only agent confirmed the brief exists on
disk (33,532 chars) with all 8 claimed source URLs present in it.

---

## 1. Research-gate summary -- and the three things it found that the step text did not

**Confirmed at source.** `_BUY_RECS = {"BUY","STRONG_BUY"}` (underscore) is
tested after `.upper()` only, which folds CASE but never the SEPARATOR. The
full-pipeline producer emits spaced title case, so `"Strong Buy"` becomes
`"STRONG BUY"` and is dropped by `continue` with no log line. Plain `"Buy"`
works by accident.

**FINDING A -- the sell side is FAIL-DANGEROUS, and this reverses the step's
severity framing.** `"Strong Sell".upper() == "STRONG SELL"` is in **neither**
`_SELL_RECS` nor `_DOWNGRADE_RECS`, so a held position carrying a full-path
`Strong Sell` matches neither the `sell_signal` branch nor the
`signal_downgrade` branch: **it is not sold at all**, and only the stop-loss can
still exit it. The buy-side half costs opportunity; the sell-side half costs
**protection**. The two halves have opposite risk polarity and must not be
armed as if they were one change.

**FINDING B -- phase-61.2's fix is silently defeated by this same mismatch.**
`paper_trader.py` writes the recommendation verbatim on the position row with no
normalisation, so the full path persists `"Strong Buy"`. `portfolio_manager.py`
then reads it back as `"STRONG BUY"`, which fails `old_rec in _BUY_RECS`, so the
`signal_downgrade` exit rule that 61.2 exists to revive stays **structurally
dead for exactly the full-path rows it was built for -- even with its flag ON**.
Read-boundary canonicalisation fixes this without touching the write path.

**FINDING C -- `conflict_detector` carries an independent SUBSTRING defect.**
It tests `"STRONG_BUY" in rec_label`, so `"STRONG BUY"` fails that test but
still PASSES the later `"BUY" in rec_label`, and is graded against the weaker
threshold rather than skipped. `"STRONG_SELL"` likewise contains `"SELL"`.
**Out of scope here** -- filed as 86.22 with the rest of the cross-module class.

**The underscore dialect is already schema-enforceable in this repo**, which
settles the producer question: `agents/schemas.py` already pins
`consensus: Literal["STRONG_BUY",...]`, while the synthesis `action` field is a
plain `str` whose vocabulary lives only in the prompt description -- so it
cannot reject `"strong buy"`, `"Accumulate"` or `"Strong Buy!"`.
`api/models.py` separately defines a `Recommendation` enum whose VALUES are
spaced title case. Both dialects are first-class in-repo today.

**External literature, cited per claim in the brief:** canonicalise ONCE before
validating, never after (CWE-180 -- validation order is itself the bug class);
never silently drop an unknown token (IETF `draft-thomson-postel-was-wrong`;
Google SRE on implicit errors); prefer constraining the producer with a schema
enum where possible (Anthropic structured outputs / constrained decoding;
arXiv 2502.14905); and the finance precedent is I/B/E/S, which maps many broker
spellings onto an explicit finite coarse scale rather than pattern-matching.
The robustness-principle critique is the reason this contract forbids liberal
matching: normalising MORE aggressively is exactly how an over-permissive money
gate is born.

**Correction inherited from the measurement, and NOT to be propagated
unfixed.** The step text says the `Strong Buy` at 8.36 is "HIGHER THAN ANY ROW
THAT DID MATCH (max BUY score 8.0)". That is **false as written**: `"Buy"` also
matches the gate after `.upper()` and reaches **8.8** on genuine rows. The true,
narrower statement is that 8.36 is higher than any row spelled with the literal
uppercase `BUY`, and that `Strong Buy` is the highest-scoring genuine
recommendation that FAILS to reach the gate. The defect is unchanged; only the
severity framing is corrected.

## 2. Hypothesis

The gate's membership test runs against a string that has been case-folded but
not separator-folded, so exactly one spelling family -- the producer's own --
never matches. Introducing a single pure canonicaliser that folds case AND
separator onto a CLOSED five-token vocabulary, applied at the READ boundary in
`portfolio_manager`, makes every spelling of an intent reach the same decision,
restores the sell/downgrade branches, and un-defeats phase-61.2 -- without
admitting any string that is not already that intent, because the target
vocabulary is closed and anything outside it resolves to UNKNOWN rather than to
a guess.

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. "The defect is REPRODUCED FIRST and recorded verbatim: drive decide_trades with a candidate whose recommendation is the literal 'Strong Buy' and assert that under CURRENT code it yields NO buy order, while an otherwise identical candidate with 'BUY' DOES -- isolating the difference to the vocabulary and not to any other gate"
2. "The recommendation population is RE-DERIVED at fix time from analysis_results (do not copy this step's counts) and recorded: every distinct value, its count, and how many are genuine (final_score > 0), with which values match the gate after .upper() and which do not"
3. "The fix normalises the SEPARATOR rather than adding the observed literal to the set. A test proves at least three spacing/punctuation variants of one intent reach the same decision, and that 'Buy' (safe today) is unchanged"
4. "SELL and DOWNGRADE are covered too, not only BUY -- assert the Strong-Sell equivalent is handled, since _SELL_RECS/_DOWNGRADE_RECS have identical exposure"
5. "NO STRING THAT IS NOT A BUY INTENT BECOMES A BUY. Assert explicitly that 'HOLD', 'Hold', 'Sell', 'N/A', '' and None still produce no buy order"
6. "The silent skip becomes observable: a candidate rejected for an UNRECOGNISED recommendation (as distinct from a recognised non-buy like HOLD) is logged or counted, so the next vocabulary drift is loud"
7. "MUTATION-TEST every new guard, including reverting the normalisation, and confirm each mutant is killed by the assertion that names it"

**Verification command (immutable):**
`bash -c 'source .venv/bin/activate && python -m pytest backend/tests/ -q -k "portfolio_manager or decide_trades"'`

**live_check (immutable):** "Verbatim test output for the reproduce-then-fix pair on 'Strong Buy'; the re-derived recommendation-distribution output showing every distinct value with count and genuine count; and verbatim negative-case output proving HOLD/Sell/N/A/empty/None still produce no buy."

## 4. Design decision, with the evidence that chose it

**Canonicalise at the READ boundary, and ALSO constrain the producer -- but only
the read boundary ships in this step.**

- **Read boundary (this step).** One pure function, new module
  `backend/services/recommendation_vocab.py`, mapping any input onto the closed
  set `{STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL}` or to a distinct UNKNOWN
  result. Folding is: strip, case-fold, collapse internal whitespace, and treat
  space / hyphen / underscore as the SAME separator. It is a **finite mapping
  onto a closed scale (the I/B/E/S pattern), not a pattern match** -- no
  substring tests, no prefix tests, no synonym expansion. `"Accumulate"`,
  `"Overweight"`, `"Strong Buy!"` and `"N/A"` all resolve to UNKNOWN.
- **Producer constraint (NOT this step).** Pinning the synthesis `action` field
  to a `Literal` is the durable fix and the repo already proves it is possible,
  but it changes an LLM output schema on the analysis path and deserves its own
  research-gated step. Recorded here so 86.22 and any producer step do not make
  the opposite decision.
- **Why not normalise on the write path** (`paper_trader`): it would leave every
  already-persisted row broken and would fix only rows written after the change.
  The read boundary covers both, which is what makes FINDING B fall out for free.

**UNKNOWN is fail-safe by construction and must stay that way:** it is not a
buy, not a sell, and not a downgrade. Criterion 6's observability requirement is
what makes it non-silent.

## 5. Plan

1. **[done]** Research gate -- PASSED, `research_brief_86.20.md`.
2. **[this file]** Contract, written BEFORE any code.
3. **Reproduce FIRST** (criterion 1): drive `decide_trades` with a candidate
   whose recommendation is the literal `Strong Buy` and assert it yields NO buy
   order, while an otherwise identical candidate with `BUY` DOES -- isolating
   the difference to the vocabulary and to nothing else. Record verbatim. Add
   the sell-side reproduction for FINDING A in the same module.
4. **Re-derive the recommendation population** (criterion 2) from the column the
   gate actually reads, and record every distinct value with its count, its
   genuine count, and whether it matches the gate before and after.
5. **Build the canonicaliser** with table-driven tests, then apply it at the
   three read sites.
6. **Observability** (criterion 6): a candidate rejected for an UNRECOGNISED
   recommendation must be distinguishable from one rejected for a recognised
   non-buy such as HOLD. This half changes no decision, so it is NOT flag-gated.
7. **Flag-gate the behaviour change dark** (see §6).
8. **Mutation-test every new guard** (criterion 7), including reverting the
   normalisation, using the in-memory harness pattern from
   `scripts/qa/mutation_matrix_36_17.py` -- never mutating a live production
   file while the backend is armed.
9. Q/A via the Workflow rail; transcribe the verdict verbatim; append
   `harness_log.md`; flip.

## 6. Flag posture -- DARK by default, and the operator owns the arming

This is a money path and the change ARMS orders that do not happen today: new
BUY candidates on the buy side, and new SELL/downgrade exits on the sell side.
Per the standing constraint for this session, **no flag is promoted here**. The
behaviour change ships behind a `paper_*_enabled` flag defaulting False, read
via `getattr(settings, "<flag>", False)` so flag-absent is byte-identical to
flag-OFF, matching the convention already used inside `portfolio_manager` itself.

**The two halves have opposite risk polarity** (FINDING A), so the artifacts must
state plainly what arming each half does, and the operator ask must present them
as a decision rather than as one switch.

## 7. Traps this step must not fall into (measured, from the brief)

- **Do NOT add `"STRONG BUY"` to the set.** That is the instance fix. Fold the
  separator.
- **Do NOT widen the vocabulary.** A gate that accepts more strings than it
  should is worse than one that accepts too few, because it moves money. No
  substring matching -- `conflict_detector` is the cautionary example, and
  `"STRONG_SELL"` contains `"SELL"`.
- **Canonicalise BEFORE validating, never after** (CWE-180).
- **The existing fixtures are green against a dialect production never emits.**
  `test_dod4_tier1_coverage_investment.py` feeds `"STRONG_BUY"`. New tests MUST
  use the PRODUCTION spelling (`"Strong Buy"`), or the suite stays blind in
  exactly the way it is blind today.
- **There is no dedicated `decide_trades` test module today**, so a green run of
  the immutable command proves very little about this path until the new tests
  exist. Say so rather than leaning on it.
- **Do not claim a lost trade or lost P&L.** Reaching the buy-CANDIDATE stage is
  not trading: Risk Judge sizing, sector caps and available cash all sit
  downstream.
- **Do not fix `bias_detector`, `api/portfolio`, `conflict_detector`,
  `outcome_tracker` or `memory` here** -- that is 86.22, filed 2026-08-09. If
  this step's canonicaliser lands, 86.22 REUSES it rather than minting a second.

## 8. References

- `handoff/current/research_brief_86.20.md` (8 read in full, 42 URLs).
- CWE-180 Incorrect Behavior Order: Validate Before Canonicalize -- https://cwe.mitre.org/data/definitions/180.html
- IETF `draft-thomson-postel-was-wrong-02` -- https://datatracker.ietf.org/doc/html/draft-thomson-postel-was-wrong-02
- Anthropic structured outputs -- https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- arXiv 2502.14905 (constrained decoding) -- https://arxiv.org/html/2502.14905
- Google SRE, Monitoring Distributed Systems -- https://sre.google/sre-book/monitoring-distributed-systems/
- Refinitiv I/B/E/S recommendation scale -- https://research2.fidelity.com/fidelity/research/reports/release2/Research/RefinitivIBES.asp
- API enum design -- https://tyk.io/blog/api-design-guidance-enums/
- Internal: `portfolio_manager.py`, `paper_trader.py`, `agents/schemas.py`,
  `api/models.py`, `autonomous_loop.py`, `config/settings.py`.
