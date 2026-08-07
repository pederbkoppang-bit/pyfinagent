# Experiment results — Step 83.1: design pack + pre-registration (DESIGN ONLY)

Date: 2026-08-07 (autonomous drain, cycle 171). Contract: `contract_83.1.md`.

## What was built (no production signal code — design artifacts + their tests)

1. **`backend/backtest/experiments/preregistration_phase83_ranking.json`** — the machine-readable pre-registration consumed by 83.1.1/83.5: ranking criteria (measured-V DSR with the variance-not-sd caveat; checked-PBO gate reading PromotionGate at runtime; tie-breakers), trial-budget cap 45 with rationale, label horizon **126 trading days** (distinct from the engine's 135-calendar-day 1.5× purge horizon), entry rule (acceleration, never birth/confirmation), the **pre-registered artifact-population rule** (globs `*_phase_83_*.json` + `phase83*.json`; naive `*83*` REJECTED — measured 71 false positives), kill-rule pointer, append-only amendment policy, and the pre-registered expected outcome (descope; a failing 83.5 is the planned-for null).
2. **`handoff/current/research_brief_phase83.md`** — the design pack landing the 2026-08-04 corpus: gate thresholds READ FROM SOURCE with the promotion_gate.py:37 contrast quoted (criterion 2); design decisions with their grounding numbers; the 7-candidate cost table with closed-vocabulary classifications, 0 unclassified (criterion 7); 7 negative-evidence failure modes each with source + number (criterion 3); the four-row reference-case table with measured coverage windows and DERIVED-labelled cost-to-hold cells (criterion 4); killed options; corpus limits incl. the completeness-critic's materially_flawed verdict and the 13-verified-vs-87-self-reported source count; **anchor corrections** (purge horizon at backtest_engine.py:274×:962, NOT :665; `variance_of_srs` 0.5 is a VARIANCE — sqrt at analytics.py:429); the recorded PREREGISTRATION_SHA256; and the criterion-1 envelope with **`coverage.dry: null` + stated reason** (the 2026-08-04 run persisted no coverage object — grep-verified; null is the honest value, contract D2) and every number's derivation rule stated (D3).
3. **`backend/tests/test_phase_83_1_design_pack.py`** — 7 tests, one per criterion. Conventions from `test_phase_82_6_bridge_design.py`: HTML comments STRIPPED before matching (the 82.6 stub trap), anti-stub size floor, runtime `PromotionGate()` attribute comparison (C2 fails if the pack hardcodes a value the module doesn't carry). The ordering guard uses strict `<` on `st_mtime_ns` (equal PASSES — fresh clones stamp identically; no pytest.skip escape); the C6 mutation test writes a glob-matching backdated artifact, **asserts the population sees it first** (an empty population cannot be vacuously green), asserts the SAME helper the C5 test calls then fails, and unlinks in `finally`.

## Verification (verbatim, re-derived after the final edit)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_1_design_pack.py -q
.......
7 passed in 0.02s
```

Lint gate over the git-derived changed-file scope: **"All checks passed!"**. Hash comparison and the artifact-population `ls` are in `live_check_83.1.md` (computed == recorded: `a22cb12f...`; population empty under the pre-registered globs).

## Follow-up — cycle 2 (2026-08-07, after Q/A CONDITIONAL wf_e69008fa-faa)

Cycle-1 verdict (verbatim in `evaluator_critique_83.1.md`): all 7 criteria MET (20 of the Q/A's 22 independent mutants killed incl. the max_pbo=0.50 shape; the one survivor proven equivalent); capped by four evidence-quality WARNs. All four closed:

1. **Ordering guard blind to canonical result_store naming** → the ranking file gained an append-only AMENDMENT (its own policy followed: amendments[] entry + hash recomputed and re-recorded in the pack — new hash `7b346492...`): a CONTENT rule (`phase_tag == 'phase_83'` at top level) as the binding backstop, plus a binding naming requirement on 83.5 and successors. New test `test_c6b` executes the Q/A's exact escape shape (backdated `20260810T120000Z_a1b2c3d4.json`), asserts the content rule catches it, asserts the ordering guard then FAILS, and runs the untagged negative control (no over-matching).
2. **urls_collected 85 → 79** — corrected under the internally consistent http-only-distinct rule (which also reproduces the pack's own 63 and 16), and the "16 vs 22" alternative-rule disagreement is repaired: both rules agree at 16; the 22 was arithmetic on the wrong 85. The origin figure in `research_brief_83.1.md` §A9 is annotated with a [MAIN CORRECTION].
3. **C7 unnumbered-row escape** → the row filter now counts EVERY table row in section 4 except header/separator, so a candidate cannot escape the census by omitting its ordinal.
4. **Negative-evidence label drift (entries 1-2)** → corrected against the corpus verbatim: entry 1 now reads "-3.1%/yr risk-adjusted after fees; ~-6%/yr first five years; FFC4 -3.24 vs -0.24%/yr; 0.13%/yr fee gap"; entry 2 relabelled as Cohen-Frazzini ECONOMIC-LINK (customer-supplier) predictability decay, VW 1.30→0.62%/mo (t=1.54), 52% decline.

Re-verification after all fixes (every number re-derived after the final edit): immutable command → **8 passed in 3.10s** (7 criterion tests + c6b); hash computed == recorded (`7b346492...`); lint gate **"All checks passed!"**; envelope urls_collected reads 79.

## Follow-up — cycle 3 (2026-08-07, after Q/A CONDITIONAL #2 wf_f30208e5-6c7; the first cycle-2 spawn wf_87c3c3f6-1ba dropped its return = NO VERDICT)

The completed cycle-2 verdict confirmed all four cycle-1 WARNs closed under independent execution (its own recall battery sampled a live filename, not the author's shape) and capped on ONE finding: my 85→79 correction swept the two locations the cycle-1 Q/A NAMED instead of the DERIVED population — the retired 85 survived in the contract (D3), twice more in the gate brief (:340, :636 — the :636 site was found only by my own derived sweep, not named by any Q/A), the gate brief's JSON envelope summary, and a self-contradicting honesty-ledger line here. All closed append-only in cycle 3: [MAIN CORRECTION] annotations at contract D3, gate brief :340 and :636, a header cover over the gate brief's JSON summary (kept parseable per the standing memory rule), and the ledger line below rewritten to agree with the Follow-up above. The `feedback_measure_dont_assert_claims` lesson ("a Q/A's list of instances is a sample, never the scope — grep the whole repo for the claim") applied literally this time: the population was derived by grep before annotating. Suite re-run after the annotations: **8 passed in 3.08s** (annotations touch no tested artifact content — the pack itself was already correct at 79).

## Honesty ledger (the gaps the corpus does not close, all recorded IN the pack rather than papered over)

- `coverage.dry` = null with reason (never fabricated true).
- Source counts: 87 self-reported vs 13 auditor-verified — both stated.
- `snippet_only_sources` = 16 under the stated rule (cycle-2 correction: both candidate rules agree at 16 under the corrected urls_collected=79; the retired '22' was arithmetic on the wrong 85 and the pack no longer states it).
- Cost-to-hold cells labelled DERIVED (no per-case figure exists in the corpus); the four reference cases were never traced end-to-end and the pack says so with the three corpus records that admit it.
- The completeness-critic's `materially_flawed` verdict + 10 unresearched angles listed so no reader infers corpus completeness.
- Two Lens-7 counter-evidence findings are read_in_full=false and are NOT counted toward criterion 3's floor.

## Built-in mutation coverage

Criterion 6 IS the mutation test for the ordering guard and runs green in the suite (backdated artifact → C5 helper fails). The C5 hash assertion is self-mutation-resistant by construction: any post-hoc edit to the ranking file makes recorded ≠ actual (the append-only policy requires recomputing the recorded hash in the same commit, which is itself the audit trail).

## Files changed

`backend/backtest/experiments/preregistration_phase83_ranking.json` (new), `handoff/current/research_brief_phase83.md` (new — the criterion-1 deliverable), `backend/tests/test_phase_83_1_design_pack.py` (new). Handoff: `contract_83.1.md`, `research_brief_83.1.md`, `live_check_83.1.md`, this file.
