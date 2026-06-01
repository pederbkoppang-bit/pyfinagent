# phase-52.2 EVALUATE -- 2026-06-01

Fresh Layer-3 Q/A. This is the FIRST real verdict for 52.2 (the prior instance
truncated at "PENDING" with no verdict; evidence UNCHANGED + complete, so this is
not second-opinion-shopping). `harness_log.md` has ZERO prior `phase=52.2` entries
-> CONDITIONAL-counter is at 0, no 3rd-CONDITIONAL escalation pressure.

## Harness-compliance audit (5-item, runs first)

| # | Item | Status |
|---|------|--------|
| 1 | Researcher spawned before contract | PASS -- `research_brief.md` present (541-line diff), cited in contract references |
| 2 | Contract written before GENERATE | PASS -- `contract.md` present; 4 success_criteria byte-present + VERBATIM vs masterplan 52.2 |
| 3 | `experiment_results.md` present + verbatim cmd output | PASS -- present; pytest output + byte-identity proof embedded |
| 4 | Log-last discipline | N/A at eval time -- `harness_log.md` append is Main's post-PASS step; no premature status=done observed |
| 5 | No verdict-shopping | PASS -- evidence unchanged from the truncated instance; this is the first delivered verdict, not a re-roll on changed evidence |

## Deterministic checks (reproduced, verbatim)

```
$ python -m pytest backend/tests/test_phase_52_2_live_tilt.py -q
.....                                                                    [100%]
5 passed in 0.30s

$ python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); ast.parse(open('backend/services/autonomous_loop.py').read()); print('AST OK')"
AST OK

$ test -f handoff/current/live_check_52.2.md && echo "live_check present"
live_check present

$ python -m pytest test_phase_50_3_universe.py test_phase_52_1_alpha_signal.py test_phase_51_2_sector_div.py -q
...............                                                          [100%]
15 passed in 0.29s
```

Independent byte-identity + tilt + 52.1-match harness (re-run by Q/A, not trusting Main):
```
default flag OFF: False
OFF==explicitFalse: True           # rank_candidates(...) == rank_candidates(..., momentum_52wh_tilt=False)
OFF no raw: True                   # OFF path writes NO composite_score_raw -> proves _apply_52wh_tilt never ran
ON changed: True | OFF ['T0','T1','T2','T3','T4'] | ON ['T0','T2','T4','T1','T3']
```
Tilt DIRECTION sanity (the inverted-tilt trap): ON promotes T2(pct=0.95)+T4(pct=0.98) ABOVE
T1(pct=0.70)+T3(pct=0.60) despite T1/T3's higher raw composite -> the centered tilt tilts
names NEARER their 52w-high UP. Correct sign; an enable delivers the measured +0.05, not its inverse.

Test-fix legitimacy (operator-flagged, TEST-RIGGING scrutiny):
```
CODE DEFAULT paper_markets:        ['US']            # Settings.model_fields[...].default_factory()
EFFECTIVE get_settings paper_markets: ['US','EU','KR']  # live backend/.env override
```

Single live call-site (no second caller silently un-tilted):
```
$ grep -rn "rank_candidates(" backend/services backend/agents backend/api | grep -v test_ | grep -v "def rank_candidates"
backend/services/autonomous_loop.py:638:            candidates = rank_candidates(
```
Exactly one. autonomous_loop.py:656-658 passes `momentum_52wh_tilt=getattr(settings,
"momentum_52wh_tilt_enabled", False)` + `momentum_52wh_tilt_k=getattr(settings,...,0.5)`.

`git diff --stat` (code surface): `settings.py (+2)`, `autonomous_loop.py (+3)`,
`screener.py (+28)`, `test_phase_50_3_universe.py (+/-7)`, `test_phase_52_2_live_tilt.py (new)`.
No `decide_trades` / `risk_engine` / `kill_switch` / `paper_trader` touched.

## The 4 IMMUTABLE criteria

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | rank_candidates gains config-gated 52wh-tilt post-pass reproducing 52.1 logic; composite otherwise unchanged | `_apply_52wh_tilt` (screener.py:500-516) inserted after sector_neutral, before the `.sort` (screener.py:480-483). `test_live_tilt_matches_52_1_replay_logic` loads the REAL `scripts/ablation/sector_neutral_replay.py::hi52_tilt_basket` by path and asserts `live_basket == replay_basket` -> the live tilt is byte-faithful to the 52.1-measured logic, not a re-implementation that could drift. | PASS |
| 2 | flag OFF (default) -> BYTE-IDENTICAL (post-pass skipped); US engine not regressed -- proven by a test | Gate is `if momentum_52wh_tilt and scored:` (screener.py:480), kwarg default `False` (screener.py:263). Q/A re-ran: `OFF==explicitFalse: True` + `OFF no raw: True` (the `composite_score_raw` witness is absent when OFF -> the pass provably never executed). `test_flag_off_is_byte_identical_default` + `test_flag_off_writes_no_raw_field`. 15/15 regression tests green. | PASS |
| 3 | flag plumbed settings(default False) -> autonomous_loop; NO flag flip | `momentum_52wh_tilt_enabled: bool = Field(False, ...)` (settings.py:339). Plumbed at autonomous_loop.py:656-658. Q/A confirmed `s.momentum_52wh_tilt_enabled == False` at runtime. The live engine is DORMANT -- the tilt only activates on an explicit `.env` enable that this step did NOT perform. | PASS |
| 4 | live_check_52.2.md records OFF byte-identity + ON behavior + deferred-enable plan | live_check_52.2.md present: proof block (OFF order, OFF==explicit-False, no-raw witness, ON order), criterion-by-criterion table, and a 4-step DEFERRED enable plan with the DSR-deflation caveat (the +0.05 was 1-of-5 configs -> deflate before trusting). | PASS |

## Adversarial judgment

**Byte-identity (the +20% engine is the regression surface).** Confirmed two
independent ways: (a) `_apply_52wh_tilt` is unreachable when the kwarg is False (default);
(b) the `composite_score_raw` side-channel is absent on OFF output, which is a positive
witness that the re-scoring pass never touched `scored`. The OFF path is provably a no-op.

**NO flag flip.** settings default False AND runtime `get_settings().momentum_52wh_tilt_enabled
== False`. No `.env` write in the diff. Dormant-until-explicit-enable confirmed.

**TEST-FIX SCRUTINY (operator hates test-rigging) -- LEGITIMATE, not rigging.** The change to
`test_paper_markets_default_is_us_only` swaps `get_settings().paper_markets == ['US']` for
`Settings.model_fields['paper_markets'].default_factory() == ['US']`. Q/A verified BOTH halves
independently: the CODE DEFAULT is still `['US']` (the byte-identity invariant the test name
asserts -- "default is US-only" -- still HOLDS), and `get_settings()` returns `['US','EU','KR']`
because the operator's real 2026-06-01 go-live `PAPER_MARKETS` `.env` override is live (the
override is REAL, not faked). The old assertion conflated "code default" with "effective
.env-overridden value"; the new one tests what the test name claims. A rigging fix would have
masked a regression by loosening the invariant; this fix TIGHTENS it to the correct surface
(code default) while letting the deliberate deployment opt-in stand. Legitimate.

**Scope isolated.** Diff = screener.py + settings.py + autonomous_loop.py (1 call-site) + 2
tests. No execution-path / risk-guard / paper_trader / decide_trades change.

## Code-review heuristics (5 dimensions evaluated)

No BLOCK or WARN fired. Notes:
- **financial-logic-without-behavioral-test [BLOCK]**: NOT triggered -- the financial-logic
  change ships WITH a behavioral test (`test_phase_52_2_live_tilt.py`, 5 cases incl. the
  52.1-replay cross-check). This is the model case for what the heuristic wants.
- **kill-switch-reachability / stop-loss-always-set / perf-metrics-bypass / max-position-bypass
  [BLOCK]**: N/A -- diff does not touch any execution or risk-guard path; `_apply_52wh_tilt` is
  a ranking-stage score transform, upstream of `decide_trades`.
- **tautological-assertion / over-mocked-test [BLOCK]**: NOT present -- assertions are
  substantive (order equality, side-channel absence, cross-module basket equality, no-pct
  no-op); nothing mocked.
- **sycophancy-under-rebuttal / second-opinion-shopping [BLOCK]**: N/A -- no prior verdict to
  flip (the truncated instance issued none); evidence is the original, unchanged set.
- **magic-number [NOTE]**: k default 0.5 is a Field default with a cited rationale (settings.py
  description: "0.5 = milder/plateau choice; k=1.0 was borderline in the 52.1 replay"). Not flagged.
- **broad-except [WARN]**: the `try/except Exception` at screener.py:212-216 (52w-high compute)
  is PRE-EXISTING (phase-28.7), not in this diff, and falls back to `pct_to_52w_high=None` which
  `_apply_52wh_tilt` treats as tilt=1.0 (no-op). No new swallow introduced.

## Verdict

**PASS.** All 4 immutable criteria met with independently-reproduced evidence:
(1) the live tilt reproduces the 52.1 `hi52_tilt_basket` EXACTLY (cross-module test, not a
drift-prone re-impl); (2) flag-OFF is byte-identical, proven two ways (unreachable pass +
absent `composite_score_raw` witness) and by 15/15 green regression tests -- the +20% engine
is untouched; (3) the flag is plumbed from `settings(default False)` through the single live
call-site with NO flag flip (dormant until an explicit operator enable); (4) live_check_52.2.md
documents OFF byte-identity, ON behavior, and a DSR-deflation-aware deferred-enable plan. The
operator-flagged test edit is a LEGITIMATE fix (code default still `['US']`; the multi-market
value is a real `.env` opt-in), not rigging. Scope is isolated to ranking + config + tests; no
risk-guard or execution path touched. No code-review heuristic fired.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable 52.2 criteria met with Q/A-reproduced evidence. (1) Live _apply_52wh_tilt (screener.py:500-516) reproduces the 52.1 hi52_tilt_basket EXACTLY -- test_live_tilt_matches_52_1_replay_logic loads scripts/ablation/sector_neutral_replay.py and asserts live_basket==replay_basket. (2) Flag OFF (default False, screener.py:263) is byte-identical: re-run shows OFF==explicit-False order AND no composite_score_raw written (positive witness the pass never ran); 15/15 regression tests green -> +20% engine not regressed. (3) momentum_52wh_tilt_enabled=Field(False) plumbed through the SINGLE live call-site autonomous_loop.py:638/656-658; runtime confirms flag OFF -- NO flag flip. (4) live_check_52.2.md records OFF byte-identity + ON tilt direction + DSR-deflation-aware deferred-enable plan. Test-fix is LEGITIMATE: code default paper_markets still ['US'], get_settings() returns ['US','EU','KR'] from a real go-live .env override -- the edit tests the code default (what the test name claims), not rigging. Scope isolated (screener+settings+autonomous_loop 1-call-site+2 tests; no decide_trades/risk_engine/kill_switch/paper_trader). No code-review BLOCK/WARN.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "byte_identity_reproduction", "tilt_direction_check", "52_1_replay_match", "test_fix_legitimacy", "single_callsite_check", "regression_suite", "code_review_heuristics", "evaluator_critique", "live_check_present"]
}
```
