STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.88
WRITTEN: 2026-08-16T10:21:13Z

# Q/A write-first record -- step 86.88 (cycle 1)

## Prior-attempt evidence
- `qa_wip.py 86.88 --spawned-at 2026-08-16T10:21:13Z`: source_present=true,
  identity_checked=true, attempt_number=1 (status "ok", not a lower bound),
  prior_attempts=0, records_retained=1 (gauge, = my own record).
- `verdict_history_86_21.py --step 86.88 --evidence-only`: status
  `no_rows_for_step`, verdicts `(none)`.
- Cross-check: attempt_number(1) - me = 0 prior attempts == ledger 0 rows. No
  staleness contradiction. Cycle 1; nothing to verdict-shop.

## A. Harness compliance (5 items) -- CLEAN
1. research-gate-before-contract: research_brief_86.88.md 38,587 bytes,
   brief_status COMPLETE, external_sources_read_in_full 15 (floor 5),
   urls_collected 30 (floor 10), recency_scan true, coverage.dry (audit-class),
   gate_passed true. mtime 12:08:43 < contract 12:11:20. OK
2. contract-before-generate: contract 12:11:20 < checker 12:13:13 < tests
   12:17:59 < autonomous_loop.py 12:19:46 (post-commit restore; sha == HEAD). OK
3. experiment_results_86.88.md present (252 lines) + live_check_86.88.md. OK
4. log-last: masterplan 86.88 status=`pending`; `grep -cF phase=86.88
   handoff/harness_log.md` = 0. OK
5. no-verdict-shopping: cycle 1. OK

## B. Deterministic
- IMMUTABLE: `bash -c 'source .venv/bin/activate && python
  scripts/qa/verify_lite_risk_seam_86_86.py'` -> **exit 0**, checks emitted 9
  (PASS 9 / FAIL 0). Branch fires on real matches at 3214/3219/3448/3453.
- pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q -> **72 passed**.
- ruff F821,F401,F811 over the COMMIT-derived scope (3 .py files, xargs-quoted,
  non-empty asserted) -> **All checks passed! exit 0**.
- ast.parse OK x3; `import backend.services.autonomous_loop` OK.
- Scope: `git diff HEAD` on the 3 subject files = EMPTY (tree == HEAD).
  Commits 03386529 + 786b5a55 touch only 3 code files + 4 handoff artifacts
  (+ CHANGELOG via hook). No .env / settings / masterplan / flag / threshold.
- sha256: claimed pre-fix `5b714a9e...` == `git show 03386529^:...` EXACT.
  Working tree == HEAD blob `644c751b...`. Restore byte-identical CONFIRMED.
- Unrelated dirty files (sovereign_api.py + 5 frontend) are a peer session's
  `1y` window work; NOT in either commit. Not this step's doing.

## C. Independent re-derivations (all reproduce)
- Enumeration: HEAD has 16 ast.Name refs; **exactly 4** `dict()` Call routes
  (3214/3219/3448/3453). Parent commit: **12** refs, 1 Assign, 7 subscript-reads
  at 2359 2362 2392 2394 2402 2403 2461, 4 Call routes at 3177/3182/3411/3416 --
  byte-for-byte the artifact's table. REPRODUCES.
- Criterion-7 table re-derived by loading BOTH module versions in memory:
  all 7 rows identical pre vs post; moved = NONE. REPRODUCES.
- Test count: 51 `def test_` at parent -> 61 at HEAD (+10); 62 -> 72 collected.

## D. MY OWN mutation matrix (in-memory module injection; disk sha verified
   unchanged after every cell -- the tree was never written)
   CONTROL through the harness: rc=0, collected=72, **72 passed**.
| cell | mutation | result |
|---|---|---|
| Q1 | delete the whole-default seam guard | KILLED (5 failed) |
| Q2 | `==` -> `is` (identity; defeats dict() copy) | KILLED (4 failed) |
| Q3 | N1 pre-mangle @ Claude producer call | **KILLED (1 failed)** = criterion 3 |
| Q4 | N1 pre-mangle @ Gemini producer call | **KILLED (1 failed)** |
| Q4b | N1 pre-mangle @ BOTH | KILLED (2 failed) |
| Q5 | guard fires unconditionally (over-fire) | KILLED (3 failed) |
| Q6 | guard returns 0.0 instead of the default | KILLED (5 failed) |
| Q7 | reword the log phrase, behaviour identical | KILLED (4 failed) -- brittle-but-not-vacuous |
| Q8 | one route adds a key -> value equality fails | KILLED (1 failed) |
| Q11 | weaken condition to reasoning-only match | **SURVIVED (72 passed)** |
| Q12 | weaken to pct+decision (REACHABLE differential) | KILLED (2 failed) |
| Q13 | subset match ignoring `reasoning` | **SURVIVED (72 passed)** |
Q11/Q13 differentials are only on effectively-unreachable inputs (a judge
emitting the default's exact reasoning string). The REACHABLE weakening (Q12) is
killed by `test_a_real_judge_verdict_is_NOT_recorded_as_absent`.

## E. THE CENTRAL MEASUREMENT (criterion 5)
Driving the shipped code:
```
_build_lite_risk_assessment(dict(_LITE_RISK_DEFAULT)) -> recommended_position_pct = 3.0
_resolve_position_pct(that persisted record) -> PositionVerdict(kind='SIZE', pct=3.0)
_resolve_position_pct(a REAL 3% judge's record) -> PositionVerdict(kind='SIZE', pct=3.0)   IDENTICAL
```
And through the real route into `decide_trades`, both flag states:
```
judge 0%          binding=False orders=0    binding=True orders=0
judge 3%          binding=False orders=1 BUY binding=True orders=1 BUY
whole-default     binding=False orders=1 BUY binding=True orders=1 BUY
```
=> post-fix a judge FAILURE STILL persists as SIZE 3.0, not ABSENT, in the
persisted record and downstream, and still emits the same BUY. The ONLY thing
the fix changes is a `logger.warning`. The code never constructs an ABSENT
verdict; it early-returns the default float BEFORE `_resolve_position_pct`.
Contradicted claims: contract Sec6 P1 "the resolver receives ABSENT rather than
SIZE"; experiment_results Sec4 "resolves ABSENT"; Sec9 "now recorded as ABSENT
rather than as an explicit SIZE".
Criterion 7 does NOT force this: an ADDITIVE provenance key on the persisted
risk_assessment (downstream reads only `recommended_position_pct`) would make
judge-failure distinguishable while moving no order.

## F. CRITERION 2 -- the order assertion does not exist
`decide_trades` appears in the new `TestLiteRouteEndToEnd` class exactly once --
inside a DOCSTRING (line 920). No test calls it. Yet:
- class docstring (856-858): "and asserts the downstream ORDER outcome" -- FALSE
- test name `test_judge_zero_pct_survives_the_route_and_produces_no_order` --
  the "produces no order" half is never asserted
- artifacts: `grep decide_trades` over contract/experiment_results/live_check =
  NO MENTION; Sec10 "Stated gaps" omits it (Main disclosed it to the evaluator
  in the spawn prompt, not in the durable record).
I drove it myself: judge 0% -> **0 orders** under both binding states. Substance
holds; the assertion and the disclosure do not.

## G. EVIDENCE STALENESS -- the post-fix matrix is not shipped-tree evidence
live_check Sec4 / experiment_results Sec7: "CONTROL: 69 passed"; every row sums
to 69 (1+68, 2+67, 15+54, 14+55). The shipped suite is **72**. So the 7-cell
matrix ran against a 69-test tree -- 3 tests short. Corroborated: my Q2-shaped
cell kills 4 tests at HEAD where M4 reported 1, and mtimes show the test file
(12:17:59) and commit (12:18:26) both postdate the matrix. The 3 absent tests
are the criterion-6 route/ABSENT-log tests. I re-ran the load-bearing cells at
HEAD (Q3/Q4/Q4b) -- they KILL -- so criterion 3 holds; the ARTIFACT's matrix
does not correspond to what shipped.

## H. STATED BOUND IS NARROWER THAN REALITY
Sec10/Sec8: "covers `dict()`, `copy()` and `deepcopy()` call shapes". MEASURED
against the shipped `or_default_sites`:
```
dict(_LITE_RISK_DEFAULT)          seen=True     deepcopy(_LITE_RISK_DEFAULT)  seen=True
copy.deepcopy(_LITE_RISK_DEFAULT) seen=False    copy.copy(...)                seen=False
{**_LITE_RISK_DEFAULT}            seen=False    dict(**_LITE_RISK_DEFAULT)    seen=False
_LITE_RISK_DEFAULT.copy()         seen=False
```
Only the BARE-NAME forms are covered; the idiomatic `copy.deepcopy(...)` is not.
4 blind shapes, 1 disclosed. MITIGATION: the runtime value-equality guard fires
for ALL six shapes (measured), so the money path is protected; only the
"a fifth route announces itself" tripwire is weaker than advertised.

## I. Criterion-by-criterion
1 MET. 2 NOT MET as worded (drive yes, order assertion no; 3 false claims).
3 MET (re-derived at HEAD; artifact matrix stale -- see G). 4 MET (premise
correction legitimate; branch shown firing on 4 real matches -- NOT a
reinterpretation to avoid deletion). 5 MET at the seam / claim reach
OVERSTATED (see E). 6 MET (calls["n"]==2 proven load-bearing by Q8). 7 MET and
strengthened (order-level, both flag states). 8 MET (checker got STRICTER,
8->9 checks).

## J. Answers to Main's four questions
A) Legitimate. The BoolOp branch does fire on `x or _LITE_RISK_DEFAULT`;
   widening is strictly additive plus a new count assertion. Deleting a working
   branch would have removed coverage.
B) The value-equality hole is real but unreachable and harmless (resolves to the
   same 3.0) -- driven, not reasoned. Note `dict()` is a SHALLOW copy so
   `risk_limits` is shared by reference with the module-level default; no site
   mutates it and the producer copies it, so no live hazard today.
C) Yes, they drive production; `calls["n"]==2` is load-bearing (proved by Q8,
   not by reading).
D) 8th cells found: Q11 and Q13 SURVIVE (near-equivalent). The important eighth
   finding is not a mutant -- it is section E.

## Verdict formed: CONDITIONAL (worst-of-3-lenses: correctness ~PASS,
## reproduce CONDITIONAL (G), scope-honesty CONDITIONAL (E/F/H)).

COMPLETED: 2026-08-16T10:34:14Z
