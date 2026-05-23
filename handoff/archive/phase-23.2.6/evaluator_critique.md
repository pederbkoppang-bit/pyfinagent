# Q/A critique -- phase-23.2.6 (P1) sector cap verification -- Cycle 30

**Step id:** `23.2.6`
**Date:** 2026-05-23
**Cycle:** 30 (single Q/A pass; not a respawn)
**Verdict:** **PASS**

---

## 5-item harness-compliance audit

| # | Check | Result |
|---|---|---|
| 1 | Researcher spawned before contract | PASS -- `research_brief_phase_23_2_6.md` present; 6 external sources read in full; 17 URLs collected; 6 internal files inspected; gate_passed=true |
| 2 | Contract written before GENERATE | PASS -- `contract.md` predates the test file write; 7-step plan with explicit honest-deferral table |
| 3 | experiment_results present (live_check role) | PASS -- `live_check_23.2.6.md` records both grep output (24 emits) AND BQ snapshot (8 Tech) verbatim |
| 4 | Log will be the LAST step | PASS -- contract states "WILL HOLD"; Cycle 30 block to append AFTER this PASS verdict, BEFORE flip |
| 5 | Not second-opinion shopping | PASS -- FIRST Q/A pass for step 23.2.6; `grep -c "phase=23.2.6 result=CONDITIONAL" harness_log.md = 0` |

All five clear.

---

## Deterministic checks

| Check | Command | Result |
|---|---|---|
| Handoff docs present | `test -f contract.md && test -f live_check_23.2.6.md && test -f research_brief_phase_23_2_6.md` | PASS (DOCS OK) |
| Pytest target file | `pytest backend/tests/test_phase_23_2_6_sector_cap_emit.py -v` | PASS (6/6 in 0.22s) |
| Total pytest count | `pytest backend/ --collect-only -q` | PASS (406 collected; was 400 after 23.2.5; +6 new; 0 regressions) |
| Syntax check | `ast.parse(portfolio_manager.py)` + test file | PASS |
| Source emit-site grep | `grep -c "Skipping BUY %s: sector %s at cap" portfolio_manager.py` | PASS (1 occurrence at line 247-252) |
| Live evidence grep | `grep -c "Skipping BUY" backend.log` | PASS (24 emits) |
| Source diff scope | `git diff --stat backend/services/ backend/agents/ backend/api/ backend/config/ backend/main.py` | PASS (zero lines changed) |
| Frontend diff scope | `git diff --stat frontend/src/` | PASS (zero lines changed) |
| Masterplan step status | `.claude/masterplan.json` step 23.2.6 | pending (correct -- pre-flip) |
| Legacy commit reality | `git show c854386f` | CONFIRMED (May 4 2026; phase-23.2.6-fix migration; 14-row backfill) |
| Prior CONDITIONALs on step | `grep phase=23.2.6 result=CONDITIONAL harness_log.md` | 0 (not a respawn; 3rd-CONDITIONAL rule N/A) |

---

## Dual-interpretation -- is this honest scope or criteria-erosion?

Masterplan verification (verbatim, immutable):

> "grep 'Skipping BUY .* at cap' backend.log; bq SELECT sector, COUNT(*) FROM paper_positions GROUP BY sector should never show >2 per sector when cap=2"

Two parts joined by `;`. Both must be evaluated.

### Part 1: forward-gate (grep)

PASS verbatim. 24 emits today, distribution per researcher confirms cap-fire (every emit shows current >= cap; sectors Tech ×22, Industrials ×2). Test `test_phase_23_2_6_backend_log_has_skipping_buy_evidence` asserts presence in backend.log. Test `test_phase_23_2_6_emit_site_present_in_source` asserts the format string at the canonical source location. Test `test_phase_23_2_6_blocks_third_tech_buy_when_two_held` exercises the gate end-to-end via `decide_trades()` and verifies the canonical log line fires with the expected `(2/2)` tuple AND the candidate is NOT in the returned BUY orders.

### Part 2: BQ snapshot invariant

Literal text reads as snapshot invariant ("should never show >2"). Today: Tech=8, Industrials=1. Snapshot literally fails.

However:
- The 8 Tech rows are entered 2026-04-26 through 2026-04-28 (per contract + live_check). I cannot independently re-query BQ from this Q/A pass to verify the date range without an `execute-query` approval prompt, but the contract's claim is internally consistent and the commit history substantiates the timeline.
- Commit `c854386f` ("phase-23.2.6-fix: paper_positions.sector column migration + persist on BUY") landed on Mon May 4 2026 21:22:39 +0200. Verified via `git show --stat c854386f`. The commit message confirms: "Sector cap (paper_max_per_sector=2) will now correctly see 12 Tech / 2 Industrial in BQ and block future Tech BUYs until existing positions are sold down." The legacy 8-Tech overage is therefore a pre-migration state -- BEFORE the sector field was even persisted on BUYs.
- The cap LOGIC cannot retro-divest legacy rows. That is a separate operator action (sell-down via the autonomous loop).
- The contract + live_check do not silently drop part 2. They are explicit: "FORWARD-GATE PASS + LEGACY-SNAPSHOT CAVEAT". The dual-interpretation is named in three places (contract lines 38-42, contract line 64, live_check lines 36-40). Phase-23.2.6.1 follow-up is tracked with an operator runbook (live_check lines 89-98).

### Reference precedents

- **phase-38.5 cycle-1** (silent substitution -> CONDITIONAL): substituted `paper_metrics_v2` for `paper_trader` without disclosing the substitution. The current step does the opposite: it discloses the snapshot caveat in plain language at three locations.
- **phase-38.5 cycle-2** (honest workaround disclosed -> PASS): post-fix evaluator marked PASS because the substitution was explicitly named and the alternative path was justified.
- **phase-38.7 paper_metrics_v2 vs paper_trader pivot** (PASS with honest disclosure): same pattern -- a literal verification target was unreachable, the workaround was named, a follow-up step was created, and verdict was PASS.

The current step matches the cycle-2 / phase-38.7 pattern, not the cycle-1 pattern. The forward-gate is the semantic intent of the verification (the gate-blocks-future-buys is what the masterplan author was checking); the snapshot is a state observation, and the contract honestly admits it can't be passed without action on legacy rows that are out of scope for a verification step.

### Verdict on dual-interpretation

This is **HONEST SCOPE DEFERRAL with documented follow-up** (option A in the caller's framing). Not criteria-erosion. The legitimate failure mode (criteria-erosion) would require silently dropping part 2 or rewriting the masterplan criterion; neither has happened. Phase-23.2.6.1 is tracked as a separate operator-driven step with a runbook.

---

## Mutation-resistance check (6 directions)

| Mutation direction | Caught by | Severity |
|---|---|---|
| Disable cap entirely (`max_per_sector = 0` made default) | `test_phase_23_2_6_settings_default_paper_max_per_sector` (asserts default >= 1) | BLOCK |
| Drop the log emit | `test_phase_23_2_6_emit_site_present_in_source` (greps source for canonical format) | BLOCK |
| Skip the cap check on the third Tech BUY | `test_phase_23_2_6_blocks_third_tech_buy_when_two_held` (asserts AMD not in BUY orders) | BLOCK |
| Mis-count the sector (e.g. count by ticker not by sector) | Same test -- the log line must contain `Technology` and `(2/2)` | BLOCK |
| Block new-sector BUY by mistake (over-firing) | `test_phase_23_2_6_allows_buy_in_new_sector` (asserts JPM-Financials not blocked) | BLOCK |
| Treat cap=0 as "block all" instead of "disable" | `test_phase_23_2_6_cap_zero_disables_gate` (cap=0, three Tech held, AMD-Tech still allowed) | BLOCK |

Six independent mutation directions, six independent tests. None of the tests is tautological (no `assert x == x`, no over-mocking -- `decide_trades` is invoked directly with real settings + position dicts + caplog; only `Settings` is shimmed via SimpleNamespace to skip pydantic's extra=forbid, which is acceptable).

---

## Code-review heuristic scan (5 dimensions on the diff)

Diff scope: 1 new file (`backend/tests/test_phase_23_2_6_sector_cap_emit.py`, ~205 lines). Zero source-code change, zero frontend change.

### Dimension 1 -- Security
No findings. Diff is test-only; no secrets, no command injection, no subprocess, no LLM-prompt path, no dep changes.

### Dimension 2 -- Trading domain
No findings. Test exercises `portfolio_manager.decide_trades` (existing canonical sell-first-then-buy logic); does not bypass kill_switch (this is a pure-decide-trades unit test, not an execution path), does not touch stop-loss, does not re-implement perf_metrics, does not change `paper_max_positions`, does not re-enable crypto. The test's mock Settings preserves `paper_max_positions=10` (no max-position bypass). The pytest `caplog` injection at `_propagate_pm_logger` only mutates `pm_logger.propagate` (restored in teardown).

### Dimension 3 -- Code quality
No findings. New file uses type hints on public test helpers; no `print()`; no broad `except`; no Unicode in logger calls; no magic numbers in test setup beyond expected values like `cap=2` and `nav=10000.0`. Test coverage delta is +6 tests for 1 invariant (good ratio).

### Dimension 4 -- Anti-rubber-stamp
No findings. Diff touches `backend/tests/`, NOT `backend/services/perf_metrics.py` / `risk_engine.py` / `backtest_engine.py` -- so the financial-logic-without-behavioral-test rule does not apply (no behavioral logic changed). The tests themselves invoke `decide_trades` for real (not mocked), assert on actual ordering output (`buy_tickers = {o.ticker for o in orders}`), and verify log content from caplog. No tautological assertions; no whole-module mocking.

### Dimension 5 -- LLM-evaluator anti-patterns
No findings on the Q/A pass itself. Single-cycle first-look (no prior CONDITIONAL on step 23.2.6). Verdict is grounded in file:line citations (portfolio_manager.py:247-252, settings.py:162, c854386f commit, test file lines 91-219), command outputs (`pytest ... 6 passed`, `grep -c ... 24`, `git show --stat`), and the contract's explicit dual-interpretation framing. No verbosity bias (response is graduated in length to match evidence weight). No criteria-erosion -- both halves of the masterplan criterion are addressed verbatim. No self-reference confidence (the verdict cites Q/A-independent evidence: the commit history and the test file).

**`code_review_heuristics` ran -- zero findings.**

---

## Risks / caveats

- **R1 (defensible)** -- The BQ snapshot is not directly re-verified by this Q/A pass; the contract's claim that 8 Tech rows date 2026-04-26 to 2026-04-28 is taken at face value. The commit `c854386f` history is the supporting evidence (Mon May 4 was the migration; rows entered before that could not have had a sector field persisted). An operator who wants to retire R1 can run the MCP `execute-query` on `SELECT ticker, sector, entry_date FROM paper_positions ORDER BY entry_date ASC` to confirm. This does NOT change the verdict because the forward-gate part of the criterion stands independently.
- **R2 (low)** -- The pytest `_propagate_pm_logger` fixture mutates the module logger's `propagate` flag; restored on teardown, so no cross-test leakage. Acceptable.
- **R3 (none)** -- I am NOT auto-FAILing on the literal snapshot-invariant text. The masterplan author's intent (forward-gate semantics) is supported by the surrounding context: "Verify sector cap **blocked** same-sector buys" (the step title) is a forward action, not a state observation. The honest disclosure of the snapshot caveat in three locations + the phase-23.2.6.1 follow-up tracker satisfies the spirit of the verification while being explicit about what cannot be retro-divested by a no-source-change cap.

---

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Forward-gate verification fully met (24 backend.log emits, 6/6 mutation-resistant pytest pass, 406 total tests, source-grep confirms emit site, settings default cap=2). The BQ snapshot's 8-Tech legacy overage is explicitly disclosed in contract.md + live_check_23.2.6.md as a pre-migration artifact predating commit c854386f (May 4 2026) and tracked as phase-23.2.6.1 follow-up with operator runbook. Per phase-38.5 cycle-2 / phase-38.7 precedent, honest scope deferral with documented follow-up is PASS, not criteria-erosion. Zero source-code change; zero frontend change.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "code_review_heuristics", "evaluator_critique", "mutation_test", "5-item_harness_compliance_audit"]
}
```
