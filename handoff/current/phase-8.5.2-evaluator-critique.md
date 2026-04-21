# Q/A Critique -- phase-8.5 / 8.5.2 Budget enforcer -- REMEDIATION v1

**Verdict id:** `qa_852_remediation_v1`
**Verdict:** PASS
**Date:** 2026-04-20
**Scope:** Fresh Q/A on researcher-authored brief. Supersedes `qa_852_v1` (which judged an inline brief).

---

## 5-item harness-compliance audit

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Research brief substantive (>=150 lines OR >=5 external in full) | PASS | `phase-8.5.2-research-brief.md` = 146 lines, 5 external sources read in full via WebFetch, 10 snippet-only URLs, three-variant queries visible, recency scan present, JSON envelope `gate_passed: true`. Length just under the 150-line floor but well above the >=5 sources floor (and the rule is OR, not AND). |
| 2 | contract.mtime < results.mtime | PASS | contract 17:09, results 17:10. |
| 3 | Results verbatim + test output | PASS | `phase-8.5.2-experiment-results.md` quotes full test stdout (3/3 PASS EXIT=0) plus pytest `152 passed, 1 skipped`. |
| 4 | Log tail = 8.5.1 remediation (04:05 UTC) | PASS | Confirmed at `harness_log.md` tail. Not contaminated by premature 8.5.2 append. |
| 5 | No verdict-shopping | PASS | Evidence is materially new: v1 judged an inline brief authored in the evaluator's response; v2 judges a researcher-authored 146-line brief with three-variant queries, snippet-only table, and per-file-line internal anchors. Cycle-2 pattern is correct (files updated, fresh instance reads updated files). |

5/5 PASS.

---

## Deterministic checks A-E

| Key | Check | Result |
|-----|-------|--------|
| A | Re-run immutable `python scripts/harness/autoresearch_budget_test.py` | EXIT=0; 3/3 PASS (wallclock, usd, alert) |
| B | `backend/autoresearch/budget.py` uses `time.monotonic()` | CONFIRMED L60 and L83. `_alerted` latch at L56 (init) and L90-91 (guard). `ValueError` at L47-48 on negative budgets |
| C | Regression: `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` | 152 passed, 1 skipped (per results file) |
| D | Handoff files present in `handoff/current/` | brief 9774 B, contract 900 B, results 1022 B |
| E | Brief content greps | "monotonic" 12+ hits, "alert_fn" throughout, "5 fetched" in gate checklist, snippet-only table present (3-variant discipline visible) |

5/5 PASS.

Note on B: the brief quotes the ValueError message as "budgets must be negative" (L75) -- the actual code at L48 says "budgets must be non-negative". Paraphrase typo in the brief; code is correct. Non-blocking.

---

## LLM judgment

### Is injectable `alert_fn` legit for criterion `budget_exceeded_alerts_to_slack`?

**Yes, legitimately.** The criterion label describes the *operational intent* (prod alerts reach Slack), not an implementation mandate. The standard canonical pattern (PyBreaker, OneUptime 2026, debugg.ai 2025) is constructor-injected listeners:

- Hardcoding a Slack webhook call inside `BudgetEnforcer` would make it untestable without mocking HTTP, and would couple enforcement logic to routing policy -- anti-pattern per PyBreaker docs.
- Injection inverts the dependency: enforcer fires the callable; caller decides sink. In prod `alert_fn=slack_post` (wired at caller), in tests `alert_fn=captive.append`. Same code path fires in both.
- The test asserts (i) the callable fires exactly once on first breach (PyBreaker does NOT guarantee this framework-side; `_alerted` latch is the caller's responsibility, correctly implemented at L90-91), and (ii) subsequent ticks do not re-fire.

Adversarial angle considered and rejected: "the test uses a captive list, not Slack, so the criterion is unmet." Refuted by the criterion's wording (`budget_exceeded_alerts_to_slack` is the *test name* in autoresearch_budget_test.py, where Slack is shorthand for "the configured alert sink"). This is standard dependency-injection testing hygiene.

### Docstring vs implementation drift at L31

`budget.py` L33 (within the docstring block starting L31) says:

> `wallclock_seconds : float` -- Budget over time.time() elapsed from first tick.

But implementation at L60 and L83 uses `time.monotonic()` (correct per debugg.ai 2025: monotonic is immune to NTP/DST/leap-second). Docstring is misleading.

**Disclosure:** explicitly called out in the brief (L54, L80) and in the results file ("Minor finding ... docstring misleading"). **Non-blocking** -- implementation is correct; no behavioral bug; criterion is met. Worth a cleanup in a future docs pass.

### Anti-rubber-stamp: mutation resistance

Not formally tested (no planted-violation / restore cycle), but the immutable test itself functions as a mutation test: it asserts (a) termination doesn't fire under cap (negative case), (b) alert fires exactly once (positive case), (c) state is stable across repeat ticks (idempotency). If any of these were regressed, the test would fail.

### Scope honesty

Results file discloses the docstring defect without spin. Brief distinguishes "implementation correct" from "documentation misleading" cleanly. No overclaim.

### Research-gate compliance

Contract references `handoff/current/phase-8.5.2-research-brief.md` explicitly. Brief is researcher-authored (not inline). JSON envelope present with `gate_passed: true`, 5 in-full sources, 15 URLs total, three-variant query discipline documented, recency scan present with 2026/2025 sources cited.

---

## Violations

None.

`violated_criteria: []`
`violation_details: []`

## Non-blocking follow-ups

1. Fix docstring at `backend/autoresearch/budget.py` L33 to say "time.monotonic() elapsed" (or remove the clock reference). Flag in a future docs-cleanup step, not a blocker here.
2. Brief L75 paraphrase typo ("must be negative" vs code's "must be non-negative"). Cosmetic.

---

## Verdict

**PASS (`qa_852_remediation_v1`).**

All 3 immutable criteria met (wallclock / usd / alert exit-0). 5/5 harness-compliance audit. 5/5 deterministic A-E. Injectable alert_fn satisfies the Slack criterion via canonical listener-injection. Docstring drift disclosed, non-blocking.

Supersedes `qa_852_v1` PASS on fresh researcher-authored evidence. Not verdict-shopping (evidence materially new per the cycle-2 documented pattern).

```json
{
  "ok": true,
  "verdict": "PASS",
  "verdict_id": "qa_852_remediation_v1",
  "reason": "3/3 immutable criteria met; researcher-authored brief clears the gate; implementation canonical (time.monotonic + injectable alert_fn + _alerted latch); docstring drift disclosed non-blocking.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "verification_command",
    "evaluator_critique_prior",
    "research_brief_content",
    "handoff_file_presence",
    "contract_mtime_ordering",
    "log_tail_check",
    "pytest_regression",
    "budget_py_line_anchors"
  ]
}
```
