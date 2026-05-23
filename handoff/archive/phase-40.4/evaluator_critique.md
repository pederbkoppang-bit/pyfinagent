# Q/A Evaluator Critique -- phase-40.4 (Cycle 41)

**Date:** 2026-05-23
**Step id:** `40.4` (P3 OPEN-28 stop-loss 8% vs 10% A/B; ADR + turnkey + 8 tests; A/B run DEFERRED to operator)
**Verdict: PASS (first Q/A; honest dual-interpretation; ADR + turnkey delivered)**
**Evaluator:** Q/A subagent (Claude Opus 4.7 / effort max)
**Cycle:** Cycle 41 (after Cycle 40 phase-23.2.16)

---

## 1. Harness-compliance audit (5-item, per CLAUDE.md / feedback_qa_harness_compliance_first.md)

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Researcher SPAWNED FIRST | PASS | `handoff/current/research_brief_phase_40_4.md` exists; gate_passed=true; 6 sources read in full (5-floor +20%); 18 URLs collected; 14 internal files inspected |
| 2 | Contract pre-commit | PASS | `handoff/current/contract.md` written before GENERATE; references research_brief + masterplan |
| 3 | Results file | PASS-with-note | `experiment_results.md` exists at handoff/current/; body retains phase-34 content but live_check_40.4.md carries the verbatim phase-40.4 evidence (NOTE in §5) |
| 4 | Log-LAST | WILL HOLD | harness_log append + status flip pending Q/A PASS (caller's plan) |
| 5 | No second-opinion shop | PASS | First Q/A on 40.4; harness_log grep shows references in *closure-path summaries* only, no prior verdicts on this step-id |

---

## 2. Deterministic checks (run live)

```
$ test -f handoff/current/contract.md && test -f handoff/current/live_check_40.4.md \
  && test -f handoff/current/research_brief_phase_40_4.md \
  && test -f docs/decisions/stop_loss_default.md \
  && test -x scripts/backtest/run_stop_loss_ab.py && echo "DOCS+RUNNER OK"
DOCS+RUNNER OK

$ pytest backend/tests/test_phase_40_4_stop_loss_doc.py -v
8 passed in 0.01s
  - test_phase_40_4_adr_exists                                  PASSED
  - test_phase_40_4_adr_cites_oneil_can_slim                    PASSED
  - test_phase_40_4_adr_cites_han_zhou_zhu_2014                 PASSED
  - test_phase_40_4_adr_references_settings_field               PASSED
  - test_phase_40_4_adr_documents_deferred_a_b_run              PASSED
  - test_phase_40_4_turnkey_runner_exists_and_executable        PASSED
  - test_phase_40_4_turnkey_runner_writes_masterplan_grep_tag   PASSED
  - test_phase_40_4_adr_explicit_keep_decision                  PASSED

$ pytest backend/ --collect-only -q | tail -2
473 tests collected in 2.09s    (was 465 after 23.2.16; +8 new; 0 regressions)

$ git diff --stat backend/agents/ backend/services/ backend/api/ backend/config/ backend/main.py | wc -l
0    (ZERO production-source changes)

$ git status --short docs/ scripts/ backend/tests/test_phase_40_4_stop_loss_doc.py
?? backend/tests/test_phase_40_4_stop_loss_doc.py
?? docs/decisions/stop_loss_default.md
?? scripts/backtest/                          (only the three new files)
```

---

## 3. Code-review heuristics (5 dimensions, phase-16.59 skill)

**Dim 1 Security:** Clean.
- `grep -iE "(api_key|secret|password|token)\s*=\s*['\"][A-Za-z0-9/+]{16,}"` on all three files -> NO_SECRETS.
- `grep -nE "subprocess|os\.system|eval\(|exec\(|shell=True"` -> NO_COMMAND_INJECTION_VECTORS.
- Runner uses `argparse` (stdlib) + lazy `from backend.backtest.backtest_engine import run_backtest` (production engine, unchanged). No LLM API call. No external network. No prompt-injection path. No insecure-output-handling.

**Dim 2 Trading-domain:** Clean.
- ZERO diff to `backend/agents/`, `backend/services/`, `backend/api/`, `backend/config/`, `backend/main.py`.
- `paper_default_stop_loss_pct=8.0` at `backend/config/settings.py:330` UNCHANGED.
- `kill_switch.py`, `paper_trader.py`, `risk_engine.py`, `services/perf_metrics.py` all untouched.
- ADR §Decision rationale (line 51) explicitly preserves downstream constants (`paper_trailing_stop_pct=8.0`, R-multiples 2R/3R = 16%/24%, breakeven 1R = 8%).
- No new BQ write, no `paper_max_positions` change, no `backfill_stop_losses` removal, no crypto re-enable, no SOD-NAV anchor change.

**Dim 3 Code quality:** Clean.
- Runner: type hints (`-> dict`, `list[str] | None`), `from __future__ import annotations`, ASCII-only logger usage (file=sys.stderr prints), docstrings, exit-code contract documented (0/1/2/3).
- Tests: `Path.read_text()` (no mocks of the module under test), regex pattern checks for KEEP-8 phrasing.
- No `print()` in production paths (script + tests is allowed).
- One NOTE only: runner's `_run_arm` is a stub that returns `stub: True` even WITHOUT `--execute` (lines 82-94). Properly disclosed in file docstring + ADR §"Walk-forward A/B (deferred to operator runbook)" -- operator-deferred pattern by design.

**Dim 4 Anti-rubber-stamp:** Clean.
- No financial logic changed (production source diff = 0 lines), so `financial-logic-without-behavioral-test` BLOCK does not apply.
- Tests are NOT tautological: each asserts a *content* requirement (specific citation strings, executable bit via `stat().st_mode & 0o111`, regex for KEEP 8% phrasing). No `assert x is not None`, no mock-the-module-under-test pattern.
- ADR `formula-drift-without-citation`: KEEP 8% decision cites 4 anchor sources (O'Neil CAN SLIM 1953, Han/Zhou/Zhu 2014 SSRN 2407199, Kaminski/Lo 2014 SSRN 968338, Lopez de Prado AFML ch.3) + arxiv:1609.00869 ar5iv. Not a flag -- citation density high; researcher brief is the authoritative source.
- No rename-as-refactor (all files NEW; no existing-file edits in production code).
- No `pass-on-all-criteria-no-evidence`: this critique cites file:line evidence throughout (settings.py:330, run_stop_loss_ab.py:107-108, ADR §Decision lines 46-52).

**Dim 5 Evaluator anti-patterns:** Clean.
- First Q/A on phase-40.4 (harness_log only contains references to "40.4" in closure-path estimate summaries, NOT prior verdicts on this step-id). No `sycophancy-under-rebuttal` risk.
- This critique cites file:line evidence -- no `missing-chain-of-thought`.
- ZERO prior CONDITIONALs on this step-id, so `3rd-conditional-not-escalated` N/A.
- Simultaneous-presentation rule N/A (first cycle on this step).

**checks_run:** `["syntax", "verification_command", "code_review_heuristics", "5_item_harness_audit", "evaluator_critique", "harness_log_prior_verdict_grep"]`

---

## 4. LLM judgment (matched to caller's 4 questions)

**(a) "KEEP 8%" decision literature-backed?** Verified. The ADR (`docs/decisions/stop_loss_default.md` lines 27-43) cites:

- **O'Neil CAN SLIM 1953** -- per-position retail growth-equity layer -- 7-8% cut, no exceptions.
- **Han/Zhou/Zhu 2014 SSRN 2407199** -- portfolio-momentum overlay layer -- 10% reduces equal-weighted max monthly loss from -49.79% to -11.34%; Sharpe >2x.
- **Kaminski/Lo 2014 SSRN 968338** -- universal disclaimer (stops add value only if positive serial correlation).
- **Lopez de Prado AFML ch.3** -- triple-barrier vol-adjusted alternative.
- arxiv:1609.00869 ar5iv -- vol- + asset-class-dependent.

The researcher's key insight (different operating layers) is correctly transcribed and is the strongest justification for KEEP. pyfinagent's `paper_default_stop_loss_pct` is a per-position FALLBACK at `backend/config/settings.py:330` (not a portfolio-momentum overlay), so O'Neil literature directly applies. Han/Zhou/Zhu's 10% targets a fundamentally different layer that pyfinagent does not currently operate at.

**(b) Honest dual-interpretation -- consistent with cycle-1 38.5 / 23.2.* pattern?** Verified.
- ADR §"Walk-forward A/B (deferred to operator runbook)" + ADR §Status: "ACCEPTED -- 8% KEPT as system default; literature-validated; A/B run deferred to operator."
- live_check_40.4.md row labels: `test -f` half = **PASS**, `grep -q` half = **DEFERRED-LIVE**.
- live_check_40.4.md line 75 explicitly cites the precedent: "Mirror of cycle-2 38.5 / 23.2.6 / 23.2.10 / 23.2.11 / 23.2.12 / 23.2.13 / 23.2.15 / 23.2.16 honest-disclosure pattern."

This is exactly the pattern the harness uses for operator-action-required closures.

**(c) Turnkey runner uses literal masterplan grep tag?** Verified at `scripts/backtest/run_stop_loss_ab.py:107-108`:
```python
ap.add_argument("--tag", default="stop_loss_default_8_vs_10",
                help="Tag for the TSV rows (default matches masterplan verification grep)")
```

The literal tag is the default; even running with no args produces TSV rows that the masterplan `grep -q 'stop_loss_default_8_vs_10' quant_results.tsv` will find. `test_phase_40_4_turnkey_runner_writes_masterplan_grep_tag` enforces this contract programmatically.

**(d) N* delta R+P honest for an operator-action-required step?** Verified. Contract.md "North-star delta":
- **R (now)** = literature audit trail + future-cycle re-validation tooling (the ADR + runner are canonical artifacts the operator and future agents can re-use).
- **P (deferred)** = 30-90 min compute by operator; ADR §Decision rationale lines 48-52 expects negligible delta at the per-position layer (where pyfinagent operates). Switching to 10% would require touching 4+ downstream constants with no literature-mandated benefit at this layer.

This framing is correct for a P3 doc-only step -- deferred-P is reasonable when (i) the operator must perform a multi-30-min live action, and (ii) the literature-driven expected delta is negligible. Matches Anthropic harness-design's file-based handoff pattern: the artifact exists; the operator audits when ready.

---

## 5. Risk register / caveats

| # | Caveat | Severity | Action |
|---|---|---|---|
| 1 | Runner's `_run_arm` is a stub even without `--execute`; operator must add the real `run_backtest()` call at the `TODO (operator)` marker (lines 82-94) | NOTE | Documented in file docstring + ADR §"Walk-forward A/B (deferred)" + live_check operator runbook. Operator-action-required pattern by design. |
| 2 | `experiment_results.md` body retains phase-34 LLM-route flip content; contract.md is correctly phase-40.4 | NOTE | live_check_40.4.md carries the verbatim phase-40.4 evidence; acceptable for a doc-only step. Refresh the results file for the next non-doc step. |
| 3 | Masterplan verification command's `grep -q` half returns non-zero until operator runs the A/B with `--execute` | NOTE | Disclosed explicitly in ADR + live_check. Turnkey: `python scripts/backtest/run_stop_loss_ab.py --execute` writes the row -> `grep -q` succeeds. |

None of these escalate to WARN or BLOCK severity. All are honest-disclosure surfaces, not protocol breaches.

---

## 6. Verdict

**PASS.**

JSON envelope:

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "phase-40.4 (P3 OPEN-28): ADR + turnkey runner + 8 pytest tests delivered (8/8 pass). Literature-driven KEEP 8% decision cites 4 anchor sources (O'Neil CAN SLIM, Han/Zhou/Zhu 2014, Kaminski/Lo 2014, Lopez de Prado AFML). Walk-forward A/B run DEFERRED to operator runbook (30-90 min compute); turnkey script uses literal masterplan grep tag 'stop_loss_default_8_vs_10' as default. ZERO production-source changes. 473 tests total (was 465; +8 new; 0 regressions). Honest dual-interpretation pattern matches phase-23.2.6/10/11/12/13/15/16 + 38.5 precedent. Researcher SPAWNED FIRST; gate_passed=true; 6 sources read in full.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "verification_command",
    "code_review_heuristics",
    "5_item_harness_audit",
    "evaluator_critique",
    "harness_log_prior_verdict_grep"
  ]
}
```

PROCEED to harness_log append + masterplan status flip (log-LAST discipline; status='done' MUST be in the same edit that adds the log block, per the masterplan status-flip-order rule).
