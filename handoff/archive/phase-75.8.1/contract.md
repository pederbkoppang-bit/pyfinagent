# Contract — phase-75.8.1: stub-fingerprint + dry-run-label guard for the SECOND gauntlet-report consumer

- **Step id:** 75.8.1 (phase-75 follow-up queue, **P1 latent, money-integrity** — the promotion path; executor: opus-tagged → Main-on-Fable GENERATE; gates opus/max via Workflow)
- **Date:** 2026-07-24
- **Boundary (from step text):** same as 75.8 — evaluator gate thresholds, kill-switch code, DSR/PBO constants, limits.yaml all byte-untouched. Change surface: exactly 4 files (new `backend/backtest/gauntlet/report_integrity.py`, edit `scripts/risk/promotion_gate.py`, edit `backend/autonomous_harness.py`, new `backend/tests/test_phase_75_8_1_harness_consumer.py`). `test_phase_75_promotion_gate.py` stays byte-untouched (C2 requires its 14 tests pass UNCHANGED).

## Research-gate summary (gate PASSED — wf_3d29e4a9-bd7)

Envelope: `tier=moderate, external_sources_read_in_full=7, snippet_only_sources=33, urls_collected=40, recency_scan_performed=true, internal_files_inspected=10, gate_passed=true`. Brief: `handoff/current/research_brief_75.8.1.md`.

Load-bearing findings (each measured, with a step-text correction):

1. **Consumer census (repo-wide):** exactly TWO consumers of `handoff/gauntlet/<strategy>/report.json` — `scripts/risk/promotion_gate.py` (fingerprint check at :118-141) and `backend/autonomous_harness.py::promote_strategy` (NO integrity checks). Writer: `scripts/risk/gauntlet.py`. Only caller of promote_strategy: `scripts/risk/phase4_9_redteam.py:68` → P1 latent, no live impact.
2. **Re-anchored:** `promote_strategy` def at autonomous_harness.py:251 (step text said 258-289; no real drift, HEAD 22409053). Insertion point: AFTER report load (:277), BEFORE `evaluate()` (:278). Existing refusal shape at :269/:281: `_append_blocklist(strategy, reason)` + `raise PromotionBlocked` — the new refusal mirrors it (no exception swallowed; C1).
3. **STEP-TEXT CORRECTION (measure-don't-assert):** promotion_gate.py has ONLY the stub-fingerprint check. Its `dry_run` refs (:190,192,211,213,237) are the CLI's own `--dry-run` no-write flag, NOT a gauntlet-report label check. This step ADDS the `dry_run:true`-label refusal to BOTH consumers via the shared module — "port both rejections" was partially inaccurate.
4. **Ordering constraint (C2):** the composite must check **fingerprint FIRST, dry_run label SECOND** — `test_phase_75_promotion_gate.py:245-257` feeds a report that is BOTH dry_run:true AND stub and asserts the 'stub fingerprint' reason; a label-first composite breaks it. All 14 existing tests verified green under fingerprint-first (every pass-test uses dry_run:False).
5. **Both sub-guards are load-bearing:** skipped regimes carry NEITHER drawdown key (gauntlet.py:77-85) → `None==None` false-positives without the skipped-filter; `all([])==True` is the vacuous trap without the `non_skipped and` empty-guard.
6. **Import safety:** promotion_gate.py already reaches backend.* via sys.path insert (:37-44). A PURE-LEAF report_integrity.py (imports only `typing.Any`, like evaluator.py) has zero cycle risk. Do not touch gauntlet/__init__.py.
7. **Report shape:** writer refuses non-dry_run:true today (gauntlet.py:147; live mode raises NotImplementedError :163) → the label refusal correctly makes NOTHING promotable until a live gauntlet exists. Non-skipped regimes: id/drawdown/bt_drawdown/forced_exits (dry-run sets bt_drawdown==drawdown at :97).
8. **External canon (2025-2026 reward-hacking):** ImpossibleBench 2510.20270 (structural test-access-control; LLM monitors caught only 42-50% of fabrication → the guard must stay a DETERMINISTIC code gate), SpecBench 2605.21384 (fabricated intermediate artifacts = most common exploit), LLMs-Gaming-Verifiers 2604.15149 (extensional checks ignore whether work happened). SSOT/DRY canon (Fowler Pull-Up; Pragmatic Programmer) mandates the single shared predicate at N=2 identical consumers.

## Hypothesis

One pure-leaf `check_report_integrity(report) -> (ok, reason|None)` (fingerprint-first, byte-identical reason string, skipped-filter + empty-guard) imported module-attr-style by BOTH consumers closes the second consumer's fabricated-evidence hole with zero duplicated predicate logic — proven "single implementation" by monkeypatching the shared predicate and observing BOTH consumers flip (a behavioral proof, not a source-scan), with the 14 pre-existing promotion-gate tests untouched and green.

## Plan

1. **New `backend/backtest/gauntlet/report_integrity.py`** (pure leaf, imports only `typing.Any`): `is_dry_run_report()`, `has_stub_fingerprint()` (skipped-filter + `non_skipped and` empty-guard), `check_report_integrity()` — fingerprint checked FIRST, reason string byte-identical to promotion_gate.py:135-137; dry_run label second.
2. **Edit `scripts/risk/promotion_gate.py`**: replace the inline fingerprint block (:118-141) with `report_integrity.check_report_integrity(report)` (module-attr import beside the evaluator import at :44). Gains the label refusal; keeps rc/output contract.
3. **Edit `backend/autonomous_harness.py::promote_strategy`**: integrity check after report load (:277), before evaluate() (:278); on not-ok → `_append_blocklist(strategy, reason)` + `raise PromotionBlocked` (parity with :269/:281 — blocklist-parity choice flagged here for Q/A; raise-only would also satisfy C1).
4. **New `backend/tests/test_phase_75_8_1_harness_consumer.py`** (offline; monkeypatch `_GAUNTLET_ROOT`/`_BLOCKLIST_PATH`/`_HARNESS_LOG` to tmp_path per the phase4_9_redteam pattern): (1) stub-through-promote raises + blocklists; (2) dry_run:true divergent-through-promote raises; (3) realistic divergent dry_run:false PROMOTES through the existing evaluator; (4) all-skipped/empty NOT fingerprinted through promote; (5) anti-fixture-divorce: feed REAL `gauntlet.run(dry_run=True)` output bytes; (6) promotion_gate label coverage (dry_run:true divergent → rc==1); (7) monkeypatch the shared predicate → BOTH consumers change (the C2 single-implementation behavioral proof).
5. **Mutation matrix** (experiment_results per C4 + qa.md §4c): {drop the check call in promotion_gate; drop it in promote_strategy; drop the dry_run branch; drop the skipped-filter; drop the empty-guard; stub the predicate to (True, None)} — each fails ≥1 test through EACH consumer where applicable; plus a fixture/stub mutation.
6. **C5 proof**: `git diff --stat` (4 files only) + empty `git diff -- backend/backtest/gauntlet/evaluator.py backend/governance/limits.yaml` in live_check_75.8.1.md, alongside the verification command exit-0 verbatim.
7. Q/A via qa-verdict Workflow; log; flip; push.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.8.1)

> command: `.venv/bin/python -m pytest backend/tests/test_phase_75_promotion_gate.py backend/tests/test_phase_75_8_1_harness_consumer.py -q`

1. "New backend/tests/test_phase_75_8_1_harness_consumer.py passes offline and asserts promote_strategy refuses (with a logged/returned reason, no exception swallowed into a promote) a report whose every non-skipped regime has bt_drawdown exactly equal to drawdown, AND refuses a report labeled dry_run:true, while a realistic divergent dry_run:false report still promotes through the existing evaluator"
2. "The fingerprint predicate is a SINGLE shared implementation imported by both scripts/risk/promotion_gate.py and backend/autonomous_harness.py (no duplicated predicate logic), and all pre-existing tests in backend/tests/test_phase_75_promotion_gate.py still pass unchanged"
3. "Empty and all-skipped per_regime lists are NOT fingerprinted by the shared predicate (the all([]) trap), proven by test through BOTH consumers"
4. "Mutation matrix in experiment_results.md: dropping the shared predicate call, the dry_run-label check, or the skipped-filter each fails at least one test through EACH consumer (no vacuous guards)"
5. "git diff shows zero edits to evaluator gate thresholds, kill-switch enforcement code, DSR/PBO constants, or limits.yaml values"

live_check spec (verbatim): "handoff/current/live_check_75.8.1.md: verbatim output of this step's verification command (exit 0) + git diff --stat proving the change surface. No UI surface; no flag-gated live-loop behavior expected."

## References

- `handoff/current/research_brief_75.8.1.md` (7 read-in-full: ImpossibleBench arXiv 2510.20270, SpecBench 2605.21384, LLMs-Gaming-Verifiers 2604.15149, Fowler Pull-Up, Pragmatic Programmer SSOT)
- autonomous_harness.py:251-289, promotion_gate.py:118-141, gauntlet.py:77-97/:147/:163, evaluator.py, test_phase_75_promotion_gate.py:245-257, phase4_9_redteam.py:63-68
- 75.8 contract + the anti-vacuous-guard doctrine (qa.md §4c)
