# Contract -- Step 75.8: promotion-gate stub-fabrication refusal + governance-limits divergence observability

- **Step id**: 75.8 (phase-75, Audit75 S8) -- P0, executor opus-tier
- **Date**: 2026-07-23
- **Author**: Main (Claude Code session; GENERATE executed by Main per the executor tag -- toughest-work-on-best-model directive)
- **BOUNDARY (from step text)**: kill-switch/stops/sector-caps/DSR/PBO gate code and thresholds byte-untouched -- stub-refusal and WARNING-only additions; historical_macro frozen, no backtest runs.

## Research-gate summary (gate PASSED)

Workflow `wf_26a12896-e0c` (researcher role, opus/max, tier=complex, structured-output launch).
Envelope: `external_sources_read_in_full=7, snippet_only_sources=18, urls_collected=35, recency_scan_performed=true, internal_files_inspected=20, gate_passed=true`.
Brief: `handoff/current/research_brief_75.8.md` (177 lines, write-first).

**Step-text corrections adopted from research (binding for this contract):**
1. **Priority framing**: gap6-01=P1, gap6-10=P3, gap3-02=P1 per `audit_phase75/confirmed_findings.json`; P0 is the bundle priority. None touches live capital today (gauntlet `run` has zero importers; the alloc-init path cannot advance an existing stage; 4%-vs-2% is a governance-enforcement gap, not active loss). All three are latent/defense-in-depth fixes.
2. **Docstring debt**: guarding the promotion_gate writes makes the module docstring (lines 3-7, "--dry-run ... ensures optimizer_best.json has allocation_pct set") FALSE -- it must be rewritten in the same change.
3. **Scope gap**: `backend/autonomous_harness.py::promote_strategy` (:258-289) is the OTHER gauntlet-report consumer (and the one with the eventual real caller); it gets NO fingerprint/dry_run guard in this step. Per `feedback_queue_discovered_defects_in_masterplan` it is queued as its own research-gated step **75.8.1** -- NOT silently folded in (the immutable criteria scope the fingerprint check to promotion_gate).
4. **Unit mismatch**: limits.yaml stores FRACTIONS (0.02, 0.10); settings stores PERCENTS (4.0, 10.0). divergence.py MUST normalize (fraction x100) before comparing. Only daily-loss diverges (2 governed vs 4 live); trailing-dd MATCHES (10 == 10) and must be reported NON-divergent.
5. **Anchor drift**: the main.py lifespan governance-load block is :277-286 in the current file (audit's :252-260 is stale).

**Key research findings load-bearing for the design:**
- Stub reports pass all four `backend/backtest/gauntlet/evaluator.py` hard gates BY CONSTRUCTION (`bt_drawdown == drawdown` -> ratio 1.0; `forced_exits=0`; `breaches=0`). The live `optimizer_best.json` already carries a stub-derived `gauntlet_report_hash`.
- No python code imports `gauntlet.run` -- the NotImplementedError breaks zero importers.
- REGIMES catalog = 7 regimes, exactly 1 `intraday_only` (skipped in dry-run) -> the fingerprint check must filter skipped regimes AND guard the empty list (`all([]) is True`).
- `limits_schema.load()` (lru-cached frozen pydantic) is the sanctioned value API; `lint_limits_usage.py` is a source AST scanner, not a runtime comparator -- divergence.py is non-duplicative.
- External consensus (7 full reads incl. Bailey/Borwein/Lopez-de-Prado PBO paper): refuse-over-fabricate; dry-run = zero side effects with EVERY writer gated (angular-cli #6810 precedent: "reassuring message while still writing" is a P1 bug class); warn-before-enforce for config-drift rollout (terraform-plan idiom).

## Hypothesis

The promotion gate can be made fail-safe against fabricated gauntlet evidence, and the governance-vs-runtime limits divergence made visible, WITHOUT touching any live gate threshold, kill-switch line, or limits.yaml value -- via (a) two independent refusal guards in gauntlet.py, (b) a consumer-side stub-fingerprint rejection + true no-write --dry-run in promotion_gate.py, and (c) a pure, WARNING-only divergence checker wired into lifespan -- all provable offline by a new pytest file with a mutation matrix in which every guard can fail.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.8)

verification.command:
```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_promotion_gate.py -q
```

1. "New backend/tests/test_phase_75_promotion_gate.py passes offline and asserts gauntlet run with dry_run=False raises NotImplementedError (or provably refuses to write a dry_run:false report), and that no code path can emit a report labeled dry_run:false from _run_regime_stub"
2. "Test asserts promotion_gate rejects a fixture report whose every regime has bt_drawdown exactly equal to drawdown (stub fingerprint) when gauntlet evidence is required, while a fixture with realistic divergent values still passes shape-validation"
3. "Test asserts promotion_gate --dry-run leaves a temp copy of optimizer_best.json byte-identical (both the allocation-stage init and the gauntlet-stamp writers are guarded)"
4. "Divergence checker returns the (settings_value, governed_value) pairs and flags 4.0-vs-2.0 daily-loss divergence with CURRENT repo values; it is invoked from lifespan as a WARNING log only -- test proves it raises nothing and mutates nothing"
5. "git diff shows zero edits to evaluator gate thresholds, kill-switch enforcement code, DSR/PBO constants, or limits.yaml values (diff file list in experiment_results.md); handoff/current/governance_limits_divergence_75.md exists with the drafted GOV-LIMITS-DECIDE operator token"
6. "All touched scripts remain runnable: python -c ast.parse passes on gauntlet.py, promotion_gate.py, and the new divergence module"

verification.live_check: "handoff/current/live_check_75.8.md: verbatim output of this step's verification command (exit 0) + git diff --stat proving the change surface; for any flag-gated live-loop behavior an ON-vs-OFF $0 diff, and for UI-touching parts a Playwright/curl capture. Findings covered: gap6-01, gap6-10, gap3-02"

## Plan steps

1. **gauntlet.py (gap6-01, two independent guards)**
   - Guard (i): top of `run()` -- `if not dry_run: raise NotImplementedError(...)` (real engine wiring pending; message names the dry-run escape). Raised BEFORE any RNG/report work, so nothing is written.
   - Guard (ii), defense-in-depth: factor the report write into `_write_report(report, out_dir)` which raises `RuntimeError` unless `report.get("dry_run") is True` (explicit `if`+`raise`, NOT `assert` -- assert strips under `-O`). All report data flows from `_run_regime_stub`, so no path can emit a `dry_run:false` report.
   - `main()` lets the NotImplementedError propagate (loud fail-safe; no swallowing).
2. **promotion_gate.py (gap6-01 consumer side + gap6-10)**
   - Stub fingerprint: after `_load_gauntlet_report`, build `non_skipped = [r for r in per_regime if not r.get("skipped")]`; reject (`{blocked: true, reason: stub fingerprint...}`, exit 1) when `non_skipped` is non-empty AND every entry has `bt_drawdown == drawdown` (exact equality is the point -- the stub copies the same float). Empty/all-skipped lists are NOT fingerprinted (they fail differently or pass through to the evaluator).
   - Dry-run guards: wrap BOTH write paths (`update_optimizer_best` alloc-init; gauntlet stamp) in `if not args.dry_run:`; the dry-run branch prints the would-be mutation (terraform-plan idiom). Guard the post-write re-read for the no-file case.
   - Rewrite the module docstring so --dry-run is documented as strictly no-write.
3. **backend/governance/divergence.py (gap3-02, observability-only)**
   - Pure `compute_divergence()` -> list of pair dicts `{name, settings_value_pct, governed_value_pct, divergent}` using `limits_schema.load()` (sanctioned cached API) + `settings.paper_daily_loss_limit_pct` / `paper_trailing_dd_limit_pct`, normalizing governed fractions x100. No I/O, no mutation.
   - `log_divergence_warnings()` -- never raises (internal try/except, fail-open), logs one ASCII WARNING per divergent pair, INFO when clean.
4. **main.py lifespan** -- invoke `log_divergence_warnings()` right after the :277-286 governance block, inside its own try/except (mirrors the existing fail-open discipline). WARNING-only; no gating, no enforcement.
5. **handoff/current/governance_limits_divergence_75.md** -- all six limits.yaml entries vs their runtime counterparts (the four without settings counterparts documented as UNMAPPED/no runtime enforcement consumer), the daily-loss divergence flagged, + drafted operator token `GOV-LIMITS-DECIDE` (which value binds; lint WARN->fail flip stays operator-gated).
6. **backend/tests/test_phase_75_promotion_gate.py** -- offline-only; imports the two CLIs via `importlib.util.spec_from_file_location` and monkeypatches module Path constants to tmp_path (NEVER touches the real optimizer_best.json). Covers criteria 1-4 + ast.parse (criterion 6). Mutation matrix (anti-vacuous-guard doctrine, incl. fixture mutations): M1 drop guard (i); M2 drop guard (ii); M3 drop fingerprint check; M4 drop skipped-filter; M5 drop non-empty check (all([]) trap); M6 unguard alloc-init write; M7 unguard stamp write; M8 drop unit normalization (trailing-dd false positive); M9 fixture mutation -- one regime with bt != dd must NOT be fingerprinted.
7. **Queue 75.8.1** (discovered defect, own masterplan step per operator rule): fingerprint/dry_run guard for `backend/autonomous_harness.py::promote_strategy` -- the second gauntlet-report consumer. `status: pending`, research-gated, executor-tagged.
8. **live_check_75.8.md** -- verbatim pytest output (exit 0) + `git diff --stat`. No UI surface; no flag-gated live-loop behavior beyond the WARNING log (a $0 no-op by construction -- documented as such).

## Explicitly NOT in scope

- `backend/backtest/gauntlet/evaluator.py` (thresholds immutable -- untouched)
- `backend/governance/limits.yaml` values (GPG-tag governed -- untouched)
- Kill-switch enforcement code (`paper_trader.py::check_and_enforce_kill_switch` etc. -- untouched)
- DSR/PBO constants -- untouched
- `autonomous_harness.py` (queued as 75.8.1, not edited here)
- Any live gauntlet/backtest run (historical_macro frozen)

## References

- `handoff/current/research_brief_75.8.md` (envelope + 35 URLs; 7 read in full incl. Bailey/Borwein/Lopez-de-Prado PBO SSRN 2326253, angular-cli #6810, terraform plan / ansible --check drift-detection docs, King "Parse, don't validate", Shingo poka-yoke)
- `handoff/current/audit_phase75/confirmed_findings.json` (gap6-01, gap6-10, gap3-02)
- Anthropic harness-design (file-based handoffs; fail-safe scaffolding)
- CLAUDE.md Harness Protocol + `.claude/rules/research-gate.md`
