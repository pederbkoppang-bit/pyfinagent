# Contract — Step 83.1: market-news design pack + pre-registration (DESIGN ONLY)

- **Step id:** 83.1 (P1, phase-83; depends_on: none in-plan; consumes the completed 2026-08-04 8-lens research. Prerequisite OF P0 83.1.1.)
- **Tier (named field):** T3 — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max).
- **Date:** 2026-08-07, autonomous drain, cycle 171

## Research-gate summary

`handoff/current/research_brief_83.1.md` — gate_passed: **true** (6 external sources read in full / 45 URLs / recency scan / 14 internal files; envelope on rail AND in brief; the researcher disclosed running over the simple-tier depth guide because the internal inventory was the bulk of the work). Decisive findings:

1. **Criterion-5 population trap**: `backend/backtest/experiments/results/` holds 438 files, 71 matching a naive `*83*` glob for unrelated reasons (exp ordinals, hash prefixes) with mtimes ALL older than any ranking file this step can create → a naive glob makes criterion 5 permanently red. The repo's phase-tag convention (`<TS>Z_phase_83_*`) has ZERO phase-83 hits today. The artifact-population RULE must itself be pre-registered.
2. **Gate thresholds at runtime**: `gate.py:19-30` frozen dataclass PromotionGate (min_dsr 0.95, max_pbo 0.20, min_pbo_trials 10); `promotion_gate.py:37` PBO_CEILING=0.5 is a DIFFERENT decision (PSR-parity staging, strict-< comparator). The 2026-08-04 spawn prompt itself carried the 2.5x error — quoted as provenance, not carried forward.
3. **Anchor corrections** (both the phase-83 step texts and auto-memory carry the wrong line): the 135-day purge horizon lives at `backtest_engine.py:274` (holding_days=90) × `:962` (1.5x), NOT `:665`; and `compute_deflated_sharpe`'s `variance_of_srs` default 0.5 is a **VARIANCE** (`sqrt` applied at `analytics.py:429`) — the "unmeasured V" all lenses argued about is the repo default, and reading 0.5 as an sd is off by √2. Feeds 83.1.1 directly.
4. **Eight gaps enumerated** where the corpus does not contain what a criterion literally asks (coverage.dry never recorded; no per-case cost-to-hold; reference cases never traced end-to-end; self-reported vs verified source counts; two defensible snippet-only counts; the completeness-critic's MATERIALLY_FLAWED verdict with 10 unresearched angles). Each is handled honestly below, none fabricated.
5. **Design-test conventions**: copy `test_phase_82_6_bridge_design.py` — anti-stub size check, concrete symbols, and STRIP HTML COMMENTS before matching (the 82.6 stub trap applies verbatim to C3/C4/C7).
6. **Pre-registration practice** (6 sources read in full): thresholds written before results; trial count recorded; plan dated/ordered relative to results (the mtime guard mechanises "locking timing"); null results planned for — `go_no_go="descope"` and a pre-registered expectation that 83.5 fails are compliant, not defeatist; append-only amendment clause instead of pretended unamendability.

## Immutable success criteria (verbatim from `.claude/masterplan.json` 83.1)

1. "handoff/current/research_brief_phase83.md exists and ends with a parseable JSON envelope reporting tier, external_sources_read_in_full, snippet_only_sources, urls_collected, recency_scan_performed, internal_files_inspected, coverage.dry and gate_passed; a test asserts the envelope parses as JSON, that external_sources_read_in_full is at least 5, and that recency_scan_performed is true"
2. "the brief records the gate thresholds READ FROM SOURCE, and a test asserts the recorded min_dsr, max_pbo and min_pbo_trials each equal the corresponding attribute of backend/autoresearch/gate.py::PromotionGate at runtime, failing if the brief hardcodes a value that module does not carry"
3. "the brief contains a negative-evidence section naming at least three documented failure modes, each with its source and its number, and a test asserts the section is non-empty and that the count of named failure modes with an attached numeric figure is at least three"
4. "the brief contains a reference-case table with exactly four rows covering COVID/pharma, AI-datacenter/memory, Ukraine/defense and Iran-US/oil, each row carrying a non-empty free-data-source cell naming the source whose recorded coverage window contains that case and a non-empty cost-to-hold cell; a test asserts no cell in those two columns is empty for any of the four rows"
5. "the pre-registered ranking for the gate step is written to a separate machine-readable file whose SHA-256 is recorded in the brief, and a test asserts the file exists, that its recorded hash matches the file's actual hash, and that no phase-83 backtest artifact exists with an mtime earlier than that file"
6. "the pre-registration ordering guard is mutation-tested: creating a phase-83 backtest artifact with an mtime earlier than the ranking file makes the criterion-5 test FAIL"
7. "every candidate design enumerated in the brief carries an explicit survives-costs classification drawn from a closed vocabulary recorded in the brief, and a test asserts the count of candidates with no classification is exactly zero"

**Verification command (immutable):** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_1_design_pack.py -q`

**live_check (immutable):** envelope transcribed verbatim; SHA-256 printed alongside the recorded value; `ls -la` full timestamps of the pre-registration file and every phase-83 backtest artifact → `handoff/current/live_check_83.1.md`.

## Explicit decisions

- **D1 — the phase-83-artifact population rule is itself pre-registered** inside the ranking file: glob patterns `backend/backtest/experiments/results/*_phase_83_*.json` (the repo's phase-tag convention) + `backend/backtest/experiments/results/phase83*.json`. The naive `*83*` glob is REJECTED with the measured 71-false-positive basis recorded in the pack. Population is empty today (measured); the criterion-6 mutation test writes a matching artifact and asserts the glob sees it FIRST, so emptiness never makes criterion 5 vacuous.
- **D2 — `coverage.dry` is reported as `null` with a stated reason**: the 2026-08-04 run was briefed audit-class (K=2) but persisted no coverage object anywhere in the three raw JSONs (grep-verified). Null + reason is honest; `true` would be fabrication.
- **D3 — envelope numbers carry their derivation rules**: external_sources_read_in_full = 87 (sum of per-lens self-reports; labelled — the citation auditor verified 13 primary in full); snippet_only_sources = 16 (rule: objects with read_in_full=false; the alternative rule giving 22 is stated); urls_collected = 85 (re-derived unique source_url values). [MAIN CORRECTION 2026-08-07, cycle 2: 85 did not reproduce -- the measured value under the internally consistent http-only-distinct rule is 79 (= 63 full-read + 16 snippet-only, zero overlap/residue); the 'alternative rule giving 22' was arithmetic on the wrong 85 and both snippet rules agree at 16. The pack envelope records 79.]
- **D4 — cost-to-hold cells are DERIVED** from the Lens-4 design-level cost table with the holding assumption stated, and labelled DERIVED (no per-case figure exists in the corpus). The pack states the four cases were never traced end-to-end (three independent corpus records say so).
- **D5 — closed classification vocabulary** for C7: `{survives_costs, marginal, fails_costs, untestable_on_free_data}`, recorded in the pack; all 7 Lens-4 candidate designs classified, count-unclassified == 0 asserted.
- **D6 — the pack records the corpus's own limits**: verdicts[2] MATERIALLY_FLAWED + the 10 never-researched angles listed as out-of-scope-for-83.1; the two Lens-7 counter-evidence findings marked read_in_full=false if cited.
- **D7 — ranking file (consumed by 83.1.1)**: JSON with pre-registered ranking criteria (DSR via `compute_deflated_sharpe` with MEASURED V — noting the variance-not-sd semantics; PBO via `compute_pbo_checked` with gate_grade + columns_diverse + always-emitted `pbo_n_trials` per the 83.0.3 disclosure), trial-budget cap, the pre-registered THEME LABEL HORIZON (taken verbatim from `synthesis.design_decision.entry_exit_timing`, explicitly distinct from the engine's 135-day 1.5×holding horizon), the kill-rule pointer for 83.1.1, the artifact-population globs (D1), and an append-only amendment clause (any amendment appends + recomputes the recorded hash).
- **D8 — mtime guard mechanics**: strict `<` on `st_mtime_ns` (equal PASSES — git checkout stamps clones identically); no pytest.skip escape; hash asserted 64-hex (immutable_limits_audit predicate, NOT the 16-char gauntlet truncation); the mutation test calls THE SAME helpers the criterion-5 test calls.

## Plan

1. Write the pre-registration ranking file `backend/backtest/experiments/preregistration_phase83_ranking.json` (D7).
2. Write the design pack `handoff/current/research_brief_phase83.md` from the inventoried corpus (sections: envelope-fed summary, gate thresholds read-from-source, design decisions, candidate-design table with C7 classifications, negative evidence (Lens 7 numbers), reference-case table (D4), killed options, residual risks + corpus limits (D6), anchor corrections (finding 3), pre-registration section recording the SHA-256 + population rule, ending with the C1 JSON envelope).
3. Write `backend/tests/test_phase_83_1_design_pack.py` (7 criteria; comment-stripping; anti-stub; runtime PromotionGate attribute reads; shared `_phase83_artifacts()` + `_assert_ranking_predates_all_artifacts()` helpers; the named C6 mutation test with glob-liveness assert + finally-unlink).
4. Run suite + lint; capture live_check (envelope verbatim, hash print, ls -la ordering).
5. `experiment_results_83.1.md` → qa-verdict → transcribe → harness_log → flip. Re-derive every fenced measurement after the final edit.

## References

`research_brief_83.1.md` (raw-corpus inventory with per-lens counts; external: unbiased-alpha, Palomar ch.8, arXiv 2603.09219, arXiv 2601.07852, JOP pre-registration policy, AlgoXpert — all read in full, URLs therein).
