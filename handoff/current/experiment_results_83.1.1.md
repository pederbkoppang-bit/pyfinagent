# Experiment results — Step 83.1.1: the go/no-go gate arithmetic (measured, recorded, kill-rule-first)

Date: 2026-08-07 (autonomous drain, cycle 172). Contract: `contract_83.1.1.md`.

## What was built (order matters — the kill rule came FIRST)

1. **`backend/backtest/experiments/killrule_phase83.json`** (written 13:25:01, BEFORE any result) — four clauses, each (recorded_quantity, comparison, threshold, threshold_source): K1 required-SR at the trial cap vs 0.8246 (the best Sharpe the repo ever produced); K2 spans@126 < 16 (= CSCV S); K3 checked-PBO vs the runtime PromotionGate ceiling; K4 trial count vs the pre-registered cap 45. Write-once by construction — any later edit turns the criterion-6 ordering test red. Deliberately NOT under `results/` (it must never join the 83.1 artifact population).
2. **`backend/backtest/gate_feasibility.py`** (new module) — `required_annualized_sr()` (bisection over the REPO deflation, bracket [1e-9, 50], 200 iters; FUNCTION-SCOPED `analytics` attribute lookup so the criterion-2 spy observes real routing — the 83.0.3 lesson applied at authoring time); `measure_v_from_82_3()`; `span_counts()` (horizon READ AT RUNTIME from the 83.1 preregistration); `pbo_feasibility()` (multi-seed); `run_all()` asserting the kill rule pre-exists before writing the results artifact.
3. **`backend/backtest/experiments/gate_feasibility_phase83_results.json`** (first written 13:28:25, AFTER the kill rule; regenerated in cycle 2 and re-stat'd in the cycle-3 capture below) — the criterion-1 record: the full 60-cell required-SR grid (N={1,2,10,45,100} × T={GDELT 2883, EDGAR 6434, GPR 10477, control 2011} × V={0.008169 measured-full, 0.167921 measured-short, 0.5 default-NOT-MEASURED}); the V measurement with its trial count and all 24 trial Sharpes; span counts; 8 verbatim compute_pbo_checked payloads. A real JSON artifact, not markdown fences (the multi-fence regex trap from the research).
4. **`backend/tests/test_phase_83_1_1_gate_feasibility.py`** — 7 tests, one per criterion. C1 checks shape/completeness/type only — NO value is asserted against a target (criterion 1's own wording); C2 routing spy incl. periods_per_year==252; C3 integer count ≥2 AND len(trial_sharpes)==count; C4 prereg-vs-engine assertion with the engine horizon COMPUTED from `inspect.signature(BacktestEngine.__init__)` (never hardcoded lore); C5 verbatim payload fields; C6 strict st_mtime_ns with the population asserted NON-EMPTY; C7 mutates the REAL kill-rule file forward and restores the original mtime in `finally`, with restoration proven by assert.

## Verification (cycle-1 capture — superseded; the CURRENT run is in the cycle-4 final capture below)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_1_1_gate_feasibility.py -q
.......
7 passed in 1.35s
```

(The heading previously claimed "re-derived after the final edit"; the cycle-3 Q/A caught it stale — the tree at close has 11 tests. Preserved unedited as the cycle-1 record.)

Lint gate over the git-derived scope: **"All checks passed!"**. The producing run's console table, the V line, the spans line, a verbatim PBO payload, and the `ls -laT` ordering evidence are in `live_check_83.1.1.md`.

## Mutation matrix — 4/4 KILLED + criterion 7 in-suite (runner: scratchpad/mutation_matrix_83_1_1.py; anchors/hashes verified per mutant)

| id | mutation | result |
|---|---|---|
| m1 | inline reimplementation bypasses `analytics.compute_deflated_sharpe` | KILLED (C2 spy: "did not invoke") |
| m2 | V recorded without its trial count (artifact-level) | KILLED (C3) |
| m3 | horizon silently equated with the engine's 135 (artifact-level) | KILLED (C4) |
| m5 | one required-SR grid cell dropped (artifact-level) | KILLED (C1 cell-count) |
| m4 | kill rule touched forward | == `test_c7`, runs green IN-SUITE every run |

m2/m3/m5 are artifact-level mutants by design: those criteria guard the RECORD, and the record is the artifact.

## The measured answer (recorded here; the STOP/CONTINUE decision belongs to the kill rule's future evaluations)

- **V measured = 0.008169 over n=24** actually-run trials (82.3 full-sample pooled; mean SR 0.5733, max 0.8246); short-window V = 0.167921 over n=32 — recorded both, never averaged; caveat recorded (K=8 configs of existing families — a prior, not the phase-83 V).
- **Required SR at the pre-registered trial cap (N=45, GDELT T=2883, measured V) = 0.6886** — BELOW the repo's best-ever 0.8246 (K1 does not fire). At the module-default V=0.5 the same cell reads **2.0692** (STOP). The 2026-08-04 verdict contradiction is fully explained: both lenses were arithmetically right at different V; the true measured V lands below the entire 1.38-3.18 range the step text worried about.
- **The deflation floor contains no T — but which lever DOMINATES is V-CONDITIONAL** [CORRECTED cycle 4 — the cycle-3 Q/A caught the previous sentence quoting the V=0.5 slice as "now measured"; the -11%/-23% pair reproduces UNIQUELY at the un-measured default]: at the MEASURED V=0.008169, +263% history (GDELT→GPR, N=45) buys **−33.6%** required SR (0.6886→0.4572) while cutting N 45→10 buys only **−8.7%** (0.6886→0.6288) — archive depth buys ~3.9× more than trial frugality. At V=0.5 the ordering reverses (−11.2% / −22.6%), which is where the step text's "trial budget, not archive depth" thesis and the pre-registration's trial_budget_rationale came from. **The V-dependence of the lever ordering IS the finding**: at the measured V, longer free history (GPR/EDGAR) is worth more than trial austerity — this changes 83.2+ prioritisation and is queued against the 83.1 pre-registration text as 83.1.5.
- **PBO is the BINDING gate** [CORRECTED cycle 2 — the figures previously here were the research brief's prototype ranges, not this run's; the cycle-1 Q/A FAILED the step on it]: at (2883, 45, S=16), MEASURED pure noise spans **0.2027-0.6524** (5 seeds — seed 4 lands essentially AT the 0.20 ceiling, so the false-pass risk is real) and the one-real-edge case spans **0.3025-0.8095** — **no measured edge seed clears the 0.20 ceiling at this shape**, harsher than previously claimed. A single-seed PBO is not a statistic; the phase-83 gate evaluation must be multi-seed and trial-diverse, and the PBO leg of kill-rule K3 is the hard constraint for 83.5.
- **Spans@126**: GDELT 22.88 / EDGAR 51.06 / GPR 83.15 pass K2's ≥16; the 82.3 control window (15.96) would NOT — recorded as a constraint on 83.5's evaluation window.
- **N=1 ≡ N=2** in every cell (`max(num_trials, 2)` clamp at analytics.py:430) — recorded as the collapse it is; the paper's formula is undefined at N=1.

## Follow-up — cycle 2 (2026-08-07, after the Q/A FAIL wf_f3d90599-f10)

The cycle-1 verdict (verbatim in `evaluator_critique_83.1.1.md`) FAILED the step on evidence integrity while confirming the arithmetic (it re-derived all 20 measured-V grid cells, V, spans, and all 8 payloads byte-identically). Two blocking findings, both closed at root:

1. **The live_check's "verbatim" PBO payload was fabricated** — carried from the research brief's separate prototype run and mislabelled as the artifact's seed-1 payload. Fixed: the live_check gained a superseding section whose payload and ranges are PIPED from the artifact programmatically (never retyped); the cycle-1 block is preserved unedited as the record of the defect.
2. **The PBO ranges were the brief's, not this run's, in three places incl. the production reading_note.** Fixed at root: `pbo_feasibility()` now DERIVES the reading-note string and a `measured_ranges` object from the payloads it just computed — a prose constant can no longer disagree with the record. `run_all()` re-executed, still after the kill rule's 13:25:01 (the CURRENT results mtime is in the cycle-3 capture below — the q7/q8 mutation runner's byte-identical restore moved the stat after the earlier figure was captured). The measured story is HARSHER: no edge seed clears the 0.20 ceiling; noise seed 4 lands at 0.2027.
3. **The WARN (no narrative-vs-payload guard)** — new `test_c5b` asserts the recorded min/max equal the payload-derived min/max AND that the prose note carries the measured values; the Q/A's q7 (falsified payload) and q8 (falsified reading_note) mutant shapes both now die (matrix below).

Re-verification after all fixes: immutable command → **8 passed in 1.39s** (7 criterion tests + c5b). Extended matrix: q7-class (payload value falsified in artifact) KILLED by c5b; q8-class (reading_note falsified) KILLED by c5b — run via the artifact-mutation runner, restore hash-verified.

## Disclosures

- **Discovered defect queued as 83.1.4**: the repo's LIVE V (analytics.py:752-754) is `np.var(window_sharpes)` — a cross-WINDOW dispersion inside one run with a silent 0.5 fallback below 2 windows — not the Bailey/LdP cross-TRIAL dispersion. Out of 83.1.1 scope.
- **Step-text corrections carried from 83.1** (not propagated into this contract): the `:665` purge-horizon anchor is wrong (real: `:274`×`:962`); a WebFetch summary of the Bailey/LdP PDF FABRICATED "V is a standard deviation" — the research brief's claims come from pypdf source-text extraction.
- The 82.3-control T column is labelled `free: False` in both the module and the artifact — it is a comparison window, not a candidate source.

## Files changed

`backend/backtest/gate_feasibility.py` (new), `backend/backtest/experiments/killrule_phase83.json` (new, first), `backend/backtest/experiments/gate_feasibility_phase83_results.json` (new, second), `backend/tests/test_phase_83_1_1_gate_feasibility.py` (new). Handoff: contract, research brief, live_check, this file. Masterplan: +1 pending step (83.1.4). Agent memory (cycle-2 Q/A disclosure-completeness fix — these ride the step commit): `.claude/agent-memory/researcher/project_gate_feasibility_83_1_1.md` (new; its prototype PBO figures corrected per the cycle-2 verdict) and `.claude/agent-memory/researcher/MEMORY.md` (index line, corrected likewise).

## Cycle-3 final capture (2026-08-07 — the tree is FROZEN after this block)

The cycle-2 Q/A found the fenced mtime captures stale: the q7/q8 artifact-mutation
runner restores content byte-identically but REWRITES the file, moving its stat —
so the 13:50:43 figures above no longer reproduced. The kill rule is byte-unchanged
(2545 bytes since 13:25:01). Final capture, taken AFTER the last suite run
(`8 passed in 1.34s`) with no further artifact-touching operation to follow:

```
$ ls -laT backend/backtest/experiments/killrule_phase83.json backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  8783  7 aug. 13:51:56 2026 backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  2545  7 aug. 13:25:01 2026 backend/backtest/experiments/killrule_phase83.json
```

Ordering: kill rule 13:25:01 < results 13:51:56 — criterion 6 holds on the live stat.
The prior fenced captures are preserved unedited as the record of their moments; this
block supersedes them for the CURRENT tree.

## Follow-up — cycle 4 (2026-08-07, after Q/A CONDITIONAL #2 wf_48465ea7-38e; streak 2 — this cycle must close everything)

The cycle-3 verdict's blocking finding: the "trial budget beats archive depth — now measured" sentence quoted the V=0.5 slice. Closed with the DERIVED-population sweep (grep across all 83.1.1 artifacts + agent memory): corrected at experiment_results (the V-conditional bullet), contract finding 3 ([MAIN CORRECTION]), the gate brief :261-263 (annotation) AND its envelope summary (header cover — which also covered the prototype PBO ranges still sitting there), and the researcher memory's lever bullet (Q/A-directed). The 83.1-owned sites (preregistration trial_budget_rationale + pack section 7) are QUEUED as step 83.1.5 — hash-committed artifacts of a closed step need their own append-only amendment cycle, not a mid-step edit.

The three guard gaps closed with tests + executed kills: `test_c1b` spot-cell equality-to-recomputation (qa15 KILLED); `test_c5c` seed-tuple pinning (qa14 KILLED — pruning the harshest seed with consistent re-derivation dies); `test_c3b` V-vs-own-sharpes at 2e-6 tolerance (measured 4dp-rounding gap 1.4e-7; one dropped trial moves it 1e-5 = 5.00x the tolerance, label corrected from '50x' per the cycle-4 verdict's NOTE) PLUS the count pinned to the SOURCE artifact's run count — both qa6 (inconsistent truncation) and qa6b (fully-consistent truncation with recomputed V) KILLED. The stale "7 passed" cycle-1 block re-labelled as superseded.

## Cycle-4 FINAL capture (the tree is FROZEN after this block — the qa-mutation runs above moved the results stat again, so this supersedes the cycle-3 capture)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_1_1_gate_feasibility.py -q
11 passed in 1.36s

$ ls -laT backend/backtest/experiments/killrule_phase83.json backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  8783  7 aug. 14:32:53 2026 backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  2545  7 aug. 13:25:01 2026 backend/backtest/experiments/killrule_phase83.json
```

Ordering: kill rule 13:25:01 (byte-unchanged since creation) < results 14:32:53. No artifact-touching operation follows this capture.
