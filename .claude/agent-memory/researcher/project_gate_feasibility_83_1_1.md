---
name: gate-feasibility-83-1-1
description: Phase-83.1.1 measured gate arithmetic -- V measured 20x apart on two windows, PBO (not DSR) is the binding gate, max(N,2) collapses N=1, and a WebFetch PDF summary fabricated the opposite of the paper
metadata:
  type: project
---

Measured 2026-08-07 for step 83.1.1 (all figures reproduced from the repo's own
functions, not re-derived formulas).

**Fact:** `compute_deflated_sharpe`'s `variance_of_srs` IS a variance -- confirmed
against Bailey & Lopez de Prado Eq.(1) primary text (`E[max{SR_n}] ~ E[{SR_n}] +
sqrt(V[{SR_n}])(...)`) and the authors' own code (`return mu+sigma*maxZ`).
`analytics.py:429` applies `math.sqrt`. The module default 0.5 is a DEFAULT, never
a measurement.

**Why:** the two contradictory 2026-08-04 feasibility verdicts were both arithmetically
correct -- they just chose different V. Measured V from the actually-run 82.3 trial
sets: full-sample (2018-2025, 24 runs) **0.008169** (ddof=1) vs short-window
(2024-07..2025-12, 32 runs) **0.167921** -- a **20x** spread, because short-window
trial Sharpes are dominated by estimation noise. Required annualized SR at N=45 /
T=2883 therefore swings **0.6886 to 2.0692**, which flips the verdict.

**How to apply:**
- Never quote a required-Sharpe number without the V and the window it came from.
- The deflation FLOOR (`sqrt(V) * E[max_N]`) contains no T. Measured: +263% history
  buys -11% on the required SR; cutting N 45->10 buys -23%. "Find a longer corpus"
  is the wrong remedy -- cut the trial budget. [MAIN CORRECTION 2026-08-07, directed
  by the 83.1.1 cycle-3 Q/A wf_48465ea7-38e: these percentages are the V=0.5 slice;
  at the MEASURED V=0.008169 the ordering REVERSES (-33.6% history vs -8.7% trials)
  and archive depth dominates. The lever ordering is V-CONDITIONAL -- never repeat
  the unconditional form.]
- **`max(num_trials, 2)` at `analytics.py:430-431` makes N=1 byte-identical to N=2.**
  An N=1 row is not an undeflated baseline. (The paper's formula is genuinely
  undefined at N=1: `Z^-1(1-1/1) = -inf`.)
- **PBO, not DSR, is the binding gate.** [MAIN CORRECTION 2026-08-07, directed by Q/A
  wf_a5e6e718-48d: the seed values previously recorded here (0.5983/0.7678/0.5640/
  0.4083/0.6407 noise; 0.1800/0.3822/0.5130 edge) were this researcher session's own
  PROTOTYPE run, superseded by the step's recorded measurement in
  gate_feasibility_phase83_results.json.] MEASURED at the intended (T=2883, N=45, S=16)
  shape: pure noise 0.4723/0.6524/0.4141/0.2027/0.4289 (range 0.2027-0.6524 -- seed 4
  lands essentially AT the 0.20 ceiling); one superior column among 44 noise:
  0.4304/0.3025/0.8095 (range 0.3025-0.8095 -- NO measured edge seed clears the
  ceiling). A single-seed PBO is not a statistic -- always report seeds.
- `columns_diverse` is a WEAK sentinel: a 0.918-correlated matrix passed the
  `corr_mean < 0.99` test and produced PBO 0.776. Record `column_corr_mean` verbatim.
- `compute_pbo_checked` REFUSAL payloads carry only 5 keys -- `column_corr_*` and
  `columns_diverse` are ABSENT. `r["columns_diverse"]` KeyErrors on a refusal.
- Independent spans must use TRADING sessions: horizon 126 is trading days, so a
  calendar-day numerator overstates GDELT by +45% (33.2 vs the correct 22.88).
  Measured spans @126: GDELT 22.88, EDGAR 51.06, GPR 83.15, 82.3 control 15.96.
- `exchange_calendars.get_calendar("XNYS")` defaults to a first session of
  **2006-08-07** and raises `DateOutOfBounds` for EDGAR (2001) / GPR (1985) starts.
  Pass `start="1980-01-01"`.
- The repo's LIVE V (`analytics.py:752-754`) is `np.var(window_sharpes)` -- a
  cross-WINDOW dispersion inside one run, NOT the Bailey/LdP cross-TRIAL dispersion.
  Separate defect; do not conflate with a measured V.
- `backtest_engine.py:665` is macro-coverage logging, NOT the purge horizon. Real
  anchors `:274` (`holding_days=90`) x `:962` (`int(holding_days*1.5)`) = 135.
  The step text still carries the wrong anchor.

**Two research-method traps found the same session:**
- A `WebFetch` on a binary PDF returned a **confidently fabricated** summary: it
  claimed `E[max SR_N] ~ (1-g)/sqrt(T)*sqrt(2 ln N)`, that V is a standard deviation,
  and that the deflation threshold depends on T. All three are FALSE against the
  extracted text. A fabricated summary is more dangerous than an empty result --
  always extract PDFs with pypdf/pdfplumber and quote the `.txt`.
- The 83.1 envelope regex `re.findall(r"```json\s*(\{.*?\})\s*```", DOTALL)`
  (`test_phase_83_1_design_pack.py:73`) is **not multi-fence safe**: an earlier fence
  with a trailing `// comment` after its closing brace makes the non-greedy match run
  on and swallow the envelope, so `f[-1]` is the wrong text. Assert the fence COUNT,
  or keep recorded figures in a real `.json` file rather than markdown fences.

See also [[phase83-design-pack-83-1]], [[dsr-trial-count-reset-82-25]],
[[pbo-level-and-dead-gate-82-27]], [[psr-dsr-formulas]].
