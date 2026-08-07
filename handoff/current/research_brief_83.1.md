# Research Brief -- step 83.1 (tier: simple)

Status: IN PROGRESS (write-first; grows incrementally)
Started: 2026-08-07 | Researcher: Layer-3 researcher (Workflow rail)

Step 83.1 = DESIGN-ONLY. It lands the COMPLETED 2026-08-04 8-lens market-news
research as an auditable design pack at `handoff/current/research_brief_phase83.md`
(criterion 1's named deliverable). It does NOT re-run the research.

NOTE ON TWO DIFFERENT BRIEFS -- do not conflate:
- `handoff/current/research_brief_83.1.md` (THIS FILE) = the research-gate artifact
  for step 83.1, written by the Layer-3 researcher.
- `handoff/current/research_brief_phase83.md` = the DESIGN PACK, the step's
  deliverable, written by Main in GENERATE. **MEASURED 2026-08-07: it does not
  exist yet** (`ls` -> "No such file or directory").

---

## Part A -- Internal audit (the bulk of this step)

### A1. Gate thresholds READ FROM SOURCE (criterion 2)

Runtime-verified, not quoted from memory.

`backend/autoresearch/gate.py:19-30` -- `@dataclass(frozen=True) class PromotionGate`:

| attribute | value | line |
|---|---|---|
| `min_dsr` | `0.95` | gate.py:21 |
| `max_pbo` | `0.20` | gate.py:22 |
| `min_pbo_trials` | `10` | gate.py:30 |

Because the dataclass is `frozen=True` with plain defaults, the criterion-2 test
can read them either as class attributes (`PromotionGate.min_dsr`) or from an
instance (`PromotionGate().min_dsr`) -- both resolve to the same float. The
in-code comment at gate.py:23-29 attributes `min_pbo_trials=10` to
Bailey/Borwein/Lopez de Prado/Zhu ("if the investor is sensitive to values of
[phi] < 1/10 ... N >> 10 is required"; the R reference implementation uses N=100)
and was added in phase-82.23.

**The OTHER gate (the 0.50 figure every research lens quoted):**
`backend/services/promotion_gate.py:37` -- `PBO_CEILING = 0.5`, a MODULE-LEVEL
CONSTANT (not a dataclass attribute), sitting alongside `STAGES = [0.05, 0.25, 1.0]`
(promotion_gate.py:33), `MIN_LIVE_DAYS = [14, 30]` (:34) and `PSR_PARITY = 0.0`
(:35). Its consumer is `evaluate_stage(...)` at :40 which compares
`ch_psr >= cp_psr + PSR_PARITY` (:56) and `ch_pbo < PBO_CEILING` (:57) -- i.e. it
is a **PSR-parity live-allocation staging decision**, not a DSR research-promotion
decision. Two structural differences worth quoting in the pack:
1. Different statistic: `promotion_gate.py` gates on **PSR relative to the
   incumbent champion** (:51-52); `autoresearch/gate.py` gates on **absolute DSR
   >= 0.95** (gate.py:59).
2. Different comparator: `promotion_gate.py` uses **strict `<`** on PBO
   (`ch_pbo < 0.5`, :57), `autoresearch/gate.py` uses **`>` to reject**
   (`pbo_f > self.max_pbo`, gate.py:61) -- i.e. PBO exactly 0.20 PASSES the
   research gate but PBO exactly 0.5 FAILS the staging gate. Do not describe them
   as "the same rule with a different number".

Ratio to quote: 0.50 / 0.20 = **2.5x**, matching the step name's "2.5x
calibration error".

### A2. `handoff/current/phase83_research_raw/` inventory (MEASURED 2026-08-07)

Three files, all mtime `2026-08-04 11:39`:

| file | bytes | JSON shape |
|---|---|---|
| `research.json` | 287,529 | LIST of 8 lens objects |
| `synthesis.json` | 122,865 | DICT, 7 keys |
| `verdicts.json` | 77,514 | LIST of 3 verifier objects |

`research.json` -- each of the 8 elements has keys: `lens`, `bottom_line`,
`key_findings`, `negative_or_null_results`, `recommendation_for_pyfinagent`,
`rejected_options`, `open_questions`, `sources_read_in_full_count`.

| # | lens | sources_read_in_full | key_findings | negative_or_null | rejected_options | open_q |
|---|---|---|---|---|---|---|
| 1 | theme representation (LDA / embeddings / LLM taxonomy / keyword+confirmation) | 16 | 16 | 9 | 7 | 8 |
| 2 | beneficiary mapping without paid supply-chain data | 13 | 16 | 16 | 10 | 8 |
| 3 | timing + crowding (birth / acceleration / confirmation) | 8 | 14 | 14 | 11 | 7 |
| 4 | cost + turnover: does the edge survive? | 7 | 10 | 13 | 8 | 7 |
| 5 | free-source feasibility, six-axis, $0 metered | 8 | 18 | 11 | 10 | 8 |
| 6 | lookahead defence + point-in-time discipline | 9 | 12 | 8 | 9 | 6 |
| 7 | NEGATIVE EVIDENCE (adversarial) | 16 | 17 | 11 | 8 | 8 |
| 8 | INTERNAL CODEBASE AUDIT (read-only, no web) | 10 | 19 | 7 | 6 | 6 |
| | **TOTAL** | **87** | **122** | **89** | **69** | **58** |

`synthesis.json` keys: `go_no_go` (str, value **`"descope"`**),
`go_no_go_rationale` (4,814 chars), `headline_findings` (10),
`design_decision` (5 sub-keys: `theme_representation`, `beneficiary_mapping`,
`entry_exit_timing`, `data_sources`, `expected_sample_adequacy`),
`step_changes` (14), `killed_options` (15), `residual_risks` (9).

`verdicts.json`: 3 verifier objects, keys `verifier`, `overall`,
`refuted_or_corrected`, `confirmed_load_bearing`, `missing_coverage`,
`bottom_line`.

(Continues -- sections A3..A6 and Part B below are appended as measured.)

### A3. The corpus's own candidate-design + cost table (feeds criterion 7)

`research.json[3]` (LENS 4, cost/turnover) `key_findings[7]` is a DERIVED
TURNOVER/COST TABLE that already enumerates SEVEN candidate designs with a
turnover figure and three cost columns. Model: `T = 2 x rebalances/yr x
fraction-of-book-replaced`; `annual cost = T x one-way-cost`. It self-reconciles
two ways (repo phase-53.1 replay 133 bps/yr; KKX published 46%/yr).

| candidate design | T (xNAV/yr, one-way) | @1.6bp | @3bp | @10bp (repo default) |
|---|---|---|---|---|
| Continuous rank tilt (~8% extra names/mo) | 1.9 | 3 | 6 | 19 bps/yr |
| Quarterly full thematic basket | 8.0 | 13 | 24 | 80 |
| MONTHLY top-N (LIVE TODAY, measured 0.555 replaced/mo) | 13.3 | 21 | 40 | 133 |
| Event entry + 8-week time exit | 13.0 | 21 | 39 | 130 |
| Event entry + 4-week time exit | 26.0 | 42 | 78 | 260 |
| Weekly rotation | 104 | 166 | 312 | 1,040 |
| Daily headline (KKX VW 91.4%/day) | 460.7 | 737 | 1,382 | 4,607 |

The classification RULE is `key_findings[8]` (THE CROSSOVER): max sustainable
`T = alpha budget / one-way cost`. Alpha budget anchor = Chen & Velikov
**4 bps/month, SE 3 bps => ~48 bps/yr central** for a cost-optimized
value-weighted large-cap anomaly (`key_findings[2]`, negative result [2]).
At c=3bps: alpha 48 -> Tmax 16.0x; 120 -> 40.0x; 240 -> 80.0x.
At c=1.6bps: 30.0 / 75.0 / 150.0x. At c=10bps: 4.8 / 12.0 / 24.0x.

=> The closed vocabulary criterion 7 needs is derivable from this table
directly. Recommended 3-value vocabulary (see Part C).

### A4. Negative-evidence material (criterion 3 needs >=3; corpus has far more)

`research.json[6]` = LENS 7, "NEGATIVE EVIDENCE (adversarial case)".
MEASURED: **17 of 17 `key_findings` carry a non-empty `numbers` field**, and
each carries `source_url` + `source_tier` + `read_in_full`. Plus 11
`negative_or_null_results` strings. Criterion 3 (>=3 failure modes each with
source + number) is over-satisfied ~5x. Strongest candidates:

1. **Thematic ETFs measured NEGATIVE alpha** -- specialized/thematic ETFs
   -3.1%/yr risk-adjusted after fees, ~-6%/yr in the first five years,
   FFC4 alpha -3.24%/yr vs -0.24%/yr broad-based, fee gap only 0.13%/yr.
   (NBER w28369, `research.json[6].key_findings[1]`, read_in_full=True)
2. **The economic-link foundation is dead in our slice** -- Cohen-Frazzini
   value-weighted L/S 1.30%/mo (t=3.03, 1978-2004) -> 0.62%/mo (t=1.54,
   2005-2018); 4-factor alpha 0.97 (t=2.47) -> 0.43 (t=1.09) once
   shared-analyst-coverage momentum is controlled.
   (ar5iv 2301.11394, `key_findings[2]` + `negative_or_null_results[0]`)
3. **LLM look-ahead: the alpha vanishes, not decays** -- DeepSeek 3.2
   +20.73% annualised in-sample (inside training window) -> -1.04%
   out-of-sample; -21.77pp. LLM post-cutoff S&P 500 directional accuracy
   45.70% vs 80.58% pre-cutoff. Prompt mitigation fails (97.6% vs 98.0%).
   (arXiv 2512.23847 + 2504.14765, `key_findings[3..4]`, `null[4..6]`)
4. **Sentiment-model choice FLIPS the sign** -- RoBERTa +23.67%,
   FinBERT -1.73%, DualGCN -11.47% alpha on the same window; two of three
   below passive buy-and-hold. (CORIA-TALN 2025, `key_findings[10]`)
5. **Thematic fund base rate** -- over 15 years to 2021-12-31 "more than
   three fourths of thematic funds globally have shuttered and just one in
   10 survived and outperformed"; 5-yr success 39%, 3-yr 57%.
   (Morningstar 2022 Global Thematic Funds Landscape, `key_findings[12]`)
6. **Timing penalty ~10x worse for thematic** -- 4.9pp/yr investor-vs-fund
   return gap ("more than two thirds of total returns") vs 0.5pp for all
   equity funds. (Morningstar Mind the Gap 2023, `key_findings[13]`)
7. **Multiple-testing base rate** -- Hou-Xue-Zhang 452 anomalies: 65% fail
   |t|>=1.96 value-weighted with NYSE breakpoints, 82% fail |t|>=2.78, 96%
   of trading-frictions anomalies fail; BHY cutoffs 3.47/4.27.
   (`key_findings[6]`)

The corpus also carries **two explicitly-labelled HONEST COUNTER-EVIDENCE
findings** (`key_findings[14]` Chen-Zimmermann 98% reproduce t>1.96,
publication-bias adjustment only -12.3%; `key_findings[15]` Hong-Torous-Valkanov
cross-industry diffusion up to two months) -- both `read_in_full=False`. If the
pack cites them, mark them as not-read-in-full.


### A5. Reference-case + free-source coverage material (criterion 4)

The corpus DOES carry the source-and-window facts, in `synthesis.json ->
design_decision.data_sources`, verbatim:

- **GDELT** `gdelt-bq.gdeltv2.gkg_partitioned` -- licence quoted verbatim as
  "unlimited and unrestricted use for any academic, commercial, or governmental
  use of any kind without fee"; **measured 4,169 daily partitions spanning
  2015-02-17 to 2026-08-04 (~99.6% calendar completeness), "covering ALL FOUR
  reference cases including COVID and Ukraine."** HARD RULE recorded: always
  filter `_PARTITIONTIME` (379 MB/day); NEVER `gdeltv2.gkg` (one V2Themes query
  scans 1.88 TB; `LIMIT` does not reduce cost). Aggregate theme intensity ONLY,
  never firm-level (measured recall 0.111 / precision 0.144 at stressed nodes).
  Dedup first: ~20.6% redundancy is virality-correlated (a BIAS, not noise).
- **Caldara-Iacoviello GPR** -- CC-BY, daily since **1985**, 44 countries,
  already fetched; consume DATED VINTAGES; 2021 methodology break = regime split.
- **SEC EDGAR full-text** -- public domain, measured floor **2001**, 10 req/s,
  filings immutable once accessioned (best PIT integrity assessed). Requires a
  declaring User-Agent (measured: empty UA -> 403, declared UA -> 200).
- **FRED via ALFRED vintages** -- macro controls; 1,221 INDPRO vintages to 1927.
- **Alpha Vantage -- BLOCKED** (operator decision owed): ToS "personal,
  non-commercial use", "investment analysis, research, testing" named as
  commercial; measured 25 req/day; >=19x archive-vs-live density discontinuity
  (AAPL 53 articles Jan-2025 vs >=1000 Jan-2026) => PIT-invalid regardless.
- **Alpaca** -- live-forward only (measured 2015 floor, `hasContent=False` on
  2016 rows, no derived-works licence grant found).
- **Wikipedia pageviews** -- REJECT as alpha (index-level, contrarian sign,
  concentrated in 2008, gross of costs; modern API floor 2015-07-01 does not
  overlap the founding study window). Wikipedia EDIT counts rejected on two
  nulls (p>0.9, p=0.19).

**GAP -- Main must know before writing the contract.** Two independent
corpus records say the four cases were NOT traced:
- `synthesis.json -> residual_risks[6]`: "THE FOUR REFERENCE CASES ARE STILL NOT
  TRACED END-TO-END, AND NOBODY IN THE CORPUS DID IT. No lens produced a single
  walkthrough of detection date, membership set, entry, exit and measured return
  for any one case."
- `verdicts.json[2] (completeness-critic, overall = "materially_flawed")`
  `missing_coverage[2]`: "REFERENCE CASE 2 (AI rush -> datacenter -> memory) IS
  THE OPERATOR'S MOST IMPORTANT CASE AND THE CORPUS CANNOT EXPLAIN IT
  END-TO-END."
- `verdicts.json[1] (feasibility-audit)` `missing_coverage[5]`: "THE FOUR
  REFERENCE CASES ARE NOT BACKTESTABLE AND THE CORPUS BURIES IT" -- but this was
  reached against the ALPHA VANTAGE floor (2022-02-24); the GDELT 2015-02-17
  floor is the correction the step name already encodes.

Criterion 4 asks only for a **free-data-source cell** and a **cost-to-hold
cell** per row -- not an end-to-end trace -- so it is satisfiable. But the
**cost-to-hold cell has NO per-case measured figure in the corpus**: the only
cost figures are the DESIGN-LEVEL table in A3. Main must therefore derive
cost-to-hold per case from the A3 design table (state the holding assumption
used, e.g. the 6-month clock exit -> the "Quarterly full thematic basket" or
"Event entry + 8-week time exit" row) and LABEL IT AS DERIVED, not measured.
Writing a per-case bps figure as if the corpus measured it would be a fabricated
number of exactly the class `feedback_measure_dont_assert_claims` warns about.

### A6. Pre-registration / SHA-256 / mtime prior art in the repo

**SHA-256 prior art (closest templates, in descending order of fit):**

| file:line | pattern | fit |
|---|---|---|
| `scripts/risk/promotion_gate.py:78-94` | `_load_gauntlet_report()` reads bytes, `hashlib.sha256(raw).hexdigest()`, returns `(report, sha, path)` | **BEST fit** -- hash a whole file's raw bytes, carry it alongside the decision |
| `scripts/housekeeping/quarantine_phantom_archives.py:42-49,131-139` | `_dir_sha256()` -> manifest key `"dir_sha256"`; verified on restore at `restore_from_quarantine.py:58,80` | write-hash + verify-later round trip already proven in-repo |
| `scripts/harness/phase10_housekeeping_test.py:117-138` | test `case_manifest_written_with_sha256_per_dir` asserts a 64-hex regex + required manifest keys | **the assertion style to copy** for criterion 5 |
| `scripts/audit/immutable_limits_audit.py:120-132` | `digest_ok = isinstance(v,str) and len(v)==64 and all(c in "0123456789abcdef" ...)` | the exact hex validation predicate |
| `scripts/risk/gauntlet.py:186-188` | `hashlib.sha256(json.dumps([...]).encode()).hexdigest()[:16]` | TRUNCATED to 16 -- do NOT copy this for criterion 5, which says SHA-256 |

**mtime-ordering guard prior art: NONE.** Grep across `backend/tests/` for
`st_mtime` / `getmtime` returns ZERO test files. The only `st_mtime` consumers
are runtime code (`scripts/smoketest/steps/chaos_watchdog.py:115,200`,
`backend/harness_self_audit_report.py:147,161`, `backend/tools/screener.py:695`,
`backend/config/prompts.py:188`, `scripts/go_live_drills/smoke_test_4_17_2.py:64-66`,
`scripts/go_live_drills/smoke_test_4_17_11.py:85`). So 83.1 writes the FIRST
mtime-ordering test in the repo -- there is no house convention to inherit.

**Design-only-step test convention DOES exist:**
`backend/tests/test_phase_82_6_bridge_design.py` is the template. Load-bearing
idioms to copy: (a) `DESIGN.stat().st_size > 2000, "the design is a stub"` (:51);
(b) require CONCRETE SYMBOLS not headings -- "a heading can be written without
content" (:56-57); (c) **strip HTML comments before matching** (`re.sub(r"<!--.*?-->", ...)`,
:60-63) because the 82.6 Q/A found a 2,100-byte stub whose only content was the
required tokens inside `<!-- -->` and it passed every criterion-1 test. That trap
applies verbatim to 83.1 criteria 3, 4 and 7.

### A7. Phase-83 backtest artifacts today (criterion 5) -- and a criteria-killing trap

**MEASURED 2026-08-07: ZERO phase-83 backtest artifacts exist.** The only
phase-83 paths repo-wide are three test modules
(`backend/tests/test_phase_83_0{,_1,_3}_*.py`), three closed handoff archives
(`handoff/archive/phase-83.0{,.1,.3}/`), `handoff/current/phase83_research_raw/`
and this brief. `backend/backtest/experiments/results/` contains NO
`*phase_83*` file.

**THE PHASE-TAG CONVENTION ALREADY EXISTS** and 83.1 should pre-register it
rather than invent one. Measured in `backend/backtest/experiments/results/`:
```
20260803T175308Z_phase_82_3_candidate_comparison.json
20260804T025319Z_phase_82_3_full_sample_3strat.json
20260804T041628Z_phase_82_3_short_window_4strat.json
```
i.e. `<UTC-TS>Z_phase_<major>_<minor>_<label>.json`. The phase-83 pattern is
therefore `**/results/*_phase_83_*.json` (plus whatever additional roots the
ranking file pre-registers).

**TRAP -- a naive `*83*` glob makes criterion 5 PERMANENTLY RED.**
`backend/backtest/experiments/results/` holds 438 files, of which **71 contain
the substring `83` in the filename** for reasons unrelated to phase 83 --
experiment ordinals (`-exp83.json`, `-exp183.json`), run-id hash prefixes
(`0083971f`), and timestamps. Their mtimes run **26 Mar 2026 to 4 Aug 2026**,
i.e. all BEFORE any ranking file 83.1 creates. A `glob("*83*")` therefore
returns 71 artifacts all older than the ranking file and criterion 5 can never
go green -- the exact structurally-uncloseable failure mode of
`feedback_immutable_criteria_must_be_green_able`. **The ranking file MUST
pre-register the exact artifact pattern, and the test MUST use that pattern.**

**Second trap -- git checkout resets mtimes.** A fresh clone / worktree (the
documented CI path) stamps EVERY file with the checkout time, so ranking file
and artifacts become mtime-EQUAL. Use strict `<` (artifact strictly older than
ranking file => FAIL). Equal mtimes then pass on a fresh clone, while the
mutation (explicitly backdated) still fails. Do NOT add a `pytest.skip` escape
for the equal case -- `feedback_guards_stop_one_seam_short` names the skip
trapdoor as a guard failure.

### A8. Facts 83.1.1 will consume -- two anchors in the step text are WRONG

- **`backtest_engine.py:665` is NOT the purge horizon.** Measured: line 665 sits
  inside macro-coverage logging. The real anchors are
  `backend/backtest/backtest_engine.py:274` (`holding_days: int = 90`) and
  **`:962`** (`horizon_days = int(self.holding_days * 1.5)`), with the doc
  comment at `:876` ("label horizon 1.5*holding_days ~= 90-135d") and
  `:564` (`global_end` uses the same 1.5x). 1.5 x 90 = **135 days** is correct;
  the LINE NUMBER in the 83.1.1 step text and in auto-memory
  `project_phase83_market_news` is not. The pack should cite `:962`.
- **The "unmeasured V" is the repo function's DEFAULT, not an arbitrary guess.**
  `backend/backtest/analytics.py:384-391`:
  `compute_deflated_sharpe(observed_sr, num_trials, variance_of_srs: float = 0.5,
  skewness=0.0, kurtosis=3.0, T=252, periods_per_year=1)`. Lens 7's V=0.5 is that
  default. Note the parameter is a **VARIANCE** (used as `math.sqrt(var_srs)` at
  `:429`), so "V=0.5" means sd ~= 0.707 -- if a lens treated 0.5 as a standard
  deviation the arithmetic is off by sqrt(2). 83.1.1 must state which it measures.
  The docstring at `:413-415` carries the Bailey numerical example
  (`observed_sr=2.5, T=1250, N=100, V=0.5, skew=-3, kurt=10, ppy=250 -> DSR ~= 0.90`),
  which is a free unit-regression fixture for 83.1.1.
- `compute_pbo_checked(pnl_matrix, S: int = 16)` at `analytics.py:208` returns
  `{"pbo", "n_trials", "n_obs", "gate_grade", "columns_diverse", "refused"}`;
  `PBO_MIN_TRIALS_GATE_GRADE = 10` at `:205`; refusals fire at `:229` (not 2-D),
  `:233` (N<2, "compute_pbo would return a false-good 0.0") and `:236`
  (T < S*2 = 32). `columns_diverse` at `:271` = `corr_mean < 0.99`.

### A9. Envelope inputs the pack's own JSON envelope needs (criterion 1)

Criterion 1 requires the PACK to end with an envelope carrying
`tier, external_sources_read_in_full, snippet_only_sources, urls_collected,
recency_scan_performed, internal_files_inspected, coverage.dry, gate_passed`,
and a test asserting `external_sources_read_in_full >= 5` and
`recency_scan_performed == true`. These are properties of the **2026-08-04
research**, and they are measurable from the raw JSON rather than asserted:

| field | MEASURED value | how |
|---|---|---|
| `external_sources_read_in_full` | **87** | sum of `sources_read_in_full_count` over the 8 lenses |
| unique `source_url` values in `research.json` | **85** [MAIN CORRECTION 2026-08-07: does NOT reproduce -- the 83.1 cycle-1 Q/A's 12-variant sweep measured **79** under the internally consistent http-only-distinct rule, which also yields this table's own 63 and 16; the pack envelope records 79] | recursive URL extraction |
| URLs on objects with `read_in_full: true` | **63** | |
| URLs on objects with `read_in_full: false` | **16** | => `snippet_only_sources` = 16 [MAIN CORRECTION: the `22 if 85 - 63` alternative was arithmetic on the wrong 85; under the corrected 79, both rules agree at 16] |
| `source_tier` histogram | peer_reviewed **66**, official_docs **18**, practitioner **10**, researcher_blog **4**, community **1** | satisfies the source-quality hierarchy |
| `coverage.audit_class` | **true** | the spawn prompt at `handoff/current/research_prompt_market_news.md` sets `coverage.audit_class: true`, `K=2` |

Recommendation: report `external_sources_read_in_full: 87` (self-reported by the
lenses) and `urls_collected: 85` [MAIN CORRECTION 2026-08-07: NOT re-derivable -- no variant reproduces 85; the measured value is 79 (http-only-distinct rule)] , and
say in the pack which is self-reported vs re-derived. NOTE the spawn prompt at
`research_prompt_market_news.md` line ~"tier: complex" declares
`PBO<=0.5` in its Objective -- that is the 2.5x error the pack corrects; quote it
as the provenance of the miscalibration, do not carry it forward.


---

## Part B -- External research

### B0. Query variants run (3-variant discipline)

| # | query | variant class |
|---|---|---|
| 1 | `pre-registration hash-committed analysis plan quantitative finance backtest overfitting` | year-less canonical |
| 2 | `pre-registration registered reports finance research 2026 replication analysis plan` | current-year frontier |
| 3 | `thematic ETF underperformance 2025 news-driven strategy failure evidence numbers` | last-2-year window |
| 4 | `kill criteria stopping rule negative results documentation research protocol` | year-less canonical (cross-domain) |
| 5 | `deflated Sharpe ratio number of trials pre-specified 2025 strategy research protocol quant` | last-2-year window |

### B1. Read in full (6; counts toward the gate) -- all accessed 2026-08-07

| URL | Kind | Fetched how | Key finding |
|---|---|---|---|
| https://arxiv.org/html/2603.09219 | preprint (quant protocol) | WebFetch (arXiv HTML) | Pre-commitment is operationalised as a **benchmark vector fixed before opening OOS**: "minimum benchmark vector for PASS/FAIL (committed before opening OOS)" with `SR >= 2.0, Calmar >= 1.5, MDD_eq < 7%, trades/day >= 5`; majority-pass `q = 2/3`; "After completing WFA and locking the final parameters theta*, no further optimization or tuning is performed when opening OOS (strict no-tuning)"; and a mandated disclosure: "Search budget, grid/trial size, permitted parameter dimensions, and **locking timing** must be reported to allow readers to assess degrees of freedom, selection bias, and reproducibility." |
| https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html | textbook (Palomar, Cambridge UP) | WebFetch | "**Not reporting the number of trials involved in identifying a successful backtest is a similar kind of fraud.**" Two operational rules: "never backtest until your model has been fully specified" and "Keep track of the number of backtests conducted on a dataset so that the probability of backtest overfitting may be estimated and the Sharpe ratio may be properly deflated." Also: "most of the claimed research findings in financial economics are likely false due to p-hacking"; "even if it is flawless, it is probably wrong." |
| https://www.cos.io/initiatives/prereg | official docs (Center for Open Science) | WebFetch | Preregistration = "specifying your research plan in advance of your study and submitting it to a registry"; its function is to "distinguish planned from unplanned work" so that "the same data cannot be used to generate _and_ test a hypothesis." Registered Reports differ: the plan is peer-reviewed BEFORE outcomes are observed and earns "in principle acceptance". Cited evidence: "the number of NHLBI trials reporting positive results declined after the year 2000" following preregistration adoption. |
| https://www.journals.uchicago.edu/journals/jop/registered-report-guidelines | official docs (journal policy) | WebFetch | Stage-1 freeze covers "all pre-processing steps and a clear description of the planned analyses" incl. "any covariates or regressors"; branching must be pre-committed: "If analysis decisions depend on the outcomes of prior analyses or specific data characteristics, these contingencies should be clearly defined and strictly followed." Acceptance is outcome-independent: "editorial decisions will not be based on the final empirical results." **Deviation protocol:** "authors should report in an appendix the results of the analyses as originally planned **and** as revised." Null results: "Plan for the possibility of null results ... Design your research to remain informative even if outcomes deviate from expectations." |
| https://arxiv.org/html/2601.07852 | preprint (utility-weighted forecasting under frictions) | WebFetch (arXiv HTML) | Analysis-plan section fixes "Primary and secondary endpoints (fixed)" plus "model lists, decision objectives, friction assumptions, evaluation protocols, robustness suites, and falsification suites"; enforces a "**No-touch test rule**" with "No adaptive data cleaning" and hard embargoes; controls a "Family of comparisons" rather than cherry-picking wins. Costs are decomposed into fee / spread / impact with "participation-rate or liquidity-linked bounds"; reports a 30.4% decision-loss reduction (t = -30.31) and binding-constraint frequency falling 16.0% -> 5.1%. |
| https://unbiased-alpha.com/how-to-avoid-backtest-overfitting-hypothesis-driven-strategy-discovery | practitioner blog | WebFetch | The most directly transferable statement of WHY 83.1 pre-registers: "**If you don't write it down first, you'll unconsciously adjust the definition of 'pass' based on what you see.**" Required pre-commitments: "The exact signal formula, the exact entry/exit rules, the holding period, the performance metric you'll use to evaluate it, **the threshold that would constitute 'passing'**". On trial counting: "If you need to correct for 50 tests, you've already run 50 tests. That's 49 opportunities to have found noise." Cumulative programmes get Holm-Bonferroni: "the kth most significant needs p < alpha / (n - k + 1)". "A failing hypothesis is information." |

### B2. Attempted but NOT read in full (do not count toward the gate)

| URL | why not |
|---|---|
| https://royalsocietypublishing.org/rsos/pages/registered-reports | HTTP 403 |
| https://www.morningstar.com/lp/global-thematic-fund-landscape | fetched, body empty (JS-rendered landing page) |
| https://magazine.morningstar.com/issues/q1-2024/investors-in-thematic-etfs-show-terrible-timing | fetched, body empty (paywalled/JS) |
| https://jhuccs1.us/clm/PDFs/stoprules.pdf | binary PDF, no text extracted (pdfplumber not attempted -- outside a `simple`-tier budget; the clinical stopping-rule concept is already carried by the JOP + COS sources above) |

### B3. Snippet-only (context only) -- 39 further unique URLs

arXiv: `2607.00276`, `2605.04135`, `2512.22476`, `2008.09481`.
Journals/publishers: `sciencedirect.com/.../S0950705124011110` (backtest overfitting in the ML era),
`sciencedirect.com/.../S0927538X24002051` (finance-replication pre-registered report),
`sciencedirect.com/.../S0927538X22001329` (PBFJ pre-registration initiative),
`pubmed.ncbi.nlm.nih.gov/37877375/` (statistical rules for safety monitoring).
Pre-registration guidance: `cos.io/blog/choosing-preregistration-template-guide-for-researchers`,
`aje.com/arc/pre-registration-vs-registered-reports`, `tesify.app/how-to-preregister-study-osf-aspredicted-2026/`,
`journalmetrics.org/blog/registered-reports-clinical-research-guide-2026`.
DSR reference: `en.wikipedia.org/wiki/Deflated_Sharpe_ratio`, `quanterlab.com/articles/foundations-dsr`,
`rdrr.io/github/braverock/quantstrat/man/SharpeRatio.deflated.html`,
`github.com/Nikhil-Kumar-Patel/The-deflated-sharpe-ratio`,
`medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464`, `rollbrains.com/quant/`.
Thematic-fund evidence: `morningstar.com/funds/how-newest-wave-etfs-is-leading-mini-bubbles`,
`morningstar.com.au/etfs/young-invested-are-thematic-etfs-worth-it`,
`thedailyupside.com/etf/thematics-sectors/rising-tide-of-thematic-etfs-could-put-investors-underwater/`,
`ishares.com/us/insights/thematic-investing-mid-year-outlook-2025`,
`etfstream.com/articles/are-we-too-harsh-on-etf-issuers-for-launching-themes-before-maturity`,
`kaiserpartner.bank/news/thematic-etfs-megatrend-or-fad-trap/`,
`investing.com/analysis/thematic-etfs-the-good-the-bad-and-the-ugly-200658721`,
`linkedin.com/posts/neil-clare-cfa-3983828_thematic-etfs-disappoint-...`.
Stopping rules: `numberanalytics.com/blog/ultimate-guide-stopping-rules-research-ethics`,
`tfscro.com/glossary/stopping-rules/`, and 6 clinicaltrials.gov protocol PDFs
(`NCT05741346`, `NCT05116787`, `NCT05116774`, `NCT05162066`, `NCT04702568`, `NCT03616184`).
Other: `community.portfolio123.com/.../3WHpAUOzhCG8QAUez71HpoWnA62.pdf` (SSRN 2745220).

**Unique URLs collected: 45.** Read in full: 6. Snippet-only: 39.

### B4. Recency scan (last 2 years, 2024-2026) -- PERFORMED

Searches 2, 3 and 5 were scoped to the 2024-2026 window. **Result: 3 new
findings that COMPLEMENT (none supersede) the canonical sources.**

1. **Pre-registration is now institutional in finance, not just psychology**
   (2024-2026). The Pacific-Basin Finance Journal runs a standing
   pre-registration publication initiative covering replication studies, full
   original studies and industry case studies; a 2024 PBFJ pre-registered report
   studies researchers' perceptions of finance replication. Registered Reports
   were extended by Nature to all disciplines in 2026, though fewer than 1% of
   MEDLINE-indexed journals offer the format. => the 83.1 pre-registration is
   consistent with current-frontier practice in this exact field, not a novelty.
2. **A 2026 quant-specific pre-commitment protocol exists** (arXiv 2603.09219,
   read in full) that names the same three primitives 83.1 needs: a benchmark
   vector committed before opening OOS, strict no-tuning after locking, and
   mandatory disclosure of **locking timing** -- which is precisely what the
   mtime-ordering guard mechanises. Nothing older says "record WHEN you locked".
3. **The thematic-fund negative evidence has been RE-CONFIRMED post-2023 and is
   now the 2025 vintage of the same finding the corpus cites from 2021-2023.**
   Morningstar's Global Thematic Fund Landscape has a 2025 edition; secondary
   coverage (2024-2025) restates the 4.9pp/yr investor-return gap over the five
   years to 2023-06-30 and the "sector equity funds ... returns gap is 2.6%, the
   largest of the observed group" figure, and 2025 commentary notes AI and crypto
   ETF launches following the same hype-timed pattern. No 2024-2026 source was
   found that REVERSES the thematic-underperformance finding. The corpus's
   negative-evidence section is therefore not stale.

**No 2024-2026 source was found that overturns any canonical claim the pack
relies on.** The one live tension is internal, not literature-driven: the
unmeasured trial-Sharpe dispersion V (see A8), which 83.1.1 exists to resolve.

### B5. Consensus vs debate

**Consensus (unanimous across all 6 read-in-full sources):**
1. The pass threshold must be written down BEFORE the test is run, and the
   written form is what prevents unconscious threshold drift.
2. The NUMBER OF TRIALS must be recorded, not just the winner. Palomar calls
   failing to do so "fraud"; AlgoXpert requires reporting "search budget, grid/
   trial size ... and locking timing"; the practitioner source frames the
   correction as evidence of the search itself.
3. A pre-registration is only credible if it is DATED/ORDERED relative to the
   results -- COS's registry timestamp, AlgoXpert's "locking timing",
   arXiv 2601.07852's "no-touch test rule". **This is the literature basis for
   criterion 5's mtime guard.**
4. Null results must be planned for and remain publishable/informative
   (JOP: "Plan for the possibility of null results"; practitioner: "A failing
   hypothesis is information"). => the corpus's `go_no_go: "descope"` and the
   pre-registered expectation that 83.5 will FAIL are compliant, not defeatist.

**Debate:** how much freedom a pre-registration may retain. JOP permits
"reasonable" deviations if reported BOTH ways in an appendix; AlgoXpert permits
in-fold re-selection but only "within the controlled scope from Stage I". So the
83.1 ranking file should include an explicit AMENDMENT clause (append-only, with
both the original and revised ranking reported) rather than pretending it is
unamendable -- an unamendable artifact will just be quietly ignored.

### B6. Pitfalls the literature names (mapped to 83.1)

| pitfall | source | how it bites 83.1 |
|---|---|---|
| Threshold drift after seeing results | unbiased-alpha | the whole reason for the ranking file + hash |
| Not reporting trials | Palomar 8.3 | the ranking file must pre-register the TRIAL BUDGET, not only the ranking; 83.1.1's DSR grid is `N in {1,10,45,100}` |
| Locking-time undisclosed | arXiv 2603.09219 | the mtime guard IS the locking-time disclosure; record it in the brief too, because mtime is not durable across clones (A7) |
| Adaptive data cleaning / no-touch violation | arXiv 2601.07852 | the ranking file must fix the friction assumptions (which `c` in bps) before any cost classification is computed |
| Un-planned branching | JOP | if the ranking depends on 83.1.1's measured V, the CONTINGENCY must be written now ("if V > x then ranking becomes ...") |
| Deviations hidden | JOP | amendment clause: report both original and revised |


---

## Part C -- Application to pyfinagent (contract-ready recommendations)

### C1. Recommended ranking-file schema (machine-readable, JSON)

Suggested path: `handoff/current/phase83_preregistered_ranking.json`
(NOT under `backend/backtest/experiments/results/` -- it must not itself match
the phase-83 artifact pattern it guards).

```
{
  "preregistration_id": "phase-83.1-ranking-v1",
  "created_at_utc": "<ISO8601>",
  "frozen": true,
  "amendment_policy": "append-only; a revision adds a new object to
     `amendments[]` and NEVER edits a prior field. Any amendment reports BOTH
     the original and the revised ranking (JOP registered-report deviation rule).",

  "gate_thresholds_read_from_source": {
     "module": "backend/autoresearch/gate.py",
     "class": "PromotionGate",
     "min_dsr": 0.95, "max_pbo": 0.20, "min_pbo_trials": 10,
     "not_this_one": {"module": "backend/services/promotion_gate.py",
                      "constant": "PBO_CEILING", "value": 0.5,
                      "why_different": "PSR-parity live-allocation staging, not DSR research promotion"}
  },

  "cost_model": {"formula": "T = 2 * rebalances_per_year * fraction_of_book_replaced;
                  annual_cost_bps = T * one_way_cost_bps",
                 "one_way_cost_bps_primary": 3.0,
                 "one_way_cost_bps_sensitivity": [1.6, 10.0],
                 "alpha_budget_bps_per_year": {"central": 48, "optimistic": 120, "best_case": 240,
                    "source": "Chen & Velikov 4 bps/mo SE 3 bps, cost-optimized value-weighted"}},

  "survives_costs_vocabulary": ["SURVIVES", "MARGINAL", "REJECTED"],
  "survives_costs_rule": "SURVIVES iff annual_cost_bps <= alpha_budget.central at
     one_way_cost_bps_primary; MARGINAL iff it clears only under optimistic/best_case
     or only at 1.6bps; REJECTED otherwise. Applied BEFORE any result is seen.",

  "candidates": [ {"id": "...", "design": "...", "turnover_T": 1.9,
                   "annual_cost_bps": {"1.6": 3, "3": 6, "10": 19},
                   "survives_costs": "SURVIVES", "rank": 1} , ... ],

  "ranking_criteria_for_the_gate_step": [
     {"rank": 1, "criterion": "...", "direction": "max|min", "tie_break": "..."} ],

  "preregistered_for_83_1_1": {
     "theme_label_horizon_days": <int, MUST NOT equal 135>,
     "engine_derived_horizon_days": 135,
     "engine_anchor": "backend/backtest/backtest_engine.py:962 (int(holding_days*1.5), holding_days=90 at :274)",
     "trial_counts_N": [1, 10, 45, 100],
     "sample_windows_T": [{"source": "GDELT", "start": "2015-02-17", "end": "2026-08-04", "days": <int>},
                          {"source": "GPR", "start": "1985-01-01", ...},
                          {"source": "SEC EDGAR", "start": "2001-01-01", ...}],
     "intended_pbo_matrix_shape": {"T": <int>, "N": <int>, "S": 16},
     "kill_rule_file": "handoff/current/phase83_kill_rule.md"
  },

  "phase83_artifact_pattern": {
     "globs": ["backend/backtest/experiments/results/*_phase_83_*.json",
               "handoff/current/phase83_*/**/*.json"],
     "why_not_substring_83": "438 files live in results/, 71 contain '83' as an
        experiment ordinal or run-id substring with mtimes from 2026-03-26; a
        substring glob makes criterion 5 permanently red."
  }
}
```

### C2. SHA-256 + mtime-guard mechanics (criteria 5 and 6)

**Order of operations (non-negotiable):**
1. Write the ranking file. 2. FREEZE it. 3. `shasum -a 256 <file>` (or
`hashlib.sha256(path.read_bytes()).hexdigest()`). 4. Paste the 64-hex digest into
`research_brief_phase83.md` on a single greppable line, e.g.
`PREREGISTRATION_SHA256: <64 hex>`. 5. Never touch the ranking file again --
a trailing newline added by an editor changes the digest. Any later change =
new `amendments[]` entry + a NEW hash line, both recorded.

**Criterion-5 test (single shared helper -- do not duplicate the logic):**
```python
def _phase83_artifacts() -> list[pathlib.Path]:      # uses the PRE-REGISTERED globs
def _assert_ranking_predates_all_artifacts() -> None:
    ranking_ns = RANKING.stat().st_mtime_ns
    older = [p for p in _phase83_artifacts() if p.stat().st_mtime_ns < ranking_ns]
    assert not older, f"phase-83 artifact(s) predate the pre-registration: {older}"
```
- Read the recorded digest out of the brief with a regex on the
  `PREREGISTRATION_SHA256:` line; assert `len == 64` and hex-only (copy the
  predicate at `scripts/audit/immutable_limits_audit.py:122`), then assert it
  equals `hashlib.sha256(RANKING.read_bytes()).hexdigest()`.
- Use **strict `<`** on `st_mtime_ns`. Equal mtimes PASS -- required so a fresh
  clone / worktree (where checkout stamps every file identically) does not go
  red. Do NOT add a `pytest.skip` for the equal case.

**Criterion-6 mutation test (the NAMED mutation):**
`test_criterion5_fails_when_a_backdated_phase83_artifact_exists`
1. `MUT = REPO/"backend/backtest/experiments/results/20260101T000000Z_phase_83_1_MUTATION.json"`
   (a path that MATCHES a pre-registered glob -- assert that first, so a broken
   glob is caught rather than silently producing a false green).
2. Write it, then `os.utime(MUT, ns=(t, t))` with
   `t = RANKING.stat().st_mtime_ns - 60_000_000_000` (60 s earlier).
3. Assert `MUT in _phase83_artifacts()` -- **proves the glob is live**, which
   matters because the real population is currently EMPTY (A7) and an empty
   population makes criterion 5 vacuous.
4. `with pytest.raises(AssertionError): _assert_ranking_predates_all_artifacts()`.
5. `finally: MUT.unlink(missing_ok=True)`.
The mutation MUST call the same `_assert_ranking_predates_all_artifacts()` the
criterion-5 test calls -- mutating a copy is the "one seam short" failure
(`feedback_guards_stop_one_seam_short`).

### C3. Closed vocabulary for criterion 7, applied to the seven candidates

Vocabulary: `SURVIVES` / `MARGINAL` / `REJECTED` (record the vocabulary IN the
brief, as criterion 7 requires). Applying the C1 rule at c=3 bps against the
48 bps/yr central alpha budget (Tmax = 16.0x) to the A3 table:

| candidate | T | cost@3bp | classification |
|---|---|---|---|
| Continuous rank tilt | 1.9 | 6 bps/yr | SURVIVES |
| Quarterly full thematic basket | 8.0 | 24 | SURVIVES |
| Event entry + 8-week time exit | 13.0 | 39 | SURVIVES (Tmax 16.0) |
| Monthly top-N (live today) | 13.3 | 40 | SURVIVES at 3bp; **MARGINAL** at the repo-default 10bp (133 bps/yr vs 48) |
| Event entry + 4-week time exit | 26.0 | 78 | REJECTED (central), MARGINAL only at alpha 120-240 |
| Weekly rotation | 104 | 312 | REJECTED |
| Daily headline (KKX VW) | 460.7 | 1,382 | REJECTED (29x the central budget) |

Zero unclassified -- criterion 7 satisfiable with no new measurement. If Main
adds candidates beyond these seven (e.g. a price-mediated variant from
`verdicts.json[2].missing_coverage[0]`), each ADDED candidate needs its own T,
or it must be classified `REJECTED` for want of a turnover estimate -- never
left blank.

### C4. GAPS -- what the 2026-08-04 research does NOT contain (read before contracting)

1. **No `coverage` / `dry_rounds` / `audit_class` record anywhere.** MEASURED: a
   grep for `"coverage"`, `dry_round`, `audit_class`, `"dry"`, `K_required`
   across all three raw JSON files returns **nothing**, and `harness_log.md`
   Cycle 1137 (line 30317) records method + verdict but no envelope.
   `research_prompt_market_news.md:28` DID set `coverage.audit_class: true` and
   `:114-120` DID require `coverage.dry == true` for `gate_passed`. So criterion
   1 asks the pack to report a `coverage.dry` that **was never recorded**. Main
   must state this honestly (e.g. `coverage: {audit_class: true, dry: null,
   note: "not recorded by the 2026-08-04 run"}`) rather than assert `true`.
   Note the criterion only requires the envelope to REPORT the field and asserts
   on `external_sources_read_in_full >= 5` and `recency_scan_performed` -- so a
   `null`/`false` with a stated reason still passes the test while staying honest.
2. **No per-reference-case cost-to-hold figure exists.** Only the design-level
   table (A3). Criterion 4's cost-to-hold cell must be DERIVED and labelled as
   derived, with the holding assumption stated.
3. **The four reference cases were never traced end-to-end** -- three independent
   corpus records say so (A5). Criterion 4 does not require a trace, but the pack
   must not imply one exists.
4. **`backtest_engine.py:665` is the wrong anchor** for the 135-day purge horizon
   (A8). Correct: `:962` (+ `:274`, `:876`, `:564`).
5. **`snippet_only_sources` has no single measured value.** Two defensible
   derivations from `research.json`: `16` (URLs on objects flagged
   `read_in_full: false`) or `22` [MAIN CORRECTION 2026-08-07: arithmetic on the wrong 85; under the corrected 79 both rules agree at 16] (`85` unique URLs minus `63` distinct
   read-in-full URLs). State the rule with the number
   (`feedback_normalization_rule_must_be_stated_with_the_ratio`); do not report a
   bare figure.
6. **`external_sources_read_in_full: 87` is SELF-REPORTED by the lenses**, and
   `verdicts.json[0]` (citation-and-number-auditor) says it verified only 13
   primary sources in full plus a 23-ID arXiv sweep. Label 87 as self-reported.
7. **The spawn prompt itself carries the 2.5x error** (`research_prompt_market_news.md:32`
   "PBO<=0.5"). Quote it as the provenance of the miscalibration; do not carry it.
8. **`verdicts.json[2]` overall = `materially_flawed`** with 10 missing-coverage
   items (never-researched angles: price-mediated propagation, return-based theme
   construction, options-flow propagation, free analyst-revision proxies,
   concentration x risk-gate interaction). The pack should list these as
   out-of-scope-for-83.1 rather than let a reader infer the corpus was complete.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6**
- [x] 10+ unique URLs total -- **45**
- [x] Recency scan (last 2 years) performed + reported -- Part B4
- [x] Full pages read (not abstracts) for the read-in-full set; 4 failed
      fetches disclosed in B2 and NOT counted
- [x] file:line anchors for every internal claim (Part A)

Soft checks:
- [x] Internal exploration covered every module the caller named, plus
      `analytics.py`, `backtest_engine.py` and the results directory
- [x] Contradictions noted (B5 debate; A8 two wrong anchors; C4 eight gaps)
- [x] Claims cited per-claim
- [ ] `coverage.dry` for the ORIGINAL 2026-08-04 audit-class run is
      unrecoverable (C4 item 1). This step (83.1) is NOT itself audit-class, so
      it does not gate here.

## JSON envelope

[MAIN CORRECTION 2026-08-07 covering the JSON summary below without editing it (kept parseable): its '85 unique URLs' figure is superseded -- measured 79 under the http-only-distinct rule; see the annotated A9 tables above.]

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 39,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "83.1 is design-only and consumes the 2026-08-04 corpus. Gate thresholds read from source: PromotionGate min_dsr=0.95, max_pbo=0.20, min_pbo_trials=10 (gate.py:21-30); the 0.50 is PBO_CEILING at promotion_gate.py:37, a PSR-parity staging decision with a strict-< comparator. Corpus inventory: 8 lenses, 87 self-reported full reads, 85 unique URLs, 122 key findings, 89 nulls, 15 killed options, 3 verdicts (one materially_flawed). Lens 4 already supplies 7 candidate designs with turnover+cost, so criterion 7 needs no new measurement; lens 7 supplies 17 numbered failure modes for criterion 3. GDELT's measured 4,169 partitions (2015-02-17..2026-08-04) cover all four reference cases. Eight gaps found, incl. no coverage.dry recorded anywhere, no per-case cost-to-hold, and backtest_engine.py:665 being the wrong purge anchor (real: :962). Criterion-5 trap: 71 of 438 files in results/ contain '83' as a substring with March-April mtimes, so a naive glob makes the criterion permanently red.",
  "brief_path": "handoff/current/research_brief_83.1.md",
  "gate_passed": true
}
```
