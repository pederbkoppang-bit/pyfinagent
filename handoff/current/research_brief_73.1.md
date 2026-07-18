# Research Brief — phase-73.1 "D2a LEAKAGE-INTEGRITY DESIGN"

Tier: **moderate** (caller-specified). NOT audit-class.
Role: design-input research for the D2a leakage-integrity DESIGN doc. Covers
FOUR components: (1) PURGE-VERIFICATION, (2) POST-CUTOFF HARNESS,
(3) COUNTERFACTUAL-AUDIT GATE (pilot), (4) CPCV WIRING (robustness complement).

Provenance carried forward: `frontier_map_73.md` #3 RE-SCORE + baseline
corrections; `research_brief_73.0.md` D1 gate. The AFML purge+embargo already
shipped in phase-69.2 — this step DOCUMENTS + REGRESSION-TESTS it, and designs
the residual LLM-side guards.

## Status: IN PROGRESS (write-first; filled as sources are read)

---

## 1. External sources — READ IN FULL (>=5 required; counts toward gate)

| # | Source | arXiv/URL | Accessed | Fetched how | Kind |
|---|--------|-----------|----------|-------------|------|
| E1 | Profit Mirage / FactFin (Li, Zeng, Xing et al.) | arxiv.org/html/2510.07920 | 2026-07-18 | `/html/` OK | leakage / counterfactual audit |
| E2 | Detecting Lookahead Bias in LLM Forecasts (LAP) | arxiv.org/html/2512.23847 | 2026-07-18 | `/html/` OK | lookahead diagnostic |
| E3 | Look-Ahead-Bench README (arXiv:2601.13770) | github.com/benstaf/lookaheadbench | 2026-07-18 | GitHub README (paper `/html/` 404) | agentic lookahead benchmark |
| E4 | AFML purged K-fold + embargo + CPCV (practitioner writeup) | towardsai.com/p/l/the-combinatorial-purged-cross-validation-method | 2026-07-18 | `/html/` OK | purged CV / CPCV |
| E5 | Time Machine GPT (point-in-time LLM) | arxiv.org/html/2404.18543 | 2026-07-18 | `/html/` OK | time-stamped model |
| E6 | The New Quant §7.1 + §7.10 | arxiv.org/html/2510.05533 | 2026-07-18 | `/html/` OK | temporal-leakage / reporting std |

## 2. Snippet-only sources (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| en.wikipedia.org/wiki/Purged_cross-validation | encyclopedia | corroborates E4 purge/embargo defn |
| skfolio.org/.../CombinatorialPurgedCV | library docs | CPCV now standard tooling (2025) — confirms buildability |
| github.com/eslazarev/purged-cross-validation | code/paper | CPCV reference impl (context) |
| grokipedia.com/page/purged_cross_validation | encyclopedia | secondary corroboration |
| garp.org/.../a1Z1W0000054x6lUAA.pdf (LdP "10 reasons ML funds fail") | industry | PBO/overfitting context |
| arxiv.org/abs/2601.13770 (Look-Ahead-Bench) | preprint | `/html/` still 404 Jan-2026; README used instead (E3) |
| papers.ssrn.com/…4686376 (CPCV comparison) | preprint | 403 Forbidden (carryover from 73.0) |
| arxiv.org/pdf/2407.17645 (Hopfield asset alloc) | preprint | CPCV usage example (context) |

## 3. Search queries run (3-variant discipline)

Per topic, 3 variants (current-year `2026`, last-2-year `2025`, year-less canonical):
- "LLM financial agent information leakage counterfactual audit 2026" / "...2025" / "LLM backtest information leakage" (year-less)
- "purged k-fold cross validation embargo Lopez de Prado" (year-less canonical) / "combinatorial purged cross validation 2025"
- "point-in-time LLM knowledge cutoff financial forecasting 2026" / "time machine GPT temporal" (year-less)

## 4. Recency scan (2024-2026 — mandatory)

Searched 2026 / 2025 / year-less variants per topic. Findings:
- **Leakage frontier MOVED in 2025-26 and STRENGTHENS all four components.** Profit Mirage (Oct-2025), Detecting-Lookahead-Bias (Dec-2025), Look-Ahead-Bench + Pitinf models (Jan-2026) all post-date casual backtest practice and independently confirm: web-scale LLMs memorize post-hoc market narratives, so pre-cutoff LLM backtests are a "profit mirage." The New Quant (Oct-2025) newly CODIFIES "strict publication cutoffs per evaluation fold" as a minimum standard (§7.1). No 2025-26 finding overturns the design; they reinforce it.
- **Point-in-time line**: Time Machine GPT (Apr-2024) is the canonical PiT-LLM; 2026 work (Pitinf trained trading models in Look-Ahead-Bench) extends it to trained PiT models we CANNOT build — but the "trust only post-cutoff windows" principle is now consensus, and for API-only backbones the free inverse (our post-cutoff live trail is clean) is the practical path.
- **CPCV**: purge+embargo is canonical (AFML 2018); NO 2025-26 method supersedes it — instead skfolio (2025) shipped `CombinatorialPurgedCV` as standard tooling, confirming C4 is a wiring job not research.
- **No-logprob constraint reconfirmed**: LAP (2512.23847) needs logprobs → BLOCKED for Claude; FactFin PC (text-only) is the usable substitute. No 2026 method removes the Anthropic-no-logprob barrier.

## 5. Key external findings (per-claim, implementation depth)

### E1 — Profit Mirage / FactFin (arxiv.org/html/2510.07920), read in full at impl depth
1. **Algorithm 1 structure**: RAG(state→features) → SCG(strategy code-gen) → iterative loop {MCTS optimize; `D_cf = Perturb(D, δ)` (Eq 18); evaluate PC/CI/IDS on both `φ(C,D)` and `φ(C,D_cf)`; update by loss min}. Perturbation is applied per-iteration; **no fixed "M copies per candidate" is stated in the paper** — M is a design knob WE choose (73.0's "M~20-50" is our number, not the paper's).
2. **Leakage metrics (exact)**:
   - **PC (Eq 4)** `= (1/N) Σ 1[ŷ_i^orig = ŷ_i^cf]` — prediction-consistency; **higher PC = MORE memorization** (bad). Operates on **text predictions alone** — no logprobs needed. This is the cheap, Claude-compatible metric.
   - **CI (Eq 5)** `= 1 − (1/M) Σ |s_j^orig − s_j^cf|` — confidence-invariance; near 1 = insensitive to input (bad). Needs a **confidence scalar** `s` (our verbalized conviction 1-10 works — NOT a logprob).
   - **IDS (Eq 6)** `= (1/N) Σ D_KL(P_i^orig ‖ P_i^cf)` — input-dependency; **higher = LESS leakage** (good). Needs a **distribution** `P` — for a text-only model this requires a class-probability proxy (e.g. multi-sample vote shares over BUY/HOLD/SELL), not available from a single T=0 call.
   - **Objective (Eq 20)** `C* = argmin {α·PC + β·CI − γ·IDS}`.
3. **Perturbation operators (Sec 3.6 / App C)**: prices → Gaussian noise `ε~N(0,σ²)`; factors → RSI/MACD adjusted; **news → content swap** (earnings strong→disappointing). Concrete NVDA-May-2022 example: revenue "$8.29B (+46% YoY)" → "$7.64B (below expectations)", price/technicals HELD CONSTANT — the swap isolates whether the model recites the known outcome.
4. **NO formal accept/reject thresholds in the paper** (correction to 73.0's "PC>0.7 / IDS>0.6" — those are interpretive, not the paper's). Anchor bands from Table 5 (AAPL): leaky SCG-only **PC=0.6213, CI=0.6457, IDS=0.4361**; clean full-FactFin **PC=0.3115, CI=0.2548, IDS=0.7781** (TSLA full: PC=0.3427, IDS=0.7544). So a defensible LOCAL cutoff (our choice, disclosed) is ~PC≥0.55 / IDS≤0.45 flags "leaky-like"; must be calibrated on our own perturbation set, not lifted.

### E2 — Detecting Lookahead Bias / LAP (arxiv.org/html/2512.23847), read in full
1. **LAP** `= P_up + P_down` from a **date-only recall query** (firm, ticker, target date; NO headline). Detection regression **Eq 7**: `Y_{t+1} = β1·μ̂ + β2·LAP + β3·(LAP×μ̂) + ε`; **Prop 1**: β3=0 under null (no contamination), β3>0 under contamination — a one-sided diagnostic.
2. **REQUIRES token logprobs** — verbatim: "requires only the model's first-token probabilities…"; impl reads `out.outputs[0].logprobs[0]` with `logprobs=20`. **CONFIRMED NOT FEASIBLE for Claude** (Anthropic Messages API exposes no logprobs) — validates 73.0's no-logprob constraint. LAP is a REJECT for our backbone.
3. **The transferable gift is NOT LAP — it is the pre/post-cutoff PLACEBO**: post-cutoff (calendar-2024, strictly after Llama-3.3-70B's documented **Dec-2023** cutoff) β3≈0 (t=1.06); Fig IV shows mean LAP "collapses essentially to zero" post-cutoff — the "smoking gun." **Method we CAN adopt**: use each backbone's documented training cutoff as a hard pre/post boundary and trust ONLY strictly-post-cutoff eval windows. Alignment example: earnings-call recall quarter 2024Q1 ↔ call q=2023Q3 (label horizon must also be post-cutoff). This is the spine of C2.

### E3 — Look-Ahead-Bench README (github.com/benstaf/lookaheadbench, arXiv:2601.13770), README-level (disclosed: paper `/html/` still 404s Jan-2026)
1. Measures **end-to-end agentic** lookahead — "how model performance degrades when moving from potentially memorized historical periods to genuinely unseen market regimes." Metric = **Alpha + Alpha Decay** (no formula/threshold in README).
2. Two fixed windows: **P1 in-sample Apr–Sep 2021**, **P2 OOS Jul–Dec 2024**. Confirms the two-window pre/post design C2 needs — but the README does NOT specify per-model cutoff enforcement (gap disclosed).
3. Contrasts standard LLMs (Llama-3.1 8B/70B, DeepSeek-3.2 — "severe alpha decay OOS; larger models generalize WORSE due to stronger memorized priors") vs **Point-in-Time models (Pitinf-Small/Medium/Large — stable/improving OOS)**. Corroborates the PiT thesis but is not itself buildable at our scale (needs bespoke PiT-trained models). Logprob requirement not stated in README.

### E5 — Time Machine GPT (arxiv.org/html/2404.18543), read in full
1. Core idea: a **series of point-in-time LLMs**, each trained ONLY on data published before Dec-31 of its year (2004–2023), so "no future information leaks into the model's representations." Problem it names verbatim: implicit associations like "Enron→bankrupt" or knowing "COVID-19" contaminate any test on data predating the event.
2. Construction: PiT Wikipedia snapshots (most-recent revision as of Dec-31) + WMT news with a negative-exponential 5-yr decay excluding future articles; fixed 0.6:0.4 domain ratio so only NEW information differs across yearly models.
3. Validation = leader-name / COVID perplexity spikes ONLY after the event date (conventional temporally-adapted models cheat with unrealistically low pre-event perplexity). Limitation: GPT-2-small scale, news+Wiki only.
4. **Application-at-our-scale**: we CANNOT train PiT models (API-only backbones). The transferable principle is the INVERSE and it is FREE for us: since our Gemini/Claude backbones have FIXED documented cutoffs, our **LIVE paper trail accumulated strictly AFTER each backbone's cutoff is inherently a Time-Machine-clean eval set** — no training needed, just correct window labeling. This is exactly what C2 formalizes.

### E4 — AFML purged K-fold + embargo + CPCV (Towards AI practitioner writeup of Lopez de Prado AFML Ch.7/Ch.12), read in full
1. **Purging (overlap condition)**: drop a training sample whenever its label-generation interval `[t, t+horizon]` intersects the test fold `[t_test_start, t_test_end]`. This is EXACTLY our `_label_overlaps_test` predicate `(s<=te) and (label_end>=ts)` — code-verified match.
2. **Embargo**: a buffer AFTER the test fold removing training samples before the next train block, for leakage purging alone can't catch (delayed reactions, autocorrelated features). Selection: "embargo should match or exceed the label-horizon length"; commonly a small % of observations. **NOTE our config**: our embargo is 5 DAYS while the label horizon is 135d — so the embargo handles autocorrelation and the PURGE handles the long label horizon; both are required and non-redundant (the purge is the load-bearing one for our 135d barrier).
3. **CPCV path count**: N groups, k test groups → φ = **C(N,k)·k / N** unique backtest PATHS. Example N=6,k=2 → C(6,2)=15 splits → 30 test-assignments → **5 paths**. Each observation is tested exactly once across paths; yields a DISTRIBUTION of OOS Sharpe vs walk-forward's single path.
4. **Why it cuts PBO**: "walk-forward tests ONE scenario (easily overfit); CPCV tests k scenarios uniformly across time… multiple paths make the probability of false discoveries negligible (given enough paths)." Grounds C4 as a robustness COMPLEMENT.

### E6 — The New Quant §7.1 + §7.10 (arxiv.org/html/2510.05533), read in full (targeted)
1. **§7.1 temporal leakage (verbatim)**: "Return prediction with general web pretraining risks look-ahead leakage because models may memorize future facts and surface them during prompting." Mitigation = "corpora with **strict publication cutoffs per evaluation fold**, training data filtered by crawl date and source type, **embargo windows** for validation, and **rationales that cite evidence published before the decision timestamp**." Latent knowledge leaks even without explicit future docs — grounds C2 + C3 (cutoff windows AND counterfactual audit needed, since post-cutoff windowing alone can't catch latent recall).
2. **§7.10 minimum reporting standard**: "time-safe data and splits with document availability at the decision timestamp"; full cost model (commissions/spreads/impact); turnover/capacity/net-of-cost; "wall-clock latency and **compute cost per decision**"; "release seeds and code to reconstruct time splits and point-in-time indices." (The cost half feeds D2b/#4, not this step; the time-safe-split half is C1/C2.)

## 6. Internal code inventory (file:line seams a build must touch)

| File | Lines | Role | Status |
|------|-------|------|--------|
| backtest_engine.py | :570-582 `_label_overlaps_test` | purge predicate `(s<=te) and (label_end>=ts)` == AFML overlap condition (E4) | LIVE (phase-69.2) — C1 documents |
| backtest_engine.py | :656 `horizon_days=int(holding_days*1.5)`; :659-664 purge loop; :428-430 call site passes test_start/test_end | purge wiring | LIVE — C1 documents |
| backtest_engine.py | :585+ `_build_predict_features` | was baseline's mis-cited `:587` — NOT a leak (predict-time imputation) | LIVE — C1 clears the stale F |
| walk_forward.py | :36 `embargo_days=5`; :61 `test_start=train_end+embargo+1` | 5-day embargo (autocorrelation only; purge covers the 135d horizon) | LIVE — C1 documents |
| autoresearch/gate.py | :21-22 `min_dsr=0.95`/`max_pbo=0.20`; :26-27 reads `trial['dsr']/['pbo']`; :42-59 `cpcv_folds()` DEFINED; :62 exported | promotion gate + CPCV enumerator defined-but-UNUSED | C4 seam (wire cpcv_folds); C3 bolt-on point |
| autoresearch/strategy_backtest_adapter.py | :38-41 CPCV-complement note; :95-152 CSCV K-variant column-stack; :247 `compute_pbo()` | today's PBO = K-variant column-stacking (NOT CPCV) | C4 wiring seam (add robustness path alongside, not replace) |
| services/meta_scorer.py | :203 `unwrap_secret(anthropic_api_key)`; :221 `ClaudeClient(claude-haiku-4-5)`; :237 `temperature=0.0` | conviction scorer on METERED direct Anthropic API | C3 perturbation seam (cheapest single-call decision surface) + cost flag |
| config/model_tiers.py | :50 `GEMINI_WORKHORSE="gemini-2.5-flash"`; :57-87/:223-228 claude model ids | model-id constants — **NO cutoff-date constant anywhere** | C2 gap (add MODEL_CUTOFFS registry here) |
| agents/llm_client.py | grep `cutoff` = EMPTY | no knowledge-cutoff handling exists | C2 gap |
| tests/regression/ (only `test_no_calendar_shadow.py`); tests/autoresearch/`test_phase_48_2_backtest_adapter.py` | — | NO purge/leak/embargo regression test exists | C1 gap (net-new test) |
| Live decision path | portfolio_manager.py, autonomous_loop.py, meta_scorer.py, signal_attribution.py, risk_overrides.py | where perturbed-context re-runs would execute | C3 targets (pick the cheapest surface) |

## 7. Design inputs per component (the design doc's skeleton)

### C1 — PURGE-VERIFICATION ($0)
**Spec skeleton:** (1) DOCUMENT the shipped mechanics as an invariant: purge predicate `_label_overlaps_test` (:570-582) drops train samples whose label span `[s, s+horizon]` overlaps test `[ts,te]`; `horizon_days=int(holding_days*1.5)` (:656) — at holding_days=90 → 135d, which must **≥ the triple-barrier vertical-barrier span (126-135d)**; the 5-day embargo (walk_forward.py:36) is a SEPARATE, weaker guard (autocorrelation) — the purge is load-bearing for the 135d horizon (E4). (2) REGRESSION TEST (net-new; home `tests/backtest/test_purge_no_leak.py` or `tests/regression/test_purge_embargo_invariant.py`): (a) truth-table unit test of `_label_overlaps_test` incl. boundary cases `s==te`, `label_end==ts`; (b) **property test — for a synthetic split, assert NO retained training sample has `label_end >= test_start`** (zero-overlap survives); (c) integration guard — call `_build_training_data(...,test_start,test_end)`, assert purged_count>0 when overlaps exist AND `max(retained label_end) < test_start`; (d) horizon invariant — assert `int(holding_days*1.5) >= vertical_barrier_days`. **Seams:** backtest_engine.py:570-582/:656/:428-430; walk_forward.py:36,61. **Cost:** $0 (pure code + pytest; quant-only path, no LLM, no macro).

### C2 — POST-CUTOFF HARNESS ($0)
**Spec skeleton:** (1) Add a `MODEL_CUTOFFS: dict[str,date]` registry in config/model_tiers.py mapping each backbone id (gemini-2.5-flash, claude-opus-4-8, claude-haiku-4-5, claude-sonnet-4-6, …) → its documented knowledge-cutoff date (operator-sourced; unknown → conservative/None = treat all history as suspect). (2) Eval-window selector: for any LLM-touched strategy, `trusted_start = max(cutoff of every backbone in the path) + purge_embargo_buffer`; the trustworthy eval window = `[trusted_start, now]`. LABEL each eval window with the governing cutoff + which backbones it clears (New Quant §7.1 "strict publication cutoffs per evaluation fold"). (3) Our LIVE paper trail since each cutoff is inherently Time-Machine-clean (E5) — so this is windowing/labeling, not retraining. (4) Quant-only GBM backtest is EXEMPT (no LLM features — rules/backend-backtest.md) — C2 governs only the live 20-agent signal-promotion path. **Seams:** config/model_tiers.py (registry, alongside :50/:223-228); promotion path gate.py / strategy_backtest_adapter.py (window guard); outcome_tracker EVAL_WINDOWS as the labeled substrate. **Cost:** $0 (date arithmetic + a constants table; no metered calls).

### C3 — COUNTERFACTUAL-AUDIT GATE (PILOT; METERED)
**Spec skeleton:** (1) Adopt FactFin's **PC only** for v1 (Eq 4, text-only → Claude-compatible): perturb the candidate decision agent's context M times (news content-swap strong↔weak per E1 App C; numeric feature Gaussian noise), re-run the SAME decision agent, `PC = fraction of predictions unchanged`; **high PC ⇒ memorization ⇒ reject** at promotion. (2) DEFER CI/IDS: CI needs a confidence scalar (our verbalized conviction works) but IDS needs a class-distribution (multi-sample vote-share → M×S calls, ~10× cost) — v1 is PC-only. (3) **NO paper threshold exists** (correction to 73.0) — calibrate a LOCAL cutoff on our own perturbation set; Table-5 anchor bands: leaky≈PC 0.62/IDS 0.44, clean≈PC 0.31/IDS 0.78 (E1). (4) Scope = tiny PILOT on a handful of promotion candidates, NOT the full universe; gate behind D2b/#4's cost meter before any scale-up. **Seams:** bolt onto gate.py (promotion time) / strategy_backtest_adapter.py; perturbed re-runs execute at meta_scorer.py:221 (`ClaudeClient(claude-haiku-4-5)`, metered anthropic_api_key). **Cost (refined from 73.0 "M~20-50"):** METERED (direct Anthropic API, confirmed meta_scorer.py:203/:221). PC-only = **M single Haiku calls/candidate** (~M×~2.25K tok; Haiku ≈$1/$5 per Mtok → ~$0.05-0.10/candidate at M=20). Full PC+CI+IDS = M×S calls ≈ ~10× (low $/candidate). Which live-decision rail (metered vs flat-fee Max) the 20-agent path uses is an UNRESOLVED data-gap (carried from 73.0) — design must let the operator pick the cheapest decision surface and hard-cap M.

### C4 — CPCV WIRING ($0; build-dark, validate-at-unfreeze)
**Spec skeleton:** (1) Wire the existing `cpcv_folds(n,k)` (gate.py:42, currently defined-but-unused) into a NEW robustness path that replays STORED prices across φ=**C(N,k)·k/N** paths (E4), applying purge+embargo per fold (reuse `_label_overlaps_test`). (2) Emit an **OOS-Sharpe DISTRIBUTION** (mean, std, worst-path, %paths Sharpe>0) as a robustness COMPLEMENT reported ALONGSIDE the existing K-variant CSCV **PBO scalar the gate consumes — do NOT replace it** (strategy_backtest_adapter.py:38-41; frontier-map REAL REFINEMENT). gate.py stays byte-unchanged; the gate keeps reading `trial['pbo']`. (3) Build under the macro freeze (code-only over cached OHLC — NO historical_macro), validate the distribution at un-freeze. **Seams:** gate.py:42 (cpcv_folds); strategy_backtest_adapter.py:95-152/:247 (new sibling to the CSCV column-stack). **Cost:** $0 (numpy/pandas over cached prices; no LLM, no macro, no metered spend).

## 8. Research Gate Checklist

Hard blockers (all satisfied):
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch: E1 Profit Mirage (2510.07920), E2 Detecting-Lookahead (2512.23847), E4 AFML CPCV (Towards AI practitioner writeup), E5 Time Machine GPT (2404.18543), E6 The New Quant §7.1/§7.10 (2510.05533) — 5 clean paper-body/practitioner reads, + E3 Look-Ahead-Bench README (README-level, disclosed).
- [x] 10+ unique URLs total (~17: 6 full + 8 snippet + carryovers).
- [x] Recency scan (last 2 years) performed + reported (§4).
- [x] Full papers/pages read (not abstracts) for the read-in-full set.
- [x] file:line anchors for every internal claim (§6, spot-verified this session via reads/greps of gate.py, walk_forward.py, backtest_engine.py, meta_scorer.py, model_tiers.py, llm_client.py, strategy_backtest_adapter.py, tests/).

Soft checks:
- [x] 3-variant query discipline (§3): current-year, last-2-year, year-less canonical.
- [x] Contradictions/consensus noted: LAP-needs-logprobs (blocked) vs PC-text-only (usable); FactFin has NO formal thresholds (73.0's PC>0.7 corrected); embargo≠purge (5d vs 135d).
- [x] Internal exploration covered every C1-C4 seam.

**Correction to 73.0 carried forward:** FactFin publishes NO accept/reject thresholds — the "PC>0.7 / IDS>0.6" figures were interpretive, not the paper's; C3 must calibrate a local cutoff on our own perturbation set.

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "D2a leakage-integrity DESIGN inputs, 4 components. C1 PURGE-VERIFICATION ($0): document the shipped AFML purge (_label_overlaps_test backtest_engine.py:570-582, horizon=int(holding_days*1.5)=135d, wired :428-430/:656) whose overlap condition (s<=te)and(label_end>=ts) EXACTLY matches E4's canonical purge; add a net-new regression test (no purge/leak test exists in tests/) asserting zero retained sample has label_end>=test_start. C2 POST-CUTOFF HARNESS ($0): NO cutoff constant exists in model_tiers.py/llm_client.py -> add MODEL_CUTOFFS registry + an eval-window selector trusting only strictly-post-cutoff windows; our live paper trail since each backbone cutoff is inherently Time-Machine-clean (E5) so it's windowing not retraining; New Quant §7.1 mandates 'strict publication cutoffs per evaluation fold'. C3 COUNTERFACTUAL-AUDIT GATE (METERED pilot): adopt FactFin PC only (Eq4, text-only, Claude-compatible; LAP/2512.23847 needs logprobs=BLOCKED); FactFin gives NO thresholds (corrects 73.0) so calibrate locally off Table-5 bands; perturbed re-runs fire meta_scorer.py:221 metered Haiku -> refined cost = M single calls/candidate ~$0.05-0.10 at M=20 (PC-only), ~10x for PC+IDS; keep a tiny pilot, cap M. C4 CPCV WIRING ($0): wire the defined-but-unused cpcv_folds (gate.py:42) as an OOS-Sharpe robustness DISTRIBUTION (phi=C(N,k)k/N paths, E4) reported ALONGSIDE the K-variant CSCV PBO scalar the gate consumes -- complement not replace (adapter :38-41); build-dark on cached prices under macro freeze, validate at un-freeze; gate.py byte-unchanged. 5 sources read in full + README; recency scan strengthens all 4.",
  "brief_path": "handoff/current/research_brief_73.1.md",
  "gate_passed": true
}
```

