# Research Brief — phase-73.0 "D1 DEEP FRONTIER STUDY" (GATE)

Tier: **complex** (caller-specified). NOT audit-class.
Role: This is the D1 GATE — anchor reads + syllabus validation + the reading
plan for the GENERATE fan-out. Parallel dimension-readers read the rest of the
~25-item syllabus in full during GENERATE; this session does NOT read all of it.

Context read in full first:
- `handoff/current/frontier_baseline_2026-07-18.md` (graded dimension map,
  read-in-full syllabus, binding critic verdicts).
- `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` (role).

Scale constraints binding on every adopt/reject question below:
- 2-person local paper fund on Peder's Mac (no fleet, `project_local_only_deployment`).
- `historical_macro` FROZEN until operator token (no optimizer runs that need it).
- $0 dev metered budget; Claude Max flat-fee rail only.
- Existing debate/architecture (#8, A-) and statistical gates (#9, B+) STAY — do
  NOT rebuild (critic verdict 1).
- Compounding chain #3 → #2 → #1 → #4 (clean backtests → learn-loop → calibrated
  sizing → net-of-cost objective); #5/#6/#7 are judged pilots behind them.

---

## Status: IN PROGRESS (write-first; sections filled as sources are read)

---

## 1. Anchor reads (READ IN FULL — counts toward the gate)

| # | Source | arXiv | Accessed | Fetched how | Kind |
|---|--------|-------|----------|-------------|------|
| A1 | "The New Quant: A Survey of LLMs in Financial Prediction and Trading" (Weilong Fu, Columbia) | 2510.05533 | 2026-07-18 | `/html/` OK | meta-survey |
| A2 | "Large Language Model Agent in Financial Trading: A Survey" (Ding, Li, Wang et al.) | 2408.06361 | 2026-07-18 | `/html/` OK (v2 revised 2026-03-01) | agent survey |
| A3 | "Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents" (Li, Zeng, Xing et al.) | 2510.07920 | 2026-07-18 | `/html/` OK | leakage (FinLake-Bench/FactFin) |
| A4 | "FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design" (Yu, Li, Chen et al., Stevens) | 2311.13743 | 2026-07-18 | **ar5iv** (`/html/` returned a broken PRIME-AI LaTeX template — pre-Dec-2023) | memory/reflection |
| A5 | "Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution" (Tian, Han, Chen et al.) | 2508.06225 | 2026-07-18 | `/html/` OK (v3) | calibration |

### Anchor key findings (per-claim, at-our-scale lens)

**A3 Profit Mirage — the single most consequential anchor; it validates the baseline's #3→#2→#1 chain and its leakage-skepticism verdict.**
1. Mechanism, verbatim: *"the model does not learn why prices move; it learns that they already moved, and simply recites the answer during back-testing."* "Pre-training contamination is lethal in finance."
2. Quantified decay: agents evaluated pre-cutoff (Q2-Q3 2021) vs post-cutoff (Q3-Q4 2024) show **Sharpe decay 51.48% (QuantAgent) → 62.23% (FinCON)** and **Total-Return decay 50.18% (TradingAgents) → 71.85% (FinMem)** — despite near-identical market tape (+13.79% vs +13.35%). This is direct evidence that ALL the frontier return numbers in the syllabus are pre-cutoff-inflated.
3. FinLake-Bench memorization audit: GPT-4o/Claude-3.7/Grok-3 answer historical price/trend/event trivia at **85.37%–92.94% accuracy** — encyclopedic recall, not forecasting.
4. FactFin mitigation = counterfactual perturbation + leakage metrics (Prediction-Consistency, Confidence-Invariance, Input-Dependency-Score); reject a strategy with PC>0.7 / require IDS>0.6 before deploy. AT OUR SCALE: the *counterfactual-audit idea* (perturb history, measure prediction stability) is a cheap, LLM-agnostic gate we could bolt onto our promotion pipeline; the full MCTS/SCG code-gen apparatus is out of scope.

**A4 FinMem — the memory-architecture blueprint for dimension #2 (our D-grade, coded-but-dark).**
1. 3-layer decay memory: shallow (news, half-life Q=14, α=0.9, decays to threshold in 30d), intermediate (10-Q, Q=90, α=0.967, 90d), deep (10-K, Q=365, α=0.988, 365d). Retrieval score = recency(exp(−δ/Q)) + relevancy(cosine on embeddings) + importance(v×θ, v∈{40,60,80}).
2. Reflection: immediate (merge market signal + top-K events per layer → Buy/Sell/Hold + rationale + influential events) and extended (re-evaluate over M-day trace, promoted into deep layer). A pivotal event gets +5 importance and recency reset to 1.0.
3. LEAKAGE FLAG for our fan-out: FinMem test window Oct-2022→Apr-2023 sits AT GPT-4-Turbo's Apr-2023 cutoff; reported TSLA CR **61.78% / Sharpe 2.68**. Profit Mirage (A3) independently rates FinMem the WORST post-cutoff decayer (71.85% TR loss). Adopt the *layered-decay + written-reflection mechanism*; treat the *return numbers as leakage-suspect* (critic verdict 2).

**A5 Calibration (LLM-as-a-Judge overconfidence) — the mechanism anchor for dimension #1 (our F-grade).**
1. Core finding: *"predicted confidence levels significantly overstate actual correctness"* — models cluster at 90-100% confidence but sit well below the calibration line (GPT-4o: 49.71% acc but ECE 39.25).
2. Methods that matter for us: Self-Confidence vs Multiple-Prompting (10 samples @ T=0.7, confidence = vote share) vs Logprob; metrics ECE/ACE/Brier/TH-Score; the LLM-as-a-Fuser ensemble cut ECE to 6.42% and lifted accuracy to 86.29%.
3. CAVEAT for our adopt/reject: this paper's calibration target is judge CORRECTNESS, not trade WIN-RATE, and it explicitly gives *"No explicit position-sizing guidance."* So it is a **mechanism anchor** (how to measure/repair overconfidence) — the actual conviction→hit-rate-bucket→size mapping is OUR build (fan-out reader C must design it, not lift it).

**A1 "The New Quant" — the organizing meta-survey; confirms our dimension map.**
1. Three closing principles map 1:1 onto our posture: *"Separate concerns"* (signal vs portfolio construction — we do), *"Bind language to evidence"* (timestamped citations + tool-verified calc before any position change — our debate/RAG), *"Evaluate like a practitioner"* (time-safe splits, realistic costs, turnover/capacity, **report latency + compute cost per bp of excess return** — directly our dimension #4 north star).
2. Explicitly flags temporal leakage (§7.1 "models may memorize future facts") and says *"evaluation practice often falls short of trading standards"* — corroborates A3.
3. Cost-as-objective is a *stated production requirement* here (§7.5/§7.10 minimum reporting standard = compute cost per decision) — supports building dimension #4 (net-of-token-cost objective).

**A2 Agent survey (v2, revised 2026-03) — the taxonomy anchor; the honest-limitations source.**
1. Taxonomy: LLM-as-Trader (news/reflection/debate/RL-driven) vs LLM-as-Alpha-Miner; memory via FinMem layered buckets; reflection from cognitive science (interact→feedback→memory→lesson).
2. Two limitations that bind our judgment: **median test period only 1.3 years** ("a short and single backtesting period may diminish the credibility of the results") and *"few studies consider trading costs in their evaluations."* Both are arguments FOR our #3 (clean/longer eval) and #4 (cost objective), and AGAINST trusting the raw 15-30% annualized frontier returns.

---
## 2. Syllabus validation (HEAD/snippet-level — the fan-out reads these in full)

Every non-anchor syllabus reference was resolved to a live source + best full-text route. **3 corrections / flags** worth Main's attention are bolded.

| # | Reference (baseline label) | Resolves? | Best full-text access |
|---|---|---|---|
| S1 | TradingAgents (arXiv:2412.20138) | OK — "TradingAgents: Multi-Agents LLM Financial Trading Framework", Xiao/Sun/Luo/Wang, 28-Dec-2024 (v7 Jun-2025). Cross-confirmed: A3 tested it. | `arxiv.org/html/2412.20138v7` |
| S2 | FinCon (arXiv:2407.06567) | OK — "FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement…", manager-analyst hierarchy. Cross-confirmed: A3 tested it (62.23% SR decay). | `arxiv.org/html/2407.06567v3` |
| S3 | QuantAgent (arXiv:2402.03755) | OK — "QuantAgent: Seeking Holy Grail in Trading by Self-Improving LLM". The only token-cost-budgeted line (A1, A2). | `arxiv.org/html/2402.03755` |
| S4 | Awesome-Self-Evolving-Agents (XMUDeepLIT) | OK — GitHub repo `github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents`; companion survey "A Systematic Survey of Self-Evolving Agents" (SSRN 6626878). NB a second repo `EvoAgentX/Awesome-Self-Evolving-Agents` also exists — use XMUDeepLIT per baseline. | GitHub README + survey PDF |
| S5 | AlphaAgent (arXiv:2502.16789, "KDD'25") | OK — "AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay", Tang et al., Feb-2025. **FLAG: abstract lists NO IR/return numbers; the baseline's "IR 1.5" and "KDD'25" venue are UNCONFIRMED at abstract level — fan-out must verify in the body.** | `arxiv.org/html/2502.16789v2` |
| S6 | QuantaAlpha (arXiv:2602.07085) | OK — 2026 preprint CONFIRMED real: "QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining", Feb-2026 (v2 May-2026). Headline IC 0.1501 / ARR 27.75% / MDD 7.98% on GPT-5.2, CSI300→CSI500/S&P500 transfer. **HYPE-FLAGGED (self-reported, pre-cutoff).** | `arxiv.org/html/2602.07085v1` |
| S7 | Alpha-GPT (EMNLP 2025) | OK — ACL Anthology `2025.emnlp-demos.14` (System Demonstrations); underlying paper arXiv:2308.00016. WorldQuant top-10/41000 claim. | ACL Anthology HTML/PDF |
| S8 | MarketSenseAI 2.0 (arXiv:2502.00415) | OK — "MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents", Feb-2025. Residual alpha ~15%; **125.9% cum. return (S&P100, 2023-24) — the leakage-suspect number baseline #6 flagged.** | `arxiv.org/html/2502.00415v2` |
| S9 | ECC Analyzer (ICAIF'24, DOI 10.1145/3677052.3698689) | OK — Cao/Chen/Pei/Lee/Subbalakshmi/Ndiaye, ICAIF 2024 (Brooklyn, Nov-2024). Earnings-call trading-signal extraction. | `dl.acm.org/doi/fullHtml/10.1145/3677052.3698689` (as baseline) |
| S10 | market-feedback adaptive retrieval (arXiv:2605.31201) | OK — 2026 preprint CONFIRMED. **CORRECTION: real title is "Point-in-Time Financial RAG with Frozen LLMs and Market-Feedback Adaptive Retrieval" (Zhao & Welsch, MIT), May-2026.** Bayesian source memory updated from MATURED RESIDUAL-RETURN feedback — a tighter #6 match than the baseline description; also directly relevant to #2's feedback plumbing. | `arxiv.org/html/2605.31201v2` |
| S11 | Multi-agent deliberation calibration (arXiv:2404.09127) | OK — "Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation", Yang/Rajagopal/Hayati/Hu/Kang, Apr-2024. Training-free post-hoc "Collaborative Calibration"; code `minnesotanlp/collaborative-calibration`. Directly usable — WE ALREADY RUN DEBATE. | `arxiv.org/html/2404.09127` |
| S12 | Amazon Science "Label with Confidence" | OK — "Label with Confidence: Effective Confidence Calibration and Ensembles in LLM-Powered Classification", GenAIECommerce 2024. Logit-based calibration → 46% ECE reduction; cost-aware cascading ensemble → >2x cheaper. **Ties #1 (calibration) to #4 (cost) in one method.** | `assets.amazon.science/.../label-with-confidence-…pdf` (open PDF) |
| S13 | Deflated Sharpe Ratio (SSRN 2460551) | OK — Bailey & López de Prado, JPM 2014. Canonical, ours. | `davidhbailey.com/dhbpapers/deflated-sharpe.pdf` (open PDF) |
| S14 | PBO (davidhbailey.com backtest-prob) | OK — "The Probability of Backtest Overfitting", Bailey/Borwein/LdP/Zhu (SSRN 2326253; same one cited by our `pyfinagent-risk` MCP `pbo_check`). Canonical, ours. | `davidhbailey.com/dhbpapers/backtest-prob.pdf` (open PDF) |
| S15 | CPCV comparison (ScienceDirect S0950705124011110) | OK — "Backtest overfitting in the ML era: A comparison of out-of-sample testing methods…", Arian/Norouzi/Seco, Knowledge-Based Systems 2024. **ScienceDirect is paywalled; free full text on SSRN 4686376.** CPCV beats Walk-Forward on PBO/DSR — directly indicts our un-purged walk-forward (#3). | SSRN 4686376 (open) |
| S16 | Look-Ahead-Bench (baseline: "ResearchGate 399953316") | OK but **CORRECTION: the canonical source is arXiv:2601.13770** "Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance" (2026); ResearchGate 399953316 is a mirror. Code `github.com/benstaf/lookaheadbench`. End-to-end agentic look-ahead measurement — the operational test for our #3. | `arxiv.org/html/2601.13770` |
| S17 | Kelly (AQR) "Limits to (Machine) Learning" (arXiv:2512.12735) | OK — "Limits To (Machine) Learning", Chen/Kelly/Malamud, 14-Dec-2025. Introduces the "Limits-to-Learning Gap" lower bound; extends Hansen-Jagannathan. Industry/theory reality-check on ML predictability ceilings. | `arxiv.org/html/2512.12735` |
| S18 | Man Group AlphaGPT (ai-street.co + Bloomberg 2025-07-10) | OK — ai-street.co multi-part series + Bloomberg "Man Group Says Agentic AI Is Now Devising Quant Trading Signals" (2025-07-10) + Hedgeweek. Dozens of AI signals APPROVED for live trading under IDENTICAL promotion thresholds; NO published returns. Validates our #10 human-gate + identical-gate posture. | ai-street.co articles (open); Bloomberg (may paywall) |

**Syllabus validation verdict:** all 18 non-anchor references resolve. Two ID/access corrections (S10 real title; S16 → arXiv:2601.13770) and one unverified-claim flag (S5 IR/venue) are handed to the fan-out. No syllabus item is dead/hallucinated — the 2026-dated IDs (S6, S10, S16) are all genuine fresh preprints.

---

## 3. Reading plan for the GENERATE fan-out (5 dimension-readers)

Each reader must independently clear the ≥5-read-in-full floor during GENERATE (anchors already read by THIS session can be re-read for section-level depth and count for that reader). Every question is scoped **AT OUR SCALE**: 2-person local paper fund on Peder's Mac, `historical_macro` frozen until token, $0 dev metered, existing debate/gates STAY, chain order #3→#2→#1→#4.

**R-A — Leakage integrity (dimension #3, our F-grade — FIRST in the chain).**
Sources: Profit Mirage 2510.07920 (FactFin/counterfactual detail), Look-Ahead-Bench 2601.13770, CPCV comparison SSRN 4686376, Detecting-Lookahead-Bias 2512.23847 (recency), The-New-Quant §7.1 + Time-Machine-GPT ref, + internal `backend/backtest/backtest_engine.py:587` and `autoresearch/gate.py`.
Adopt/reject questions: (1) Confirm walk-forward has no purge/embargo and triple-barrier 90-135d labels leak — quantify the fix. (2) Adopt CPCV-over-walk-forward, or purge+embargo the existing WF? Which is buildable without unfreezing macro? (3) Is a cheap counterfactual-audit gate (perturb history, measure prediction-consistency, reject PC>0.7) bolt-on-able to our promotion pipeline? (4) Can we add an LLM-pretraining-cutoff guard (only trust post-cutoff eval windows) given Gemini/Claude backbones — is this even measurable locally?

**R-B — Memory/reflection + self-improvement (dimensions #2 D-grade + #5 C-grade).**
Sources: FinMem 2311.13743 (exact decay params), agent-survey 2408.06361, QuantAgent 2402.03755 (token-budgeted self-improve), FinCon 2407.06567 (conceptual verbal reinforcement), Self-Evolving-Agents survey (XMUDeepLIT).
Adopt/reject questions: (1) Adopt FinMem's 3-layer decay memory (shallow/intermediate/deep, half-lives 14/90/365) over our flat BM25 `memory.py`? Cost/benefit at our ticker count. (2) Add a written reflection per CLOSED trade into `outcome_tracker` — and first, what exactly crashes `outcome_tracker.py:50` (P3 sheet)? (3) Is the champion-bridge wiring (best_params→`decide_trades`) safe to build while macro is frozen, or does validation require the un-freeze token? (4) Reject or defer? — flag anything that needs metered spend or a fleet.

**R-C — Calibration → sizing (dimension #1, our F-grade — the direct P&L multiplier).**
Sources: LLM-as-Judge-overconfidence 2508.06225 (methods), Collaborative-Calibration 2404.09127, Amazon "Label with Confidence" (logit calibration + cost), Autorater-calibration 2510.00263 (recency), + internal meta-scorer + advisory RJ-pct sizing path.
Adopt/reject questions: (1) Which elicitation fits our 1-10 meta-scorer — verbalized, multi-sample vote-share, or logprob — given Gemini/Claude access locally? (2) Design the conviction→realized-hit-rate-bucket map: how many buckets, what history window, where stored (BQ)? (3) How does a calibrated conviction feed position size WITHOUT breaking existing sector caps / risk vetoes? (4) Can our EXISTING bull/bear/DA debate double as the multi-agent-deliberation calibrator (2404.09127) at zero extra agents? (5) Explicit non-lift: A5/S11/S12 calibrate CORRECTNESS not WIN-RATE — the trade-outcome mapping is our build; confirm no paper hands it to us.

**R-D — Cost objective + industry reality (dimension #4 C-grade + #10 reality-check).**
Sources: The-New-Quant 2510.05533 (cost-per-bp reporting standard), QuantAgent 2402.03755 (token budgeting), Kelly "Limits to (Machine) Learning" 2512.12735, Man Group AlphaGPT (ai-street + Bloomberg), Amazon "Label with Confidence" (cost-aware ensemble).
Adopt/reject questions: (1) Concretely, how do we fold token + slippage + fees into the promotion objective (`autoresearch/gate.py`) — additive penalty, or a net-of-cost DSR? We already track ~$51/window + a turnover penalty. (2) What is the minimum cost-reporting shape (cost per bp of excess return) we should log per decision? (3) Does the Limits-to-Learning-Gap bound imply our DSR≥0.95 / PBO≤0.20 gates are already appropriately conservative, or mis-set? (4) Man Group runs identical gates for AI vs human signals + publishes NO returns — what does that confirm about our recommend-only posture and about discounting academic return claims?

**R-E — Pilots literature: news/RAG evidence-weighting (#6 B) + factor mining (#7 C-) — HONEST-JUDGMENT reader, default-DEFER.**
Sources: MarketSenseAI 2.0 2502.00415, Point-in-Time Financial RAG 2605.31201, ECC Analyzer (ICAIF'24), AlphaAgent 2502.16789, QuantaAlpha 2602.07085, Alpha-GPT (EMNLP 2025), QuantEvolve 2510.18569 (recency).
Adopt/reject questions: (1) #6: is matured-residual-return evidence weighting (2605.31201's Bayesian source memory, frozen LLM) a CHEAP upgrade that reuses #2's feedback plumbing — or premature before #2 works? (2) #7: every factor-miner here reports pre-cutoff, self-reported returns (QuantaAlpha 27.75%, MarketSenseAI 125.9%) — judge honestly whether ANY factor-mining pilot has positive expected value AT OUR SCALE before #3 (clean backtests) exists; the critic's prior is "low plausibility, small pilot at most." (3) State plainly what to DEFER and why — do not manufacture adopt-recommendations to look productive.

---

## 4. Recency scan (2025-2026 — mandatory; per dimension)

Searched 2025 + 2026 windows plus year-less canonical for each dimension. Findings:

- **Leakage (#3) — the frontier MOVED here in 2025-26, and it supersedes casual backtest practice.** Profit Mirage (Oct-2025), Look-Ahead-Bench arXiv:2601.13770 (Jan-2026), "Detecting Lookahead Bias in LLM Forecasts" arXiv:2512.23847 (Dec-2025), and Point-in-Time (Pitinf) LLMs are all NEW. Consensus: web-scale LLMs memorize post-hoc market narratives → pre-cutoff backtests are a "profit mirage." Our #3 F-grade is exactly the gap this fresh literature calls lethal. **This is the highest-value recency finding.**
- **Calibration (#1) — active 2025-26 area.** 2508.06225 (Aug-2025, LLM-as-Judge overconfidence + Fuser), 2510.00263 "Judging with Confidence: Calibrating Autoraters" (Oct-2025), 2605.11954 miscalibration mitigation (2026) complement the 2024 Collaborative-Calibration line. The reusable thread: multi-agent deliberation improves calibration — and we already run debate.
- **Cost-as-objective (#4) — newly formalized.** The New Quant (Oct-2025) codified "compute cost per bp of excess return" as a MINIMUM reporting standard (§7.10). QuantAgent remains the only token-budgeted agent. Net: our north star ("net, cost-adjusted") is now the field's stated standard, not an eccentricity.
- **Factor mining (#7) — hottest but most leakage-suspect 2025-26 area.** QuantaAlpha (Feb-2026), AlphaAgent (Feb-2025), Alpha-GPT EMNLP (2025), QuantEvolve arXiv:2510.18569 (Oct-2025), EFS arXiv:2507.17211. High publication volume, uniformly self-reported pre-cutoff returns → treat as mechanism evidence only (critic verdict 2).
- **News/RAG (#6) — freshest concrete upgrade is 2026.** Point-in-Time Financial RAG arXiv:2605.31201 (May-2026, matured-residual-return Bayesian memory + frozen LLM) is the most directly adoptable #6 mechanism and overlaps #2's feedback plumbing. MarketSenseAI 2.0 (Feb-2025) is the leakage-suspect return exemplar.
- **Memory/reflection (#2) — mechanism stable; frontier is decay-tiered + self-evolving.** FinMem (2023) is now the canonical baseline; 2025-26 movement is the self-evolving-agents survey wave (XMUDeepLIT / EvoAgentX) + FlashEvolve arXiv:2605.08520. No supersession of the layered-decay idea — it's the accepted design.
- **Industry reality (#10) — went public 2025.** Man Group AlphaGPT (Bloomberg 2025-07-10): agentic AI generating LIVE-approved signals under identical promotion gates, no published returns. Kelly/Malamud "Limits to (Machine) Learning" (Dec-2025) sets a theoretical predictability ceiling. Both reinforce our human-gate + gate-everything posture.
- **Bonus surfaced (not in syllabus; hand to fan-out if useful):** QuantEvolve 2510.18569, Detecting-Lookahead-Bias 2512.23847, Autorater-calibration 2510.00263, financial-LLM eval suite 2602.19073.

**Recency verdict:** the 2025-26 window materially STRENGTHENS the baseline's compounding-chain thesis — the leakage literature is newer and more damning than the baseline conveyed, and the cost-objective is now a codified field standard. No 2025-26 finding overturns the "do-not-rebuild debate/gates" verdict.

---

## 5. Research Gate Checklist

Hard blockers (all satisfied):
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (A1 2510.05533, A2 2408.06361, A3 2510.07920, A4 2311.13743 via ar5iv, A5 2508.06225) — 5 anchors, full-text.
- [x] 10+ unique URLs total — ~40 collected (5 anchors + 18 syllabus refs + ~15 recency/bonus/mirror URLs).
- [x] Recency scan (last 2 years) performed + reported (Section 4).
- [x] Full papers read (not abstracts) for the read-in-full set.
- [x] file:line anchors for internal claims (backtest_engine.py:587, outcome_tracker.py:50, autoresearch/gate.py, multi_agent_orchestrator.py:1238 — carried from baseline, not independently re-read this session; fan-out R-A/R-B verify).

Soft checks:
- [x] Contradictions/consensus noted (A3 vs the frontier return claims; Kelly ceiling vs factor-mining hype).
- [x] All claims cited per-claim with URL + access date 2026-07-18.
- Note: this is a GATE session by design — internal code was NOT deep-read (baseline already did the graded inventory with file:line evidence); the fan-out readers do the in-full internal + remaining-syllabus reads.

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 20,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 0,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "D1 GATE for phase-73.0. Read 5 anchors in full: The-New-Quant (2510.05533), agent survey (2408.06361), Profit Mirage (2510.07920), FinMem (2311.13743 via ar5iv — /html/ gives a broken template), calibration/LLM-as-Judge (2508.06225). Profit Mirage is decisive: frontier agents lose 50-72% of return post-cutoff, so all syllabus return claims are pre-cutoff-inflated (mechanism evidence only). Validated all 18 non-anchor syllabus refs — every one resolves incl. the 2026 IDs (QuantaAlpha 2602.07085, PiT-RAG 2605.31201, Look-Ahead-Bench). Three corrections handed to fan-out: S10 real title is PiT Financial RAG, S16 canonical source is arXiv:2601.13770 (not the ResearchGate mirror), S5 AlphaAgent IR/KDD claims unverified at abstract level. Reading plan = 5 GENERATE readers (leakage-integrity; memory+self-improve; calibration->sizing; cost+industry; pilots-DEFER) with at-our-scale adopt/reject questions honoring the #3->#2->#1->#4 chain, frozen macro, $0 metered, keep debate/gates. Recency scan: leakage lit is newer+more damning than baseline conveyed; cost-per-bp is now a codified field standard.",
  "brief_path": "handoff/current/research_brief_73.0.md",
  "gate_passed": true
}
```
