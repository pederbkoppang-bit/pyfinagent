# Research Brief — phase-70.2: SOFT profit-aware cross-sector diversification (S2)

**Tier:** moderate (focused implementation-specifics; builds on the 70.0 complex pack) ·
**Date:** 2026-07-17 · **Author:** Layer-3 Researcher subagent · **HEAD:** 0bf7bb9b

**Step objective (S2 "no new stock from other sectors"):** implement SOFT, profit-aware
cross-sector diversification, flag-gated default-OFF (DARK-until-token), backend-only, $0,
paper-only, `historical_macro` FROZEN. Three parts from the 70.0 design:
1. PRIMARY — soft diversity penalty at rank time in `screener.rank_candidates` (shade, never
   zero, `composite_score` by sector representation; `w_d=0` ⇒ byte-identical).
2. SECONDARY — min-K-sector round-robin on the analyze slice at `autonomous_loop.py:838`
   (`K=0` ⇒ byte-identical).
3. Unknown-bucket fix — exempt the `"Unknown"` sector from the count/NAV caps so an
   enrichment failure cannot freeze the funnel.

**Binding (unchanged from 70.0):** flag-gated default-OFF (`w_d=0`/`K=0` byte-identical); $0;
paper-only; `historical_macro` FROZEN; NO risk threshold moved; HARD sector-neutral REJECTED
(2026-06-01 replay = −0.166 long-only Sharpe). Validation gate (Sharpe ≥ incumbent + DSR≥0.95
/ PBO≤0.5) before any activation token.

**Prior pack (READ FIRST, built on):** `handoff/current/research_brief_70.0.md`
(gate_passed=true, 7 sources in full) + `handoff/current/design_trade_diversity_70.md`
(design pack). This 70.2 brief does NOT re-derive the "why soft not hard" argument (settled
in 70.0 via Ehsani-Harvey-Li + the −0.166 replay); it focuses on the EXACT formula,
θ-scaling, min-K algorithm, and the validation harness mechanics.

---

## 1. Three-variant query disclosure (research-gate mandatory)

Per `.claude/rules/research-gate.md`, ≥3 query variants per topic (current-year 2026 /
last-2-year 2025-24 / year-less canonical).

**Topic A — soft/penalty diversification formula + θ scaling (HHI penalty on a rank score)**
- 2026 frontier: `portfolio diversification HHI penalty objective auto-scaled weight 2026`
- 2025 window: `soft sector constraint penalty portfolio optimization slack variable 2025`
- Year-less canonical: `Herfindahl Hirschman index portfolio concentration penalty objective function`

**Topic B — min-K / round-robin / stratified / diversified top-N selection**
- 2026 frontier: `diversified top-k selection re-ranking maximal marginal relevance 2026`
- 2025 window: `stratified selection diversity constraint ranking 2025`
- Year-less canonical: `maximal marginal relevance diversity re-ranking round-robin selection`

**Topic C — validating a ranking change does not lower risk-adjusted OOS P&L**
- 2026 frontier: `deflated Sharpe ratio strategy selection overfitting 2026`
- 2025 window: `probability backtest overfitting ablation out-of-sample validation 2025`
- Year-less canonical: `deflated Sharpe ratio Bailey Lopez de Prado multiple testing`

<!-- source table, recency scan, design recs appended below as sources are read -->

---

## 2. Source table

### Read IN FULL via WebFetch (6 — floor is 5)
| # | Source | Tier | Topic | Key takeaway (implementation-specific) |
|---|--------|------|-------|----------------------------------------|
| 1 | **arXiv 2601.08717v1** — *Portfolio Optimization with 'Physical' Decision Variables & Non-Linear Performance Metrics: Diversification Challenge* (Jan 2026), `arxiv.org/html/2601.08717v1` | 1 (peer-reviewed preprint) | A | The EXACT soft-penalty: `max_x [(1−w)·ROI(x) − w·Risk_β(x) − w_d·θ₁·HHI(x)]`. **θ₁ = [w·\|ROI(x)\|_mean + (1−w)·\|Risk_β(x)\|_mean] / \|HHI(x)\|_mean**, averaged over the Pareto-front baseline — rescales HHI to the ROI/risk order of magnitude. **`w_d∈[0,1]`; `w_d=0` ⇒ HHI term vanishes ⇒ baseline reproduced EXACTLY** (byte-identity). `HHI(x)=Σ(xᵢ/B)² ∈ [1/n, 1]`. Penalty **SHADES, never zeroes** — high-ROI assets keep substantial share, "compete against diversification gain." |
| 2 | **arXiv 2212.14464** — *Result Diversification in Search & Recommendation: A Survey* (via `ar5iv.labs.arxiv.org/html/2212.14464`) | 1 (peer-reviewed survey) | B | MMR greedy selection: `d_k = argmax_{d} [o^rel(d\|u) + λ·o^div(d\|σ^(1:k-1))]` with `o^div = −max_{d_j∈selected} sim(d,d_j)`. Greedy **sequential** selection "maximizes a joint measure of relevance and diversity at each position" ⇒ progressive category coverage. Coverage objective `S-Coverage = \|∪ subtopics(selected)\| / n_S`. Grounds the min-K round-robin as a post-processing re-rank layer. |
| 3 | **Deflated Sharpe Ratio** — Wikipedia (canonical: Bailey & López de Prado, *JPM* 2014, SSRN 2460551) | 2 (reference of a Tier-1 primary) | C | `DSR = Φ((SR̂*−SR₀)·√(T−1) / √(1 − γ̂₃·SR₀ + ((γ̂₄−1)/4)·SR₀²))`. **SR₀ = √V[SR̂] · ((1−γ)·Φ⁻¹[1−1/N] + γ·Φ⁻¹[1−1/(N·e)])** — the expected-max Sharpe under the null, rising with **N trials** (γ = Euler-Mascheroni). Deflates for selection bias / multiple testing / non-normality / sample length. Higher DSR ⇒ stronger evidence vs the zero-skill null. |
| 4 | **CFA Institute** — *Momentum Investing: A Stronger, More Resilient Framework for Long-Term Allocators* (2025) | 2/3 (official-body blog) | A | Multidimensional momentum (price + 10 signals) "delivers higher average returns, stronger t-stats, substantially improved drawdowns vs price momentum alone"; vol-scaling "cuts drawdowns nearly in half" WITHOUT sacrificing return. Composite Sharpe 0.38–0.94 (median 0.61). Does NOT neutralize sector — the modern long-only recipe is *keep* the exposure and diversify the SIGNAL, aligning with a shade-not-neutralize tilt. |
| 5 | **Dvara Research** — *Generalized HHI across Multiple Correlated Sectors* | 4 (practitioner) | A | `HHI = Σᵢ cᵢ² ∈ [1/N, 1]`; **`1/HHI` = effective number of positions/sectors**; correlation-aware `GHHI = Σcᵢ² + ΣΣ_{j≠i} 2·cᵢ·cⱼ·ρᵢⱼ`. HHI rises with concentration, falls with diversification. Gives the concentration metric for the validation harness's "avg sectors" column and any HHI-subtractive variant. |
| 6 | **arXiv 2408.09168** — *Ranking Across Different Content Types: The Robust Beauty of Multinomial Blending* (Aug 2024), `arxiv.org/html/2408.09168` | 1 (peer-reviewed preprint) | B | Round-robin done right: at each slot **sample a content type ~ p, take the highest-scoring remaining item of that type**; renormalize when a type empties. `p_c` = expected exposure share of type `c`. **Appendix lower-bound variant: if the baseline ranker already meets the desired exposure, blending doesn't fire — never drops a previously-favored type below its baseline.** Preserves within-type relevance order. This is the sector-round-robin blueprint (sectors = content types). |

### Snippet-only (evaluated, not read in full)
| Source | Topic | Why not read in full / value |
|--------|-------|------------------------------|
| **Springer, *J. Asset Management* 2026** — *Diversification effects of ESG penalties in mean–variance portfolios* (10.1057/s41260-026-00446-2) | A | **Auth-walled (303 → idp.springer.com).** Snippet (high value, 2026): "**linear penalties induce rapid and systematic concentration** as the effect of the sustainability objective grows"; proposes a **nonlinear/endogenous** penalty instead. ⇒ FUNCTIONAL FORM matters: a naive linear-subtractive tilt can concentrate pathologically; a bounded **multiplicative rank-decay** avoids it. Directly supports the design's `(1−w_d)^(j−1)` choice. |
| fffinstill — *HHI & Effective Positions* | A | Corroborates Dvara HHI + `1/HHI` effective-N. |
| MOSEK *Portfolio Optimization Cookbook* v1.6 (from 70.0) | A | "risk/weight/sector limits **softened with a penalty term** / slack variables" — official-docs corroboration of soft-over-hard. |
| Grokipedia *Maximal Marginal Relevance*; aayushmnit *Diversity MMR* (Dec 2025) | B | MMR still the standard diversified-selection primitive (recency + implementation refs). |
| SSRN 2460551 + davidhbailey.com/dhbpapers/deflated-sharpe.pdf | C | DSR canonical primary (read the Wikipedia distillation in full instead; formulas match). |
| PBO / CSCV — Bailey, Borwein, López de Prado, Zhu (2016), SSRN 2326253 | C | Basis of the project's `pyfinagent-risk::pbo_check` MCP (PBO ≤ 0.5 promotion gate). |
| arXiv 2603.20319 — *Implementation Risk in Portfolio Backtesting* (2026) | C | Recency: previously-unquantified backtest error sources — reinforces DSR/PBO discipline on the replay. |
| MDPI *Sector Rotation TSX 60 (2000–2025)* | A | OOS-validated sector rotation; tangential (rotation is 70.3, not 70.2). |
| arXiv 2512.24526 — *Generative-AI Sector-based Portfolio Construction* | A | Sector-bucket construction; tangential. |
| Ehsani-Harvey-Li FAJ 2023 + QuantPedia (from 70.0) | A | The settled "long-only keep sector exposure" result — not re-read (70.0 read in full). |

**URLs collected this session:** ~34 unique across 4 searches (floor 10; 70.0 added ~48 more).
6 read in full; ~10 snippet-only recorded above.

---

## 3. Recency scan (last 2 years) — MANDATORY SECTION

**New findings in the 2024–2026 window (this focused session):**
1. **The auto-scaled HHI-penalty is the Jan-2026 frontier** (arXiv 2601.08717): supplies the
   exact `−w_d·θ₁·HHI` term AND the θ₁ auto-rescaling formula AND the `w_d=0 ⇒ exact baseline`
   property this step's byte-identity requirement depends on. Newer than / operationalizes the
   70.0 canon.
2. **Functional form of the penalty matters — a 2026 result** (Springer *J. Asset Mgmt* 2026):
   LINEAR penalties "induce rapid and systematic concentration." ⇒ prefer a **bounded
   multiplicative rank-decay** `(1−w_d)^(j−1)` over a linear `−λ·count`. Supersedes a naive
   linear-subtractive default.
3. **Probabilistic round-robin with exposure guarantees** (arXiv 2408.09168, Aug 2024): the
   modern way to guarantee ≥K category representation while preserving within-category order,
   with a "don't drop a favored category below baseline" lower-bound variant — a cleaner
   blueprint than a naive hard interleave for the min-K sector slice.
4. **Long-only momentum: diversify the signal, keep the sector** (CFA Institute 2025): the
   2025 practitioner consensus is multidimensional momentum + vol-scaling for risk-adjusted
   returns; it does NOT sector-neutralize — consistent with the 70.0 "shade, don't neutralize"
   thesis and the −0.166 internal replay.

**Canonical prior art still valid:** MMR (Carbonell-Goldstein 1998), HHI (Herfindahl/Hirschman),
Deflated Sharpe (Bailey-López de Prado 2014 JPM), PBO/CSCV (Bailey et al. 2016), and
Ehsani-Harvey-Li (FAJ 2023, the long-only keep-sector result). No 2024–2026 work overturns the
long-only conclusion; the newer papers operationalize *how* to shade concentration while keeping
the across-sector signal.

**Internal recency anchors:** (i) 2026-06-01 replay = **−0.166** hard-neutral long-only Sharpe
(`screener.py:71-73`) — the sign the soft variant must beat; (ii) phase-69.2 note — **DSR was
de-annualized to Bailey's per-period definition (0.9004)**: feed the replay's *monthly* returns
to the DSR test, NOT the `×√12` annualized Sharpe (`ann_sharpe`, replay `:116-120`).

---

## 4. Internal code audit (EXACT file:line on HEAD 0bf7bb9b)

Files inspected: `backend/tools/screener.py`, `backend/services/autonomous_loop.py`,
`backend/services/portfolio_manager.py`, `backend/config/settings.py`,
`scripts/ablation/sector_neutral_replay.py`, plus the 70.0 pack + `design_trade_diversity_70.md`.

### 4a. The rank/slice funnel — the EXACT two truncations

The candidate funnel truncates **TWICE**, and this is the load-bearing fact for the design:

**Truncation A (rank level) — `screener.py:483-484`:**
```
scored.sort(key=lambda x: x["composite_score"], reverse=True)
return scored[:top_n]
```
Called from `autonomous_loop.py:703-705` with `top_n=settings.paper_screen_top_n` (**default
10**, `settings.py:366`). In a semis-led regime the top-10 by pure-momentum composite is
mono-/duo-sector. **This is where cross-sector names are structurally discarded.** The
`composite_score` is built at `screener.py:299-303` (`mom_1m*0.40 + mom_3m*0.35 +
mom_6m*0.25`, RSI/vol penalties `:305-311`), then mutated by ~14 optional overlays
(`:318-409`), appended at `:411`, and the sort/truncate is `:483-484`. **The soft penalty
must be inserted just BEFORE `:483` so it reshuffles which names survive Truncation A.**

**Truncation B (analyze slice) — `autonomous_loop.py:837-838`:**
```
new_candidates = [c for c in candidates if c["ticker"] not in held_tickers]
analyze_tickers = [c["ticker"] for c in new_candidates[:settings.paper_analyze_top_n]]
```
`paper_analyze_top_n` **default 5** (`settings.py:367`). Naive top-5-by-score of the non-held
top-10. **This is where the min-K round-robin replaces the plain slice.**

### 4b. Sector availability at each stage (CRITICAL WIRING FINDING)

- **`screen_data` rows carry a sector ONLY when `_sector_lookup` is built**, which is GATED at
  `autonomous_loop.py:432-436`:
  ```
  _sector_lookup = None
  if getattr(settings,"sector_neutral_momentum_enabled",False) or getattr(settings,"multidim_momentum_enabled",False):
      _sector_lookup = build_sector_map(universe)
  screen_universe(..., sector_lookup=_sector_lookup, ...)   # :440-446
  ```
  `screen_universe` attaches `row["sector"]` only when `sector_lookup` is truthy
  (`screener.py:233-239`). **⇒ With both flags OFF (default), every `screen_data` row has NO
  sector, so `rank_candidates` sees `stock.get("sector")=None` on every row.** The soft
  penalty groups by sector → **it is a silent no-op unless the new flag ALSO triggers the
  `_sector_lookup` build.** MAIN MUST add `paper_soft_sector_diversity_enabled` to the OR at
  `:433`. `build_sector_map` (`screener.py:64-88`) is one Wikipedia read, $0, cheap.
- **The top-10 `candidates` ARE enriched unconditionally** via `_fetch_ticker_meta`
  (`autonomous_loop.py:744-761`, `if candidates:` — NO flag gate) — sets `c["sector"]`. This
  runs at `:744`, i.e. BEFORE Truncation B at `:838`. **⇒ The min-K round-robin at `:838` has
  sectors available on `new_candidates` today, no extra wiring.** (Enrichment is ALREADY
  before the `:838` slice; the 70.0 "move enrichment before truncation" phrasing describes
  the existing order — the real change at `:838` is making the *slice* sector-aware, not
  moving enrichment.)
- **Reach asymmetry (why both levers are needed):** the min-K round-robin at `:838` operates
  only on the already-truncated top-10 (Truncation A done). If the top-10 is mono-sector it
  cannot introduce a new sector. **The soft penalty (rank level, before Truncation A) is the
  lever that pulls cross-sector names INTO the top-10; the round-robin then guarantees they
  reach the top-5 analyze slice.** Shipping only min-K without the soft penalty (or without
  its `_sector_lookup` wiring) achieves ~nothing in a mono-sector top-10.

### 4c. The EXISTING hard `sector_neutral` lever (REJECTED — do not flip)

`screener.py:444-474`: when `sector_neutral=True`, replaces `composite_score` with the
within-sector percentile rank in [0,1] (`_apply_pct_rank`, `:464-469`); raw preserved on
`composite_score_raw`; groups < `sector_neutral_min_group_size` (default 3) + `_UNKNOWN_`
fall back to a global percentile pool (`:459-462`). This DESTROYS across-sector momentum (a
+40% semi and a +2% utility each top their sector → both ≈1.0). 2026-06-01 replay = −0.166
long-only Sharpe (docstring `screener.py:71-73`). **The soft penalty is a NEW code path;
leave this lever untouched.** Existing flags: `sector_neutral_momentum_enabled` (`:427`),
`sector_neutral_min_group_size` (`:428`).

### 4d. The Unknown-sector bucket in the count/NAV caps (`portfolio_manager.py`)

`decide_trades` sector caps collapse missing sectors into a single `"Unknown"` bucket at
FIVE sites — so an enrichment failure lumps ALL candidates into one bucket and the count cap
(default 2) freezes the funnel at 2 buys:
- `:272` candidate build: `"sector": cand_sector or "Unknown"`
- `:319` held-position seed: `s = (pos.get("sector") or "").strip() or "Unknown"` →
  `sector_counts[s]`, `sector_market_values[s]`
- `:360` COUNT cap check: `cand_sector = cand.get("sector") or "Unknown"`; block if
  `sector_counts[cand_sector] >= max_per_sector` (`:362`, default 2, `settings.py:269`)
- `:395` NAV-pct cap check: `cand_sector_nav = cand.get("sector") or "Unknown"`; block if
  projected sector NAV% > `max_sector_nav_pct` (`:400`, default 30, `settings.py:277`)
- `:456` post-BUY increment: `cs = cand.get("sector") or "Unknown"`
**Fix:** treat `"Unknown"` as cap-exempt (an UNKNOWN sector is NOT evidence of concentration).
Guard the count-cap block (`:359-369`) and NAV-cap block (`:394-408`) with
`and cand_sector != "Unknown"` (and skip seeding/incrementing the Unknown bucket, or simply
never enforce against it). Flag-gate so OFF ⇒ byte-identical.

### 4e. Validation harness — `scripts/ablation/sector_neutral_replay.py` (RUNS NOW, $0, macro-free)

**Confirmed the OOS-P&L check for criterion 2 can run NOW, without the activation token and
without `historical_macro`:**
- `$0`: free yfinance batch download (`:148`) + one Wikipedia `read_html` for GICS sectors
  (`load_universe_sectors`, `:37-52`). **No LLM, no BQ, no `historical_macro`, no optimizer.**
  Docstring `:9-12` states this verbatim.
- Replays the **PRODUCTION** `screener.rank_candidates` (imported `:23`) over monthly
  rebalances 2022-2025 (`:164-171`), S&P 500, top_n=10 (`:25`), 21-day forward returns
  (`basket_fwd_return :101-113`), annualized Sharpe (`ann_sharpe :116-120`), sector spread
  (# distinct GICS) + turnover.
- Already parameterized by config: `configs = {"baseline": False, "sector_neutral": True}`
  (`:173`) → **the soft-penalty variant is added by extending `configs`/`rank_candidates`
  kwargs and re-running** (same pattern the 52.1 tilt used, `hi52_tilt_basket :123-138`).
- Emits a machine verdict block (`:247-267`) and dumps paired monthly return arrays
  (`:272-281`) for a deterministic Ledoit-Wolf / DSR SR-difference test.
- **⇒ Criterion 2 (no OOS P&L drop) is provable NOW via a `w_d × K` grid replay; the
  activation TOKEN is a separate operator gate, not a data dependency.** This is the single
  most important internal finding for the validation plan.

### 4f. Flags to model the new ones on (`backend/config/settings.py`)

- `sector_neutral_momentum_enabled: bool = Field(False, description="phase-28.4: ...")` (`:427`)
- `momentum_52wh_tilt_enabled: bool = Field(False, ...)` (`:442`) +
  `momentum_52wh_tilt_k: float = Field(0.5, ...)` (`:443`) — the exact `enabled`+`float-knob`
  pair to mirror for `paper_soft_sector_diversity_enabled` + `_w`.
- `paper_max_per_sector: int = Field(2, ge=0, le=20, ...)` (`:269`) — int-with-bounds model
  for `paper_min_k_sectors_analyzed`.
- `paper_analyze_top_n: int = Field(5, ...)` (`:367`).
- **Confirmed: no pre-existing `paper_soft_sector_diversity*` / `paper_min_k_sectors*` flag**
  anywhere under `backend/` (grep clean).

---

## 5. Design recommendations

All three parts flag-gated **default-OFF ⇒ byte-identical**; backend-only; $0; paper-only;
`historical_macro` FROZEN; NO risk threshold moved.

### 5.0 New flags (`backend/config/settings.py`) — model on `momentum_52wh_tilt_enabled`/`_k` (`:442-443`)

```python
paper_soft_sector_diversity_enabled: bool = Field(False,
    description="phase-70.2: PRIMARY soft cross-sector diversity. When True, rank_candidates shades composite_score by same-sector rank-position (the j-th same-sector name x (1-w)^(j-1)) BEFORE the top-N truncation, so cross-sector names surface. SHADES never zeroes (dominant sector's #1 keeps full score) -> respects the -0.166 hard-neutral replay + Ehsani-Harvey-Li long-only. w=0 => byte-identical. Profit-aware analog of arXiv 2601.08717 -w_d*theta1*HHI. Default OFF. Also triggers the build_sector_map wiring at autonomous_loop.py:433.")
paper_soft_sector_diversity_w: float = Field(0.0, ge=0.0, le=1.0,
    description="phase-70.2: w_d rank-decay strength in [0,1]. 0 => byte-identical (multiplier (1-0)^j=1). ~0.10-0.20 = mild shade. Grid-validate via scripts/ablation/sector_neutral_replay.py before any live enable.")
paper_min_k_sectors_analyzed: int = Field(0, ge=0, le=11,
    description="phase-70.2: SECONDARY. Min distinct GICS sectors guaranteed to reach the deep-analyze slice (autonomous_loop.py:838), best-effort/capped by available sectors. 0 => byte-identical top-N slice. Max 11 (GICS). Round-robin fill preserving within-sector score order (arXiv 2408.09168).")
paper_unknown_sector_cap_exempt: bool = Field(False,
    description="phase-70.2: when True, the 'Unknown' sector is EXEMPT from the per-sector COUNT + NAV caps (portfolio_manager.py) -- a ticker-meta enrichment failure that lumps every candidate into 'Unknown' can no longer freeze the funnel at max_per_sector. Default OFF => byte-identical (Unknown enforced as today).")
```
Recommend FOUR separate flags (not one) so the Unknown-bucket fix + the min-K slice can each
ship/activate independently of the rank-time penalty. All default-OFF.

### 5.1 PRIMARY — soft diversity penalty in `screener.rank_candidates` (the profit-aware tilt)

**Formula (RECOMMENDED — multiplicative rank-decay; self-scaling, exact byte-identity):**
within each sector, the *j*-th candidate (by raw composite order) is multiplied by
`(1 − w_d)^(j−1)`. j=1 ⇒ ×1 (sector leader untouched — keeps across-sector momentum);
j=2 ⇒ ×(1−w_d); etc. `w_d=0 ⇒ ×1` everywhere ⇒ byte-identical. **Why multiplicative not
linear-subtractive:** it is dimensionless/scale-free (no θ₁ needed — it is automatically
commensurate with any score magnitude, unlike the paper's additive `−w_d·θ₁·HHI` which
*requires* θ₁ rescaling), bounded, monotone, and cannot flip a sign or over-concentrate the
way a LINEAR penalty does (Springer 2026). It SHADES not zeroes (arXiv 2601.08717) → a
genuinely dominant sector still wins its share (respects −0.166).

**Insertion point:** `screener.py` in `rank_candidates`, a NEW block placed AFTER `:411` (the
`scored.append`) and AFTER the existing `sector_neutral`/`multidim`/`52wh` blocks
(`:436-481`), immediately **BEFORE `scored.sort(...); return scored[:top_n]` (`:483-484`)**:
```python
# phase-70.2: soft cross-sector diversity (default OFF -> byte-identical).
if soft_sector_diversity and soft_sector_diversity_w > 0 and scored:
    _apply_soft_sector_diversity(scored, soft_sector_diversity_w)
```
New signature params on `rank_candidates`: `soft_sector_diversity: bool = False,
soft_sector_diversity_w: float = 0.0` (append to the kwargs list ~`:274`).
```python
def _apply_soft_sector_diversity(scored: list[dict], w: float) -> None:
    """In-place rank-decayed same-sector shade. j-th same-sector name (by raw
    composite order) *= (1-w)^(j-1). w=0 => no-op. Preserves raw on
    composite_score_raw (mirrors _apply_52wh_tilt). Missing sector -> 'Unknown'
    bucket shaded together (conservative; the Unknown-cap-exempt flag handles the
    downstream caps separately)."""
    order = sorted(range(len(scored)),
                   key=lambda i: scored[i].get("composite_score") or 0.0, reverse=True)
    seen: dict[str, int] = {}
    for i in order:
        s = scored[i]
        key = (s.get("sector") or "").strip() or "Unknown"
        j = seen.get(key, 0)                    # 0-based count already-seen higher
        s["composite_score_raw"] = s.get("composite_score")
        s["composite_score"] = round((s.get("composite_score") or 0.0) * ((1.0 - w) ** j), 4)
        seen[key] = j + 1
```
One-pass over the raw-sorted list (deterministic, O(n log n)); the subsequent `:483` sort
re-orders by the shaded score so cross-sector names rise. (A fully-greedy MMR re-selection —
recompute after each pick, arXiv 2212.14464 — is the more faithful variant but heavier and
unnecessary here; the one-pass matches the design pack's `(1−w_d)^(j−1)`.)

**CRITICAL WIRING (else silent no-op):** `screen_data` rows have NO sector unless
`_sector_lookup` is built. **Main MUST extend the gate at `autonomous_loop.py:433`:**
```python
if (getattr(settings,"sector_neutral_momentum_enabled",False)
    or getattr(settings,"multidim_momentum_enabled",False)
    or getattr(settings,"paper_soft_sector_diversity_enabled",False)):   # <-- ADD
    _sector_lookup = build_sector_map(universe)
```
and thread the two new kwargs into the `rank_candidates(...)` call at `:703-711`:
`soft_sector_diversity=getattr(settings,"paper_soft_sector_diversity_enabled",False),
soft_sector_diversity_w=getattr(settings,"paper_soft_sector_diversity_w",0.0),`.
`build_sector_map` is one Wikipedia read ($0). Held-awareness is intentionally NOT seeded at
rank time (positions are fetched later, `:835`); it is covered end-to-end by (a) the min-K
round-robin operating on the held-excluded `new_candidates` and (b) the held-aware
per-sector caps (`portfolio_manager.py:319`). Threading `held_sectors` into rank time is an
OPTIONAL higher-diff enhancement, not required for 70.2.

**Optional SUBTRACTIVE variant (if a true HHI term is wanted later):** `score −= w_d·θ₁·HHI`,
with `θ₁ = [w·mean|score| ] / mean|HHI|` computed over the current `scored` set (arXiv
2601.08717). NOT recommended for 70.2 — needs magnitude bookkeeping and does not give the
clean `×1` byte-identity; the multiplicative form is strictly simpler for the same intent.

### 5.2 SECONDARY — min-K-sector round-robin on the analyze slice (`autonomous_loop.py:838`)

Enrichment already runs on the top-10 `candidates` at `:744-761` BEFORE this slice, so
`new_candidates` carry sectors. Replace the naive slice at `:838`:
```python
# phase-70.2: min-K-sector round-robin (default K=0 -> byte-identical slice).
_K = int(getattr(settings, "paper_min_k_sectors_analyzed", 0) or 0)
_N = settings.paper_analyze_top_n
if _K > 0 and new_candidates:
    analyze_tickers = _min_k_sector_slice(new_candidates, _N, _K)
else:
    analyze_tickers = [c["ticker"] for c in new_candidates[:_N]]
```
```python
def _min_k_sector_slice(cands, n, k):
    """cands already score-desc. Take the leader of each of the K highest-'peak'
    distinct sectors (guarantees >=min(K, #sectors) sectors reach the analyzer),
    then fill remaining slots by pure score. Best-effort: fewer sectors than K =>
    take what exists + score-fill. Preserves within-sector order (arXiv 2408.09168
    multinomial-blend leader pick). K=0 never reaches here."""
    from collections import OrderedDict
    by_sector = OrderedDict()
    for c in cands:                                   # cands is score-desc
        by_sector.setdefault((c.get("sector") or "Unknown"), []).append(c)
    picked, ids = [], set()
    for s in list(by_sector.keys())[:k]:              # K strongest distinct sectors' leaders
        if len(picked) >= n: break
        c = by_sector[s][0]; picked.append(c); ids.add(c["ticker"])
    for c in cands:                                   # fill remainder by score
        if len(picked) >= n: break
        if c["ticker"] in ids: continue
        picked.append(c); ids.add(c["ticker"])
    picked.sort(key=lambda x: x.get("composite_score") or 0.0, reverse=True)
    return [c["ticker"] for c in picked[:n]]
```
The round-robin can pull a lower-score cold-sector leader into *analysis* — that is
funnel-widening, NOT a forced buy (the downstream deep-analyze + RiskJudge + BUY-lean still
filter it). Reach note: min-K only diversifies among the top-10 sectors, so it is
**complementary to 5.1** (5.1 widens the top-10's sector span; 5.2 guarantees that span
reaches the top-5). Shipping 5.2 alone in a mono-sector top-10 gains ~nothing — validate them
together.

### 5.3 Unknown-bucket fix (`portfolio_manager.py`, findings #5/#14)

Exempt `"Unknown"` from the COUNT cap (`:359-369`) and NAV cap (`:394-408`) when
`paper_unknown_sector_cap_exempt` is ON. Minimal diff — guard only the ENFORCEMENT (leave the
`:319` seed / `:456` increment as-is; harmless since the check is skipped):
```python
# COUNT cap (:359):
if max_per_sector > 0:
    cand_sector = cand.get("sector") or "Unknown"
    _unk_ok = getattr(settings, "paper_unknown_sector_cap_exempt", False) and cand_sector == "Unknown"
    if not _unk_ok and sector_counts.get(cand_sector, 0) >= max_per_sector:
        ... sector_blocked.append(cand); continue
# NAV cap (:394):
if max_sector_nav_pct > 0 and nav > 0:
    cand_sector_nav = cand.get("sector") or "Unknown"
    _unk_ok = getattr(settings, "paper_unknown_sector_cap_exempt", False) and cand_sector_nav == "Unknown"
    if not _unk_ok:
        ... existing projected-NAV check ...
```
OFF ⇒ `_unk_ok=False` ⇒ enforcement byte-identical. (`settings` is already in scope in
`decide_trades`.) Rationale: an UNKNOWN sector is missing-data, NOT evidence of concentration;
capping it collapses N distinct real sectors into one bucket and starves the funnel at 2.

### 5.4 Validation plan — criterion 2 (no OOS risk-adjusted P&L drop)

**Provable NOW, $0, macro-free — NOT deferred to the activation token.**
`scripts/ablation/sector_neutral_replay.py` is confirmed **$0** (free yfinance batch + one
Wikipedia `read_html`; docstring `:9-12`), **no LLM, no BQ, no `historical_macro`, no
optimizer** — so running it does NOT violate the FROZEN constraint. It replays the PRODUCTION
`rank_candidates` (`:23`, `:194`) over 2022-2025 monthly rebalances and already dumps paired
monthly return arrays (`:272-281`) for a deterministic SR-difference test.

**Plan:**
1. Extend `configs` (`:173`) with soft-penalty variants: call
   `rank_candidates(rows, top_n=TOP_N, strategy="momentum", soft_sector_diversity=True,
   soft_sector_diversity_w=w)` for a grid `w ∈ {0.05, 0.10, 0.20, 0.30}` (and optionally apply
   `_min_k_sector_slice` to the basket for `K ∈ {2,3,4}`). Same pattern the 52.1 tilt used
   (`hi52_tilt_basket`, `:123-138`).
2. Report per-config annualized Sharpe + avg distinct sectors + turnover vs `baseline`
   (the harness already prints this + the `dSharpe`/`dSectors` verdict, `:247-267`).
3. **Promotion gate (immutable):** grant the activation token ONLY if the chosen `(w_d, K)`:
   **OOS Sharpe ≥ incumbent baseline** AND **DSR ≥ 0.95** AND **PBO ≤ 0.5**. DSR via the
   Bailey-López de Prado formula (§2 #3) on the paired **monthly** returns (NOT the `×√12`
   annualized SR — phase-69.2 de-annualization); PBO via `pyfinagent-risk::pbo_check` on the
   paired-config PnL matrix. Same harness that produced the −0.166 hard-neutral verdict ⇒
   apples-to-apples sign comparison.

**Now vs token:** the OOS-P&L EVIDENCE is produced in 70.2 (or a 70.2 grid-replay follow-on) —
criterion 2 is NOT blocked on the token. The **token is the operator's sign-off on that
evidence** (flip `paper_soft_sector_diversity_enabled`/`_w`/`paper_min_k_sectors_analyzed` in
prod). Until then the flags stay OFF and the live loop is byte-identical. If the grid replay
shows NO `(w_d,K)` clears the gate, the feature ships DARK and no token is requested (do-no-harm
honored — same disposition as hard-neutral).

### 5.5 What Main must NOT do (binding guardrails)
- Do NOT flip `sector_neutral_momentum_enabled` (the hard −0.166 lever) — it is a separate,
  rejected path; the soft penalty is a NEW code block.
- Do NOT move any risk threshold (`paper_max_per_sector`, NAV-pct cap, stops, kill-switch,
  DSR/PBO gates) — the Unknown fix EXEMPTS a bucket, it does not change a limit value.
- Do NOT enable any flag by default; every default is OFF and `w_d=0`/`K=0` must be
  byte-identical (add a quick ON-vs-OFF `$0` diff check to `experiment_results.md`).
- Do NOT run the optimizer or any `historical_macro` write; the replay is read-only cached data.

---

## 6. Gate envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```

`gate_passed: true` — external_sources_read_in_full (6) ≥ 5, recency_scan_performed, brief
written (incrementally). Read in full: arXiv 2601.08717 (θ₁/HHI formula), arXiv 2212.14464
(MMR), DSR (Wikipedia), CFA momentum 2025, Dvara GHHI, arXiv 2408.09168 (multinomial blend).
Internal files inspected: screener.py, autonomous_loop.py, portfolio_manager.py, settings.py,
sector_neutral_replay.py.
