# research_brief -- phase-52.2: wire the 52wh tilt live (config-gated, default OFF)

Tier: **moderate** (caller specified full gate floor: >=5 sources read in full,
3-variant queries, recency scan, >=10 URLs, internal audit). $0 LLM.

## Objective

Wire the phase-52.1-MEASURED centered multiplicative 52-week-high momentum tilt
into LIVE `screener.rank_candidates` as a CONFIG-GATED cross-sectional post-pass,
DEFAULT OFF, byte-identical when off, tested. The enable (flag flip) is DEFERRED
to a separate post-Monday-baseline operator action -- do NOT enable here.

Tilt formula (measured at k=0.5 in 52.1: +0.05 ann Sharpe, turnover-neutral, on
the S&P-500 replay):
```
tilted = composite * (1 + k*(pct_to_52w - universe_mean_pct_to_52w))   # k=0.5
```
`pct_to_52w = close / trailing-252d-high`, already in prod (screener.py:213-214).
Cross-sectional post-pass: needs the universe mean over `scored` -> runs AFTER
all composites + RSI/vol multipliers, BEFORE the final sort.

#1 CONSTRAINT: DON'T REGRESS the working live engine (+20% NAV). Byte-identical
when the flag is OFF (the default), proven by a test.

---

## Part A -- INTERNAL CODE AUDIT (file:line, all CONFIRMED by reading)

### A1. The EXACT insertion point in `rank_candidates` (screener.py:249-475)
`rank_candidates` builds `scored` (a list of `{**stock, "composite_score": ...}`
dicts) in the per-stock loop (:287-409), then runs two EXISTING gated
cross-sectional post-passes, then sorts:
- per-stock loop ends at **:409** `scored.append({**stock, "composite_score": round(score, 3)})`
- news-only surfacing (:411-426)
- **multidim post-pass** (:434-440) -- `if multidim_momentum and scored: _apply_multidim_momentum(...)`
- **sector_neutral post-pass** (:448-472) -- `if sector_neutral and scored: ...`
- **final sort** at **:474** `scored.sort(key=lambda x: x["composite_score"], reverse=True)`
- return `scored[:top_n]` (:475)

=> **INSERT the gated 52wh-tilt post-pass at line 473** -- AFTER the
sector_neutral block (ends :472) and IMMEDIATELY BEFORE the `scored.sort` (:474).
This guarantees the tilt runs after every composite is computed AND after both
existing re-scoring passes, and that the tilt's mutation of `composite_score` is
honoured by the single final sort. The cross-sectional mean is computed over the
SAME `scored` set being ranked (cross-sectional, exactly like the replay).

Recommended shape (mirrors the existing `_apply_multidim_momentum`/sector_neutral
idiom -- a helper that mutates `scored` in place + preserves the raw score):
```python
# phase-52.2: optional centered 52-week-high multiplicative tilt (George-Hwang
# 2004). Default OFF. When ON, tilts composite_score UP for names nearer their
# 52w high, DOWN for names far below it, centered on the universe mean so the
# average tilt ~= 1.0 (turnover-neutral). Measured +0.05 ann Sharpe @ k=0.5 on
# the S&P-500 replay (phase-52.1). Missing pct_to_52w_high -> tilt 1.0 (no-op).
if momentum_52wh_tilt and scored:
    _apply_52wh_tilt(scored, k=momentum_52wh_tilt_k)
```
with the helper (faithful to `sector_neutral_replay.py:123-138 hi52_tilt_basket`):
```python
def _apply_52wh_tilt(scored: list[dict], k: float = 0.5) -> None:
    pcts = [s.get("pct_to_52w_high") for s in scored
            if s.get("pct_to_52w_high") is not None]
    mp = (sum(pcts) / len(pcts)) if pcts else 1.0     # universe mean (cross-sectional)
    for s in scored:
        p = s.get("pct_to_52w_high")
        tilt = (1 + k * (p - mp)) if p is not None else 1.0   # missing -> 1.0 no-op
        s["composite_score_raw"] = s.get("composite_score")
        s["composite_score"] = round((s.get("composite_score") or 0.0) * tilt, 3)
```
This reproduces the replay's ranking EXACTLY: same centered tilt
`composite * (1 + k*(p - mean))`, same missing->1.0 rule, same `scored` set for
the de-meaning. NB: the replay's `hi52_tilt_basket` computes the mean over rows
WITH a non-None pct (`:130`); the helper above matches that (filters Nones from
the mean), so the live ranking == the 52.1-measured ranking.

### A2. `pct_to_52w_high` is on EVERY screen_data row at rank time -- CONFIRMED
`screen_universe` (screener.py) computes `pct_to_52w_high` INSIDE the per-ticker
loop (the `row = {...}` dict built per ticker), at **:210-214 / :228**:
```python
high_52w = float(close.rolling(252, min_periods=20).max().iloc[-1])   # :213
pct_to_52w_high = round(current_price / high_52w, 4) if high_52w > 0 else None  # :214
...
"pct_to_52w_high": pct_to_52w_high,  # phase-28.7                      # :228
```
It is set for ALL screened names (not just top-N) -- it is a field on the `row`
dict appended to `results` for every ticker that clears the basic filters (:240),
and `rank_candidates` receives that full `screen_data` list. So the tilt has the
feature at rank time for every name. Names whose close window is < ~20 trading
days get `pct_to_52w_high=None` (the `min_periods=20` guard + the except), which
the helper treats as tilt=1.0 (no-op) -- correct robustness. **No threading
needed: the feature already flows screen_universe -> rank_candidates.** (This is
the live equivalent of `sector_neutral_replay.py:87/97` which puts the same field
on every replay row.)

### A3. The FLAG pattern (byte-identical when OFF) -- mirror multidim/sector_neutral
Three pieces, exactly paralleling the existing `multidim_momentum` /
`sector_neutral_momentum` wiring:

(1) **New `rank_candidates` kwargs** (add to the signature, screener.py:249-273,
alongside `multidim_momentum: bool = False`):
```python
momentum_52wh_tilt: bool = False,
momentum_52wh_tilt_k: float = 0.5,
```
Defaults `False`/`0.5` -> when the caller passes nothing, the `if
momentum_52wh_tilt and scored:` guard is False -> the post-pass is SKIPPED ->
`scored` is untouched -> identical to today. (Same inert-when-disabled pattern as
multidim at :434 and sector_neutral at :448.)

(2) **New settings** (backend/config/settings.py, alongside the
`multidim_momentum_*` block at :334-338):
```python
momentum_52wh_tilt_enabled: bool = Field(False, description="phase-52.2: centered multiplicative 52-week-high tilt on the screener composite (George-Hwang 2004). Measured +0.05 ann Sharpe @ k=0.5 on S&P-500 replay (phase-52.1). Default OFF.")
momentum_52wh_tilt_k: float = Field(0.5, description="phase-52.2: tilt strength k. tilted = composite * (1 + k*(pct_to_52w - universe_mean)). k=0.5 was the 52.1-measured setting.")
```

(3) **Call-site** (autonomous_loop.py -- the ONE live caller). `rank_candidates`
is called at **:638-666** (confirmed the only live call; `grep` shows the import
at :25 and the call at :638). Add two args alongside the existing
`multidim_momentum=getattr(settings, "multidim_momentum_enabled", False)` at :649:
```python
momentum_52wh_tilt=getattr(settings, "momentum_52wh_tilt_enabled", False),
momentum_52wh_tilt_k=getattr(settings, "momentum_52wh_tilt_k", 0.5),
```
The `getattr(..., False)` default means even if settings is stale/missing the
field, the flag is False -> no-op. Identical defensive idiom to the existing
overlays (:646-654).

### A4. Byte-identity proof when OFF (the test)
Because the post-pass is the LAST mutation before the sort and is fully behind
`if momentum_52wh_tilt and scored:`, with the kwarg defaulting False, the OFF
path executes the EXACT same code as today (the new helper is never called, the
signature gains two unused defaulted params). Prove it with a test that asserts
the ranked output is identical with the flag absent vs explicitly False:
```python
data = [ {with pct_to_52w_high}, ... ]   # synthetic screen_data
out_default = rank_candidates(data, top_n=10, strategy="momentum")
out_off     = rank_candidates(data, top_n=10, strategy="momentum",
                              momentum_52wh_tilt=False)
assert out_default == out_off            # byte-identical, flag-absent == flag-False
# and the tilt is NOT applied when off:
assert all("composite_score_raw" not in r for r in out_off)   # helper never ran
# ON changes the ranking (sanity, not a regression):
out_on = rank_candidates(data, top_n=10, strategy="momentum",
                         momentum_52wh_tilt=True, momentum_52wh_tilt_k=0.5)
assert any(r.get("composite_score_raw") is not None for r in out_on)
```
Stronger still: snapshot the full `out_off` list and assert it equals a pre-52.2
golden (the existing screener tests already exercise `rank_candidates`; add the
flag-off equality assertion to lock byte-identity). The helper writes
`composite_score_raw` ONLY when it runs, so its ABSENCE on every row is a clean
witness that the OFF path never touched `scored`.

### A5. Centering robustness -- CONFIRMED matches the 52.1 reference
- De-meaning is over the SAME `scored` set being ranked (cross-sectional): the
  helper computes `mp` from `scored` and tilts `scored` in place -> identical to
  `hi52_tilt_basket` which de-means over `ranked_all` and ranks `ranked_all`.
- Missing `pct_to_52w_high` -> `tilt = 1.0` (no-op for that name) -- matches
  `sector_neutral_replay.py:135` exactly.
- The live version therefore produces the SAME ranking the replay measured (same
  formula, same mean basis, same None-handling). The only nuance: prod
  `pct_to_52w_high` is `round(...,4)` (:214) vs the replay's full-precision float
  (`:87`); rounding to 4dp is immaterial to the ranking (sub-0.01% tilt deltas).

---

## Part B -- EXTERNAL RESEARCH

### Read in full (>=5 required; counts toward the gate) -- 6 sources

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://martinfowler.com/bliki/DarkLaunching.html | 2026-06-01 | authoritative blog (Fowler) | WebFetch (full) | Canonical dark-launch definition. "If we use a Feature Flag we can switch the recommendation on and off easily in production, so if we do see a worrying impact on performance we switch it off." Code "does all the work it would do when it's released, but nobody can see that it's doing it" -> decouple deploy from release; the disabled path is INERT. |
| https://launchdarkly.com/blog/guide-to-dark-launching/ | 2026-06-01 | official (vendor) | WebFetch (full) | Default-OFF + instant reversibility are the core safety properties: "the feature flag can be disabled with one click (without having to restart your application)". Deploy code with flag OFF, validate, then enable gradually (1%/5%/10%). |
| https://arxiv.org/html/2603.09219 (AlgoXpert Alpha Research Framework, IS/WFA/OOS) | 2026-06-01 | peer-reviewed (arXiv) | WebFetch (arXiv HTML) | Promotion-gate protocol: chronological IS -> WFA -> OOS gates; "if WFA FAILs, the strategy does not proceed to OOS holdout"; params "locked after WFA with no further tuning"; mandatory artifact logging + decision trace; prefer "plateaus" over "cliff" zones (robustness > peak metric). Maps directly to: wire-now (deploy, OFF) then ENABLE only after the Monday OOS baseline. |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf (Bailey & Lopez de Prado, DSR) | 2026-06-01 | peer-reviewed | WebFetch (PDF, partial text) | DSR corrects the observed Sharpe DOWN for (a) the effective number of independent trials and (b) skew/kurtosis. "testing many strategies inflates the probability of finding spuriously high-performing candidates." => the +0.05 dSharpe from 52.1 must be deflated by the number of configs tried (k=0.5, k=1.0, baseline, vol_scaled, sector_neutral) before it justifies ENABLING -- the wire-now/enable-later split is the right discipline. |
| https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf (George & Hwang 2004, J.Finance LIX:5) | 2026-06-01 | peer-reviewed | pdfplumber (binary->text) | THE 52wh paper. Winner-Loser (30% high/low ratio of current price to 52w high): 52wh 0.45%/mo **(t=2.00)** vs JT individual-stock momentum 0.48% (t=2.35) vs MG industry 0.45% (t=3.43). "Nearness to the 52-week high dominates and improves upon the forecasting power of past returns." "Future returns forecast using the 52-week high DO NOT REVERSE in the long run" (short-term momentum + long-term reversal are separate). Monthly formation/holding (J,K) construction; CRSP 1963-2001 (all sizes, equal-weighted). |
| https://arxiv.org/html/2512.11913v1 (Not All Factors Crowd Equally, alpha decay, 2025) | 2026-06-01 | peer-reviewed (arXiv) | WebFetch (arXiv HTML) | RECENCY. Momentum alpha decays hyperbolically `alpha(t)=K/(1+lambda*t)`, R^2=0.65, "from ~1.5 in the mid-1990s to ~0.25 today"; post-2015 acceleration ("0.30 predicted vs 0.15 actual"), ETF-volume-correlated crowding (rho=-0.63). KEY for a long-only momentum book: **crowded momentum has LOWER crash risk** (0.38x crash prob, p=0.006) -- "many investors reinforcing the trend tends to sustain rather than reverse it"; reversal factors are the crowded-crash risk, not momentum. "crowding-based factor selection fails to generate alpha (Sharpe 0.22 vs 0.39)" -> don't time on crowding; just monitor decay. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.digitalapplied.com/blog/feature-flag-rollout-strategies-2026-engineering-playbook | industry | 2026 feature-flag playbook; corroborates default-OFF + ramp; lower-tier than Fowler/LaunchDarkly read in full. |
| https://docs.getunleash.io/guides/gradual-rollout | official (vendor) | Open-source flag platform; gradual-rollout mechanics corroborate LaunchDarkly. |
| https://learn.microsoft.com/en-us/training/modules/implement-canary-releases-dark-launching/ | official (MSFT) | Canary + dark-launch training; corroborates the inert-when-off pattern. |
| https://www.efmaefm.org/.../EFMA%202021_...id-368.pdf (Wang, "What explains price momentum and 52-week high momentum") | peer-reviewed | ADVERSARIAL-adjacent on large caps; the large-cap-mute concern captured from George-Hwang internal controls + Barroso-Wang snippet. |
| https://www.sciencedirect.com/science/article/abs/pii/S0165188926000321 ("Proximity to 52-week high and risk-return trade-off", 2026) | peer-reviewed | RECENCY hit; abstract-gated; recency captured below. |
| https://onlinelibrary.wiley.com/doi/10.1111/jofi.13501 (Muravyev, "Anomalies and Their Short-Sale Costs", J.Finance 2025) | peer-reviewed | RECENCY; short-sale-cost decay -- less relevant to a LONG-ONLY book; snippet only. |
| https://afajof.org/management/viewp.php?n=46984 ("What Drives Anomaly Decay?") | peer-reviewed | Decay-mechanism context; corroborates McLean-Pontiff; snippet. |
| https://www.sciencedirect.com/science/article/abs/pii/S1386418122000465 ("race to exploit anomalies, cost of slow trading") | peer-reviewed | Implementation-speed decay; snippet. |
| https://www.quantvps.com/blog/guide-to-quantitative-trading-strategies-and-backtesting | industry | Robustness/plateau guidance; corroborates AlgoXpert; snippet. |
| https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf (PBO) | peer-reviewed | PBO primary; the PBO mechanism captured via search + DSR read; snippet. |

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026/2025):** "feature flag gradual rollout machine
   learning model in production A/B test reversible default off 2026";
   "incremental alpha signal live A/B test shadow mode canary deployment quant
   strategy avoid overfit 2025"; "deflated Sharpe ratio PBO promoting backtested
   factor to live trading overfit 2025 2026"; "McLean Pontiff factor anomaly
   decay out of sample post publication arbitrage 2024 2025". (-> arXiv:2512.11913
   alpha-decay 2025, Muravyev 2025, AlgoXpert arXiv:2603.09219, digitalapplied
   2026 playbook, the 2026 J.Econ.Dyn.&Control 52wh paper.)
2. **Last-2-year window:** covered by the above; recency findings reported below.
3. **Year-less canonical:** "dark launch canary release backward compatible code
   path no-op when flag disabled best practice"; "52-week high momentum live
   implementation turnover rebalance frequency George Hwang large cap". (-> the
   canonical Fowler dark-launching post, LaunchDarkly guide, George-Hwang 2004,
   DSR primary.)

### Recency scan (2024-2026) -- PERFORMED
Searched last-2-year window on (a) live factor promotion / A-B of alpha signals,
(b) factor/anomaly decay, (c) 52wh momentum, (d) feature-flag rollout. **Findings:**
1. **COMPLEMENTS (alpha decay, 2025):** "Not All Factors Crowd Equally"
   (arXiv:2512.11913, 2025) -- momentum alpha decays hyperbolically and has
   accelerated post-2015 with ETF-driven crowding, BUT crowded momentum carries
   LOWER crash risk (it self-reinforces); the crowded-crash risk is in REVERSAL
   factors, not momentum. Reinforces: a long-only momentum tilt is on the
   safer side of the crowding-tail-risk ledger; just monitor for decay.
2. **COMPLEMENTS (52wh, 2026):** "Proximity to the 52-week high and the
   risk-return trade-off" (J.Econ.Dyn.&Control, 2026) -- the 52wh effect induces
   cross-sectional heterogeneity in the risk-return trade-off (not a uniform
   premium). Refines, does not overturn, George-Hwang -> a GENTLE centered tilt
   (k=0.5), not a hard winner-take-all, is the right live form.
3. **STABLE (promotion discipline, 2026):** AlgoXpert (arXiv:2603.09219, 2026)
   re-confirms the IS->WFA->OOS chronological-gate + lock-after-WFA + plateau-over-
   cliff doctrine. Directly endorses "wire now (OFF), enable only after the
   Monday OOS baseline" as the disciplined path.
4. **STABLE (decay magnitude):** McLean-Pontiff (50-58% post-publication decay)
   remains the canonical haircut; recent work (Muravyev 2025; "race to exploit
   anomalies") attributes more of the residual decay to short-sale costs +
   slow-trading -- both LESS binding on a long-only large-cap book, but the
   directional message (haircut the historical edge) stands.
No 2024-2026 result OVERTURNS the wire-it-as-a-gated-default-OFF-tilt plan; the
recency literature strengthens both the "enable cautiously" and the "monitor
decay" sides.

### Key findings (per-claim, cited)
1. **Wire-now-enable-later behind a default-OFF flag is the canonical
   dark-launch pattern -- the disabled path must be inert.** "New code paths
   exist in the production environment but are not executed because the flags
   controlling them are disabled" (Source: LaunchDarkly,
   https://launchdarkly.com/blog/guide-to-dark-launching/). Fowler: switch
   "on and off easily in production ... if we do see a worrying impact on
   performance we switch it off" (https://martinfowler.com/bliki/DarkLaunching.html).
   => the phase-52.2 default-False kwarg + getattr-False settings read is textbook;
   the byte-identity test is the validation that the OFF path is truly inert.
2. **Promote a backtested edge to live only through chronological gates with
   params locked and an audit trail -- prefer robustness over peak metric.**
   "if WFA FAILs, the strategy does not proceed to OOS holdout"; params "locked
   after WFA with no further tuning"; prefer "plateau" over "cliff" zones
   (Source: AlgoXpert arXiv:2603.09219). => deferring the ENABLE to a separate
   post-Monday-baseline operator action (an OOS gate) is exactly this doctrine;
   k=0.5 (the milder of the two tested) is the plateau choice over k=1.0.
3. **Deflate the measured edge for the number of configs tried before enabling.**
   DSR corrects the observed Sharpe down for the effective number of independent
   trials + skew/kurtosis (Source: Bailey & Lopez de Prado,
   davidhbailey.com/dhbpapers/deflated-sharpe.pdf). The 52.1 +0.05 dSharpe was
   selected from {baseline, sector_neutral, vol_scaled, hi52_k0.5, hi52_k1.0} ->
   a multiple-testing haircut applies; the Monday live baseline is the OOS check
   that the edge survives deflation. This is the project's north-star metric
   (DSR 0.9984) discipline applied to the enable decision (NOT this wiring step).
4. **52wh is real and does NOT reverse long-run, but is weakest in large caps ->
   a gentle centered tilt is the right live form.** 52wh (6,?) winner-loser
   0.45%/mo t=2.00; "future returns forecast using the 52-week high do not reverse
   in the long run" (Source: George & Hwang 2004, J.Finance). The 52.1 replay
   already MEASURED +0.05 ann Sharpe at k=0.5 turnover-neutral on OUR S&P-500
   universe -> the large-cap-mute risk (Barroso-Wang) was already tested and the
   edge was small-but-positive, which is why it is being wired as an OPT-IN tilt,
   not made default.
5. **A long-only momentum tilt sits on the SAFER side of the crowding-tail-risk
   ledger; the residual risk is decay, not a crowded crash.** "Crowded momentum
   means many investors are reinforcing the trend, which tends to sustain rather
   than reverse it" -- 0.38x crash prob (Source: arXiv:2512.11913, 2025). The
   tail risk is in crowded REVERSAL factors, not momentum. => the regression risk
   of the tilt (once enabled) is gradual alpha decay (monitor), not a sudden
   crash injection. When OFF (this step) there is ZERO added risk by construction.

### Consensus vs debate (external)
- **Consensus:** (a) default-OFF feature flag + instant kill-switch is the
  standard way to dark-launch a change reversibly (Fowler, LaunchDarkly, MSFT,
  Unleash). (b) Promote backtested edges only through chronological OOS gates
  with locked params + audit trail and a robustness (plateau) bias (AlgoXpert,
  quantvps). (c) Deflate for multiple testing before believing a small edge
  (DSR/PBO, Bailey-Lopez de Prado). (d) Published anomalies decay ~50-58%
  post-publication (McLean-Pontiff).
- **Debate:** (a) 52wh additivity in LARGE caps -- George-Hwang say it dominates
  and does not reverse; Barroso-Wang/Du say it is largely a small-cap effect.
  RESOLVED for us by the 52.1 replay on the actual S&P-500 universe (small-but-
  positive). (b) Whether crowding can be TIMED -- arXiv:2512.11913 says no
  (crowding predicts tail risk, not returns). Not relevant to this wiring step.

### Pitfalls (from literature) -> applied to phase-52.2
1. **A flag that isn't truly inert when OFF is the classic dark-launch failure.**
   Mitigate: the post-pass is behind `if momentum_52wh_tilt and scored:`, the
   kwarg defaults False, the helper writes `composite_score_raw` only when it
   runs -> a test asserts flag-absent == flag-False AND no `composite_score_raw`
   on any OFF-path row. (Fowler/LaunchDarkly inert-path requirement.)
2. **Enabling on a single in-sample number is overfitting.** Mitigate: the ENABLE
   is DEFERRED to a post-Monday-baseline operator action (an OOS gate), and the
   +0.05 must be viewed through DSR (deflate for the 5 configs tried). This step
   does NOT enable. (AlgoXpert OOS-gate + DSR multiple-testing.)
3. **Momentum alpha decays (faster post-2015).** Mitigate: once enabled, monitor
   the live contribution; do not assume the 0.45%/mo (1963-2001) or even the
   52.1 +0.05 persists. (arXiv:2512.11913 + McLean-Pontiff.) Out of scope for the
   wiring step, but flag it for the enable runbook.
4. **Turnover creep if k too large.** 52.1 measured k=0.5 as turnover-neutral;
   the default is k=0.5 (not k=1.0). Keep the default mild. (52.1 verdict +
   plateau-over-cliff.)
5. **Look-ahead/causality:** `pct_to_52w_high` is computed from the trailing-252d
   close window inside screen_universe (causal); the tilt only re-weights the
   existing causal composite -> no new look-ahead introduced. (Confirmed
   screener.py:213.)

---

## Application to pyfinagent (external -> internal anchors)
- Dark-launch default-OFF inert path (Fowler/LaunchDarkly) -> `momentum_52wh_tilt:
  bool = False` kwarg (screener.py:~250) + `if momentum_52wh_tilt and scored:`
  guard at the insertion point (screener.py:473) + `getattr(settings,
  "momentum_52wh_tilt_enabled", False)` at the call-site (autonomous_loop.py:649).
- OOS-gate / lock-after-WFA / plateau (AlgoXpert) -> ENABLE deferred to the
  post-Monday-baseline operator action; default k=0.5 (the milder tested value).
- DSR multiple-testing deflation (Bailey-Lopez de Prado) -> the 52.1 +0.05 is one
  of 5 configs; deflate before the enable decision (enable step, not this step).
- 52wh formula + no-reversal (George-Hwang) -> reuse the prod
  `pct_to_52w_high` (screener.py:213-214) verbatim; the helper replicates the
  52.1 `hi52_tilt_basket` (sector_neutral_replay.py:123-138) so live == measured.
- Momentum crowding = lower crash risk (arXiv:2512.11913) -> the only post-enable
  regression risk is gradual decay (monitor), not a crash; OFF = zero risk now.

---

## Research Gate Checklist
Hard blockers (gate_passed=false if any unchecked):
- [x] >=5 authoritative external sources READ IN FULL (6: Fowler dark-launching,
  LaunchDarkly dark-launch guide, AlgoXpert arXiv:2603.09219 (HTML), Bailey-Lopez
  de Prado DSR PDF, George-Hwang 2004 PDF (pdfplumber), arXiv:2512.11913 alpha-
  decay 2025 (HTML))
- [x] 10+ unique URLs total (16 incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported (alpha-decay 2025, 52wh-2026,
  AlgoXpert 2026, Muravyev 2025)
- [x] Full papers/pages read (pdfplumber for binary George-Hwang; arXiv HTML for
  the two arXiv papers; full WebFetch for the blogs), not abstracts
- [x] file:line anchors for every internal claim (screener.py:213-214/228/249-273/
  287-409/434-440/448-472/473/474-475/478-488/491-550; autonomous_loop.py:25/638-666/
  649; settings.py:334-338; sector_neutral_replay.py:123-138)

Soft checks:
- [x] Internal exploration covered rank_candidates + screen_universe feature flow +
  both existing gated post-passes + the call-site + the settings block + the 52.1
  replay reference
- [x] Contradictions/consensus noted (52wh large-cap debate resolved by the 52.1
  replay; crowding-timing debate noted)
- [x] 3-variant query discipline visible (current-year / last-2-year / year-less)

## GATE ENVELOPE
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 9,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "gate_passed": true
}
```
`gate_passed: true` -- 6 sources read in full (floor 5); recency scan performed;
3-variant queries run; 15 unique URLs; internal audit pinned to file:line across
screener.py + autonomous_loop.py + settings.py + sector_neutral_replay.py. The
EXACT insertion point is screener.py:473 (after sector_neutral, before the final
sort); the flag pattern mirrors multidim_momentum (default-False kwarg + getattr-
False settings + call-site arg); `pct_to_52w_high` is already on every screen_data
row (screener.py:228) so no threading is needed; byte-identity when OFF is proven
by a flag-absent==flag-False test plus the absence of `composite_score_raw` on
OFF-path rows. No reason NOT to wire it: when OFF (the default + this step) the
engine is byte-identical, so zero regression risk; the only risk (gradual momentum
decay once enabled) belongs to the DEFERRED enable decision, not this step.
