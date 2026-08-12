# Deep-research prompt — make the candidate picker actually earn

Paste this as the task. It is written to be handed to a research agent or a fresh
session. **Everything under MEASURED is already established — do not re-derive it from
scratch, but re-verify any single number before you restate it as your own.**

---

## Your job

Decide **what change to the candidate pipeline produces the most risk-adjusted P&L**,
and prove it out-of-sample. **Do not assume the answer is a better ranking.** Your
first deliverable is a decision about *where the binding constraint actually is*.

## MEASURED (pyfinagent, 2026-08-11/12 — verified against source and live logs)

**Capital**
- NAV **$23,881.12**, invested **$1,060.48 (4.4%)**, idle cash **95.6%**. One position (NTAP).

**Funnel, one cycle**
```
583 universe → 577 pass filters → 10 scored → 5 deep-analysed → 1 signal → 0 trades
```
- Deep analysis touches **0.87%** of qualifying names.
- At the stated 90-day BUY rate of 21.1%, expected BUYs ≈ **1.05 per cycle** *before*
  the risk gate.
- Last night the risk gate rejected **3 of 3**: `REJECT/HIGH/position=0%` ×2 and
  `Lite risk judge NTAP: REJECT/EXTREME`.

**Ranking is frozen** — `backend/tools/screener.py:299-305`
```python
score = mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25
```
- All three terms are trailing returns over 21/63/126 sessions; one day moves each by
  ~1/21, 1/63, 1/126 of its window. Ranks are stickier than levels (a rank change
  needs a *crossing*).
- **Result: 8 distinct tickers analysed across 8 cycles, out of 583.** DELL/HPE/NTAP/
  PANW in 8 of 8. Four consecutive days byte-identical. **88% one sector** (Technology).

**The declared weights are not the effective weights**
- No cross-sectional standardisation on the live path: `_zscore` is defined at
  `screener.py:532` but called **only** at `:607-610`, inside a dark path.
- So weights apply to **raw** returns of differing dispersion; 6-month has ~2.4× the
  dispersion of 1-month. **The smallest declared weight contributes the most ranking
  variance.** Reweighting without standardising first tunes a disconnected knob.

**Every alternative signal is structurally blind** — `autonomous_loop.py:749,769,833,860,884,910,938,967`
- All eight overlays (news, PEAD, insider, options flow, social velocity, peer
  lead-lag, M&A pre-announce, analyst revisions) slice
  `screen_data[: 2 * paper_screen_top_n]`.
- `screen_universe` returns **unsorted** (`screener.py:147 → :240 → :246`), and nothing
  reorders it before the slice.
- **So the overlays see the head of the universe list, not the top of any ranking.**
  They are score *adjustments inside a set momentum already chose*, never *entry
  paths*. ~557 of 577 names can never receive one.
- The one overlay that ran, failed: `400 raw → 100 deduped headlines`, then
  `parse failed` ×2 on an **empty model response** → `0 ticker signals`.

**Literature already established (86.59 gate, brief on disk)**
- **Gârleanu–Pedersen**: a slow predictor *correctly* yields low turnover. Low turnover
  is not itself the defect — **the absence of any fast signal is**.
- **Novy-Marx & Velikov**: below ~50% one-sided monthly turnover, most strategies
  survive costs; few above. **Target a fast signal with bounded turnover, not churn.**
- Prefer **slate composition** (min-K sector round-robin) over **score mutation**
  (soft diversity), because mutating the score contaminates the DSR/PBO gates any
  change must still clear.
- **Declared gap**: residual/idiosyncratic momentum was NOT researched — no fetchable
  source found. This is likely the strongest candidate for the missing fast signal.
  **Close this gap first.**

## Answer these in order. Do not skip to Q3.

**Q1 — WHERE IS THE BINDING CONSTRAINT?** Decide between, with evidence:
  (a) **selection quality** — the 5 names chosen are the wrong 5;
  (b) **throughput** — 5/577 is too few for any hit rate to matter;
  (c) **the risk gate** — good candidates are produced and then rejected;
  (d) **capital deployment** — even accepted BUYs cannot consume 95.6% cash at the
      current position sizing.
  Quantify each. **If the answer is (c) or (d), say so plainly and stop optimising the
  ranking** — the standing prohibition on loosening the risk judge is unchanged, so
  (c) becomes "why is the evidence reaching the judge producing EXTREME/HIGH", which is
  a *pipeline-quality* question, not a threshold question.

**Q2 — WHAT IS THE HIGHEST RETURN PER UNIT OF COST?** Rank candidate changes by
  expected P&L per dollar and per engineering-hour. Note explicitly that redirecting the
  overlay slice from head-of-universe to top-by-score is **~$0 marginal cost** — the
  API and LLM calls are *already being paid for*, on the wrong 20 names.

**Q3 — ONLY THEN: what should the picker actually compute?** For each proposal give
  the formula, the data it needs, its expected one-sided monthly turnover, and how it
  clears **DSR ≥ 0.95** and **PBO ≤ 0.20**.

## Constraints — these are hard

- **Paper trading only.** No flag promotions, no `.env` writes, no manual cycles.
- **Never loosen a gate or weaken an assertion to get green.** A picker producing
  better candidates is the goal; a gate that rejects fewer is not.
- **Metered spend is a STOP.** Overlays are paid calls; widening naively multiplies
  cost. Report the per-cycle cost delta of anything you propose.
- Three diversity mitigations already exist and are **dark**
  (`sector_neutral_momentum_enabled`, `paper_soft_sector_diversity_enabled`,
  `paper_min_k_sectors_analyzed`). **Measure what they do before building anything new.**
  Do not rebuild an existing mitigation.

## Output requirements

- **Every number derived from a command, with the rule stated beside it.** A count with
  no stated population rule is not evidence.
- **Any claim of absence gets the same proof as a claim of presence.** "X does not
  exist" from one failed probe is not a finding.
- **State what you could not verify.** A declared gap beats a padded section.
- Out-of-sample or it did not happen. In-sample improvement to a momentum screen is
  the single easiest thing to fake in this entire domain.

## Traps specific to this codebase

- Two files are named `autonomous_loop.py`; line numbers above are
  `backend/services/`.
- The shell's `grep` is a function wrapping `ugrep`; `subprocess` gets
  `/usr/bin/grep`. They disagree on binary files. Pin the binary in anything you publish.
- Two production comments cite arXiv papers from unrelated domains (86.61) — do not
  inherit their claimed guarantees.
- Reading `backend/.env` is denied; read live values from the **running process**
  (`GET /api/settings/`), not from defaults in `settings.py`.
